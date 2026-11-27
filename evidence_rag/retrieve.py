"""Hybrid retrieval: BM25, dense cosine, reciprocal rank fusion."""

from __future__ import annotations

import json

import numpy as np
from rank_bm25 import BM25Okapi

from evidence_rag.config import INDEX, PROCESSED, ROOT, load_config
from evidence_rag.utils import tokenize

_STATE: dict | None = None


def _load_state() -> dict:
    global _STATE
    if _STATE is not None:
        return _STATE
    config = load_config()
    rows = []
    with (PROCESSED / "corpus.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    by_id = {row["arxiv_id"]: row for row in rows}
    ids = json.loads((INDEX / "ids.json").read_text())
    embeddings = np.load(INDEX / "embeddings.npy")
    tokenized = []
    with (INDEX / "tokens.jsonl").open() as handle:
        for line in handle:
            tokenized.append(json.loads(line)["tokens"])
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config["embedding_model"])
    state = {
        "config": config,
        "rows": rows,
        "by_id": by_id,
        "ids": ids,
        "embeddings": embeddings,
        "bm25": BM25Okapi(tokenized),
        "model": model,
    }
    state["neural"] = _load_neural(ids)
    _STATE = state
    return _STATE


def _load_neural(ids: list[str]) -> dict | None:
    """Load the from-scratch family tagger and per-document family scores.

    Returns None when the checkpoint or score file is missing, so
    retrieval degrades gracefully to the two-signal hybrid.
    """
    ckpt_path = ROOT / "outputs" / "neural" / "model.pt"
    scores_path = ROOT / "outputs" / "neural" / "family_scores.json"
    if not ckpt_path.exists() or not scores_path.exists():
        return None
    try:
        import torch

        from evidence_rag.neural.data import encode
        from evidence_rag.neural.model import FamilyTagger
        from evidence_rag.neural.predict import load_model

        model, vocab, families, cfg = load_model(ckpt_path)
        bundle = json.loads(scores_path.read_text())
        score_map = bundle["scores"]
        matrix = np.array([score_map.get(doc_id, [0.0] * len(families)) for doc_id in ids], dtype=np.float32)
        return {
            "model": model,
            "vocab": vocab,
            "families": families,
            "max_len": int(cfg["max_len"]),
            "doc_matrix": matrix,
        }
    except Exception:
        return None


def _neural_ranking(query: str, state: dict, pool: int) -> tuple[list[str], np.ndarray] | None:
    neural = state.get("neural")
    if neural is None:
        return None
    import torch

    from evidence_rag.neural.data import encode

    ids_tensor = torch.tensor([encode(query, neural["vocab"], neural["max_len"])], dtype=torch.long)
    with torch.no_grad():
        query_probs = neural["model"].predict_proba(ids_tensor)[0].numpy()
    # Family-agreement score: dot product between the query's predicted
    # family distribution and each document's predicted family distribution.
    agreement = neural["doc_matrix"] @ query_probs
    order = np.argsort(-agreement)[:pool]
    ranked_ids = [state["ids"][int(i)] for i in order]
    return ranked_ids, agreement


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def retrieve(query: str, top_k: int | None = None, signals: set[str] | None = None) -> dict:
    """Hybrid retrieval. `signals` optionally restricts the RRF inputs to
    a subset of {"bm25", "dense", "neural"} for ablation runs."""
    state = _load_state()
    config = state["config"]
    top_k = top_k or int(config["top_k"])
    pool = int(config["candidate_pool"])
    query_tokens = tokenize(query)
    bm25_scores = state["bm25"].get_scores(query_tokens)
    bm25_order = np.argsort(-bm25_scores)[:pool]
    dense_query = state["model"].encode([query], normalize_embeddings=True)[0]
    cosine = state["embeddings"] @ dense_query
    dense_order = np.argsort(-cosine)[:pool]
    ids = state["ids"]
    lexical_ids = [ids[int(i)] for i in bm25_order]
    dense_ids = [ids[int(i)] for i in dense_order]
    rankings = []
    if signals is None or "bm25" in signals:
        rankings.append(lexical_ids)
    if signals is None or "dense" in signals:
        rankings.append(dense_ids)
    use_neural = config.get("use_neural_signal") and (signals is None or "neural" in signals)
    neural_info = _neural_ranking(query, state, pool) if use_neural else None
    if neural_info is not None:
        neural_ids, neural_scores = neural_info
        rankings.append(neural_ids)
    else:
        neural_scores = None
    fused = reciprocal_rank_fusion(rankings, int(config["rrf_k"]))
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    id_to_pos = {doc_id: pos for pos, doc_id in enumerate(ids)}
    results = []
    for doc_id, rrf_score in ranked:
        pos = id_to_pos[doc_id]
        row = state["by_id"][doc_id]
        item = {
            "arxiv_id": doc_id,
            "title": row["title"],
            "abstract": row["abstract"],
            "primary_category": row["primary_category"],
            "published": row["published"],
            "bm25": float(bm25_scores[pos]),
            "cosine": float(cosine[pos]),
            "rrf": float(rrf_score),
            "url": f"https://arxiv.org/abs/{doc_id}",
        }
        if neural_scores is not None:
            item["neural_family_agreement"] = float(neural_scores[pos])
        results.append(item)
    max_cosine = max((item["cosine"] for item in results), default=0.0)
    return {
        "query": query,
        "top_k": top_k,
        "max_cosine": max_cosine,
        "low_evidence": max_cosine < float(config["low_evidence_cosine"]),
        "neural_signal": neural_scores is not None,
        "results": results,
    }
