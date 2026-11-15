"""Hybrid retrieval: BM25, dense cosine, reciprocal rank fusion."""

from __future__ import annotations

import json

import numpy as np
from rank_bm25 import BM25Okapi

from evidence_rag.config import INDEX, PROCESSED, load_config
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
    _STATE = {
        "config": config,
        "rows": rows,
        "by_id": by_id,
        "ids": ids,
        "embeddings": embeddings,
        "bm25": BM25Okapi(tokenized),
        "model": model,
    }
    return _STATE


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def retrieve(query: str, top_k: int | None = None) -> dict:
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
    fused = reciprocal_rank_fusion([lexical_ids, dense_ids], int(config["rrf_k"]))
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    id_to_pos = {doc_id: pos for pos, doc_id in enumerate(ids)}
    results = []
    for doc_id, rrf_score in ranked:
        pos = id_to_pos[doc_id]
        row = state["by_id"][doc_id]
        results.append(
            {
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
        )
    max_cosine = max((item["cosine"] for item in results), default=0.0)
    return {
        "query": query,
        "top_k": top_k,
        "max_cosine": max_cosine,
        "low_evidence": max_cosine < float(config["low_evidence_cosine"]),
        "results": results,
    }
