"""Lexical and embedding indexes over the cleaned corpus."""

from __future__ import annotations

import json

import numpy as np
from rank_bm25 import BM25Okapi

from evidence_rag.config import INDEX, PROCESSED, ensure_dirs, load_config
from evidence_rag.utils import tokenize, write_json


def load_corpus() -> list[dict]:
    path = PROCESSED / "corpus.jsonl"
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("Clean corpus is empty. Run the prepare phase after download.")
    return rows


def build_index(config: dict | None = None) -> dict:
    ensure_dirs()
    config = config or load_config()
    rows = load_corpus()
    documents = [f"{row['title']}. {row['abstract']}" for row in rows]
    tokenized = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    # BM25Okapi stores the corpus; we persist tokens and rebuild on load.
    # Rebuilding 800 short documents is cheap and avoids pickling issues.
    token_path = INDEX / "tokens.jsonl"
    with token_path.open("w") as handle:
        for row, tokens in zip(rows, tokenized):
            handle.write(json.dumps({"arxiv_id": row["arxiv_id"], "tokens": tokens}) + "\n")

    model_name = config["embedding_model"]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        documents,
        batch_size=int(config["embedding_batch_size"]),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(INDEX / "embeddings.npy", embeddings)
    ids = [row["arxiv_id"] for row in rows]
    (INDEX / "ids.json").write_text(json.dumps(ids))
    meta = {
        "documents": len(rows),
        "embedding_model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "lexical": "BM25Okapi on title + abstract",
        "bm25_ready": bm25 is not None,
    }
    write_json(INDEX / "index_meta.json", meta)
    return meta
