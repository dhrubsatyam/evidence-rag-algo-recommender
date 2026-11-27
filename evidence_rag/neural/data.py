"""Dataset and weak-label construction for the neural family tagger.

Labels are derived from algorithm-name mentions in the title and
abstract. An abstract can carry several family labels; families with no
mention get a 0. This is weak supervision: a mention is not proof the
paper is *about* that algorithm, so the model learns mention patterns,
not ground-truth algorithm relevance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

FAMILIES = [
    "linear",
    "instance_kernel",
    "tree_ensemble",
    "neural",
]

FAMILY_PATTERNS = {
    "linear": [
        "logistic regression",
        "regularized logistic regression",
        "linear discriminant analysis",
        "naive bayes",
    ],
    "instance_kernel": [
        "k-nearest neighbors",
        "k-nearest neighbour",
        "support vector machine",
        "support vector machines",
    ],
    "tree_ensemble": [
        "decision tree",
        "decision trees",
        "random forest",
        "random forests",
        "extra trees",
        "gradient boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    ],
    "neural": [
        "multilayer perceptron",
        "neural network",
        "neural networks",
        "deep learning",
        "deep neural",
    ],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def weak_labels(title: str, abstract: str) -> np.ndarray:
    """Multi-hot family labels from keyword mentions. Returns float32 vector."""
    blob = f"{title} {abstract}".lower()
    labels = np.zeros(len(FAMILIES), dtype=np.float32)
    for idx, family in enumerate(FAMILIES):
        for pattern in FAMILY_PATTERNS[family]:
            if re.search(re.escape(pattern), blob):
                labels[idx] = 1.0
                break
    return labels


def load_corpus(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_vocab(records: list[dict], max_vocab: int = 15000) -> dict[str, int]:
    """Most-frequent tokens -> id. id 0 is reserved for padding."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for record in records:
        counts.update(tokenize(record["title"] + " " + record["abstract"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(max_vocab - 2):
        vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = 200) -> list[int]:
    ids = [vocab.get(tok, 1) for tok in tokenize(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids


def make_arrays(
    records: list[dict],
    vocab: dict[str, int],
    max_len: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [encode(r["title"] + " " + r["abstract"], vocab, max_len) for r in records],
        dtype=np.int64,
    )
    Y = np.stack([weak_labels(r["title"], r["abstract"]) for r in records])
    return X, Y
