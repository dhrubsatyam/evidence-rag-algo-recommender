"""Shared helpers that do not load models."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "is", "are", "was", "were", "be", "as", "at", "that", "this", "it",
    "we", "our", "their", "using", "used", "via", "into", "than", "then", "not",
    "can", "may", "also", "such", "these", "those", "its", "which", "between",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [tok for tok in tokens if tok not in _STOP and len(tok) > 1]


def canonical_arxiv_id(value: str) -> str:
    raw = (value or "").strip()
    raw = raw.split("/abs/")[-1]
    raw = raw.split("?")[0].strip("/")
    match = re.match(r"(.+?)v\d+$", raw)
    if match:
        raw = match.group(1)
    return raw


def find_arxiv_ids(text: str) -> list[str]:
    found = re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text or "")
    seen: list[str] = []
    for item in found:
        cid = canonical_arxiv_id(item)
        if cid not in seen:
            seen.append(cid)
    return seen


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").lower()).strip()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
