"""Project paths and config."""

from __future__ import annotations

from pathlib import Path

from evidence_rag.utils import ROOT, load_json

CONFIG_PATH = ROOT / "configs" / "default.json"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVAL = DATA / "eval"
OUTPUTS = ROOT / "outputs"
INDEX = DATA / "index"


def load_config() -> dict:
    return load_json(CONFIG_PATH)


def ensure_dirs() -> None:
    for path in (RAW, PROCESSED, EVAL, OUTPUTS, INDEX, RAW / "pages"):
        path.mkdir(parents=True, exist_ok=True)
