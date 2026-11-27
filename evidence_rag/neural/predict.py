"""Score text with the trained family tagger.

Usage:
    python -m evidence_rag.neural.predict "Title. Abstract text..."
    python -m evidence_rag.neural.predict --file some_abstract.txt
    python -m evidence_rag.neural.predict --top corpus_top.txt   # score all corpus records
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evidence_rag.neural.data import encode
from evidence_rag.neural.model import FamilyTagger

CKPT = Path("outputs/neural/model.pt")


def load_model(path: Path = CKPT):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    model = FamilyTagger(
        vocab_size=len(bundle["vocab"]),
        num_labels=cfg["num_labels"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle["vocab"], bundle["families"], cfg


def score_texts(texts: list[str], model, vocab: dict, max_len: int) -> torch.Tensor:
    ids = torch.tensor([encode(t, vocab, max_len) for t in texts], dtype=torch.long)
    with torch.no_grad():
        return model.predict_proba(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score abstracts with the neural family tagger")
    parser.add_argument("text", nargs="?", help="Title + abstract text to score")
    parser.add_argument("--file", help="Read text from a file")
    parser.add_argument("--corpus", action="store_true", help="Score every record in data/processed/corpus.jsonl")
    args = parser.parse_args()

    if not CKPT.exists():
        raise SystemExit("No checkpoint found. Run: python -m evidence_rag.neural.train")

    model, vocab, families, cfg = load_model()

    if args.corpus:
        from evidence_rag.neural.data import load_corpus

        records = load_corpus("data/processed/corpus.jsonl")
        texts = [r["title"] + " " + r["abstract"] for r in records]
        probs = score_texts(texts, model, vocab, cfg["max_len"])
        scores = {}
        for record, row in zip(records, probs.tolist()):
            record["neural_family_scores"] = {
                fam: round(score, 4) for fam, score in zip(families, row)
            }
            scores[record["arxiv_id"]] = [round(s, 6) for s in row]
        out = Path("outputs/neural/corpus_scores.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        compact = {"families": families, "scores": scores}
        Path("outputs/neural/family_scores.json").write_text(json.dumps(compact))
        print(f"wrote {len(records)} scored records to {out}")
        print(f"wrote compact family scores to outputs/neural/family_scores.json")
        return

    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        parser.error("provide text, --file, or --corpus")

    probs = score_texts([text], model, vocab, cfg["max_len"])[0]
    for fam, score in zip(families, probs.tolist()):
        print(f"{fam:16s} {score:.4f}")


if __name__ == "__main__":
    main()
