"""Retrieval-signal ablation on the development cases.

Compares BM25-only, dense-only, the two-signal hybrid, and the full
three-signal hybrid (with the neural family tagger) using the
extractive generator, which is deterministic and needs no model
download. Results land in outputs/ablation.json.

Usage:
    python -m evidence_rag.ablate
    python -m evidence_rag.pipeline --phases ablate
"""

from __future__ import annotations

import json
import time

from evidence_rag.config import OUTPUTS, ensure_dirs
from evidence_rag.evaluate import load_cases
from evidence_rag.generate import answer
from evidence_rag.utils import write_json

VARIANTS: dict[str, set[str] | None] = {
    "bm25_only": {"bm25"},
    "dense_only": {"dense"},
    "hybrid_lexical_dense": {"bm25", "dense"},
    "hybrid_full_neural": {"bm25", "dense", "neural"},
}


def run_ablation(generator: str = "extractive") -> dict:
    ensure_dirs()
    cases = load_cases(include_heldout=False)
    summary = {"cases": len(cases), "generator": generator, "variants": {}}
    for variant, signals in VARIANTS.items():
        started = time.perf_counter()
        rows = []
        for case in cases:
            result = answer(
                problem=case["problem"],
                constraints=case["constraints"],
                task=case["task"],
                generator=generator,
                use_retrieval=True,
                signals=signals,
            )
            rows.append({"case_id": case["case_id"], **result})
        elapsed = time.perf_counter() - started
        cited = sum(len(r.get("cited_ids") or []) for r in rows)
        unsupported = sum(len(r.get("unsupported_ids") or []) for r in rows)
        low_evidence = sum(1 for r in rows if r.get("low_evidence"))
        ok = sum(1 for r in rows if r.get("status") == "ok")
        insufficient = sum(1 for r in rows if r.get("status") == "insufficient_evidence")
        entry = {
            "signals": sorted(signals) if signals else "all",
            "cases": len(rows),
            "status_ok": ok,
            "status_insufficient_evidence": insufficient,
            "cited_ids": cited,
            "unsupported_ids": unsupported,
            "low_evidence_cases": low_evidence,
            "seconds": round(elapsed, 2),
            "per_case": [
                {
                    "case_id": r["case_id"],
                    "status": r.get("status"),
                    "cited_ids": r.get("cited_ids") or [],
                    "low_evidence": r.get("low_evidence"),
                    "top_ids": [item["arxiv_id"] for item in (r.get("retrieval") or [])[:5]],
                }
                for r in rows
            ],
        }
        summary["variants"][variant] = entry
        print(
            f"{variant:22s} ok={ok:2d} cited={cited:2d} unsupported={unsupported} "
            f"low_evidence={low_evidence} {elapsed:.1f}s"
        )
    path = OUTPUTS / "ablation.json"
    write_json(path, summary)
    return {"variants": list(summary["variants"]), "path": str(path)}


if __name__ == "__main__":
    run_ablation()
