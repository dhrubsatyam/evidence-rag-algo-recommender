"""Run the phases in order.

Phases:
  download    arXiv abstract subset
  prepare     clean, dedupe, leakage screen
  index       BM25 tokens and dense embeddings
  retrieve    development-set retrieval traces
  generate    with-retrieval and no-retrieval answers
  probes      clarification, out-of-scope, citation checks
  analyze     automatic metrics
  feasibility adaptation go/no-go record
"""

from __future__ import annotations

import argparse

from evidence_rag.config import ensure_dirs
from evidence_rag.download import download
from evidence_rag.evaluate import analyze, retrieval_traces, run_generation, run_probes
from evidence_rag.feasibility import decide
from evidence_rag.index_corpus import build_index
from evidence_rag.prepare import prepare

PHASES = ["download", "prepare", "index", "retrieve", "generate", "probes", "analyze", "feasibility"]


def run(phases: list[str], generator: str, limit: int | None, include_heldout: bool) -> None:
    ensure_dirs()
    for phase in phases:
        print(f"\n=== {phase} ===")
        if phase == "download":
            manifest = download()
            print(f"raw records: {manifest['record_count_raw']}")
        elif phase == "prepare":
            report = prepare()
            print(f"kept records: {report['kept_records']}")
        elif phase == "index":
            meta = build_index()
            print(f"indexed documents: {meta['documents']}")
        elif phase == "retrieve":
            info = retrieval_traces(limit, include_heldout)
            print(f"traces: {info['cases']}")
        elif phase == "generate":
            info = run_generation(generator, limit, include_heldout)
            print(f"generation rows: {info['rows']}")
        elif phase == "probes":
            info = run_probes(generator)
            failed = [item["probe_id"] for item in info["results"] if not item["ok"]]
            print(f"probes failed: {failed or 'none'}")
        elif phase == "analyze":
            summary = analyze(generator)
            print(summary)
        elif phase == "feasibility":
            decision = decide()
            print(decision["decision"])
        else:
            raise SystemExit(f"Unknown phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-grounded algorithm recommendation pipeline")
    parser.add_argument("--phases", default=",".join(PHASES), help="Comma-separated phase names")
    parser.add_argument("--generator", choices=["local_lm", "extractive"], default="local_lm")
    parser.add_argument("--limit", type=int, default=None, help="Cap development cases for a short run")
    parser.add_argument(
        "--include-heldout",
        action="store_true",
        help="Also run held-out cases. Leave this off until the rubric is frozen.",
    )
    args = parser.parse_args()
    phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    run(phases, args.generator, args.limit, args.include_heldout)


if __name__ == "__main__":
    main()
