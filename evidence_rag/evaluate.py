"""Run development-set comparisons and summarize what was measured.

Held-out cases stay sealed unless --include-heldout is passed. The Week 3
checkpoint asks the team not to view held-out outputs before the rubric is frozen.
"""

from __future__ import annotations

import json
from collections import Counter

from evidence_rag.config import EVAL, OUTPUTS, ensure_dirs
from evidence_rag.generate import answer
from evidence_rag.utils import write_json


def load_cases(include_heldout: bool = False) -> list[dict]:
    cases = []
    with (EVAL / "cases.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            if case["split"] == "heldout" and not include_heldout:
                continue
            cases.append(case)
    return cases


def load_probes() -> list[dict]:
    probes = []
    with (EVAL / "probes.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                probes.append(json.loads(line))
    return probes


def run_generation(generator: str, limit: int | None, include_heldout: bool) -> dict:
    ensure_dirs()
    cases = load_cases(include_heldout)
    if limit is not None:
        cases = cases[:limit]
    rows = []
    for case in cases:
        query = case["problem"]
        constraints = case["constraints"]
        for mode, use_retrieval in (("with_retrieval", True), ("without_retrieval", False)):
            result = answer(
                problem=query,
                constraints=constraints,
                task=case["task"],
                generator=generator,
                use_retrieval=use_retrieval,
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "mode": mode,
                    **result,
                }
            )
            print(f"{case['case_id']} {mode} {result.get('status')} {result.get('generator')}")
    path = OUTPUTS / f"generations_{generator}.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {"cases": len(cases), "rows": len(rows), "path": str(path)}


def run_probes(generator: str) -> dict:
    ensure_dirs()
    results = []
    for probe in load_probes():
        result = answer(
            problem=probe["problem"],
            constraints=probe.get("constraints") or {},
            task=probe["task"],
            generator="extractive",
            use_retrieval=probe["expect"] == "citations_resolve",
        )
        ok = _probe_ok(probe, result)
        results.append({"probe_id": probe["probe_id"], "expect": probe["expect"], "ok": ok, "status": result.get("status"), "questions": result.get("questions", [])})
        print(f"probe {probe['probe_id']} ok={ok} status={result.get('status')}")
    write_json(OUTPUTS / "probes.json", {"results": results, "passed": all(item["ok"] for item in results)})
    return {"results": results}


def _probe_ok(probe: dict, result: dict) -> bool:
    if probe["expect"] == "needs_clarification":
        return result.get("status") == "needs_clarification" and len(result.get("questions") or []) >= 1
    if probe["expect"] == "insufficient_evidence":
        return result.get("status") == "insufficient_evidence"
    if probe["expect"] == "citations_resolve":
        return result.get("unsupported_ids") == [] and result.get("citation_resolution_rate", 0) == 1.0
    return False


def retrieval_traces(limit: int | None, include_heldout: bool) -> dict:
    from evidence_rag.retrieve import retrieve

    ensure_dirs()
    cases = load_cases(include_heldout)
    if limit is not None:
        cases = cases[:limit]
    path = OUTPUTS / "retrieval_traces.jsonl"
    with path.open("w") as handle:
        for case in cases:
            trace = retrieve(case["problem"])
            slim = {
                "case_id": case["case_id"],
                "split": case["split"],
                "max_cosine": trace["max_cosine"],
                "low_evidence": trace["low_evidence"],
                "results": [
                    {"arxiv_id": item["arxiv_id"], "title": item["title"], "cosine": item["cosine"], "bm25": item["bm25"], "rrf": item["rrf"]}
                    for item in trace["results"]
                ],
            }
            handle.write(json.dumps(slim) + "\n")
    return {"cases": len(cases), "path": str(path)}


def analyze(generator: str = "extractive") -> dict:
    gen_path = OUTPUTS / f"generations_{generator}.jsonl"
    if not gen_path.exists():
        raise RuntimeError(f"No generation file at {gen_path.name}. Run the generate phase first.")
    rows = [json.loads(line) for line in gen_path.read_text().splitlines() if line.strip()]
    by_mode = Counter(row["mode"] for row in rows)
    unsupported = sum(len(row.get("unsupported_ids") or []) for row in rows)
    cited = sum(len(row.get("cited_ids") or []) for row in rows)
    latencies = [row["latency_seconds"] for row in rows if row.get("latency_seconds") is not None]
    generators = Counter(row.get("generator") for row in rows)
    statuses = Counter(row.get("status") for row in rows)
    paired = {}
    for row in rows:
        paired.setdefault(row["case_id"], {})[row["mode"]] = row
    citation_gain = 0
    for modes in paired.values():
        with_ids = len((modes.get("with_retrieval") or {}).get("cited_ids") or [])
        without_ids = len((modes.get("without_retrieval") or {}).get("cited_ids") or [])
        if with_ids > without_ids:
            citation_gain += 1
    summary = {
        "rows": len(rows),
        "cases": len(paired),
        "modes": dict(by_mode),
        "generators": dict(generators),
        "statuses": dict(statuses),
        "cited_ids": cited,
        "unsupported_ids": unsupported,
        "cases_where_retrieval_added_citations": citation_gain,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "note": (
            "Human 0-2 relevance and suitability scores are not computed here. "
            "This file records automatic checks only: identifier resolution, "
            "latency, and whether retrieval changed the citation set."
        ),
    }
    summary["generation_file"] = gen_path.name
    write_json(OUTPUTS / f"metrics_{generator}.json", summary)
    return summary
