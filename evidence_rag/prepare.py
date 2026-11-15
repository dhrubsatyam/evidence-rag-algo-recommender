"""Clean the abstract corpus and write a data-quality report."""

from __future__ import annotations

import json

import pandas as pd

from evidence_rag.config import EVAL, PROCESSED, RAW, ensure_dirs, load_config
from evidence_rag.utils import jaccard, normalize_title, tokenize, write_json


def load_raw_records() -> list[dict]:
    path = RAW / "records.jsonl"
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare(config: dict | None = None) -> dict:
    ensure_dirs()
    config = config or load_config()
    allowed = set(config["ml_categories"])
    min_chars = int(config["min_abstract_chars"])
    raw = load_raw_records()

    missing = {"arxiv_id": 0, "title": 0, "abstract": 0, "published": 0, "primary_category": 0}
    withdrawn = 0
    short_abstract = 0
    outside_category = 0
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    duplicate_ids = 0
    duplicate_titles = 0
    kept: list[dict] = []

    for record in raw:
        for field in missing:
            if not record.get(field):
                missing[field] += 1
        comment = (record.get("comment") or "").lower()
        title = record.get("title") or ""
        if "withdrawn" in comment or title.lower().startswith("withdrawn"):
            withdrawn += 1
            continue
        abstract = " ".join((record.get("abstract") or "").split())
        if len(abstract) < min_chars:
            short_abstract += 1
            continue
        categories = set(record.get("categories") or [])
        primary = record.get("primary_category") or ""
        if primary not in allowed and not (categories & allowed):
            outside_category += 1
            continue
        arxiv_id = record.get("arxiv_id") or ""
        if not arxiv_id or not title:
            continue
        if arxiv_id in seen_ids:
            duplicate_ids += 1
            continue
        title_key = normalize_title(title)
        if title_key in seen_titles:
            duplicate_titles += 1
            continue
        seen_ids.add(arxiv_id)
        seen_titles.add(title_key)
        kept.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": "; ".join(record.get("authors") or []),
                "primary_category": primary,
                "categories": " ".join(record.get("categories") or []),
                "published": record.get("published") or "",
                "updated": record.get("updated") or "",
                "doi": record.get("doi") or "",
                "source_query": record.get("source_query") or "",
                "source": "arxiv_api",
            }
        )

    frame = pd.DataFrame(kept)
    out_path = PROCESSED / "corpus.jsonl"
    with out_path.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
    frame.to_csv(PROCESSED / "corpus.csv", index=False)

    leakage = _leakage_report(kept)
    report = {
        "raw_records": len(raw),
        "kept_records": len(kept),
        "missing_field_counts": missing,
        "excluded_withdrawn": withdrawn,
        "excluded_short_abstract": short_abstract,
        "excluded_outside_ml_categories": outside_category,
        "duplicate_ids_dropped": duplicate_ids,
        "duplicate_titles_dropped": duplicate_titles,
        "inclusion_rules": {
            "categories": sorted(allowed),
            "min_abstract_chars": min_chars,
            "drop_withdrawn": True,
            "dedupe": ["arxiv_id", "normalized title"],
            "full_text": "not collected",
        },
        "leakage": leakage,
        "schema": list(frame.columns) if len(frame) else [],
    }
    write_json(PROCESSED / "data_quality_report.json", report)
    return report


def _leakage_report(records: list[dict]) -> dict:
    """Flag evaluation problems that are copied from abstracts.

    Cases are authored problem statements. A high token overlap would mean a
    case was leaked from the corpus rather than written as a new decision problem.
    """
    cases_path = EVAL / "cases.jsonl"
    if not cases_path.exists():
        return {"status": "cases file missing"}
    cases = []
    with cases_path.open() as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    doc_tokens = [
        (row["arxiv_id"], set(tokenize(row["title"] + " " + row["abstract"])))
        for row in records
    ]
    flagged = []
    max_seen = 0.0
    for case in cases:
        case_tokens = set(tokenize(case["problem"]))
        best_id = ""
        best = 0.0
        for arxiv_id, tokens in doc_tokens:
            score = jaccard(case_tokens, tokens)
            if score > best:
                best = score
                best_id = arxiv_id
        max_seen = max(max_seen, best)
        if best >= 0.35:
            flagged.append({"case_id": case["case_id"], "arxiv_id": best_id, "jaccard": round(best, 4)})
    return {
        "max_jaccard_case_vs_abstract": round(max_seen, 4),
        "flagged_at_or_above_0.35": flagged,
        "rule": "Authored cases stay in the project. They are not training labels drawn from abstracts.",
    }
