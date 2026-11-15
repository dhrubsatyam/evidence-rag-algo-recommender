"""Download a pinned machine-learning abstract subset from the arXiv API.

The Group 3 plan names the Cornell arXiv dataset on Kaggle. That release is a
multi-gigabyte historical dump of the same arXiv metadata. This phase pulls a
smaller, date-pinned classification subset directly from arXiv's public API,
which is the authorized source behind that dump. Full text is not downloaded.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from evidence_rag.config import RAW, ensure_dirs, load_config
from evidence_rag.utils import canonical_arxiv_id, write_json

ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"
API = "https://export.arxiv.org/api/query"


def _text(parent: ET.Element, tag: str, ns: str = ATOM) -> str:
    node = parent.find(f"{{{ns}}}{tag}")
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def parse_feed(xml_bytes: bytes) -> tuple[int | None, list[dict]]:
    root = ET.fromstring(xml_bytes)
    total_node = root.find(f"{{{OPENSEARCH}}}totalResults")
    total = int(total_node.text) if total_node is not None and total_node.text else None
    records: list[dict] = []
    for entry in root.findall(f"{{{ATOM}}}entry"):
        entry_id = _text(entry, "id")
        if not entry_id or "arxiv.org" not in entry_id:
            continue
        authors = []
        for author in entry.findall(f"{{{ATOM}}}author"):
            name = _text(author, "name")
            if name:
                authors.append(name)
        categories = []
        for cat in entry.findall(f"{{{ATOM}}}category"):
            term = cat.attrib.get("term", "").strip()
            if term:
                categories.append(term)
        primary_node = entry.find(f"{{{ARXIV}}}primary_category")
        primary = primary_node.attrib.get("term", "").strip() if primary_node is not None else ""
        if not primary and categories:
            primary = categories[0]
        doi = _text(entry, "doi", ARXIV)
        journal = _text(entry, "journal_ref", ARXIV)
        comment = _text(entry, "comment", ARXIV)
        records.append(
            {
                "arxiv_id": canonical_arxiv_id(entry_id),
                "id_url": entry_id,
                "title": _text(entry, "title"),
                "abstract": _text(entry, "summary"),
                "authors": authors,
                "primary_category": primary,
                "categories": categories,
                "published": _text(entry, "published"),
                "updated": _text(entry, "updated"),
                "doi": doi,
                "journal_ref": journal,
                "comment": comment,
            }
        )
    return total, records


def fetch_page(search_query: str, start: int, page_size: int, user_agent: str) -> bytes:
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(API + "?" + params, headers={"User-Agent": user_agent})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # arXiv occasionally returns 429 or a reset
            last_error = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"arXiv request failed after retries: {last_error}")


def download(config: dict | None = None) -> dict:
    ensure_dirs()
    config = config or load_config()
    all_records: list[dict] = []
    query_reports = []
    for query in config["queries"]:
        kept: list[dict] = []
        start = 0
        total = None
        pages = 0
        cap = int(config["corpus_cap_per_query"])
        page_size = int(config["page_size"])
        while len(kept) < cap:
            xml_bytes = fetch_page(
                query["search_query"], start, page_size, config["user_agent"]
            )
            page_path = RAW / "pages" / f"{query['name']}_{start}.xml"
            page_path.write_bytes(xml_bytes)
            total, records = parse_feed(xml_bytes)
            pages += 1
            if not records:
                break
            for record in records:
                record["source_query"] = query["name"]
                record["source"] = "arxiv_api"
                kept.append(record)
                if len(kept) >= cap:
                    break
            start += len(records)
            if total is not None and start >= total:
                break
            time.sleep(float(config["request_interval_seconds"]))
        query_reports.append(
            {
                "name": query["name"],
                "search_query": query["search_query"],
                "reported_total": total,
                "downloaded": len(kept),
                "pages": pages,
                "selection": "newest submittedDate within the pinned window, capped",
            }
        )
        all_records.extend(kept)
        time.sleep(float(config["request_interval_seconds"]))

    raw_path = RAW / "records.jsonl"
    with raw_path.open("w") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")

    manifest = {
        "dataset": "arXiv metadata and abstracts",
        "provider": "Cornell University arXiv via the public export API",
        "kaggle_note": (
            "The course plan names the Cornell arXiv dataset on Kaggle. "
            "That file is a snapshot of this same metadata. A date-pinned "
            "API subset is used so the corpus stays small enough to index locally "
            "and does not require a Kaggle credential. Full text is excluded."
        ),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reuse": (
            "Metadata and abstracts are used for academic retrieval experiments. "
            "Papers remain the property of their authors. Do not treat retrieved "
            "abstracts as proof that an algorithm is best for a problem."
        ),
        "record_count_raw": len(all_records),
        "queries": query_reports,
        "raw_path": str(raw_path.relative_to(RAW.parents[1])),
    }
    write_json(RAW / "manifest.json", manifest)
    return manifest
