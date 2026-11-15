import json
from pathlib import Path

from evidence_rag.download import parse_feed
from evidence_rag.generate import (
    apply_citation_policy,
    clarification_questions,
    scope_problem,
)
from evidence_rag.retrieve import reciprocal_rank_fusion
from evidence_rag.utils import canonical_arxiv_id, find_arxiv_ids

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_id_strips_version_and_url():
    assert canonical_arxiv_id("https://arxiv.org/abs/2401.01234v2") == "2401.01234"
    assert find_arxiv_ids("See 2401.01234 and 2401.01234v3.") == ["2401.01234"]


def test_rrf_prefers_documents_ranked_high_by_both_lists():
    scores = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], rrf_k=60)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_unsupported_citations_are_removed():
    cleaned = apply_citation_policy("Use boosting (2401.00001, 1999.99999).", {"2401.00001"})
    assert cleaned["cited_ids"] == ["2401.00001"]
    assert cleaned["unsupported_ids"] == ["1999.99999"]
    assert "1999.99999" not in cleaned["text"]


def test_clarification_and_scope():
    assert len(clarification_questions({})) == 4
    assert clarification_questions(
        {
            "sample_size": "1000",
            "class_balance": "rare",
            "interpretability": "high",
            "latency": "batch",
        }
    ) == []
    assert scope_problem("image_generation")
    assert scope_problem("tabular_classification") is None


def test_parse_feed_reads_atom_entry():
    xml = b"""<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom' xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>1</opensearch:totalResults>
      <entry>
        <id>https://arxiv.org/abs/2401.01234v1</id>
        <title>A classification baseline</title>
        <summary>We study tabular classification with logistic regression.</summary>
        <published>2024-01-02T00:00:00Z</published>
        <updated>2024-01-03T00:00:00Z</updated>
        <author><name>Ada Lovelace</name></author>
        <arxiv:primary_category term='cs.LG'/>
        <category term='cs.LG'/>
      </entry>
    </feed>
    """
    total, records = parse_feed(xml)
    assert total == 1
    assert records[0]["arxiv_id"] == "2401.01234"
    assert records[0]["primary_category"] == "cs.LG"
    assert records[0]["authors"] == ["Ada Lovelace"]


def test_case_split_counts():
    counts = {"dev": 0, "heldout": 0}
    ids = set()
    for line in (ROOT / "data" / "eval" / "cases.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        counts[case["split"]] += 1
        ids.add(case["case_id"])
        assert case["task"] == "tabular_classification"
        for key in ("sample_size", "class_balance", "interpretability", "latency"):
            assert case["constraints"][key]
    assert counts == {"dev": 10, "heldout": 20}
    assert len(ids) == 30
