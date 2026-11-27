# Human scoring guide — held-out cases

Two reviewers score independently; a third adjudicates disagreements.
Keep first scores. Do not share scores until both reviewers finish.

## File

`scoring_template.csv` — one row per held-out case (20 rows). Each
reviewer fills `scorer` with their initials and completes every case.
Make a copy per reviewer (e.g. `scores_ar.csv`, `scores_mk.csv`) or use
the `scorer` column in a shared sheet — decide as a team and keep the
raw files.

## Columns

| Column | What to score | Scale |
| --- | --- | --- |
| `rel_p1`–`rel_p5` | Relevance of each of the top-5 retrieved abstracts to the case's decision problem, in rank order | 0–2 each (see rubric.json) |
| `suitability_with_retrieval` | Do the recommended algorithms fit the stated constraints? | 0–2 |
| `evidence_with_retrieval` | Do the cited abstracts support the claims made? | 0–2 |
| `suitability_without_retrieval` | Same, for the no-retrieval baseline answer | 0–2 |
| `evidence_without_retrieval` | Same — note: the baseline cites nothing by design; score whether its claims are appropriately hedged | 0–2 |
| `notes` | Free text: why a score was given, any confusion | — |

## Where to look

- Answers: `outputs/generations_local_lm.jsonl`, rows with
  `"split": "heldout"`. `mode` = `with_retrieval` or
  `without_retrieval`. Each row has `recommendations_text`,
  `cited_ids`, and `retrieval` (top-5 `arxiv_id`, `title`, scores).
- Case definitions (problem + constraints): `data/eval/cases.jsonl`,
  `"split": "heldout"`.
- Scoring scale definitions: `rubric.json`.

## After scoring

1. Compare the two reviewers' files; flag every cell differing by ≥1.
2. Adjudicate flagged cells, record final score + reason in `notes`.
3. Merge into `scores_final.csv` and compute: mean per criterion,
   precision@5 (share of rel_p* = 2), % cases with ≥1 useful paper
   (any rel_p* ≥ 1), and with/without-retrieval wins–ties–losses on
   suitability + evidence.
