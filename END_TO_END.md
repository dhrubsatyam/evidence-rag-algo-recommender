# End-to-End Pipeline — Evidence-Grounded Algorithm Recommendation

AAI 590 Capstone, Group 3. This document describes the full system as
implemented: data acquisition, preprocessing, indexing, retrieval,
generation, evaluation, the from-scratch neural component, and how to
reproduce every artifact in `outputs/`.

## System overview

```
                        ┌────────────────────────────────────────────┐
                        │           INPUT: problem + constraints      │
                        │  (sample size, class balance,               │
                        │   interpretability, latency)                │
                        └──────────────────┬─────────────────────────┘
                                           ▼
                        ┌────────────────────────────────────────────┐
                        │  INTAKE & CLARIFICATION                     │
                        │  missing constraint → clarifying questions  │
                        │  non-tabular task    → insufficient evidence│
                        └──────────────────┬─────────────────────────┘
                                           ▼
        ┌──────────────────────────────────────────────────────────┐
        │                    RETRIEVAL (hybrid, 3 signals)          │
        │                                                          │
        │   query ──► BM25 (lexical) ──┐                           │
        │          ──► MiniLM dense ───┼──► RRF fusion ──► top-k    │
        │          ──► neural family ──┘     (k=60)     abstracts   │
        │            agreement                                  │
        └───────────────────────────┬──────────────────────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │                    GENERATION                              │
        │  Qwen2.5-0.5B-Instruct (local, greedy)                    │
        │  fallback: extractive algorithm-mention baseline          │
        │  low-evidence flag → prefer abstention                    │
        └───────────────────────────┬──────────────────────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │              CITATION & UNCERTAINTY CONTROLS               │
        │  arXiv IDs not in retrieved set are stripped + counted    │
        │  status: ok / ungrounded_baseline / insufficient_evidence  │
        └───────────────────────────┬──────────────────────────────┘
                                    ▼
                        ┌────────────────────────────────────────────┐
                        │  OUTPUT: cited shortlist + trade-offs +     │
                        │  uncertainty note + retrieval trace         │
                        └────────────────────────────────────────────┘
```

Offline branch (built once, reused by retrieval):

```
arXiv API ──► download ──► prepare ──► index ──► corpus artifacts
                                   │
                                   └──► neural train ──► family scores
                                        (from scratch)   (per abstract)
```

## Stage-by-stage

### 1. Download — `evidence_rag/download.py`

Source: public arXiv API (the same metadata the Cornell arXiv Kaggle
dataset snapshots; the API slice keeps the corpus small and needs no
credential).

- Queries: `cat:cs.LG AND all:classification` and
  `cat:stat.ML AND all:classification`, submitted 2024-01-01 →
  2024-06-30, newest-first, capped at 400/query.
- Result: **712 raw records** (`data/raw/records.jsonl` + per-page XML
  in `data/raw/pages/`). Manifest: `data/raw/manifest.json`.

### 2. Prepare — `evidence_rag/prepare.py`

Cleaning and leakage screen (`data/processed/`):

- Dedupe on `arxiv_id` and normalized title → **671 kept records**
  (41 duplicates dropped).
- Filters: ML categories only (`cs.AI`, `cs.LG`, `stat.ML`), abstract
  ≥ 200 chars, withdrawn papers dropped, full text excluded.
- Leakage check: max Jaccard similarity between authored eval cases
  and abstracts = **0.058** (flag threshold 0.35) → no leakage.
- Missing-field audit: 0 missing across id/title/abstract/published/
  category. Report: `data/processed/data_quality_report.json`.

### 3. Index — `evidence_rag/index_corpus.py`

Two retrieval indexes over title + abstract (`data/index/`):

- **BM25Okapi** lexical index (tokens persisted in `tokens.jsonl`).
- **Dense embeddings**: `sentence-transformers/all-MiniLM-L6-v2`,
  L2-normalized, `embeddings.npy` + `ids.json`.

### 4. Neural model (from scratch) — `evidence_rag/neural/`

A small deep learning model implemented and trained in PyTorch with no
pretrained weights.

- **Task**: multi-label algorithm-family tagging — predict which of
  `linear`, `instance_kernel`, `tree_ensemble`, `neural` an abstract
  discusses.
- **Supervision**: weak labels from algorithm-name mentions in
  title + abstract (documented limitation: mentions ≠ ground truth).
- **Architecture**: `Embedding(128)` → masked mean pooling →
  `Linear(128,64)+ReLU+Dropout(0.3)` → `Linear(64,4)`; ~2M params,
  vocab 7,810, max length 200 tokens.
- **Training**: 571 train / 100 val (seed 13), `BCEWithLogitsLoss`
  with inverse-frequency `pos_weight`, AdamW (lr 1e-3, wd 1e-4),
  batch 64, 40 epochs on MPS — **4.5 s total**.
- **Result**: best validation **macro-F1 = 0.477** (against weak
  labels). Checkpoint `outputs/neural/model.pt`, per-epoch log
  `train_log.jsonl`, config `config.json`, metrics `metrics.json`.
- **Corpus scoring**: `predict --corpus` writes per-abstract family
  probabilities to `outputs/neural/family_scores.json` — the offline
  artifact retrieval consumes.

### 5. Retrieve — `evidence_rag/retrieve.py`

Hybrid retrieval, three signals fused by **Reciprocal Rank Fusion**
(`rrf_k=60`, candidate pool 30, top-k 5):

| Signal | Score | Role |
| --- | --- | --- |
| BM25 | lexical term match | exact terms: algorithm names, dataset traits |
| Dense cosine | query ↔ abstract embedding | semantic similarity |
| Neural family agreement | dot(query family probs, doc family probs) | algorithm-family relevance from the from-scratch tagger |

Each result carries `bm25`, `cosine`, `rrf`, and
`neural_family_agreement` scores plus its arXiv URL. Traces for the 10
development cases: `outputs/retrieval_traces.jsonl`.

### 6. Generate — `evidence_rag/generate.py`

Two generators, same prompt and citation policy:

- **`local_lm`**: `Qwen/Qwen2.5-0.5B-Instruct`, chat template, greedy
  decoding (`do_sample=False`), max 240 new tokens, runs on MPS.
  Prompt contains problem + constraints + top-5 abstracts (650 chars
  each) + instructions to cite only retrieved IDs and state
  uncertainty.
- **`extractive`**: deterministic baseline — counts algorithm-name
  mentions across retrieved abstracts and builds the shortlist with
  template reasons/trade-offs. No model download; used as fallback
  when the LLM fails to load.

Missing required constraints (`sample_size`, `class_balance`,
`interpretability`, `latency`) → clarifying questions, no guess.
Non-tabular tasks → `insufficient_evidence`.

### 7. Controls — citation policy + probes

- `apply_citation_policy` strips any arXiv ID not present in that
  question's retrieved set; counts `unsupported_ids`.
- `low_evidence_cosine = 0.28` on max cosine → abstention preferred.
- Probes (`data/eval/probes.jsonl`): out-of-scope refusal, tabular OK,
  missing-constraints clarification — **all pass**
  (`outputs/probes.json`).

### 8. Evaluate — `evidence_rag/evaluate.py`

Dev set: 10 authored cases × {with retrieval, without retrieval} = 20
rows per generator (`data/eval/cases.jsonl`; 20 held-out cases remain
sealed until the rubric is frozen — `--include-heldout` only then).

Automatic metrics (`outputs/metrics_*.json`):

| Metric | extractive | local_lm |
| --- | --- | --- |
| Rows | 20 | 20 |
| Status ok / ungrounded | 10 / 10 | 10 / 10 |
| Cited IDs resolved | 27 | 15 |
| **Unsupported IDs** | **0** | **0** |
| Cases where retrieval added citations | 10/10 | 9/10 |
| Mean latency | ~0.001 s | ~3.3 s (MPS) |

Human 0–2 relevance/suitability scores are collected manually per
`data/eval/rubric.json` — not fabricated by code.

### 9. Feasibility — `evidence_rag/feasibility.py`

`outputs/adaptation_decision.json`: **defer** fine-tuning. Ten dev
cases are too few to fit an adapter without memorization, CPU
fine-tuning isn't assumed feasible, and the required comparison is the
untuned model ± retrieval. This is the plan's documented gate outcome.

### 10. Deploy — `evidence_rag/serve.py` + `static/index.html`

Stdlib HTTP server, `POST /api/recommend`, `GET /api/health`, single-page
UI at `http://127.0.0.1:8000`. Generator selectable
(`extractive`/`local_lm`), retrieval toggle, constraint form.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full pipeline (download → feasibility), extractive generator
python -m evidence_rag.pipeline --generator extractive

# neural tagger: train, then score corpus for the retrieval signal
python -m evidence_rag.neural.train --epochs 40
python -m evidence_rag.neural.predict --corpus

# local-LLM generation + evaluation
python -m evidence_rag.pipeline --phases generate,probes,analyze --generator local_lm

# serve
python -m evidence_rag.serve

# tests
pytest
```

## Models used — summary

| Component | Model | Trained? | Role |
| --- | --- | --- | --- |
| Lexical retrieval | BM25Okapi | no | exact-match ranking |
| Dense retrieval | all-MiniLM-L6-v2 | no (pretrained) | semantic embeddings |
| Family tagger | FamilyTagger (PyTorch, from scratch) | **yes, this repo** | third RRF signal |
| Generator | Qwen2.5-0.5B-Instruct | no (pretrained) | grounded shortlist |
| Baseline | extractive mention counter | n/a | deterministic fallback |

## Known limitations

- Abstracts only — no full text; they support discovery, not proof of
  superiority.
- Neural tagger labels are keyword-derived (weak supervision);
  macro-F1 0.477 reflects agreement with those labels.
- 10 dev cases is small; held-out evaluation and human rubric scores
  are still pending per the evaluation plan.
- The neural signal currently re-ranks within the fused candidate set;
  it is not yet evaluated as a standalone retriever in the ablation
  matrix (BM25-only / dense-only / hybrid) planned for the report.
