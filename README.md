# Evidence-grounded algorithm recommendation

**AAI 590 Capstone — Group 3**

Team: Anugrah Rastogi, Dhrub Satyam, Mounika Konduru

Faculty guide: Haisav Chokshi

Local prototype for AAI 590 Group 3. Given a tabular classification problem and its constraints, the pipeline retrieves machine-learning abstracts and returns a short list of candidate algorithms with citations and an explicit uncertainty note. The same question can be answered with retrieval turned off, which is the comparison the project is built around.

This follows the Group 3 plan (evidence-grounded recommendation over research abstracts). 

## Run it yourself (first time)

You need **Python 3.11+** and about **3 GB of free disk** (dependencies + model weights). No GPU is required; on Apple Silicon the model runs on MPS, otherwise CPU.

```bash
# 1. Clone the repo
git clone https://github.com/dhrubsatyam/evidence-rag-algo-recommender.git
cd evidence-rag-algo-recommender

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (first install takes a few minutes)
pip install -r requirements.txt
```

**Fastest demo — no model download.** The extractive generator is deterministic and needs no language model:

```bash
# 4a. Run the full pipeline: download -> prepare -> index -> retrieve ->
#     generate -> probes -> analyze -> feasibility (uses committed data,
#     but re-download is safe to re-run)
python -m evidence_rag.pipeline --generator extractive

# 5. Launch the web interface
python -m evidence_rag.serve
```

Then open **http://127.0.0.1:8000**. Fill in all four constraints
(sample size, class balance, interpretability, latency) — leaving one
blank intentionally returns clarifying questions instead of a guess.

**Full demo — with the local LLM.** This downloads
`Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) on first use:

```bash
# 4b. Skip download/prepare/index (already built); just generate + analyze
python -m evidence_rag.pipeline --phases generate,probes,analyze --generator local_lm

# 5. Serve and pick "local_lm" in the UI generator dropdown
python -m evidence_rag.serve
```

**Check everything works:**

```bash
pytest    # 6 tests, should all pass in under a second
```

**Expected outputs** land in `outputs/`: `generations_extractive.jsonl`,
`generations_local_lm.jsonl`, `retrieval_traces.jsonl`, `probes.json`,
`metrics_extractive.json`, `metrics_local_lm.json`,
`adaptation_decision.json`. The committed copies in this repo were
generated on macOS arm64 with the commands above.

**Common issues**

- `pip install` fails on Python < 3.11 → install a newer Python (e.g. `brew install python@3.12`).
- First `local_lm` run is slow → it is downloading ~1 GB of weights from Hugging Face; subsequent runs load from cache.
- `ModuleNotFoundError` → you forgot to activate `.venv`, or installed into a different Python.
- Port 8000 busy → `python -m evidence_rag.serve --port 8080`.

## What each phase does

| Phase | Plan task | Output |
| --- | --- | --- |
| `download` | Source access | Pinned arXiv abstract subset in `data/raw/` |
| `prepare` | Data preparation | Deduplicated corpus, schema, missing-value and leakage report |
| `index` | Retrieval index | BM25 tokens and MiniLM embeddings |
| `retrieve` | Retrieval traces | Top-5 traces for development cases |
| `generate` | Local model and RAG | Answers with and without retrieval |
| `probes` | Missing-evidence tests | Clarification, out-of-scope refusal, citation check |
| `analyze` | Automatic checks | `outputs/metrics.json` |
| `feasibility` | Adaptation gate | Written deferral of fine-tuning |

Held-out cases (20) are in `data/eval/cases.jsonl` and are not run unless you pass `--include-heldout`. Leave them sealed until the rubric in `data/eval/rubric.json` is confirmed. Human 0–2 scores are not invented by this code.

## Data

The plan names the Cornell arXiv dataset on Kaggle. That release is a multi-gigabyte snapshot of arXiv metadata. This pipeline downloads a smaller slice from the public arXiv API, which is the source of that snapshot:

- Categories: `cs.LG` and `stat.ML`
- Query: classification, submitted from 2024-01-01 through 2024-06-30
- Selection: newest papers in that window, capped at 400 per query
- Fields: identifier, title, authors, abstract, categories, dates
- Full text is not downloaded

Abstracts support discovery. They are not a leaderboard, and they do not prove that one algorithm is best.

## Pipeline commands (reference)

All phases, in order (this is what "Run it yourself" step 4a executes):

```bash
python -m evidence_rag.pipeline --generator extractive
# equivalent to --phases download,prepare,index,retrieve,generate,probes,analyze,feasibility
```

Useful variations:

```bash
# Only re-run generation + evaluation with the local model
python -m evidence_rag.pipeline --phases generate,probes,analyze --generator local_lm

# Smoke test on 2 development cases
python -m evidence_rag.pipeline --phases generate,analyze --generator local_lm --limit 2

# Rebuild the index after changing configs/default.json
python -m evidence_rag.pipeline --phases index

# Serve the web UI (requires the index to exist)
python -m evidence_rag.serve            # http://127.0.0.1:8000
python -m evidence_rag.serve --port 8080
```

The extractive phase builds the shortlist from algorithm names that actually appear in the retrieved abstracts. It is deterministic and does not download a language model. The first `local_lm` run downloads the weights; generation is unsampled (`do_sample=False`) so repeated runs stay stable. Citations that are not in the retrieved set for that question are stripped and counted as unsupported. Generation files are `outputs/generations_extractive.jsonl` and `outputs/generations_local_lm.jsonl`, so the two runs do not overwrite each other. Leaving a constraint blank returns clarifying questions instead of a recommendation; choosing a non-classification task returns an insufficient-evidence response.

## Neural model (built from scratch)

In addition to the pretrained retrieval/generation components, the repo
contains a small deep learning model implemented and trained from
scratch in PyTorch — `evidence_rag/neural/`.

**Task.** Multi-label classification: given a paper title + abstract,
predict which algorithm families it discusses (`linear`,
`instance_kernel`, `tree_ensemble`, `neural`). Labels are weakly
supervised from algorithm-name mentions in the corpus text.

**Architecture** (`evidence_rag/neural/model.py`):

```
Embedding(vocab=~15k, dim=128) -> masked mean pooling
  -> Linear(128, 64) + ReLU + Dropout(0.3)
  -> Linear(64, 4) logits
```

~2M parameters, no pretrained weights.

**Training** (`evidence_rag/neural/train.py`): 571/100 train/val split
of the 671-record corpus, `BCEWithLogitsLoss` with per-label
inverse-frequency `pos_weight`, `AdamW` (lr 1e-3, weight_decay 1e-4),
batch 64, 40 epochs, best-validation checkpoint kept. Best validation
macro-F1 = **0.477** (labels are keyword-derived, so this measures
agreement with the weak labels, not human judgment). Per-epoch logs:
`outputs/neural/train_log.jsonl`; summary: `outputs/neural/metrics.json`.

```bash
# retrain from scratch
python -m evidence_rag.neural.train --epochs 40

# score one abstract
python -m evidence_rag.neural.predict "Title. Abstract text..."

# score the whole corpus -> outputs/neural/corpus_scores.jsonl
python -m evidence_rag.neural.predict --corpus
```

Intended use: an auxiliary family-relevance signal that can re-rank or
filter retrieved abstracts before generation. It is not yet wired into
the default retrieval path; the RRF hybrid remains the baseline.

## Tests

```bash
pytest
```

## Adaptation

`outputs/adaptation_decision.json` records a deferral. Ten development write-ups are not a training set, and fine-tuning is optional in the plan. The required comparison is the untuned local model with and without retrieval.
