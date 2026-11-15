# Evidence-grounded algorithm recommendation

**AAI 590 Capstone — Group 3**

Team: Anugrah Rastogi, Dhrub Satyam, Maunika Konduru
Faculty guide: Haisav Chokshi
University of San Diego — Master of Science in Applied Artificial Intelligence

Local prototype for AAI 590 Group 3. Given a tabular classification problem and its constraints, the pipeline retrieves machine-learning abstracts and returns a short list of candidate algorithms with citations and an explicit uncertainty note. The same question can be answered with retrieval turned off, which is the comparison the project is built around.

This follows the Group 3 plan (evidence-grounded recommendation over research abstracts). 
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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every phase on the 10 development cases:

```bash
python -m evidence_rag.pipeline --generator extractive
```

The extractive phase builds the shortlist from algorithm names that actually appear in the retrieved abstracts. It is deterministic and does not download a language model.

To use the local instruction model (`Qwen/Qwen2.5-0.5B-Instruct`) instead:

```bash
python -m evidence_rag.pipeline --phases generate,analyze --generator local_lm
```

A short check on two development cases:

```bash
python -m evidence_rag.pipeline --phases generate,analyze --generator local_lm --limit 2
```

The first local-model run downloads the weights. Generation is unsampled (`do_sample=False`) so repeated runs stay stable. Citations that are not in the retrieved set for that question are stripped and counted as unsupported. Generation files are `outputs/generations_extractive.jsonl` and `outputs/generations_local_lm.jsonl`, so the two runs do not overwrite each other.

Open the interface after the index exists:

```bash
python -m evidence_rag.serve
```

Then open `http://127.0.0.1:8000`. Leaving a constraint blank returns clarifying questions instead of a recommendation. Choosing a non-classification task returns an insufficient-evidence response.

## Tests

```bash
pytest
```

## Adaptation

`outputs/adaptation_decision.json` records a deferral. Ten development write-ups are not a training set, and fine-tuning is optional in the plan. The required comparison is the untuned local model with and without retrieval.
