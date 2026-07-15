# LLM Eval Pipeline

> **A production-grade, CI/CD-integrated evaluation platform for Retrieval-Augmented Generation (RAG) systems.**
> Modelled after internal platforms at Braintrust, LangSmith, and Arize — designed to gate merges on real quality regressions, not just test coverage.

---

## Overview

This project automates the evaluation of a RAG pipeline every time a prompt, model, or knowledge base changes — exactly like unit tests run when code changes.

On every `git push` or Pull Request:
1. **Runs the benchmark** against a curated 204-question tax Q&A golden dataset
2. **Measures** hallucination rate, answer faithfulness, retrieval hit rate, latency percentiles (p50/p95/p99), and cost per query
3. **Blocks the merge** if any metric regresses beyond the configured SLA bounds
4. **Tracks history** to visualize whether the system is improving or degrading over time
5. **Posts a full quality report** as a PR comment and GitHub Actions step summary

---

## Architecture

```
llm-eval-pipeline/
│
├── config.py                  # Central configuration: paths, models, thresholds, pricing
│
├── core/                      # Core evaluation library
│   ├── retrieval.py           # ChromaDB vector store, document loading, retrieval metrics
│   ├── generator.py           # Groq API client (lazy init), retry/backoff, token tracking
│   ├── judge.py               # LLM-as-a-Judge: multi-dimensional JSON scoring
│   ├── metrics.py             # Semantic similarity (cosine) via SentenceTransformers
│   ├── reporter.py            # Metrics history logging, regression detection
│   └── utils.py               # Cost calculation, Git SHA/branch tracking, logging
│
├── evaluate.py                # CLI evaluation runner (full or --smoke mode)
├── ci_gate.py                 # CI quality gate: loads runs, checks thresholds, writes report
├── generate_dataset.py        # Tool to expand the golden dataset using Groq synthesis
│
├── golden_dataset.csv         # 204-question benchmark with structured metadata
├── docs/                      # Knowledge base corpus (.txt files)
├── eval_results/              # Per-run evaluation JSON outputs
├── metrics_history.json       # Aggregated trend log across all runs
│
├── dashboard/
│   └── app.py                 # Multi-page Streamlit diagnostic dashboard
│
└── .github/workflows/
    └── eval.yml               # GitHub Actions CI/CD workflow
```

---

## Evaluation Metrics

| Metric | Description | SLA Threshold |
|---|---|---|
| **Pass Rate** | % of queries graded as correct by LLM judge | ≥ 70% |
| **Hallucination Rate** | Avg judge score for unfaithful output | ≤ 0.05 |
| **p95 Latency** | 95th percentile end-to-end query time | ≤ 3.5s |
| **Retrieval Hit Rate** | % of queries with correct source document retrieved | ≥ 80% |
| **MRR** | Mean Reciprocal Rank of retrieved documents | tracked |
| **Faithfulness** | LLM judge score for context-groundedness (0–1) | tracked |
| **Cost / Query** | USD cost of a single evaluation query | tracked |

---

## LLM-as-a-Judge

Instead of fragile word-overlap heuristics (BLEU/ROUGE), the pipeline uses **Groq itself as a judge**. For each query it generates a structured JSON score across 6 dimensions:

```json
{
  "correctness":   0.9,
  "faithfulness":  1.0,
  "completeness":  0.8,
  "hallucination": 0.0,
  "confidence":    0.9,
  "reasoning":     "Answer correctly identifies 80C limit as Rs 1.5L and cites the context."
}
```

Queries with semantic similarity ≥ 0.90 skip the LLM judge to conserve API rate limits on Groq's free tier.

---

## Golden Dataset

`golden_dataset.csv` contains **204 curated Indian income tax Q&A pairs** with full metadata:

| Column | Description |
|---|---|
| `unique_id` | Stable identifier (e.g. `Q001`) |
| `question` | The benchmark query |
| `ground_truth` | Expected correct answer |
| `category` | `factual`, `edge_case`, or `out_of_scope` |
| `difficulty` | `easy`, `medium`, or `hard` |
| `tags` | Relevant topic tags |
| `expected_sources` | Source document file that should be retrieved |
| `reasoning_type` | `direct_lookup`, `multi_step`, `comparative`, `negation`, `numerical` |
| `version` | Dataset version for tracking |
| `evaluation_notes` | Human-annotated notes for difficult edge cases |

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/eval.yml`) runs on every push to `main` and every PR:

```
git push
    │
    ▼
[evaluate.py --smoke]          ← Runs 32 balanced queries (seed=42)
    │
    ▼
[ci_gate.py]                   ← Checks absolute thresholds + regression detection
    │
    ├── PASS → merge allowed
    │          eval_summary.md posted as PR comment
    │
    └── FAIL → merge blocked
               regression details posted to PR + step summary
```

**Auto-commits** `metrics_history.json` back to `main` on successful runs (with `[skip ci]` to prevent loops).

---

## Regression Detection

`ci_gate.py` compares the latest run against the previous run across all key metrics. A build **fails** if:

- Pass rate drops by more than **5 percentage points**
- Hallucination rate increases by more than **0.02 (absolute)**
- p95 latency increases by more than **15%** or **0.3s absolute**
- Avg cost increases by more than **20%**
- Retrieval hit rate drops by more than **5 percentage points**

All thresholds are configurable in `config.py`.

---

## Streamlit Dashboard

The multi-page diagnostic dashboard provides a live view of evaluation health:

```bash
streamlit run dashboard/app.py
```

**Pages:**

| Page | What it shows |
|---|---|
| **Overview & KPI Matrix** | Core metrics, delta vs previous run, failure breakdown |
| **Metric Trends** | Multi-run historical charts for quality & performance |
| **Regression Analysis** | Side-by-side run comparison, per-question regressions |
| **Failure Explorer** | Searchable failure database with full trace reproduction |
| **Retrieval Inspector** | Visual chunk flow: Question → Retrieved Chunks → Answer |
| **Cost Analytics** | Total cost, cost by category, most expensive queries |
| **Latency Analytics** | p50/p95/p99 percentile charts, distribution histograms |
| **Prompt Playground** | Interactive sandbox: test prompts, top-k, temperature live |
| **Dataset Explorer** | Filterable view of the full 204-question benchmark |

---

## Quick Start

### 1. Clone & Set Up

```bash
git clone https://github.com/ps-keerthana/llm-litmus.git
cd llm-eval-pipeline
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Windows
$env:GROQ_API_KEY = "gsk_..."

# Or create a .env file
echo GROQ_API_KEY=gsk_... > .env
```

### 3. Run Evaluation

```bash
# Full evaluation (204 questions — may take 10–15 min on free tier due to rate limits)
python evaluate.py

# Smoke test (32 balanced questions, ~3 min)
python evaluate.py --smoke
```

### 4. Check Quality Gate

```bash
python ci_gate.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Rate Limit Handling

The pipeline is built for Groq's **free tier (6,000 TPM)**:

- **Automatic retry** with 12-second backoff on `429 RateLimitError` (up to 5 attempts)
- **Sleep-time subtraction**: cumulative API sleep is tracked and excluded from latency metrics
- **Smoke mode** (`--smoke`): runs only 32 deterministic queries to fit within CI rate limits
- **Judge bypass**: queries with semantic similarity ≥ 0.90 skip the LLM judge to save tokens

---

## Configuration

All tunable parameters are centralized in [`config.py`](config.py):

```python
# Model selection
MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Groq pricing (USD per 1M tokens)
PRICE_INPUT_1M = 0.05
PRICE_OUTPUT_1M = 0.08

# Quality SLA thresholds
THRESHOLD_PASS_RATE = 70.0
THRESHOLD_HALLUCINATION = 0.05
THRESHOLD_P95_LATENCY = 3.5
THRESHOLD_RETRIEVAL_HIT_RATE = 80.0

# Regression detection limits
REGRESSION_LIMIT_PASS_RATE = 5.0
REGRESSION_LIMIT_HALLUCINATION = 0.02
REGRESSION_LIMIT_P95_LATENCY_PERCENT = 15.0
```

---

## Expanding the Dataset

To synthesize new benchmark questions from the `docs/` corpus:

```bash
python generate_dataset.py
```

This uses Groq to generate structured Q&A pairs with full metadata, aligned with the dataset schema.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM Inference | [Groq](https://groq.com) (`llama-3.1-8b-instant`) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | [ChromaDB](https://www.trychroma.com) (in-memory) |
| Dashboard | [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com) |
| CI/CD | GitHub Actions |
| Evaluation Framework | Custom (modelled after Braintrust/LangSmith patterns) |

---

## Inspiration

This project is inspired by how production ML teams evaluate LLM systems:

- **[Braintrust](https://www.braintrustdata.com)** — Dataset management, LLM-as-a-judge, regression scoring
- **[LangSmith](https://smith.langchain.com)** — Trace visualization, evaluation datasets, CI integration
- **[Arize AI](https://arize.com)** — Retrieval diagnostics, embedding drift, latency SLAs
