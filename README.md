# LLM-Litmus

> **An automated evaluation and quality-gating platform for RAG (Retrieval-Augmented Generation) pipelines.**
> Built to catch regressions in AI assistants before they reach production — the way unit tests catch code bugs before a release.

---

## What this project does

Every time you change a prompt, swap an LLM model, or update the knowledge base, this pipeline automatically:

1. Runs a benchmark of **204 Indian income tax questions** against your RAG system
2. Measures how accurate, faithful, fast, and cost-efficient the answers are
3. **Blocks the merge** on GitHub if quality drops below your thresholds
4. Posts a full report as a PR comment and GitHub Actions summary
5. Saves history so you can track whether quality is improving or getting worse over time

Think of it as "CI/CD for your LLM outputs."

---

## Current benchmark results (latest run)

| Metric | Value |
|---|---|
| **Pass Rate** | 83.3% (170 / 204 queries) |
| **Retrieval Hit Rate** | 97.5% |
| **Mean Reciprocal Rank (MRR)** | 0.913 |
| **p95 Latency** | 19.39s (rate-limited by Groq free tier) |
| **Average Latency** | 6.81s |
| **Average Cost per Query** | $0.000046 |

---

## How it works — the 4-step pipeline

```
User Question
    │
    ▼
Step 1 — Document Ingestion
    8 tax guides are split into chunks and stored in ChromaDB
    with sentence embeddings (all-MiniLM-L6-v2)
    │
    ▼
Step 2 — Vector Retrieval (top-k = 5)
    The question is embedded and the 5 most relevant
    chunks are retrieved from ChromaDB
    │
    ▼
Step 3 — LLM Answer Synthesis
    The retrieved chunks are passed to the LLM with a
    system prompt that says: "answer only from the context"
    │
    ▼
Step 4 — Automated Quality Gate
    The answer is compared to the ground truth using
    semantic similarity. If similarity ≥ 0.90, the query
    passes. Otherwise, an LLM judge scores it on
    correctness, faithfulness, hallucination, etc.
```

---

## Project structure

```
llm-eval-pipeline/
│
├── config.py                    # All settings: models, thresholds, pricing, scheduler
│
├── core/
│   ├── retrieval.py             # ChromaDB vector store, document loading, retrieval metrics
│   ├── generator.py             # Calls the LLM provider, handles retries and token tracking
│   ├── judge.py                 # LLM-as-a-judge: scores answers on 6 dimensions
│   ├── metrics.py               # Semantic similarity (cosine) via SentenceTransformers
│   ├── reporter.py              # Saves run results, logs history, detects regressions
│   ├── attributor.py            # Diagnoses WHY a query failed (retrieval gap, hallucination, etc.)
│   ├── cache.py                 # Caches embeddings and results to avoid redundant API calls
│   ├── scheduler.py             # Proactive rate-limit scheduler (respects Groq TPM/RPM limits)
│   ├── queue.py                 # Async request queue for parallel evaluation
│   ├── utils.py                 # Cost calculation, Git SHA/branch tracking, logging
│   └── providers/
│       ├── groq.py              # Groq API provider (llama-3.3-70b-versatile)
│       ├── ollama.py            # Local Ollama provider (llama3.2:1b)
│       ├── openai.py            # OpenAI provider (gpt-4o-mini)
│       ├── anthropic.py         # Anthropic provider (claude-3-5-haiku)
│       ├── base.py              # Abstract base class all providers implement
│       └── factory.py           # Creates the right provider based on config
│
├── evaluate.py                  # Main CLI runner — runs the full benchmark or smoke test
├── ci_gate.py                   # Reads the latest run, checks thresholds, writes the report
├── generate_dataset.py          # Generates new Q&A pairs from docs/ using Groq
│
├── golden_dataset.csv           # 204-question benchmark (questions, ground truths, metadata)
├── metrics_history.json         # History of all evaluation runs (auto-updated by CI)
├── eval_results/                # Per-run JSON output files
│
├── docs/                        # Knowledge base (8 tax guide text files)
│   ├── income_tax_basics.txt
│   ├── section_80c_deductions.txt
│   ├── section_80d_health.txt
│   ├── hra_exemption.txt
│   ├── home_loan_section_24b.txt
│   ├── nps_section_80ccd.txt
│   ├── tds_rules.txt
│   └── capital_gains_tax.txt
│
├── dashboard/
│   └── app.py                   # Multi-page Streamlit dashboard
│
├── web/
│   └── index.html               # Public-facing landing page (deployable to Vercel)
│
├── api/                         # FastAPI backend service
├── db/                          # SQLite database helpers
├── tests/                       # Test suite
├── scripts/                     # Utility scripts
│
└── .github/workflows/
    └── eval.yml                 # GitHub Actions CI/CD workflow
```

---

## Knowledge base (docs/)

The RAG system answers questions from 8 tax reference documents:

| File | Topic |
|---|---|
| `income_tax_basics.txt` | Tax slabs, regimes, exemption limits, 87A rebate |
| `section_80c_deductions.txt` | PPF, ELSS, EPF, LIC — deduction limits and eligibility |
| `section_80d_health.txt` | Medical insurance deductions for self, family, parents |
| `hra_exemption.txt` | HRA calculation, metro vs non-metro, Section 10(13A) |
| `home_loan_section_24b.txt` | Home loan interest deduction limits |
| `nps_section_80ccd.txt` | NPS Tier-I, 80CCD(1), 80CCD(1B), 80CCD(2) |
| `tds_rules.txt` | TDS rates under Sections 192, 194C, 194J, 194-I |
| `capital_gains_tax.txt` | STCG, LTCG, Section 112A, indexation |

---

## Supported LLM providers

The pipeline supports 4 providers, switchable via environment variable:

| Provider | Model | When to use |
|---|---|---|
| **Groq** (default) | `llama-3.3-70b-versatile` | Fast, free tier, best for CI |
| **Ollama** | `llama3.2:1b` | Fully local, no API key needed |
| **OpenAI** | `gpt-4o-mini` | Highest quality, pay-per-use |
| **Anthropic** | `claude-3-5-haiku-20241022` | Alternative commercial option |

Switch provider:
```bash
# Use Ollama locally
LLM_PROVIDER=ollama python evaluate.py --smoke

# Use OpenAI
LLM_PROVIDER=openai python evaluate.py --smoke
```

---

## Golden dataset — what's in it

`golden_dataset.csv` has 204 Indian income tax questions with full metadata:

| Column | What it means |
|---|---|
| `unique_id` | Stable ID like `Q001` |
| `question` | The benchmark question |
| `ground_truth` | The correct expected answer |
| `category` | `factual`, `reasoning`, `multi-hop`, `adversarial`, `edge_case`, `out_of_scope` |
| `difficulty` | `easy`, `medium`, or `hard` |
| `tags` | Topic tags like `80c`, `tds`, `hra` |
| `expected_sources` | Which doc file should be retrieved |
| `reasoning_type` | `direct_lookup`, `multi_step`, `comparative`, `negation`, `numerical` |
| `evaluation_notes` | Notes for tricky edge cases |

---

## LLM-as-a-Judge scoring

Instead of simple keyword matching, the pipeline uses an LLM to judge each answer on 6 dimensions:

```json
{
  "correctness":   0.9,
  "faithfulness":  1.0,
  "completeness":  0.8,
  "hallucination": 0.0,
  "confidence":    0.9,
  "reasoning": "Answer correctly identifies 80C limit as ₹1.5L and cites the context."
}
```

**Judge is skipped** when semantic similarity ≥ 0.90 to save API tokens on Groq's free tier.

---

## Quality thresholds (what causes CI to fail)

These are the absolute minimums. If any metric misses, the merge is blocked:

| Metric | Threshold |
|---|---|
| Pass rate | ≥ 70% |
| Hallucination rate | ≤ 5% |
| p95 Latency | ≤ 3.5s |
| Retrieval hit rate | ≥ 80% |

### Regression detection (what causes CI to flag a regression)

Even if you pass all thresholds, CI fails if the *latest run is significantly worse than the previous run*:

| Metric | Max allowed drop |
|---|---|
| Pass rate | −5 percentage points |
| Hallucination rate | +0.02 (absolute) |
| p95 Latency | +15% or +0.3s |
| Average cost | +20% |
| Retrieval hit rate | −5 percentage points |

---

## CI/CD flow

```
git push to main  (or open a Pull Request)
        │
        ▼
  evaluate.py --smoke
  (runs 32 balanced queries, ~3 min)
        │
        ▼
  ci_gate.py
  (checks thresholds + regression vs last run)
        │
        ├── ✅ PASS  → merge allowed
        │            eval_summary.md posted as PR comment
        │            metrics_history.json auto-committed
        │
        └── ❌ FAIL  → merge blocked
                     regression details posted to PR
```

The workflow auto-commits `metrics_history.json` back to `main` with `[skip ci]` to avoid infinite loops.

---

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/ps-keerthana/llm-litmus.git
cd llm-eval-pipeline

python -m venv venv
.\venv\Scripts\activate        # Windows
# or: source venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the smoke test (quick, ~3 min)

```bash
python evaluate.py --smoke
```

This runs 32 balanced queries using the Groq provider.

### 4. Run the full benchmark (204 questions, ~15 min)

```bash
python evaluate.py
```

### 5. Check the quality gate

```bash
python ci_gate.py
```

### 6. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Streamlit dashboard pages

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

## Rate limit handling (Groq free tier)

Groq's free tier allows 15 RPM and 14,400 TPM for llama-3.3-70b-versatile. The pipeline handles this automatically:

- **Proactive scheduler** (`core/scheduler.py`) spaces requests to stay within limits, no manual sleep calls needed
- **Automatic retry** with 12-second backoff on `429 RateLimitError` (up to 5 attempts)
- **Smoke mode** (`--smoke`) runs only 32 queries — fits within CI rate limits
- **Judge bypass** — queries with similarity ≥ 0.90 skip the LLM judge to save tokens
- **Result cache** (`core/cache.py`) — re-uses embeddings and results to avoid redundant API calls

---

## Expanding the knowledge base

To add a new tax topic, just create a `.txt` file in `docs/` and re-run the evaluation. ChromaDB will automatically re-index it.

To generate new benchmark questions from your docs:

```bash
python generate_dataset.py
```

This uses Groq to synthesise structured Q&A pairs with full metadata, aligned with the dataset schema.

---

## Public web app (Vercel)

The `web/index.html` file is a standalone public-facing landing page that explains the project with a live interactive sandbox. No server needed — it's pure HTML/CSS/JS.

To deploy to Vercel:
1. Go to [vercel.com](https://vercel.com) and import the `ps-keerthana/llm-litmus` repo
2. Set root directory to `web/`
3. Framework preset: **None**
4. Click Deploy

Anyone with the URL will see the project explained clearly, with a live sandbox to try tax queries.

---

## Configuration reference

All settings are in [`config.py`](config.py). Key values:

```python
# Which LLM to use
LLM_PROVIDER = "groq"                        # groq | ollama | openai | anthropic
MODEL_NAME = "llama-3.3-70b-versatile"       # Groq model
OLLAMA_MODEL_NAME = "llama3.2:1b"            # Local Ollama model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # Sentence embedding model

# Retrieval
DEFAULT_TOP_K = 5                            # Number of chunks retrieved per query

# Quality thresholds
THRESHOLD_PASS_RATE = 70.0                   # Minimum pass rate (%)
THRESHOLD_HALLUCINATION = 0.05               # Maximum hallucination rate
THRESHOLD_P95_LATENCY = 3.5                  # Maximum p95 latency (seconds)
THRESHOLD_RETRIEVAL_HIT_RATE = 80.0          # Minimum retrieval hit rate (%)

# Groq rate limit safety margins
SCHEDULER_MAX_RPM = 8                        # Requests per minute (80% of free tier limit)
SCHEDULER_MAX_TPM = 10000                    # Tokens per minute (conservative)
```

---

## Tech stack

| Component | What's used |
|---|---|
| LLM inference | Groq (`llama-3.3-70b-versatile`) or Ollama (`llama3.2:1b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (in-memory, auto-built from `docs/`) |
| Evaluation runner | Custom Python (`evaluate.py`) |
| Quality gate | Custom Python (`ci_gate.py`) |
| Dashboard | Streamlit + Plotly |
| Backend API | FastAPI + Uvicorn |
| Database | SQLite (`eval_platform.db`) |
| CI/CD | GitHub Actions |
| Public web page | Vanilla HTML/CSS/JS, deployable to Vercel |

---

## Inspiration

This project is modelled after how production ML teams evaluate LLMs:

- **[Braintrust](https://www.braintrustdata.com)** — Dataset management, LLM-as-a-judge, regression scoring
- **[LangSmith](https://smith.langchain.com)** — Trace visualization, evaluation datasets, CI integration
- **[Arize AI](https://arize.com)** — Retrieval diagnostics, embedding drift, latency SLAs
