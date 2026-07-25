<div align="center">

# ⚗ LLM-Litmus

**Continuous quality gating for production RAG pipelines.**

[![CI](https://github.com/ps-keerthana/llm-litmus/actions/workflows/eval.yml/badge.svg)](https://github.com/ps-keerthana/llm-litmus/actions/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**Live Demo**](https://llm-litmus.vercel.app) · [**Streamlit Dashboard**](#streamlit-dashboard) · [**Quick Start**](#quick-start)

</div>

---

## What is LLM-Litmus?

LLM-Litmus is an automated evaluation and quality-gating platform for Retrieval-Augmented Generation (RAG) systems.

Think of it as **CI/CD for your LLM outputs** — the same way unit tests block broken code from reaching production, LLM-Litmus blocks regressions in AI answer quality before they hit your users.

Every time you change a prompt, swap a model, update a knowledge base, or modify retrieval settings, the pipeline automatically:

1. Runs a benchmark of configurable Q&A queries against your RAG system
2. Measures retrieval accuracy, answer correctness, faithfulness, and latency
3. **Blocks the merge** on GitHub if quality drops below your thresholds
4. Posts a full evaluation report as a PR comment and GitHub Actions summary
5. Saves a run history so you can track quality trends over time

---

## Why not just use LangSmith or Braintrust?

LLM-Litmus is purpose-built for teams that want:

| Capability | LLM-Litmus | LangSmith | Braintrust |
|---|---|---|---|
| **Self-hostable, no SaaS dependency** | ✅ Full source | ❌ Cloud-only | ❌ Cloud-only |
| **Multi-signal oracle (semantic + F1 + numbers + negation)** | ✅ Built-in | ❌ Manual | ❌ Manual |
| **Adversarial test suite** | ✅ Built-in | 🟡 Manual | 🟡 Manual |
| **Native GitHub Actions CI gate** | ✅ Built-in | 🟡 Webhook | 🟡 Webhook |
| **Domain-agnostic (any corpus)** | ✅ Config-driven | ❌ Custom | ❌ Custom |
| **No per-query API cost for evaluation** | ✅ Local oracle | ❌ LLM calls | ❌ LLM calls |

---

## Current benchmark results

> Evaluated on 204 Indian income tax Q&A queries · Groq `llama-3.3-70b-versatile`

| Metric | Value |
|---|---|
| **Pass Rate** | 88.7% (181 / 204 queries) |
| **Retrieval Hit Rate** | 97.5% |
| **nDCG@K** | 0.916 |
| **Mean Average Precision** | 0.725 |
| **Token F1** | 0.407 (avg) |
| **p95 Latency** | ~19s (Groq free-tier rate-limited) |
| **Avg Cost per Query** | $0.000046 |

---

## How it works

```
Query
  │
  ▼
┌─────────────────────────────────────┐
│  Step 1 — Document Ingestion        │
│  .txt/.md corpus → chunked →        │
│  ChromaDB (SHA-256 incremental)     │
└──────────────────┬──────────────────┘
                   │
  ▼
┌─────────────────────────────────────┐
│  Step 2 — Vector Retrieval          │
│  all-MiniLM-L6-v2 embeddings →      │
│  Top-K chunks · Hit Rate, nDCG,     │
│  MRR, Precision@K, MAP, Coverage    │
└──────────────────┬──────────────────┘
                   │
  ▼
┌─────────────────────────────────────┐
│  Step 3 — LLM Answer Generation     │
│  Groq / Ollama / OpenAI / Anthropic │
│  "Answer ONLY from the context."    │
└──────────────────┬──────────────────┘
                   │
  ▼
┌─────────────────────────────────────┐
│  Step 4 — Multi-Signal Oracle Gate  │
│  Semantic sim + Token F1 +          │
│  Numeric extraction + Negation →    │
│  Auto-pass · Auto-fail · LLM judge  │
└──────────────────┬──────────────────┘
                   │
  ▼
┌─────────────────────────────────────┐
│  Step 5 — CI Quality Gate           │
│  Threshold check + Regression vs    │
│  previous run → PASS / BLOCK merge  │
└─────────────────────────────────────┘
```

---

## Project structure

```
llm-eval-pipeline/
│
├── config.py                    # Central configuration: models, thresholds, paths
│
├── core/
│   ├── retrieval.py             # ChromaDB vector store · 6 retrieval metrics
│   ├── generator.py             # Multi-provider LLM client with retry logic
│   ├── judge.py                 # Oracle routing: auto-pass · auto-fail · LLM judge
│   ├── metrics.py               # Token F1 · numeric extraction · negation detection
│   ├── reporter.py              # Run persistence · regression detection
│   ├── attributor.py            # Failure root-cause attribution
│   ├── cache.py                 # Embedding and result caching
│   ├── scheduler.py             # Proactive Groq rate-limit scheduler (SQLite token bucket)
│   ├── telemetry.py             # Structured per-phase latency tracing (JSONL)
│   └── providers/
│       ├── groq.py              # Groq (llama-3.3-70b-versatile)
│       ├── ollama.py            # Local Ollama (llama3.2:1b)
│       ├── openai.py            # OpenAI (gpt-4o-mini)
│       ├── anthropic.py         # Anthropic (claude-3-5-haiku)
│       ├── base.py              # Abstract provider interface
│       └── factory.py           # Provider factory
│
├── evaluate.py                  # Main CLI — full benchmark or smoke test
├── ci_gate.py                   # Quality gate checker · threshold + regression
├── generate_dataset.py          # Domain-agnostic Q&A dataset synthesizer
│
├── golden_dataset.csv           # 204-question benchmark (questions + ground truths)
├── metrics_history.json         # Historical run metrics (auto-updated by CI)
│
├── docs/                        # Knowledge base (8 Indian tax reference documents)
├── datasets/adversarial/        # Adversarial prompt injection test suite
├── dashboard/app.py             # Multi-page Streamlit analytics dashboard
├── web/index.html               # Public landing page (Vercel-deployable)
├── api/                         # FastAPI REST backend
├── db/                          # SQLite database helpers
├── tests/                       # Unit test suite
│
└── .github/workflows/eval.yml   # GitHub Actions CI/CD pipeline
```

---

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/ps-keerthana/llm-litmus.git
cd llm-eval-pipeline

python -m venv venv
.\venv\Scripts\activate        # Windows
# or: source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the smoke test (32 queries, ~3 min)

```bash
python evaluate.py --smoke
```

### 4. Run the full benchmark (204 queries, ~15 min)

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

Open [http://localhost:8501](http://localhost:8501).

---

## Supported LLM providers

| Provider | Model | Notes |
|---|---|---|
| **Groq** (default) | `llama-3.3-70b-versatile` | Fast, free tier, best for CI |
| **Ollama** | `llama3.2:1b` | Fully local, no API key needed |
| **OpenAI** | `gpt-4o-mini` | High quality, pay-per-use |
| **Anthropic** | `claude-3-5-haiku-20241022` | Alternative commercial option |

Switch provider via environment variable:

```bash
LLM_PROVIDER=ollama python evaluate.py --smoke
LLM_PROVIDER=openai  python evaluate.py --smoke
```

---

## Multi-signal oracle gate

The evaluation uses a 4-signal composite gate instead of a simple similarity threshold. Auto-pass requires **all** signals to agree:

| Signal | What it checks | Why it matters |
|---|---|---|
| **Semantic similarity** | Cosine distance (≥ 0.85) | Overall meaning alignment |
| **Token F1** | Lexical word overlap (≥ 0.60) | Catches same-meaning, wrong-number answers |
| **Numeric consistency** | All GT numbers present in answer | Catches ₹1.5L vs ₹2L substitutions |
| **Negation detection** | Answer doesn't flip GT polarity | Catches "can be claimed" vs "cannot be claimed" |

If all signals pass → **auto-pass** (no LLM call needed, saves API tokens).  
If semantic sim ≤ 0.25 → **auto-fail** (clearly wrong or hallucinated).  
Otherwise → **LLM judge** (or ensemble judge if `JUDGE_ENSEMBLE=true`).

---

## Quality thresholds

These thresholds block merges if breached:

| Metric | Threshold |
|---|---|
| Pass rate | ≥ 70% |
| Hallucination rate | ≤ 5% |
| p95 Latency | ≤ 3.5s |
| Retrieval hit rate | ≥ 80% |

Regression detection also fires if the latest run is significantly worse than the previous:

| Metric | Max allowed decay |
|---|---|
| Pass rate | −5 pp |
| Hallucination rate | +0.02 (abs) |
| p95 Latency | +15% or +0.3s |
| Avg cost per query | +20% |
| Retrieval hit rate | −5 pp |

---

## Streamlit dashboard

| Page | Contents |
|---|---|
| **Overview & KPI Matrix** | Core metrics, delta vs previous run, failure breakdown |
| **Metric Trends** | Multi-run historical charts |
| **Regression Analysis** | Side-by-side run comparison |
| **Failure Explorer** | Searchable failure database with full traces |
| **Retrieval Inspector** | Visual chunk flow: Question → Chunks → Answer |
| **Cost Analytics** | Cost by category, most expensive queries |
| **Latency Analytics** | p50/p95/p99 percentile charts |
| **Prompt Playground** | Interactive sandbox: test prompts live |
| **Dataset Explorer** | Filterable 204-question benchmark view |

---

## Configuration reference

All settings live in [`config.py`](config.py). Key environment variables:

```bash
EVAL_DOMAIN=legal                          # Domain label (default: general)
EVAL_DOMAIN_DESCRIPTION="Legal RAG"       # Used in judge prompts
EVAL_COLLECTION_NAME=my_collection        # ChromaDB collection name

LLM_PROVIDER=groq                         # groq | ollama | openai | anthropic
GROQ_MODEL_NAME=llama-3.3-70b-versatile  # Override Groq model

CHUNK_STRATEGY=paragraph                  # paragraph | sentence | fixed_size
CHROMA_PERSISTENT=true                    # Persist ChromaDB to disk

JUDGE_ENSEMBLE=true                       # Enable multi-model ensemble judging
```

---

## Running tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=core --cov-report=term-missing
```

---

## Deploying the landing page

The `web/index.html` file is a self-contained public landing page with an interactive evaluation console. Deploy to Vercel in seconds:

1. Go to [vercel.com](https://vercel.com) and import `ps-keerthana/llm-litmus`
2. Set root directory to `web/`
3. Framework preset: **None**
4. Deploy

---

## Tech stack

| Component | Technology |
|---|---|
| LLM inference | Groq (`llama-3.3-70b-versatile`) · Ollama · OpenAI · Anthropic |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent + incremental indexing) |
| Rate limiting | SQLite token-bucket scheduler |
| Dashboard | Streamlit + Plotly |
| Backend API | FastAPI + Uvicorn |
| Database | SQLite |
| CI/CD | GitHub Actions |
| Landing page | Vanilla HTML/CSS/JS · Vercel |
| Tests | pytest |

---

## Inspiration

Built studying how production ML teams do LLM evaluation:

- **[Braintrust](https://www.braintrustdata.com)** — Dataset management and LLM-as-a-judge patterns
- **[LangSmith](https://smith.langchain.com)** — Trace visualization and evaluation datasets
- **[Arize AI](https://arize.com)** — Retrieval diagnostics and embedding drift monitoring

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding guidelines, and how to run tests.

## License

MIT — see [LICENSE](LICENSE).
