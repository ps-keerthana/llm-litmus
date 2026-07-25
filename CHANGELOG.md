# Changelog

All notable changes to LLM-Litmus are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.0.0] — 2026-07-24

### Major additions

**Multi-Signal Oracle Gate (Phase 3)**
- Replaced single-threshold embedding gate with a 4-signal composite oracle: semantic similarity, token F1, numeric claim extraction, and negation/polarity detection.
- Prevents false auto-passes on answers that are semantically plausible but numerically wrong (e.g. ₹1.5L vs ₹2L).
- `core/metrics.py`: `compute_token_f1`, `extract_numbers`, `numbers_consistent`, `is_contradicting`, `multi_signal_auto_pass`.

**Ensemble Judge (Phase 4)**
- Opt-in multi-model judge (`JUDGE_ENSEMBLE=true`) that runs two Groq models (70B at 60% weight, 8B at 40% weight) and returns weighted-average scores.
- Disagreement detection: flags results where judges differ by > `JUDGE_DISAGREEMENT_THRESHOLD`.

**Persistent ChromaDB + Incremental Indexing (Phase 6)**
- `CHROMA_PERSISTENT=true` persists the vector index to disk across runs.
- SHA-256 file hashing ensures only modified documents are re-indexed.
- Configurable chunk strategies: `paragraph`, `sentence`, `fixed_size`.

**Structured Telemetry (Phase 7)**
- `core/telemetry.py`: per-query JSONL traces recording latency for each pipeline phase (retrieval, generation, judge, attribution).
- Latency percentile aggregation excludes in-memory cache hits to avoid p95 collapsing to ~0ms.

**A/B Run Comparison (Phase 8)**
- `ci_gate.py --compare run_A.json run_B.json`: side-by-side diff of two evaluation runs with per-category pass rate deltas.

**Adversarial Evaluation (Phase 9)**
- `datasets/adversarial/adversarial_dataset.csv`: dedicated suite for prompt injection, negation traps, misleading retrieval, and missing-context handling.
- `evaluate.py --adversarial`: runs adversarial suite alongside or independently of main benchmark.

**Domain-Agnostic Architecture (Phase 10)**
- `config.py`: `EVAL_DOMAIN`, `EVAL_DOMAIN_DESCRIPTION`, `EVAL_COLLECTION_NAME` — makes every component (judge prompts, ChromaDB collections, reports) domain-configurable via environment variables.

### Fixed

- **Auto-fail faithfulness bug**: `semantic_sim <= 0.25` branch was previously returning `faithfulness=1.0` and `hallucination=0.0` (inverted). Fixed to `faithfulness=0.0`, `hallucination=1.0`.
- **MAP duplication bug**: duplicate retrieved sources from the same document were inflating MAP scores above 1.0. Fixed with source deduplication before MAP calculation.
- **Latency p95 collapse**: in-memory cache hits (~0ms) were included in percentile calculation, causing p95 to collapse to 0.0s on repeated runs. Now filtered out.

### Streamlit Dashboard (Phase 5)

Added 3 new pages:
- **Run Comparison**: side-by-side metric diff for any two stored runs.
- **Trace Replay**: step-by-step trace viewer for any individual query.
- **Adversarial Explorer**: dedicated view for adversarial test results.

### Landing page (web/)

- Complete redesign from generic chatbot demo to live interactive product console.
- 5-stage animated pipeline visualization.
- 5 domain examples (Legal, Healthcare, HR, Finance, Adversarial) with full trace data.
- Open Graph meta tags, skip-to-main-content link, `aria-live` regions, keyboard-navigable feed chips, `prefers-reduced-motion` support.
- Chart.js pass rate trend and run distribution charts.

---

## [1.3.0] — Prompt hardening

### Changed
- `VERSION_PROMPT` bumped to `1.3.0`.
- System prompt boundary-hardened: explicit out-of-scope refusal instruction added.
- Multi-hop retrieval improved by including more context chunks (top-K increased from 3 to 5).

---

## [1.2.0] — Rate limit scheduler

### Added
- `core/scheduler.py`: proactive SQLite token-bucket scheduler. Spaces requests to stay within Groq free-tier limits (15 RPM, 14,400 TPM) without manual `time.sleep` calls.
- `core/scheduler.py`: `acquire()`, `refund()`, `drain()` — thread-safe with SQLite WAL mode.
- `tests/test_scheduler.py`: 7 test classes covering cold start, debit, refund, refill, concurrency, drain, and spacing.

---

## [1.1.0] — Retrieval metrics

### Added
- `core/retrieval.py`: 6 retrieval quality metrics: Hit Rate, MRR, nDCG@K, Precision@K, MAP, Coverage.
- All metrics computed per-query and aggregated per-run.

### Fixed
- MAP bug: duplicate sources inflated score above 1.0 on queries where the same document appeared multiple times in Top-K.

---

## [1.0.0] — Initial release

### Added
- End-to-end RAG evaluation pipeline: ChromaDB ingestion → vector retrieval → LLM generation → semantic similarity scoring → CI quality gate.
- 204-question Indian income tax benchmark dataset (`golden_dataset.csv`).
- 4 LLM providers: Groq, Ollama, OpenAI, Anthropic.
- GitHub Actions CI workflow with PR comment reporting.
- Streamlit dashboard (6 pages: Overview, Trends, Regression, Failures, Retrieval Inspector, Cost, Latency, Playground, Dataset Explorer).
- FastAPI backend service.
- SQLite evaluation database.
