"""
Central Configuration Module
Defines paths, model details, API pricing, pipeline component versions,
and threshold constraints for quality checks and regression detection.
"""

# ── File System Paths ──────────────────────────────────────
DATASET_PATH = "golden_dataset.csv"
DOCS_FOLDER = "docs"
EVAL_RESULTS_DIR = "eval_results"
METRICS_HISTORY_PATH = "metrics_history.json"

# ── Model & API Settings ──────────────────────────────────
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Groq pricing details per 1 million tokens (USD)
PRICE_INPUT_1M = 0.05
PRICE_OUTPUT_1M = 0.08

# Default inference hyperparameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 3

# ── Version Tracking ──────────────────────────────────────
VERSION_DATASET = "1.0.0"
VERSION_PROMPT = "1.1.0"      # Optimized LLM-judge prompts & refusal checks
VERSION_RETRIEVER = "1.0.0"   # ChromaDB vector store
VERSION_EMBEDDING = EMBEDDING_MODEL_NAME
VERSION_LLM = MODEL_NAME

# ── Absolute Quality Thresholds ───────────────────────────
THRESHOLD_PASS_RATE = 70.0        # Minimum pass percentage (%)
THRESHOLD_HALLUCINATION = 0.05    # Maximum average hallucination rate (5%)
THRESHOLD_P95_LATENCY = 3.5       # Maximum p95 latency (seconds)
THRESHOLD_RETRIEVAL_HIT_RATE = 80.0  # Minimum retrieval hit rate (%)

# ── Regression Detection Limits ───────────────────────────
# If the metric decays by more than this limit, the build is flagged as regressed.
REGRESSION_LIMIT_PASS_RATE = 5.0            # Max drop in pass rate (percentage points)
REGRESSION_LIMIT_HALLUCINATION = 0.02       # Max increase in hallucination rate (abs)
REGRESSION_LIMIT_P95_LATENCY_PERCENT = 15.0  # Max slowdown in p95 latency (%)
REGRESSION_LIMIT_P95_LATENCY_ABS = 0.3       # Max absolute slowdown in p95 latency (sec)
REGRESSION_LIMIT_COST_PERCENT = 20.0        # Max increase in average query cost (%)
REGRESSION_LIMIT_RETRIEVAL_HIT_RATE = 5.0   # Max drop in retrieval hit rate (%)
