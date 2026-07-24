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
import os
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # Options: "groq", "ollama", "openai", "anthropic"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:1b")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-haiku-20241022")

MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Groq pricing details per 1 million tokens (USD)
PRICE_INPUT_1M = 0.05
PRICE_OUTPUT_1M = 0.08

# Default inference hyperparameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 5

# ── Version Tracking ──────────────────────────────────────
VERSION_DATASET = "1.0.0"
VERSION_PROMPT = "1.3.0"      # Multi-hop retrieval + boundary-hardened prompt
VERSION_RETRIEVER = "1.0.0"   # ChromaDB vector store

VERSION_EMBEDDING = EMBEDDING_MODEL_NAME
VERSION_LLM = MODEL_NAME

# ── Absolute Quality Thresholds ───────────────────────────
THRESHOLD_PASS_RATE = 70.0        # Minimum pass percentage (%)
THRESHOLD_HALLUCINATION = 0.05    # Maximum average hallucination rate (5%)
THRESHOLD_P95_LATENCY = 3.5       # Maximum p95 latency (seconds)
THRESHOLD_RETRIEVAL_HIT_RATE = 80.0  # Minimum retrieval hit rate (%)

# ── Failure Attribution Thresholds ───────────────────────────
ATTRIBUTION_RECALL_MIN = 0.50          # Recall below this indicates retrieval issue/KB gap
ATTRIBUTION_SIM_MIN = 0.50             # Semantic similarity below this indicates mismatch
ATTRIBUTION_JUDGE_CORRECTNESS_MIN = 0.75  # LLM judge correctness threshold
ATTRIBUTION_FAITHFULNESS_MIN = 0.70    # LLM judge faithfulness threshold
ATTRIBUTION_HALLUCINATION_MAX = 0.10   # Hallucination threshold

# ── Regression Detection Limits ───────────────────────────
# If the metric decays by more than this limit, the build is flagged as regressed.
REGRESSION_LIMIT_PASS_RATE = 5.0            # Max drop in pass rate (percentage points)
REGRESSION_LIMIT_HALLUCINATION = 0.02       # Max increase in hallucination rate (abs)
REGRESSION_LIMIT_P95_LATENCY_PERCENT = 15.0  # Max slowdown in p95 latency (%)
REGRESSION_LIMIT_P95_LATENCY_ABS = 0.3       # Max absolute slowdown in p95 latency (sec)
REGRESSION_LIMIT_COST_PERCENT = 20.0        # Max increase in average query cost (%)
REGRESSION_LIMIT_RETRIEVAL_HIT_RATE = 5.0   # Max drop in retrieval hit rate (%)

# ── Platform Database & Oracle Routing ──────────────────
DB_PATH = "eval_platform.db"
ORACLE_AUTO_PASS_THRESHOLD = 0.85
ORACLE_AUTO_FAIL_THRESHOLD = 0.25

# ── Proactive Request Scheduler ──────────────────────────
# Groq free-tier limits for llama-3.3-70b-versatile: 15 RPM, 14,400 TPM.
# Safety margins applied: 80% of RPM, 83% of TPM.
# These values are read by core/scheduler.py — change here to tune without
# touching any other module.
SCHEDULER_BUCKET_ID = "groq_default"       # Unique bucket key in SQLite
SCHEDULER_MAX_RPM: int = 8                  # Max requests per minute (more conservative safety margin)
SCHEDULER_MAX_TPM: int = 10000              # Max tokens per minute (more conservative safety margin)
SCHEDULER_ESTIMATED_OUTPUT_TOKENS: int = 256  # Conservative pre-debit for completions
SCHEDULER_MIN_SPACING_SEC: float = 6.0      # Minimum spacing in seconds between requests to avoid burst limits


