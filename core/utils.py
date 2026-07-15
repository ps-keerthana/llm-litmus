"""
Core Utilities Module
Provides helper functions for cost calculation, Git tracking (commit SHA & branch),
and robust API connection management.
"""

import subprocess
import logging
from typing import Dict, Any

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LLMEvalPipeline")


def get_git_sha() -> str:
    """
    Retrieves the current git commit SHA (short format).
    Returns 'unknown' if not inside a git repository or command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    """
    Retrieves the current git branch name.
    Returns 'unknown' if not inside a git repository or command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculates the API query cost in USD based on input/output pricing configurations.
    """
    from config import PRICE_INPUT_1M, PRICE_OUTPUT_1M
    cost = (prompt_tokens * PRICE_INPUT_1M + completion_tokens * PRICE_OUTPUT_1M) / 1_000_000
    return round(cost, 6)
