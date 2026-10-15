"""
SQLite-Backed Content-Addressable Cache Module (core/cache.py)
Implements a database cache for LLM generation and judge outputs,
storing runs and lineage metadata to maximize speed and trace reproducibility.
"""

import json
import hashlib
from typing import Dict, Any, Optional

from config import VERSION_PROMPT, VERSION_DATASET
from db.connection import get_db_connection
from core.utils import get_git_sha, logger

def get_cache_key(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Generates a stable SHA256 key for cache lookups."""
    payload = f"{model}|{prompt}|{temperature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def lookup_cache(key: str) -> Optional[Dict[str, Any]]:
    """
    Looks up a key in the SQLite cache.
    Returns the cached dictionary (decoded from JSON) or None if missing/error.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM eval_cache WHERE cache_key = ? LIMIT 1;", 
            (key,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row["value"])
    except Exception as e:
        logger.warning(f"[Cache Lookup Error] Failed to read from cache: {e}")
    finally:
        if conn:
            conn.close()
    return None

def update_cache(key: str, data: Dict[str, Any]) -> None:
    """
    Inserts or updates an entry in the SQLite cache, including prompt, dataset,
    and git lineage metadata for reproducibility.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Lineage metadata
        commit_sha = get_git_sha()
        model_name = data.get("model_name", "unknown")
        if "metrics" in data and isinstance(data["metrics"], dict):
            # If it's a judge call, we can extract the model info if present
            model_name = f"judge-{model_name}"

        value_str = json.dumps(data)

        cursor.execute(
            """
            INSERT OR REPLACE INTO eval_cache 
            (cache_key, value, model_name, prompt_version, dataset_version, commit_sha)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (key, value_str, model_name, VERSION_PROMPT, VERSION_DATASET, commit_sha)
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"[Cache Write Error] Failed to write to cache: {e}")
    finally:
        if conn:
            conn.close()
