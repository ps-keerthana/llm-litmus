"""
Core Caching Module (core/cache.py)
Implements a simple local file cache for LLM generation and judge outputs,
helping to bypass Groq API rate limits and minimize costs during development.
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional

CACHE_FILE = ".eval_cache.json"


def _load_cache() -> Dict[str, Any]:
    """Loads the cache file from disk safely."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Saves the cache dictionary back to disk safely."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"  [Warning] Failed to write cache file: {e}")


def get_cache_key(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Generates a stable SHA256 key for cache lookups."""
    payload = f"{model}|{prompt}|{temperature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lookup_cache(key: str) -> Optional[Dict[str, Any]]:
    """
    Looks up a key in the local cache.
    Returns the cached dictionary or None if missing.
    """
    cache = _load_cache()
    return cache.get(key)


def update_cache(key: str, data: Dict[str, Any]) -> None:
    """Inserts or updates an entry in the local cache."""
    cache = _load_cache()
    cache[key] = data
    _save_cache(cache)
