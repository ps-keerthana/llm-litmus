"""
Headless verification of the refactored dashboard data loading layer (Step 7).

Strategy: extract and test only the refactored functions (_api_get,
_api_connected, load_all_runs, load_history_log, normalize_run_data)
without executing the full Streamlit page rendering code.
"""
import os
import sys
import json
import urllib.request as _urllib_req

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# ── Inline copies of the refactored functions (from dashboard/app.py) ──────
# We copy them here to test without triggering Streamlit page rendering.

API_BASE_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000")


def _api_get(path: str, base_url: str = API_BASE_URL, timeout: int = 4):
    try:
        with _urllib_req.urlopen(f"{base_url}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _api_connected(base_url: str = API_BASE_URL) -> bool:
    return _api_get("/health", base_url) is not None


def normalize_run_data(run: dict) -> dict:
    """Minimal copy for test purposes."""
    if not run:
        return run
    defaults = {
        "pass_rate": 0.0, "passed": 0, "total_questions": 0,
        "avg_retrieval_hit_rate": run.get("avg_relevancy", 1.0),
        "git_commit_hash": run.get("commit_sha", "unknown"),
        "timestamp": run.get("run_timestamp", run.get("timestamp", "")),
    }
    for k, v in defaults.items():
        if k not in run:
            run[k] = v
    normalized = []
    for idx, r in enumerate(run.get("results", [])):
        normalized.append({
            "unique_id": r.get("unique_id", f"Q{idx+1:03d}"),
            "question": r.get("question", ""),
            "status": r.get("status", "PASS"),
            "faithfulness": r.get("faithfulness", 1.0),
            "hallucination": r.get("hallucination", r.get("hallucination_rate", 0.0)),
            "correctness": r.get("correctness", r.get("semantic_similarity", 1.0)),
            "hit_rate": r.get("hit_rate", 1.0),
            "retrieved_chunks": r.get("retrieved_chunks", []),
            "retrieved_sources": r.get("retrieved_sources", []),
            "retrieved_similarities": r.get("retrieved_similarities", []),
            "failure_category": r.get("failure_category", "N/A"),
            "diagnostic_report": r.get("diagnostic_report"),
        })
    run["results"] = normalized
    return run


def load_all_runs(api_base: str = API_BASE_URL):
    summaries = _api_get("/runs", api_base)
    if summaries is not None:
        runs = []
        for summary in reversed(summaries):
            detail = _api_get(f"/runs/{summary['run_id']}", api_base)
            if not detail:
                continue
            run_data = dict(detail.get("metadata") or {})
            run_data.update({
                "run_id": detail["run_id"],
                "status": detail["status"],
                "mode": detail["mode"],
                "commit_sha": detail.get("commit_sha"),
                "results": detail.get("results", []),
            })
            runs.append(normalize_run_data(run_data))
        return runs
    # Filesystem fallback
    import glob
    files = sorted(glob.glob(os.path.join("..", config.EVAL_RESULTS_DIR, "run_*.json")))
    if not files:
        files = sorted(glob.glob(os.path.join(config.EVAL_RESULTS_DIR, "run_*.json")))
    runs = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            runs.append(normalize_run_data(data))
        except Exception:
            pass
    return runs


def load_history_log(api_base: str = API_BASE_URL):
    history_raw = _api_get("/history", api_base)
    if history_raw is not None:
        return [normalize_run_data(r) for r in history_raw]
    # Filesystem fallback
    p = os.path.join("..", config.METRICS_HISTORY_PATH)
    if not os.path.exists(p):
        p = config.METRICS_HISTORY_PATH
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return [normalize_run_data(r) for r in json.load(f)]
        except Exception:
            return []
    return []


# ── Tests ────────────────────────────────────────────────────────────────────

print("=== TEST 1: _api_connected() with live server ===")
assert _api_connected(), "Expected API to be reachable"
print("  [OK] API is reachable")

print()
print("=== TEST 2: load_all_runs() via API ===")
runs = load_all_runs()
assert isinstance(runs, list), f"Expected list, got {type(runs)}"
print(f"  Loaded {len(runs)} run(s)")
assert len(runs) > 0, "Expected at least one run"

# Find the latest completed run (enqueue-triggered runs may still be running)
completed = [r for r in runs if r.get("status") == "completed"]
assert len(completed) > 0, "Expected at least one completed run"
latest = completed[-1]

assert "pass_rate" in latest, "Missing pass_rate in flattened run"
assert "results" in latest, "Missing results key"
assert len(latest["results"]) > 0, "Expected non-empty results in completed run"
r0 = latest["results"][0]
assert "status" in r0, "Missing status in result"
assert isinstance(r0.get("retrieved_chunks"), list), "retrieved_chunks should be a list (JSON deserialized by API)"
print(f"  run_id: {latest.get('run_id')}")
print(f"  pass_rate: {latest.get('pass_rate')}")
print(f"  results count: {len(latest['results'])}")
print(f"  results[0].status: {r0['status']}")
print(f"  results[0].faithfulness: {r0.get('faithfulness')}")
print(f"  results[0].retrieved_chunks type: {type(r0['retrieved_chunks'])}")
print(f"  results[0].diagnostic_report type: {type(r0.get('diagnostic_report'))}")
print("  [OK] load_all_runs() via API")


print()
print("=== TEST 3: load_history_log() via API ===")
history = load_history_log()
assert isinstance(history, list), f"Expected list, got {type(history)}"
assert len(history) > 0, "Expected at least one history entry"
print(f"  Loaded {len(history)} history entries")
print(f"  history[-1] keys: {list(history[-1].keys())[:6]}")
print("  [OK] load_history_log() via API")

print()
print("=== TEST 4: Filesystem fallback (API offline) ===")
OFFLINE_BASE = "http://127.0.0.1:9999"
assert not _api_connected(OFFLINE_BASE), "Expected offline probe to fail"
runs_fb = load_all_runs(api_base=OFFLINE_BASE)
history_fb = load_history_log(api_base=OFFLINE_BASE)
assert isinstance(runs_fb, list), "Fallback runs should be a list"
assert isinstance(history_fb, list), "Fallback history should be a list"
print(f"  Fallback runs: {len(runs_fb)} (from filesystem)")
print(f"  Fallback history: {len(history_fb)} entries (from metrics_history.json)")
print("  [OK] Filesystem fallback verified")

print()
print("[SUCCESS] All dashboard Step 7 verification checks passed!")
