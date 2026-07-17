"""
Step 8: End-to-End pipeline verification script.
Checks alignment across DB layers, FastAPI endpoints, and Dashboard data loaders.
"""
import os
import sys
import os
import sys

# Resolve project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import urllib.request as urllib_req
from db.connection import get_db_connection

API_BASE = "http://127.0.0.1:8000"

def test_api_active():
    print("Checking if FastAPI server is active...")
    try:
        with urllib_req.urlopen(f"{API_BASE}/health", timeout=5) as r:
            data = json.loads(r.read())
            assert data.get("status") == "ok", "FastAPI server health status is not ok"
            print("  [OK] FastAPI server is active and healthy.")
            print(f"       Dataset Version: {data.get('dataset_version')}")
            print(f"       LLM Model: {data.get('llm_model')}")
            return True
    except Exception as exc:
        print(f"  [FAIL] FastAPI server unreachable or error: {exc}")
        return False

def test_db_alignment_with_api():
    print("Checking alignment of SQLite database with API response...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Fetch latest completed run from DB
    cursor.execute("SELECT run_id, mode, status FROM eval_runs WHERE status = 'completed' ORDER BY completed_at DESC LIMIT 1;")
    db_run = cursor.fetchone()
    if not db_run:
        print("  [SKIP] No completed run found in database.")
        conn.close()
        return True
    
    run_id = db_run["run_id"]
    print(f"  Latest completed run in DB: {run_id}")
    
    # 2. Fetch results from DB
    cursor.execute("SELECT unique_id, status, failure_category, diagnostic_report FROM eval_results WHERE run_id = ?;", (run_id,))
    db_results = {row["unique_id"]: dict(row) for row in cursor.fetchall()}
    
    # 3. Fetch from API
    try:
        with urllib_req.urlopen(f"{API_BASE}/runs/{run_id}", timeout=5) as r:
            api_run = json.loads(r.read())
            assert api_run.get("run_id") == run_id, "API returned wrong run_id"
            api_results = {res["unique_id"]: res for res in api_run.get("results", [])}
            
            # Check length match
            assert len(db_results) == len(api_results), f"Result count mismatch: DB={len(db_results)}, API={len(api_results)}"
            print(f"  [OK] Query result counts match ({len(db_results)} rows).")
            
            # Verify details
            for uid, db_res in db_results.items():
                api_res = api_results.get(uid)
                assert api_res is not None, f"Result {uid} missing in API response"
                
                # Check status
                assert db_res["status"] == api_res["status"], f"Status mismatch for {uid}: DB={db_res['status']}, API={api_res['status']}"
                
                # Check diagnostic report JSON parsing
                db_diag_str = db_res["diagnostic_report"]
                api_diag = api_res["diagnostic_report"]
                
                if db_diag_str:
                    db_diag = json.loads(db_diag_str)
                    assert db_diag == api_diag, f"Diagnostic report mismatch for {uid}"
                    assert "failure_category" in api_diag
                    assert "counterfactual_result" in api_diag
                    assert "evidence" in api_diag
                    assert "recommended_action" in api_diag
                else:
                    assert api_diag is None, f"Expected null diagnostic_report for {uid} in API"
            
            print("  [OK] Database and API outputs match perfectly.")
    except Exception as exc:
        print(f"  [FAIL] Alignment verification failed: {exc}")
        conn.close()
        return False
        
    conn.close()
    return True

def run_verify_dashboard():
    print("Running dashboard loading verification...")
    # Executing the dashboard verification script we built in step 7
    try:
        import subprocess
        res = subprocess.run([sys.executable, "scratch/verify_dashboard.py"], capture_output=True, text=True)
        if res.returncode == 0:
            print("  [OK] Dashboard data loading verification succeeded.")
            print("\n".join("       " + line for line in res.stdout.splitlines() if line))
            return True
        else:
            print(f"  [FAIL] Dashboard loading verification failed (exit code {res.returncode}):")
            print(res.stderr)
            return False
    except Exception as exc:
        print(f"  [FAIL] Failed to run dashboard verification script: {exc}")
        return False

def main():
    print("============================================================")
    print("E2E PIPELINE VERIFICATION")
    print("============================================================")
    
    api_ok = test_api_active()
    if not api_ok:
        sys.exit(1)
        
    db_ok = test_db_alignment_with_api()
    if not db_ok:
        sys.exit(1)
        
    dash_ok = run_verify_dashboard()
    if not dash_ok:
        sys.exit(1)
        
    print()
    print("[SUCCESS] All End-to-End checks passed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
