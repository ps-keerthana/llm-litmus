"""
Verifies that diagnostic_report is correctly stored in eval_results table.
"""
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.connection import get_db_connection, init_db

def verify():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check column exists
    cursor.execute("PRAGMA table_info(eval_results);")
    columns = [row["name"] for row in cursor.fetchall()]
    assert "diagnostic_report" in columns, f"diagnostic_report column missing! Got: {columns}"
    print(f"[OK] diagnostic_report column present in eval_results table.")

    # Check actual data
    cursor.execute("SELECT unique_id, status, failure_category, diagnostic_report FROM eval_results LIMIT 10;")
    rows = cursor.fetchall()
    assert len(rows) > 0, "No rows found in eval_results!"

    fail_count = 0
    pass_count = 0
    for row in rows:
        uid = row["unique_id"]
        status = row["status"]
        failure_cat = row["failure_category"]
        diag_raw = row["diagnostic_report"]

        if status == "FAIL":
            fail_count += 1
            assert diag_raw is not None, f"diagnostic_report is NULL for FAIL record {uid}!"
            report = json.loads(diag_raw)
            assert "failure_category" in report, f"Missing failure_category in report for {uid}"
            assert "counterfactual_result" in report, f"Missing counterfactual_result in report for {uid}"
            assert "evidence" in report, f"Missing evidence in report for {uid}"
            assert "recommended_action" in report, f"Missing recommended_action in report for {uid}"
            print(f"  [FAIL] {uid}: category={report['failure_category']} | cf={report['counterfactual_result']}")
            print(f"         action: {report['recommended_action']}")
        else:
            pass_count += 1
            # PASS rows may have NULL diagnostic_report
            print(f"  [PASS] {uid}: diagnostic_report=NULL (expected)")

    print(f"\n[OK] Verified {pass_count} PASS and {fail_count} FAIL records.")
    print("[SUCCESS] diagnostic_report storage in eval_results verified!")
    conn.close()

if __name__ == "__main__":
    verify()
