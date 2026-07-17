import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import core.generator as gen_mod

# Mock generate_answer to avoid expensive/rate-limited Groq API calls during testing
def mock_generate(*args, **kwargs):
    return "The standard deduction limit is Rs 50,000 for FY 2023-24.", 50, 20

gen_mod.generate_answer = mock_generate

from db.connection import init_db, get_db_connection
from core.queue import enqueue_run, process_queue
import core.queue as queue_mod
import core.attributor as attr_mod

queue_mod.generate_answer = mock_generate
attr_mod.generate_answer = mock_generate

def test_queue() -> None:
    print("Initializing database...")
    init_db()

    # Clear run details for verification clean run
    conn = get_db_connection()
    conn.execute("DELETE FROM eval_results;")
    conn.execute("DELETE FROM eval_queue;")
    conn.execute("DELETE FROM eval_runs;")
    conn.commit()
    conn.close()

    print("Enqueueing a SMOKE run...")
    run_id = enqueue_run("smoke")
    print(f"Created run: {run_id}")

    # Verify queue count
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM eval_queue WHERE run_id = ? AND status = 'pending';", (run_id,))
    count_pending = cursor.fetchone()["cnt"]
    print(f"Pending tasks count: {count_pending} (expected 10)")
    assert count_pending == 10, "Expected exactly 10 queued items for smoke mode"
    conn.close()

    # Process queue with no_judge=True to speed up verification without calling judge API
    print("Processing run queue...")
    summary = process_queue(run_id, no_judge=True)
    print("Queue processing complete.")

    # Check tasks statuses
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT status, COUNT(*) as cnt FROM eval_queue WHERE run_id = ? GROUP BY status;", (run_id,))
    statuses = cursor.fetchall()
    print("Queue status distribution:")
    for row in statuses:
        print(f"  - {row['status']}: {row['cnt']}")
        assert row["status"] == "completed", f"All tasks should be completed, found: {row['status']}"

    # Check runs status
    cursor.execute("SELECT status, metadata, completed_at FROM eval_runs WHERE run_id = ?;", (run_id,))
    run_row = cursor.fetchone()
    print("Run Record Status:")
    print(f"  - Status:       {run_row['status']}")
    print(f"  - Completed At: {run_row['completed_at']}")
    assert run_row["status"] == "completed", "Expected run status to be completed"

    # Check results table records
    cursor.execute("SELECT COUNT(*) as cnt FROM eval_results WHERE run_id = ?;", (run_id,))
    res_cnt = cursor.fetchone()["cnt"]
    print(f"Results recorded: {res_cnt} (expected 10)")
    assert res_cnt == 10, "Expected exactly 10 result rows recorded"

    conn.close()
    print("\n[SUCCESS] SQLite Queue and Worker Step 3 verified successfully!")

if __name__ == "__main__":
    test_queue()
