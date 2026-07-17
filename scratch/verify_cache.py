import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.connection import init_db, get_db_connection
from core.cache import get_cache_key, lookup_cache, update_cache

def test_cache() -> None:
    print("Initializing database...")
    init_db()

    model = "llama-3.3-70b-versatile"
    prompt = "Compute HRA limit for Delhi."
    temp = 0.0

    key = get_cache_key(model, prompt, temp)
    print(f"Generated key: {key}")

    # 1. Verify cache miss
    print("Testing cache lookup (expected miss)...")
    res = lookup_cache(key)
    assert res is None, "Expected cache miss to return None"
    print("Cache miss verified successfully.")

    # 2. Verify cache write
    print("Writing test data to cache...")
    test_data = {
        "answer": "HRA exemption is 50% of basic salary for Delhi.",
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "model_name": model
    }
    update_cache(key, test_data)
    print("Cache write completed.")

    # 3. Verify cache hit
    print("Testing cache lookup (expected hit)...")
    res_hit = lookup_cache(key)
    assert res_hit is not None, "Expected cache hit"
    assert res_hit["answer"] == test_data["answer"], "Answer mismatch"
    assert res_hit["prompt_tokens"] == test_data["prompt_tokens"], "Token mismatch"
    print("Cache hit and validation verified successfully.")

    # 4. Check lineage metadata table records
    print("Checking database record for lineage tracking...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT model_name, prompt_version, dataset_version, commit_sha, created_at FROM eval_cache WHERE cache_key = ?;",
        (key,)
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Row must exist in eval_cache"
    print("Lineage Metadata Records:")
    print(f"  - Model:     {row['model_name']}")
    print(f"  - Prompt v:  {row['prompt_version']}")
    print(f"  - Dataset v: {row['dataset_version']}")
    print(f"  - Git SHA:   {row['commit_sha']}")
    print(f"  - Created:   {row['created_at']}")

    print("\n[SUCCESS] SQLite Cache Layer Step 2 verified successfully!")

if __name__ == "__main__":
    test_cache()
