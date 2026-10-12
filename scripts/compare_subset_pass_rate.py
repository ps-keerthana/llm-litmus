"""
scripts/compare_subset_pass_rate.py
===================================
Computes and compares evaluation pass rates between the human-verified
subset (40 rows) and the full dataset (204 rows).

Usage:
    python scripts/compare_subset_pass_rate.py [path_to_run_file.json]

If no run file is supplied, it automatically evaluates against the latest
run file found in eval_results/.
"""

import os
import sys
import glob
import json
import csv
from typing import Dict, Any, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_verified_keys(dataset_path: str = "golden_dataset.csv"):
    """Loads unique_ids and questions where human_verified is True."""
    verified_ids = set()
    verified_questions = set()
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file not found: {dataset_path}")
        return verified_ids, verified_questions

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("human_verified", "").strip().lower() in ("true", "1", "yes"):
                if row.get("unique_id"):
                    verified_ids.add(row["unique_id"])
                if row.get("question"):
                    verified_questions.add(row["question"].strip().lower())
    return verified_ids, verified_questions


def compare_pass_rates(run_path: str, dataset_path: str = "golden_dataset.csv") -> Dict[str, Any]:
    """Computes overall vs human-verified pass rate for a given evaluation run JSON."""
    verified_ids, verified_questions = load_verified_keys(dataset_path)

    with open(run_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    results = run_data.get("results", [])
    if not results:
        print("[WARNING] Run file contains 0 evaluated query results.")
        return {}

    total_count = len(results)
    total_passed = sum(1 for r in results if r.get("status") == "PASS")
    overall_pass_rate = round((total_passed / total_count) * 100.0, 2)

    def is_verified(r):
        uid = r.get("unique_id")
        if uid and uid in verified_ids:
            return True
        q = r.get("question", "").strip().lower()
        if q and q in verified_questions:
            return True
        return False

    verified_results = [r for r in results if is_verified(r)]
    verified_count = len(verified_results)
    verified_passed = sum(1 for r in verified_results if r.get("status") == "PASS")
    verified_pass_rate = round((verified_passed / verified_count) * 100.0, 2) if verified_count > 0 else 0.0

    print("=" * 60)
    print(f"RUN EVALUATION COMPARISON: {os.path.basename(run_path)}")
    print("=" * 60)
    print(f"Total Queries Evaluated:    {total_count}")
    print(f"  Overall Passed:           {total_passed}/{total_count} ({overall_pass_rate}%)")
    print(f"Human-Verified Queries:     {verified_count}")
    print(f"  Verified Passed:          {verified_passed}/{verified_count} ({verified_pass_rate}%)")
    delta = round(verified_pass_rate - overall_pass_rate, 2)
    sign = "+" if delta >= 0 else ""
    print(f"Delta (Verified vs All):    {sign}{delta} pp")
    print("=" * 60)

    return {
        "run_file": run_path,
        "total_count": total_count,
        "total_passed": total_passed,
        "overall_pass_rate": overall_pass_rate,
        "verified_count": verified_count,
        "verified_passed": verified_passed,
        "verified_pass_rate": verified_pass_rate,
        "delta_pp": delta,
    }


def main():
    if len(sys.argv) > 1:
        run_file = sys.argv[1]
    else:
        run_files = sorted(glob.glob(os.path.join("eval_results", "run_*.json")))
        if not run_files:
            print("No evaluation run files found in eval_results/.")
            sys.exit(1)
        run_file = run_files[-1]

    compare_pass_rates(run_file)


if __name__ == "__main__":
    main()
