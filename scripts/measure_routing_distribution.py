"""
scripts/measure_routing_distribution.py
=======================================
Measures the real percentage of queries resolved at each routing stage:
  - Refusal (auto-pass)
  - Auto-Fail (semantic similarity <= 0.25)
  - Auto-Pass (all 4 signals agree: sem >= 0.85, token F1 >= 0.60, numbers, negation)
  - Sent to LLM Judge (borderline or conflicting signals)
"""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.metrics import compute_token_f1, numbers_consistent, is_contradicting


def analyze_routing(run_path: str):
    with open(run_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    total = len(results)
    if total == 0:
        print("No results found in run file.")
        return

    refusals = 0
    auto_fails = 0
    auto_passes = 0
    sent_to_judge = 0

    fail_reasons = {
        "semantic_below_085": 0,
        "f1_below_060": 0,
        "numbers_inconsistent": 0,
        "contradiction_detected": 0,
    }

    for r in results:
        ans = r.get("answer", "")
        gt = r.get("ground_truth", "")
        sim = r.get("semantic_similarity", 0.0)

        # 1. Refusal check
        refusal_keywords = ["don't have information", "do not have information", "no information"]
        if any(k in ans.lower() for k in refusal_keywords):
            refusals += 1
            continue

        # 2. Auto-fail check
        if sim <= 0.25:
            auto_fails += 1
            continue

        # 3. Multi-signal check
        f1 = compute_token_f1(ans, gt)
        num_ok = numbers_consistent(ans, gt)
        neg_ok = not is_contradicting(ans, gt)

        sem_ok = sim >= 0.85
        f1_ok = f1 >= 0.60

        if not sem_ok:
            fail_reasons["semantic_below_085"] += 1
        if not f1_ok:
            fail_reasons["f1_below_060"] += 1
        if not num_ok:
            fail_reasons["numbers_inconsistent"] += 1
        if not neg_ok:
            fail_reasons["contradiction_detected"] += 1

        if sem_ok and f1_ok and num_ok and neg_ok:
            auto_passes += 1
        else:
            sent_to_judge += 1

    p_pass = (auto_passes / total) * 100.0
    p_fail = (auto_fails / total) * 100.0
    p_ref = (refusals / total) * 100.0
    p_judge = (sent_to_judge / total) * 100.0
    p_saved = ((auto_passes + auto_fails + refusals) / total) * 100.0

    print("=" * 60)
    print(f"ROUTING STAGE ANALYSIS: {os.path.basename(run_path)}")
    print("=" * 60)
    print(f"Total Queries:         {total}")
    print(f"  Auto-Pass (4 signals): {auto_passes:3d} / {total} ({p_pass:5.1f}%)")
    print(f"  Auto-Fail (sim <=0.25):{auto_fails:3d} / {total} ({p_fail:5.1f}%)")
    print(f"  Refusal (safe pass):   {refusals:3d} / {total} ({p_ref:5.1f}%)")
    print(f"  Sent to LLM Judge:     {sent_to_judge:3d} / {total} ({p_judge:5.1f}%)")
    print("-" * 60)
    print(f"Total API Judge Cost Saved (Zero Judge Calls): {p_saved:.1f}%")
    print(f"Why queries missed Auto-Pass (sent to judge):")
    for reason, count in fail_reasons.items():
        print(f"    - {reason:25s}: {count} queries")
    print("=" * 60)


if __name__ == "__main__":
    import glob
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        run_files = sorted(glob.glob(os.path.join("eval_results", "run_*.json")))
        if not run_files:
            print("No evaluation run files found in eval_results/.")
            sys.exit(1)
        path = run_files[-1]
    print(f"Analyzing routing distribution for: {path}")
    analyze_routing(path)
