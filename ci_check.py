import json
import os
import glob
import sys

# Find all run result files
result_files = sorted(glob.glob("eval_results/run_*.json"))
if not result_files:
    print("ERROR: No eval results found. Did evaluate_dataset.py run?")
    sys.exit(1)

latest_file = result_files[-1]
print(f"Loading latest run results from: {latest_file}")
with open(latest_file, "r") as f:
    latest = json.load(f)

# Define absolute thresholds
PASS_RATE_MIN = 70.0        # Min pass rate (e.g. 70%)
HALLUCINATION_MAX = 0.05    # Max average hallucination rate (5%)
LATENCY_P95_MAX = 3.5       # Max p95 latency in seconds

latest_pass_rate = latest["pass_rate"]
latest_hallucination = latest["hallucination_rate_avg"]
latest_p95_latency = latest.get("p95_latency_sec", latest.get("avg_latency_sec"))
latest_avg_cost = latest.get("avg_cost_usd", 0.0)

print("=" * 50)
print("CI QUALITY GATE - ABSOLUTE CHECKS")
print("=" * 50)
print(f"Pass rate:      {latest_pass_rate}%  (min threshold: {PASS_RATE_MIN}%)")
print(f"Hallucination:  {latest_hallucination}  (max threshold: {HALLUCINATION_MAX})")
print(f"p95 Latency:    {latest_p95_latency}s  (max threshold: {LATENCY_P95_MAX}s)")
print(f"Average Cost:   ${latest_avg_cost:.6f}")
print("=" * 50)

failures = []

# Absolute checks
if latest_pass_rate < PASS_RATE_MIN:
    failures.append(f"FAIL: Pass rate {latest_pass_rate}% is below minimum {PASS_RATE_MIN}%")

if latest_hallucination > HALLUCINATION_MAX:
    failures.append(f"FAIL: Hallucination rate {latest_hallucination} exceeds maximum {HALLUCINATION_MAX}")

if latest_p95_latency > LATENCY_P95_MAX:
    failures.append(f"FAIL: p95 latency {latest_p95_latency}s exceeds maximum {LATENCY_P95_MAX}s")

# Regression checks (if a previous run exists)
if len(result_files) > 1:
    # Find the second to last file
    # We want to skip comparing the latest with itself or files that are not valid
    previous_file = result_files[-2]
    print(f"\nComparing against previous run: {previous_file}")
    try:
        with open(previous_file, "r") as f:
            prev = json.load(f)

        prev_pass_rate = prev["pass_rate"]
        prev_hallucination = prev["hallucination_rate_avg"]
        prev_p95_latency = prev.get("p95_latency_sec", prev.get("avg_latency_sec"))
        prev_avg_cost = prev.get("avg_cost_usd", 0.0)

        # 1. Pass Rate Regression Check
        # Allow drop of at most 5%
        if (prev_pass_rate - latest_pass_rate) > 5.0:
            failures.append(f"FAIL: Pass rate regressed by {prev_pass_rate - latest_pass_rate:.1f}% (from {prev_pass_rate}% to {latest_pass_rate}%)")

        # 2. Hallucination Regression Check
        # Allow increase of at most 2% absolute
        if (latest_hallucination - prev_hallucination) > 0.02:
            failures.append(f"FAIL: Hallucination rate increased by {latest_hallucination - prev_hallucination:.3f} (from {prev_hallucination} to {latest_hallucination})")

        # 3. Latency Regression Check
        # Allow increase of at most 15% (with minimum difference of 0.3s)
        latency_increase_ratio = (latest_p95_latency - prev_p95_latency) / max(prev_p95_latency, 0.1)
        if latency_increase_ratio > 0.15 and (latest_p95_latency - prev_p95_latency) > 0.3:
            failures.append(f"FAIL: p95 latency regressed by {latency_increase_ratio*100:.1f}% (from {prev_p95_latency}s to {latest_p95_latency}s)")

        # 4. Cost Regression Check
        # Allow increase of at most 20%
        if prev_avg_cost > 0:
            cost_increase_ratio = (latest_avg_cost - prev_avg_cost) / prev_avg_cost
            if cost_increase_ratio > 0.20:
                failures.append(f"FAIL: Average cost per query increased by {cost_increase_ratio*100:.1f}% (from ${prev_avg_cost:.6f} to ${latest_avg_cost:.6f})")
    except Exception as e:
        print(f"WARNING: Could not parse previous run {previous_file} for regression check: {e}")

if failures:
    print("\n[FAIL] QUALITY GATE FAILED:")
    for f in failures:
        print(f"  {f}")
    print("\nMerge blocked. Fix the issues above before merging.")
    sys.exit(1)
else:
    print("\n[SUCCESS] QUALITY GATE PASSED - merge allowed")
    sys.exit(0)