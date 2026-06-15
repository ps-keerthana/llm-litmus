import json
import os
import glob
import sys

# find the most recent eval result
result_files = glob.glob("eval_results/run_*.json")
if not result_files:
    print("ERROR: No eval results found. Did evaluate_dataset.py run?")
    sys.exit(1)

latest = max(result_files)
with open(latest, "r") as f:
    results = json.load(f)

pass_rate = results["pass_rate"]
hallucination = results["hallucination_rate_avg"]
avg_latency = results["avg_latency_sec"]

# ── define your thresholds here ──────────────────────────
PASS_RATE_MIN = 65.0       # fail if pass rate drops below 65%
HALLUCINATION_MAX = 0.05   # fail if avg hallucination exceeds 0.05
LATENCY_MAX = 5.0          # fail if avg latency exceeds 5 seconds

print("=" * 50)
print("CI QUALITY GATE CHECK")
print("=" * 50)
print(f"Pass rate:      {pass_rate}%  (min: {PASS_RATE_MIN}%)")
print(f"Hallucination:  {hallucination}  (max: {HALLUCINATION_MAX})")
print(f"Avg latency:    {avg_latency}s  (max: {LATENCY_MAX}s)")
print("=" * 50)

failures = []

if pass_rate < PASS_RATE_MIN:
    failures.append(
        f"FAIL: Pass rate {pass_rate}% is below minimum {PASS_RATE_MIN}%"
    )

if hallucination > HALLUCINATION_MAX:
    failures.append(
        f"FAIL: Hallucination {hallucination} exceeds maximum {HALLUCINATION_MAX}"
    )

if avg_latency > LATENCY_MAX:
    failures.append(
        f"FAIL: Latency {avg_latency}s exceeds maximum {LATENCY_MAX}s"
    )

if failures:
    print("\nQUALITY GATE FAILED:")
    for f in failures:
        print(f"  {f}")
    print("\nMerge blocked. Fix the issues above before merging.")
    sys.exit(1)  # non-zero exit code = GitHub marks the check as failed
else:
    print("\nQUALITY GATE PASSED - merge allowed")
    sys.exit(0)