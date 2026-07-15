"""
Core Reporter Module
Saves run summary metrics into history tracking, checks quality thresholds,
and determines if the latest code state has regressed compared to the previous run.
"""

import os
import json
from typing import Dict, Any, List, Tuple
from config import (
    METRICS_HISTORY_PATH,
    THRESHOLD_PASS_RATE, THRESHOLD_HALLUCINATION, THRESHOLD_P95_LATENCY, THRESHOLD_RETRIEVAL_HIT_RATE,
    REGRESSION_LIMIT_PASS_RATE, REGRESSION_LIMIT_HALLUCINATION,
    REGRESSION_LIMIT_P95_LATENCY_PERCENT, REGRESSION_LIMIT_P95_LATENCY_ABS,
    REGRESSION_LIMIT_COST_PERCENT, REGRESSION_LIMIT_RETRIEVAL_HIT_RATE
)


def append_to_metrics_history(summary: Dict[str, Any]) -> None:
    """
    Appends the summary metric dictionary of the active run to metrics_history.json.
    """
    history = []
    if os.path.exists(METRICS_HISTORY_PATH):
        try:
            with open(METRICS_HISTORY_PATH, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(summary)

    with open(METRICS_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [Reporter] Appended entry to {METRICS_HISTORY_PATH} ({len(history)} total runs)")


def check_regressions(latest: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    """
    Compares the latest evaluation run against a baseline run.
    Checks both absolute thresholds and relative regression boundaries.
    Returns a list of error strings detailing any failures, or an empty list if passing.
    """
    failures = []

    # ── 1. Absolute Threshold Checks ────────────────────────
    latest_pass_rate = latest.get("pass_rate", 0.0)
    if latest_pass_rate < THRESHOLD_PASS_RATE:
        failures.append(f"Pass rate {latest_pass_rate}% falls below absolute threshold {THRESHOLD_PASS_RATE}%")

    latest_hall = latest.get("hallucination_rate_avg", 0.0)
    if latest_hall > THRESHOLD_HALLUCINATION:
        failures.append(f"Average hallucination rate {latest_hall:.3f} exceeds absolute threshold {THRESHOLD_HALLUCINATION}")

    latest_p95 = latest.get("p95_latency_sec", 0.0)
    if latest_p95 > THRESHOLD_P95_LATENCY:
        failures.append(f"p95 latency {latest_p95:.2f}s exceeds absolute threshold {THRESHOLD_P95_LATENCY}s")

    latest_hit = latest.get("avg_retrieval_hit_rate", 1.0) * 100.0  # Convert to %
    if latest_hit < THRESHOLD_RETRIEVAL_HIT_RATE:
        failures.append(f"Retrieval hit rate {latest_hit:.1f}% falls below absolute threshold {THRESHOLD_RETRIEVAL_HIT_RATE}%")

    # ── 2. Relative Regression Checks ────────────────────────
    if baseline:
        base_pass_rate = baseline.get("pass_rate", 0.0)
        pass_drop = base_pass_rate - latest_pass_rate
        if pass_drop > REGRESSION_LIMIT_PASS_RATE:
            failures.append(
                f"Pass rate regressed by {pass_drop:.1f}% "
                f"(from {base_pass_rate}% to {latest_pass_rate}%) exceeding limit of {REGRESSION_LIMIT_PASS_RATE}%"
            )

        base_hall = baseline.get("hallucination_rate_avg", 0.0)
        hall_rise = latest_hall - base_hall
        if hall_rise > REGRESSION_LIMIT_HALLUCINATION:
            failures.append(
                f"Hallucination rate increased by {hall_rise:.3f} "
                f"(from {base_hall:.3f} to {latest_hall:.3f}) exceeding limit of {REGRESSION_LIMIT_HALLUCINATION}"
            )

        base_p95 = baseline.get("p95_latency_sec", 0.0)
        if base_p95 > 0:
            latency_pct = ((latest_p95 - base_p95) / base_p95) * 100.0
            latency_abs = latest_p95 - base_p95
            if latency_pct > REGRESSION_LIMIT_P95_LATENCY_PERCENT and latency_abs > REGRESSION_LIMIT_P95_LATENCY_ABS:
                failures.append(
                    f"p95 latency slowed down by {latency_pct:.1f}% / +{latency_abs:.2f}s "
                    f"(from {base_p95:.2f}s to {latest_p95:.2f}s) exceeding limits"
                )

        base_cost = baseline.get("avg_cost_usd", 0.0)
        latest_cost = latest.get("avg_cost_usd", 0.0)
        if base_cost > 0:
            cost_pct = ((latest_cost - base_cost) / base_cost) * 100.0
            if cost_pct > REGRESSION_LIMIT_COST_PERCENT:
                failures.append(
                    f"Average query cost increased by {cost_pct:.1f}% "
                    f"(from ${base_cost:.6f} to ${latest_cost:.6f}) exceeding limit of {REGRESSION_LIMIT_COST_PERCENT}%"
                )

        base_hit = baseline.get("avg_retrieval_hit_rate", 1.0) * 100.0
        hit_drop = base_hit - latest_hit
        if hit_drop > REGRESSION_LIMIT_RETRIEVAL_HIT_RATE:
            failures.append(
                f"Retrieval hit rate regressed by {hit_drop:.1f}% "
                f"(from {base_hit:.1f}% to {latest_hit:.1f}%) exceeding limit of {REGRESSION_LIMIT_RETRIEVAL_HIT_RATE}%"
            )

    return failures
