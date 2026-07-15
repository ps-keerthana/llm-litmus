"""
CI Quality Gate Runner (ci_gate.py)
Checks the latest evaluation run metrics against absolute standards
and checks relative regressions compared to the previous run.
"""

import os
import glob
import json
import sys
from typing import Dict, Any, List, Tuple
from config import EVAL_RESULTS_DIR
from core.reporter import check_regressions
from core.utils import logger


def load_latest_runs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads the latest two runs from the evaluation results directory.
    Returns (latest_run, baseline_run).
    """
    run_files = sorted(glob.glob(os.path.join(EVAL_RESULTS_DIR, "run_*.json")))
    if not run_files:
        raise FileNotFoundError("No evaluation run files found in the results directory.")

    latest_file = run_files[-1]
    logger.info(f"Loading candidate evaluation run: {latest_file}")
    with open(latest_file, "r", encoding="utf-8") as f:
        latest = json.load(f)

    # Baseline is the previous run (if exists)
    baseline = {}
    if len(run_files) > 1:
        baseline_file = run_files[-2]
        logger.info(f"Loading baseline evaluation run: {baseline_file}")
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    return latest, baseline


def write_markdown_summary(latest: Dict[str, Any], failures: List[str]) -> None:
    """
    Generates a structured markdown summary report (eval_summary.md)
    for GitHub Actions step summaries and PR postings.
    """
    status_emoji = "✅" if not failures else "❌"
    pass_rate = latest.get("pass_rate", 0.0)
    passed = latest.get("passed", 0)
    total = latest.get("total_questions", 0)
    
    # Retrieval stats
    hit_rate = round(latest.get("avg_retrieval_hit_rate", 1.0) * 100.0, 1)
    recall = latest.get("avg_context_recall", 1.0)
    precision = latest.get("avg_context_precision", 1.0)
    mrr = latest.get("avg_retrieval_mrr", 1.0)
    
    # Generation stats
    faithfulness = latest.get("avg_faithfulness", 1.0)
    hallucination = latest.get("hallucination_rate_avg", 0.0)
    p95_lat = latest.get("p95_latency_sec", 0.0)
    avg_cost = latest.get("avg_cost_usd", 0.0)
    
    summary = f"""## {status_emoji} LLM Evaluation CI/CD Report

### Quality Summary
- **Pass Rate**: `{pass_rate}%` ({passed}/{total} queries passed)
- **Status**: {"**PASSED**" if not failures else "**FAILED**"}
- **Commit SHA**: `{latest.get('git_commit_hash', 'unknown')}` on branch `{latest.get('branch', 'unknown')}`

### Retrieval Analytics
- **Hit Rate**: `{hit_rate}%`
- **Mean Reciprocal Rank (MRR)**: `{mrr}`
- **Context Precision / Recall**: `{precision} / {recall}`

### Generation Analytics
- **Avg Faithfulness**: `{faithfulness}` (Hallucination Rate: `{hallucination}`)
- **p95 / Average Latency**: `{p95_lat}s / {latest.get('avg_latency_sec', 0.0)}s`
- **Average Cost per Query**: `${avg_cost:.6f}`

"""
    if failures:
        summary += "### ⚠️ Regression & Threshold Failures\n"
        for fail in failures:
            summary += f"- {fail}\n"
        summary += "\n**Actions**: Fix the regressions or update configuration thresholds before merging.\n"
    else:
        summary += "### 🎉 Success\nAll quality gates and regression checks passed successfully. Ready to merge!\n"

    with open("eval_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Generated markdown report summary to 'eval_summary.md'")


def main() -> None:
    try:
        latest, baseline = load_latest_runs()
    except Exception as exc:
        logger.error(f"Failed to load evaluation runs: {exc}")
        sys.exit(1)

    # Perform quality threshold and regression checks
    failures = check_regressions(latest, baseline)

    logger.info("=" * 60)
    logger.info("CI QUALITY GATES & REGRESSION AUDIT")
    logger.info("=" * 60)
    logger.info(f"Pass Rate:      {latest.get('pass_rate', 0.0)}% (threshold: >= 70.0%)")
    logger.info(f"Hallucination:  {latest.get('hallucination_rate_avg', 0.0):.3f} (threshold: <= 0.05)")
    logger.info(f"p95 Latency:    {latest.get('p95_latency_sec', 0.0):.2f}s (threshold: <= 3.5s)")
    logger.info(f"Hit Rate:       {round(latest.get('avg_retrieval_hit_rate', 1.0) * 100.0, 1)}% (threshold: >= 80.0%)")
    logger.info("-" * 60)

    # Generate the PR-comment ready markdown summary file
    write_markdown_summary(latest, failures)

    if failures:
        logger.error("[GATE FAILED] Regression check or absolute quality gate failed:")
        for fail in failures:
            logger.error(f"  - {fail}")
        logger.info("=" * 60)
        sys.exit(1)
    else:
        logger.info("[GATE PASSED] All threshold and regression boundaries satisfied.")
        logger.info("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
