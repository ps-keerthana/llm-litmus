"""
CI Quality Gate Runner (ci_gate.py)

Phase 1: Domain-agnostic — report uses config.DOMAIN in headers.
Phase 2: Extended retrieval metrics (nDCG@K, Precision@K, MAP, Coverage) appear in report.
Phase 7: --compare mode for explicit A/B run comparison (prompt A vs B, model A vs B, etc.)
         Generates comparison_report.md alongside eval_summary.md.
"""

import os
import glob
import json
import sys
from typing import Dict, Any, List, Tuple, Union

import config
from config import EVAL_RESULTS_DIR
from core.reporter import check_regressions
from core.utils import logger


# ── Helpers ───────────────────────────────────────────────────────────────

def _fmt(val: Any, fmt: str = ".3f", fallback: str = "N/A") -> str:
    """Safely format a metric that may be a float or a sentinel string."""
    if isinstance(val, (int, float)):
        return format(val, fmt)
    return str(val) if val is not None else fallback


def _pct(val: float) -> str:
    return f"{round(val * 100.0, 1)}%"


def _delta(new: Any, old: Any, higher_is_better: bool = True, fmt: str = ".3f") -> str:
    """Return a delta string with ↑/↓ arrow."""
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return "—"
    diff = new - old
    arrow = ("↑" if diff >= 0 else "↓") if higher_is_better else ("↓" if diff >= 0 else "↑")
    sign = "+" if diff >= 0 else ""
    return f"{arrow} {sign}{format(diff, fmt)}"


# ── Load Runs ─────────────────────────────────────────────────────────────

def load_latest_runs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads the two most recent runs. Returns (latest, baseline)."""
    run_files = sorted(glob.glob(os.path.join(EVAL_RESULTS_DIR, "run_*.json")))
    if not run_files:
        raise FileNotFoundError("No evaluation run files found in eval_results/.")

    with open(run_files[-1], "r", encoding="utf-8") as f:
        latest = json.load(f)

    baseline: Dict[str, Any] = {}
    if len(run_files) > 1:
        with open(run_files[-2], "r", encoding="utf-8") as f:
            baseline = json.load(f)

    return latest, baseline


def load_run_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── CI Report Markdown ────────────────────────────────────────────────────

def write_markdown_summary(latest: Dict[str, Any], failures: List[str]) -> None:
    """Generates eval_summary.md for GitHub Actions step summaries and PR comments."""
    domain = latest.get("domain", config.DOMAIN)
    status_emoji = "✅" if not failures else "❌"
    pass_rate   = latest.get("pass_rate", 0.0)
    passed      = latest.get("passed", 0)
    total       = latest.get("total_questions", 0)

    hit_rate  = round(latest.get("avg_retrieval_hit_rate", 1.0) * 100.0, 1)
    mrr       = latest.get("avg_retrieval_mrr", 0.0)
    prec      = latest.get("avg_context_precision", 0.0)
    recall    = latest.get("avg_context_recall", 0.0)
    ndcg      = latest.get("avg_ndcg_at_k", 0.0)
    map_score = latest.get("avg_map_score", 0.0)
    prec_k    = latest.get("avg_precision_at_k", 0.0)
    coverage  = latest.get("avg_coverage", 0.0)

    faith     = latest.get("avg_faithfulness", "N/A")
    hall      = latest.get("hallucination_rate_avg", "N/A")
    token_f1  = latest.get("avg_token_f1", "N/A")
    p95_lat   = latest.get("p95_latency_sec", 0.0)
    avg_cost  = latest.get("avg_cost_usd", 0.0)

    summary = f"""## {status_emoji} LLM-Litmus Evaluation Report

### Run Summary
- **Domain**: `{domain}`
- **Pass Rate**: `{pass_rate}%` ({passed}/{total} queries)
- **Status**: {"**PASSED** ✅" if not failures else "**FAILED** ❌"}
- **Commit**: `{latest.get('git_commit_hash', 'unknown')}` on `{latest.get('branch', 'unknown')}`
- **Provider/Model**: `{latest.get('provider', '?')}` / `{latest.get('llm_model', '?')}`
- **Mode**: `{latest.get('mode', 'full')}`

### Retrieval Analytics
| Metric | Value |
|---|---|
| Hit Rate | `{hit_rate}%` |
| MRR | `{_fmt(mrr)}` |
| nDCG@K | `{_fmt(ndcg)}` |
| Precision@K | `{_fmt(prec_k)}` |
| MAP | `{_fmt(map_score)}` |
| Coverage | `{_fmt(coverage)}` |
| Context Precision | `{_fmt(prec)}` |
| Context Recall | `{_fmt(recall)}` |

### Generation Analytics
| Metric | Value |
|---|---|
| Avg Faithfulness | `{_fmt(faith)}` |
| Hallucination Rate | `{_fmt(hall)}` |
| Avg Token F1 | `{_fmt(token_f1)}` |
| p95 Latency | `{p95_lat}s` |
| Avg Cost / Query | `${avg_cost:.6f}` |

"""
    if failures:
        summary += "### ⚠️ Regression & Threshold Failures\n"
        for fail in failures:
            summary += f"- {fail}\n"
        summary += "\n**Action**: Fix regressions or update thresholds in `config.py` before merging.\n"
    else:
        summary += "### 🎉 All Gates Passed\nQuality gates and regression checks passed. Ready to merge!\n"

    with open("eval_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Generated eval_summary.md")


# ── Phase 7: A/B Comparison Report ───────────────────────────────────────

def write_comparison_report(run_a: Dict[str, Any], run_b: Dict[str, Any],
                             label_a: str = "Run A", label_b: str = "Run B") -> None:
    """
    Phase 7: Generates comparison_report.md comparing two evaluation runs side-by-side.
    Useful for: prompt A vs B, model A vs B, retrieval strategy A vs B.
    """
    results_a = {r["unique_id"]: r for r in run_a.get("results", [])}
    results_b = {r["unique_id"]: r for r in run_b.get("results", [])}
    shared_ids = set(results_a.keys()) & set(results_b.keys())

    # Per-category pass rates
    cats_a: Dict[str, List] = {}
    cats_b: Dict[str, List] = {}
    for qid in shared_ids:
        cat = results_a[qid].get("category", "general")
        cats_a.setdefault(cat, []).append(1 if results_a[qid]["status"] == "PASS" else 0)
        cats_b.setdefault(cat, []).append(1 if results_b[qid]["status"] == "PASS" else 0)

    # Queries that changed status
    regressions = []   # PASS → FAIL
    improvements = []  # FAIL → PASS
    for qid in shared_ids:
        sa = results_a[qid]["status"]
        sb = results_b[qid]["status"]
        if sa == "PASS" and sb == "FAIL":
            regressions.append(qid)
        elif sa == "FAIL" and sb == "PASS":
            improvements.append(qid)

    lines = [
        f"# A/B Evaluation Comparison\n",
        f"| | {label_a} | {label_b} | Delta |",
        "|---|---|---|---|",
    ]

    def row(label, key, higher_is_better=True, fmt=".3f", scale=1.0):
        va = run_a.get(key, 0.0)
        vb = run_b.get(key, 0.0)
        if isinstance(va, (int, float)):
            va *= scale
        if isinstance(vb, (int, float)):
            vb *= scale
        d = _delta(vb, va, higher_is_better=higher_is_better, fmt=fmt)
        return f"| {label} | `{_fmt(va, fmt)}` | `{_fmt(vb, fmt)}` | {d} |"

    lines.extend([
        row("Pass Rate (%)", "pass_rate", fmt=".1f"),
        row("Hit Rate", "avg_retrieval_hit_rate", scale=100, fmt=".1f"),
        row("MRR", "avg_retrieval_mrr"),
        row("nDCG@K", "avg_ndcg_at_k"),
        row("MAP", "avg_map_score"),
        row("Coverage", "avg_coverage"),
        row("Avg Faithfulness", "avg_faithfulness"),
        row("Hallucination Rate", "hallucination_rate_avg", higher_is_better=False),
        row("Token F1", "avg_token_f1"),
        row("p95 Latency (s)", "p95_latency_sec", higher_is_better=False),
        row("Avg Cost ($)", "avg_cost_usd", higher_is_better=False, fmt=".6f"),
        "",
        f"**Provider**: {run_a.get('provider','?')}/{run_a.get('llm_model','?')} vs "
        f"{run_b.get('provider','?')}/{run_b.get('llm_model','?')}",
        "",
        "## Per-Category Pass Rate",
        "",
        "| Category | Run A | Run B | Delta |",
        "|---|---|---|---|",
    ])

    all_cats = sorted(set(cats_a.keys()) | set(cats_b.keys()))
    for cat in all_cats:
        pa = round(sum(cats_a.get(cat, [])) / len(cats_a[cat]) * 100, 1) if cats_a.get(cat) else 0.0
        pb = round(sum(cats_b.get(cat, [])) / len(cats_b[cat]) * 100, 1) if cats_b.get(cat) else 0.0
        d_str = _delta(pb, pa, fmt=".1f")
        lines.append(f"| {cat} | {pa}% | {pb}% | {d_str} |")

    lines += [
        "",
        f"## Status Changes ({len(shared_ids)} shared queries)",
        "",
        f"- **Regressions** (PASS → FAIL in {label_b}): **{len(regressions)}**",
    ]
    for qid in regressions[:10]:
        q = results_a[qid].get("question", "")[:80]
        lines.append(f"  - `{qid}`: {q}")

    lines += [
        "",
        f"- **Improvements** (FAIL → PASS in {label_b}): **{len(improvements)}**",
    ]
    for qid in improvements[:10]:
        q = results_a[qid].get("question", "")[:80]
        lines.append(f"  - `{qid}`: {q}")

    report = "\n".join(lines)
    with open("comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Generated comparison_report.md  (%d regressions, %d improvements)",
                len(regressions), len(improvements))


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = __import__("argparse").ArgumentParser(
        description="LLM-Litmus CI Quality Gate & A/B Comparison"
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="full",
        help="Gate mode: smoke=pipeline health only, full=quality+regression checks.")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
        help="Phase 7: Compare two run JSON files side-by-side. Generates comparison_report.md.")
    parser.add_argument("--label-a", default="Run A", dest="label_a",
        help="Label for the first run in comparison mode (e.g. 'Prompt v1').")
    parser.add_argument("--label-b", default="Run B", dest="label_b",
        help="Label for the second run in comparison mode (e.g. 'Prompt v2').")
    args = parser.parse_args()

    # ── Phase 7: Compare Mode ─────────────────────────────────────────────
    if args.compare:
        path_a, path_b = args.compare
        logger.info("A/B COMPARISON MODE: %s  vs  %s", path_a, path_b)
        try:
            run_a = load_run_file(path_a)
            run_b = load_run_file(path_b)
        except Exception as exc:
            logger.error("Failed to load comparison runs: %s", exc)
            sys.exit(1)
        write_comparison_report(run_a, run_b, label_a=args.label_a, label_b=args.label_b)
        logger.info("Comparison report written to comparison_report.md")
        sys.exit(0)

    # ── Standard Gate Mode ────────────────────────────────────────────────
    try:
        latest, baseline = load_latest_runs()
    except Exception as exc:
        logger.error("Failed to load evaluation runs: %s", exc)
        sys.exit(1)

    failures: List[str] = []

    if args.mode == "smoke":
        logger.info("CI QUALITY GATE — SMOKE MODE (Pipeline Health Check)")
        latest_hit = latest.get("avg_retrieval_hit_rate", 1.0) * 100.0
        if latest_hit < 80.0:
            failures.append(f"Retrieval hit rate {latest_hit:.1f}% < 80.0% smoke threshold")
        latest_p95 = latest.get("p95_latency_sec", 0.0)
        if latest_p95 > 4.0:
            failures.append(f"p95 latency {latest_p95:.2f}s > 4.0s smoke threshold")
        results = latest.get("results", [])
        min_expected = 10 if latest.get("mode") == "smoke" else 100
        if len(results) < min_expected:
            failures.append(f"Total evaluated ({len(results)}) < expected minimum ({min_expected})")
        for r in results:
            if "LLM generation failed" in r.get("answer", "") or "Error:" in r.get("answer", ""):
                failures.append(f"API generation error in {r.get('unique_id')}: {r['answer'][:80]}")
                break
    else:
        logger.info("CI QUALITY GATE — FULL MODE (Quality & Regression Audit)")
        failures = check_regressions(latest, baseline)

    # ── Logging ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Domain:       %s", latest.get("domain", config.DOMAIN))
    logger.info("Provider:     %s / %s", latest.get("provider", "?"), latest.get("llm_model", "?"))
    logger.info("Pass Rate:    %s%%", latest.get("pass_rate", 0.0))
    logger.info("Hit Rate:     %.1f%%", latest.get("avg_retrieval_hit_rate", 1.0) * 100.0)
    logger.info("nDCG@K:       %s", _fmt(latest.get("avg_ndcg_at_k", 0.0)))
    logger.info("MAP:          %s", _fmt(latest.get("avg_map_score", 0.0)))
    logger.info("p95 Latency:  %ss", latest.get("p95_latency_sec", 0.0))
    logger.info("-" * 60)

    write_markdown_summary(latest, failures)

    if failures:
        logger.error("[GATE FAILED]")
        for f in failures:
            logger.error("  - %s", f)
        sys.exit(1)
    else:
        logger.info("[GATE PASSED] All %s quality gates satisfied.", args.mode)
        sys.exit(0)


if __name__ == "__main__":
    main()
