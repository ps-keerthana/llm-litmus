"""
Phase 7: CI/CD Quality Gate & Evaluation Enforcement.

Evaluates test/benchmark results against strict production quality gates:
    1. Regression Gate: Pass rate, F1, and nDCG >= threshold
    2. Grader Agreement Gate: Cohen's Kappa >= 0.70 against golden sample
    3. Latency Gate: p95 latency <= threshold ms
    4. Cost Gate: Estimated cost <= budget USD
    5. Contamination Gate: Test queries vs Dev queries (max cosine < 0.95)
    6. Drift Gate: Population Stability Index (PSI <= 0.25) against baseline

Outputs:
    - Structured JSON summary
    - GitHub Actions PR comment markdown
    - sys.exit(0) on success, sys.exit(1) on gate failure
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from litmus_v2.gate.drift import compute_psi, compute_wasserstein_distance_1d
from litmus_v2.graders.graders import _tokenize, _cosine_similarity


@dataclass
class GateConfig:
    min_pass_rate: float = 0.75
    min_f1: float = 0.70
    min_ndcg: float = 0.70
    min_cohens_kappa: float = 0.70
    max_p95_latency_ms: float = 5000.0
    max_cost_usd: float = 1.00
    max_psi: float = 0.25
    max_contamination_cosine: float = 0.95


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    actual_value: Any
    threshold: Any
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class QualityGateSummary:
    overall_passed: bool
    gates: List[GateResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "gates_passed": sum(1 for g in self.gates if g.passed),
            "gates_failed": sum(1 for g in self.gates if not g.passed),
            "total_gates": len(self.gates),
            "gates": [g.to_dict() for g in self.gates],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generates a clean GitHub Actions PR comment in markdown format."""
        status_banner = "### :white_check_mark: litmus_v2 Quality Gate: PASSED" if self.overall_passed else "### :x: litmus_v2 Quality Gate: FAILED"

        lines = [
            status_banner,
            "",
            "| Gate | Status | Actual | Threshold | Details |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ]

        for g in self.gates:
            icon = ":white_check_mark: PASS" if g.passed else ":x: FAIL"
            lines.append(f"| **{g.gate_name}** | {icon} | `{g.actual_value}` | `{g.threshold}` | {g.message} |")

        lines.extend([
            "",
            f"**Summary**: {sum(1 for g in self.gates if g.passed)}/{len(self.gates)} gates passed.",
        ])

        if not self.overall_passed:
            lines.extend([
                "",
                "> [!CAUTION]",
                "> One or more production quality gates failed. Deployment or PR merge blocked.",
            ])

        return "\n".join(lines)


def check_contamination(
    dev_queries: Sequence[str],
    test_queries: Sequence[str],
    threshold: float = 0.95,
) -> Tuple[bool, float, Optional[Tuple[str, str, float]]]:
    """
    Checks that no test query leaks from the dev split.
    Computes pairwise word-cosine similarity.
    Returns (passed, max_sim, worst_pair).
    """
    if not dev_queries or not test_queries:
        return (True, 0.0, None)

    dev_tf = [_tokenize(q) for q in dev_queries]
    test_tf = [_tokenize(q) for q in test_queries]

    max_sim = 0.0
    worst_pair = None

    for i, t_tokens in enumerate(test_tf):
        t_text = test_queries[i]
        for j, d_tokens in enumerate(dev_tf):
            d_text = dev_queries[j]
            sim = _cosine_similarity(t_tokens, d_tokens)
            if sim > max_sim:
                max_sim = sim
                worst_pair = (t_text, d_text, sim)
                if max_sim >= 1.0:
                    break

    passed = max_sim < threshold
    return (passed, round(max_sim, 4), worst_pair)


def run_quality_gates(
    metrics: Dict[str, Any],
    config: Optional[GateConfig] = None,
    dev_queries: Optional[Sequence[str]] = None,
    test_queries: Optional[Sequence[str]] = None,
    baseline_scores: Optional[Sequence[float]] = None,
    current_scores: Optional[Sequence[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityGateSummary:
    """
    Evaluates all configured quality gates against the given run results.
    """
    cfg = config or GateConfig()
    gate_results: List[GateResult] = []

    # 1. Regression Gates (pass rate, F1, nDCG)
    pass_rate = metrics.get("pass_rate") or metrics.get("accuracy", 0.0)
    p_pass = pass_rate >= cfg.min_pass_rate
    gate_results.append(GateResult(
        gate_name="Regression Gate (Pass Rate)",
        passed=p_pass,
        actual_value=f"{pass_rate*100:.1f}%",
        threshold=f">={cfg.min_pass_rate*100:.1f}%",
        message="Pass rate meets threshold" if p_pass else "Pass rate below threshold",
    ))

    f1 = metrics.get("f1", 0.0)
    f1_pass = f1 >= cfg.min_f1
    gate_results.append(GateResult(
        gate_name="Regression Gate (F1 Score)",
        passed=f1_pass,
        actual_value=round(f1, 4),
        threshold=f">={cfg.min_f1}",
        message="F1 score meets threshold" if f1_pass else "F1 score regressed below threshold",
    ))

    ndcg = metrics.get("ndcg_at_5") or metrics.get("ndcg", 0.0)
    ndcg_pass = ndcg >= cfg.min_ndcg
    gate_results.append(GateResult(
        gate_name="Retrieval Gate (nDCG@5)",
        passed=ndcg_pass,
        actual_value=round(ndcg, 4),
        threshold=f">={cfg.min_ndcg}",
        message="Retrieval ranking quality acceptable" if ndcg_pass else "Retrieval nDCG below threshold",
    ))

    # 2. Grader Agreement Gate (Cohen's Kappa against golden labels)
    kappa = metrics.get("cohens_kappa")
    if kappa is not None:
        k_pass = kappa >= cfg.min_cohens_kappa
        gate_results.append(GateResult(
            gate_name="Grader Agreement (Cohen's Kappa)",
            passed=k_pass,
            actual_value=round(kappa, 4),
            threshold=f">={cfg.min_cohens_kappa}",
            message="Grader agreement with human labels substantial (>=0.70)" if k_pass else "Grader agreement insufficient (<0.70)",
        ))

    # 3. Latency Gate (p95 latency)
    p95 = metrics.get("latency_p95_ms", 0.0)
    lat_pass = p95 <= cfg.max_p95_latency_ms
    gate_results.append(GateResult(
        gate_name="Latency Gate (p95)",
        passed=lat_pass,
        actual_value=f"{p95:.1f}ms",
        threshold=f"<={cfg.max_p95_latency_ms:.0f}ms",
        message="Latency within SLA" if lat_pass else "p95 latency exceeds SLA",
    ))

    # 4. Cost Gate
    cost = metrics.get("total_cost_usd", metrics.get("cost_per_100_queries_usd", 0.0))
    cost_pass = cost <= cfg.max_cost_usd
    gate_results.append(GateResult(
        gate_name="Cost Gate",
        passed=cost_pass,
        actual_value=f"${cost:.4f}",
        threshold=f"<=${cfg.max_cost_usd:.2f}",
        message="Cost within budget" if cost_pass else "Cost exceeded budget threshold",
    ))

    # 5. Contamination Gate
    if dev_queries is not None and test_queries is not None:
        contam_pass, max_sim, worst_pair = check_contamination(dev_queries, test_queries, cfg.max_contamination_cosine)
        msg = f"Zero train/test leakage detected (max cosine={max_sim})" if contam_pass else f"Contamination detected! Dev/Test similarity {max_sim} >= {cfg.max_contamination_cosine}"
        gate_results.append(GateResult(
            gate_name="Contamination Gate",
            passed=contam_pass,
            actual_value=max_sim,
            threshold=f"<{cfg.max_contamination_cosine}",
            message=msg,
        ))

    # 6. Drift Gate (PSI)
    if baseline_scores is not None and current_scores is not None and len(baseline_scores) > 0 and len(current_scores) > 0:
        psi_res = compute_psi(baseline_scores, current_scores)
        psi_val = psi_res["psi"]
        psi_pass = psi_val <= cfg.max_psi
        gate_results.append(GateResult(
            gate_name="Drift Gate (PSI)",
            passed=psi_pass,
            actual_value=psi_val,
            threshold=f"<={cfg.max_psi}",
            message=f"Distribution {psi_res['status']} (PSI={psi_val})" if psi_pass else f"ALERT: Significant distribution drift detected (PSI={psi_val} > {cfg.max_psi})",
        ))

    overall_passed = all(g.passed for g in gate_results)

    return QualityGateSummary(
        overall_passed=overall_passed,
        gates=gate_results,
        metadata=metadata or {},
    )
