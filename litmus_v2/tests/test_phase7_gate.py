"""
Tests for Phase 7: Quality Gate & Drift Detection.
Tests regression gate, Cohen's kappa agreement gate, latency gate, cost gate,
contamination check, Population Stability Index (PSI), and Wasserstein distance.
"""

import pytest
from litmus_v2.gate.drift import compute_psi, compute_wasserstein_distance_1d
from litmus_v2.gate.gate import (
    GateConfig,
    GateResult,
    QualityGateSummary,
    check_contamination,
    run_quality_gates,
)


class TestDriftDetection:
    def test_identical_distributions_have_zero_psi(self):
        scores = [0.1, 0.2, 0.5, 0.8, 0.9] * 20
        res = compute_psi(scores, scores)
        assert res["psi"] == 0.0
        assert res["status"] == "STABLE"
        assert not res["is_drifted"]

    def test_moderate_noise_is_not_drifted(self):
        import random
        rng = random.Random(42)
        base = [rng.uniform(0.6, 0.9) for _ in range(100)]
        current = [min(1.0, max(0.0, v + rng.gauss(0, 0.02))) for v in base]
        res = compute_psi(base, current)
        assert res["psi"] < 0.25
        assert not res["is_drifted"]

    def test_severe_distribution_shift_triggers_drift(self):
        # Baseline: mostly high scores [0.8, 0.9]
        # Current: mostly low scores [0.1, 0.2]
        base = [0.8, 0.85, 0.9, 0.95] * 25
        current = [0.05, 0.1, 0.15, 0.2] * 25
        res = compute_psi(base, current)
        assert res["psi"] > 0.25
        assert res["status"] == "SIGNIFICANT_DRIFT"
        assert res["is_drifted"]

    def test_empty_distribution_handled_safely(self):
        res = compute_psi([], [0.5])
        assert res["psi"] == 0.0
        assert not res["is_drifted"]

    def test_wasserstein_identical_is_zero(self):
        vals = [0.1, 0.5, 0.9]
        assert compute_wasserstein_distance_1d(vals, vals) == 0.0

    def test_wasserstein_shifted_is_positive(self):
        u = [0.0, 0.1, 0.2]
        v = [0.8, 0.9, 1.0]
        dist = compute_wasserstein_distance_1d(u, v)
        assert dist > 0.5

    def test_wasserstein_is_symmetric(self):
        u = [0.1, 0.3, 0.7]
        v = [0.2, 0.6, 0.9]
        assert compute_wasserstein_distance_1d(u, v) == compute_wasserstein_distance_1d(v, u)


class TestContaminationCheck:
    def test_disjoint_queries_pass(self):
        dev = ["What is Section 80C deduction limit?", "How is HRA calculated?"]
        test = ["Can a HUF claim 80D medical insurance?", "What are LTCG tax rates on equity?"]
        passed, max_sim, _ = check_contamination(dev, test, threshold=0.95)
        assert passed
        assert max_sim < 0.95

    def test_leaked_exact_query_fails(self):
        dev = ["What is the maximum aggregate deduction under Section 80C?"]
        test = ["What is the maximum aggregate deduction under Section 80C?"]
        passed, max_sim, worst_pair = check_contamination(dev, test, threshold=0.95)
        assert not passed
        assert max_sim >= 0.95
        assert worst_pair is not None


class TestQualityGateRunner:
    @pytest.fixture
    def passing_metrics(self):
        return {
            "pass_rate": 0.88,
            "f1": 0.85,
            "ndcg_at_5": 0.82,
            "cohens_kappa": 0.78,
            "latency_p95_ms": 1200.0,
            "total_cost_usd": 0.05,
        }

    def test_all_gates_pass(self, passing_metrics):
        summary = run_quality_gates(
            metrics=passing_metrics,
            config=GateConfig(),
            dev_queries=["What is HRA exemption?"],
            test_queries=["What is Section 80C limit?"],
            baseline_scores=[0.8, 0.9] * 20,
            current_scores=[0.82, 0.88] * 20,
        )
        assert summary.overall_passed
        assert all(g.passed for g in summary.gates)
        md = summary.to_markdown()
        assert "PASSED" in md
        assert ":white_check_mark:" in md

    def test_low_pass_rate_fails_gate(self, passing_metrics):
        passing_metrics["pass_rate"] = 0.60  # Below 0.75 threshold
        summary = run_quality_gates(metrics=passing_metrics)
        assert not summary.overall_passed
        failed_gate = [g for g in summary.gates if not g.passed]
        assert any("Pass Rate" in g.gate_name for g in failed_gate)
        md = summary.to_markdown()
        assert "FAILED" in md
        assert "CAUTION" in md

    def test_low_kappa_fails_gate(self, passing_metrics):
        passing_metrics["cohens_kappa"] = 0.45  # Below 0.70 threshold
        summary = run_quality_gates(metrics=passing_metrics)
        assert not summary.overall_passed
        failed_gate = [g for g in summary.gates if not g.passed]
        assert any("Cohen's Kappa" in g.gate_name for g in failed_gate)

    def test_high_latency_fails_gate(self, passing_metrics):
        passing_metrics["latency_p95_ms"] = 8500.0  # Above 5000ms threshold
        summary = run_quality_gates(metrics=passing_metrics)
        assert not summary.overall_passed
        failed_gate = [g for g in summary.gates if not g.passed]
        assert any("Latency" in g.gate_name for g in failed_gate)

    def test_high_cost_fails_gate(self, passing_metrics):
        passing_metrics["total_cost_usd"] = 5.50  # Above $1.00 threshold
        summary = run_quality_gates(metrics=passing_metrics)
        assert not summary.overall_passed
        failed_gate = [g for g in summary.gates if not g.passed]
        assert any("Cost" in g.gate_name for g in failed_gate)

    def test_drift_fails_gate(self, passing_metrics):
        # Baseline high scores, current low scores
        base = [0.9] * 50
        curr = [0.1] * 50
        summary = run_quality_gates(
            metrics=passing_metrics,
            baseline_scores=base,
            current_scores=curr,
        )
        assert not summary.overall_passed
        failed_gate = [g for g in summary.gates if not g.passed]
        assert any("Drift" in g.gate_name for g in failed_gate)
