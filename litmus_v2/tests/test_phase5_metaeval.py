"""
Tests for Phase 5: Meta-Evaluation (evaluate the evaluators).
Tests all classification metrics, confusion matrix with abstains, Cohen's kappa,
Spearman rank correlation, bootstrap CIs, McNemar's paired test, and the
grader comparison runner.
"""

import math
import pytest
from typing import List

from litmus_v2.data.schema import GradeLabel
from litmus_v2.metaeval.metrics import (
    ConfusionMatrix,
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_cohens_kappa,
    compute_spearman_rho,
    compute_percentiles,
    bootstrap_ci,
    mcnemar_test,
)
from litmus_v2.metaeval.evaluate_graders import (
    evaluate_single_grader,
    run_meta_evaluation,
    _extract_triplet,
)
from litmus_v2.graders.graders import BaseGrader, GradeResult


class DummyGrader(BaseGrader):
    """Deterministic mock grader for tests."""
    def __init__(self, name: str, return_label: GradeLabel, score: float = 0.8, latency_ms: float = 10.0):
        self.name = name
        self.return_label = return_label
        self.score = score
        self.latency_ms = latency_ms

    def grade(self, question: str, reference_answer: str, candidate_answer: str, context=None) -> GradeResult:
        return self._make_result(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            label=self.return_label,
            score=self.score,
            rationale="dummy rationale",
            latency_ms=self.latency_ms,
        )


class PatternGrader(BaseGrader):
    """Deterministic grader that returns CORRECT if candidate equals reference, else WRONG."""
    name = "pattern_grader"

    def grade(self, question: str, reference_answer: str, candidate_answer: str, context=None) -> GradeResult:
        is_same = candidate_answer.strip() == reference_answer.strip()
        label = GradeLabel.CORRECT if is_same else GradeLabel.WRONG
        score = 1.0 if is_same else 0.0
        return self._make_result(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            label=label,
            score=score,
            rationale=f"is_same={is_same}",
            latency_ms=5.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix & Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestConfusionMatrix:
    def test_all_correct(self):
        y_true = [GradeLabel.CORRECT] * 5
        y_pred = [GradeLabel.CORRECT] * 5
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.tp == 5
        assert cm.fp == 0
        assert cm.tn == 0
        assert cm.fn == 0
        assert cm.abstain == 0
        assert cm.total == 5

    def test_all_wrong(self):
        y_true = [GradeLabel.WRONG] * 5
        y_pred = [GradeLabel.WRONG] * 5
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.tp == 0
        assert cm.tn == 5
        assert cm.fp == 0
        assert cm.fn == 0

    def test_critical_hallucination_false_positive(self):
        # Ground truth is WRONG, grader approved it as CORRECT (FP)
        y_true = [GradeLabel.WRONG] * 4
        y_pred = [GradeLabel.CORRECT] * 4
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.fp == 4
        assert cm.tn == 0
        metrics = compute_classification_metrics(cm)
        assert metrics["specificity"] == 0.0

    def test_abstain_counted_separately(self):
        y_true = [GradeLabel.CORRECT, GradeLabel.WRONG, GradeLabel.CORRECT]
        y_pred = [GradeLabel.CORRECT, GradeLabel.ABSTAIN, GradeLabel.ERROR]
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.tp == 1
        assert cm.abstain == 2
        assert cm.total == 3
        assert cm.decided_total == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_confusion_matrix([GradeLabel.CORRECT], [GradeLabel.CORRECT, GradeLabel.WRONG])


class TestClassificationMetrics:
    def test_perfect_metrics(self):
        cm = ConfusionMatrix(tp=10, fp=0, tn=10, fn=0, abstain=0)
        m = compute_classification_metrics(cm)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["specificity"] == 1.0

    def test_zero_denominator_safe(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0, abstain=0)
        m = compute_classification_metrics(cm)
        assert m["accuracy"] == 0.0
        assert m["f1"] == 0.0

    def test_abstain_penalizes_overall_accuracy(self):
        # 8 correct decisions out of 10 items (2 abstains)
        cm = ConfusionMatrix(tp=8, fp=0, tn=0, fn=0, abstain=2)
        m = compute_classification_metrics(cm)
        assert m["accuracy"] == 0.8
        assert m["precision"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's Kappa
# ─────────────────────────────────────────────────────────────────────────────

class TestCohensKappa:
    def test_perfect_agreement(self):
        y_true = [GradeLabel.CORRECT, GradeLabel.WRONG, GradeLabel.CORRECT, GradeLabel.WRONG]
        y_pred = [GradeLabel.CORRECT, GradeLabel.WRONG, GradeLabel.CORRECT, GradeLabel.WRONG]
        assert compute_cohens_kappa(y_true, y_pred) == 1.0

    def test_complete_disagreement(self):
        y_true = [GradeLabel.CORRECT] * 10 + [GradeLabel.WRONG] * 10
        y_pred = [GradeLabel.WRONG] * 10 + [GradeLabel.CORRECT] * 10
        kappa = compute_cohens_kappa(y_true, y_pred)
        assert kappa == -1.0

    def test_empty_returns_zero(self):
        assert compute_cohens_kappa([], []) == 0.0

    def test_abstains_excluded_from_agreement(self):
        y_true = [GradeLabel.CORRECT, GradeLabel.WRONG, GradeLabel.CORRECT]
        y_pred = [GradeLabel.CORRECT, GradeLabel.WRONG, GradeLabel.ABSTAIN]
        # Only 2 decided pairs, both agree perfectly
        assert compute_cohens_kappa(y_true, y_pred) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Spearman Rank Correlation
# ─────────────────────────────────────────────────────────────────────────────

class TestSpearmanRho:
    def test_perfect_positive_correlation(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        y_true = [GradeLabel.WRONG, GradeLabel.WRONG, GradeLabel.CORRECT, GradeLabel.CORRECT]
        rho = compute_spearman_rho(scores, y_true)
        assert rho is not None
        assert rho > 0.85

    def test_handles_ties(self):
        scores = [0.5, 0.5, 0.8, 0.8]
        y_true = [GradeLabel.WRONG, GradeLabel.WRONG, GradeLabel.CORRECT, GradeLabel.CORRECT]
        rho = compute_spearman_rho(scores, y_true)
        assert rho is not None
        assert rho > 0.8

    def test_zero_variance_returns_zero(self):
        scores = [0.5, 0.5, 0.5]
        y_true = [GradeLabel.CORRECT, GradeLabel.CORRECT, GradeLabel.CORRECT]
        assert compute_spearman_rho(scores, y_true) == 0.0

    def test_too_few_samples_returns_none(self):
        assert compute_spearman_rho([0.5], [GradeLabel.CORRECT]) is None


# ─────────────────────────────────────────────────────────────────────────────
# Percentiles & Bootstrap CIs
# ─────────────────────────────────────────────────────────────────────────────

class TestPercentilesAndBootstrap:
    def test_percentiles(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50, p95, mean = compute_percentiles(vals)
        assert p50 == 30.0
        assert p95 > 45.0
        assert mean == 30.0

    def test_bootstrap_ci_valid_range(self):
        y_true = [GradeLabel.CORRECT] * 20 + [GradeLabel.WRONG] * 20
        y_pred = [GradeLabel.CORRECT] * 18 + [GradeLabel.WRONG] * 2 + [GradeLabel.WRONG] * 18 + [GradeLabel.CORRECT] * 2
        lower, upper = bootstrap_ci(y_true, y_pred, metric_name="f1", n_bootstrap=200, seed=42)
        assert 0.0 <= lower <= upper <= 1.0
        assert lower >= 0.7  # High accuracy, F1 CI should be reasonably high


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's Test
# ─────────────────────────────────────────────────────────────────────────────

class TestMcNemarTest:
    def test_identical_graders_no_significance(self):
        y_true = [GradeLabel.CORRECT, GradeLabel.WRONG] * 10
        y_pred = [GradeLabel.CORRECT, GradeLabel.WRONG] * 10
        res = mcnemar_test(y_true, y_pred, y_pred)
        assert res["statistic"] == 0.0
        assert res["p_value"] == 1.0
        assert not res["significant_at_05"]

    def test_one_grader_significantly_better(self):
        # Grader A gets 30 right where B gets them wrong
        y_true = [GradeLabel.CORRECT] * 35
        y_pred_a = [GradeLabel.CORRECT] * 35  # A: 35/35
        y_pred_b = [GradeLabel.CORRECT] * 5 + [GradeLabel.WRONG] * 30  # B: 5/35
        res = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert res["b_grader_a_only"] == 30
        assert res["c_grader_b_only"] == 0
        assert res["significant_at_05"]
        assert res["p_value"] < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Grader Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateGradersRunner:
    @pytest.fixture
    def sample_items(self):
        return [
            {
                "id": "1",
                "question": "What is the 80C limit?",
                "reference_answer": "₹1,50,000",
                "candidate_answer": "₹1,50,000",
                "ground_truth": "correct",
                "category": "number_swap",
            },
            {
                "id": "2",
                "question": "What is the 80C limit?",
                "reference_answer": "₹1,50,000",
                "candidate_answer": "₹1,00,000",
                "ground_truth": "wrong",
                "category": "number_swap",
            },
            {
                "id": "3",
                "question": "Can I claim HRA?",
                "reference_answer": "Yes, if rent is paid.",
                "candidate_answer": "No, you cannot.",
                "ground_truth": "wrong",
                "category": "negation_flip",
            },
        ]

    def test_evaluate_single_grader(self, sample_items):
        grader = PatternGrader()
        report = evaluate_single_grader(grader, sample_items, dataset_name="test_dataset")
        assert report.metrics.grader_name == "pattern_grader"
        assert report.metrics.sample_size == 3
        # PatternGrader gets item 1 right (identical), item 2 right (different -> WRONG), item 3 right (different -> WRONG)
        assert report.metrics.accuracy == 1.0
        assert report.metrics.confusion_matrix.tp == 1
        assert report.metrics.confusion_matrix.tn == 2
        assert "number_swap" in report.failure_breakdown_by_category

    def test_run_meta_evaluation_compares_two_graders(self, sample_items):
        grader_a = DummyGrader("always_correct", return_label=GradeLabel.CORRECT)
        grader_b = DummyGrader("always_wrong", return_label=GradeLabel.WRONG)
        suite = run_meta_evaluation([grader_a, grader_b], sample_items, dataset_name="test_suite")

        assert len(suite.grader_reports) == 2
        assert "always_correct__vs__always_wrong" in suite.pairwise_mcnemar
        assert len(suite.ranking_by_f1) == 2
        assert suite.sample_size == 3
