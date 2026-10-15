"""
Phase 5: Meta-Evaluation Metrics.
Evaluates the evaluators (graders) against human-labeled ground truth or known perturbations.

Includes:
    - Binary classification metrics: Accuracy, Precision, Recall, F1, Specificity
    - Confusion Matrix (TP, FP, FN, TN, Abstain)
    - Cohen's Kappa for inter-annotator agreement
    - Spearman rank correlation between continuous score and binary ground truth
    - Latency statistics (p50, p95, mean)
    - Cost estimator per 100 queries
    - Statistical significance:
        - Bootstrap confidence intervals (BCa / percentile)
        - McNemar's test for paired grader comparison (with continuity correction)
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from litmus_v2.data.schema import GradeLabel


@dataclass
class ConfusionMatrix:
    """Confusion matrix with explicit Abstain tracking."""
    tp: int = 0  # Ground truth CORRECT, Grader predicted CORRECT
    fp: int = 0  # Ground truth WRONG, Grader predicted CORRECT (Critical error: approving hallucination)
    tn: int = 0  # Ground truth WRONG, Grader predicted WRONG
    fn: int = 0  # Ground truth CORRECT, Grader predicted WRONG (False alarm)
    abstain: int = 0  # Grader returned ABSTAIN or ERROR

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn + self.abstain

    @property
    def decided_total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> Dict[str, int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "abstain": self.abstain,
            "total": self.total,
        }


@dataclass
class GraderMetrics:
    """Full evaluation metrics for a single grader."""
    grader_name: str
    dataset_name: str
    sample_size: int
    confusion_matrix: ConfusionMatrix
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    cohens_kappa: float
    spearman_rho: Optional[float]
    abstain_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    cost_per_100_queries_usd: float
    f1_ci_95: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grader_name": self.grader_name,
            "dataset_name": self.dataset_name,
            "sample_size": self.sample_size,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "specificity": round(self.specificity, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "spearman_rho": round(self.spearman_rho, 4) if self.spearman_rho is not None else None,
            "abstain_rate": round(self.abstain_rate, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
            "cost_per_100_queries_usd": round(self.cost_per_100_queries_usd, 4),
            "f1_ci_95": [round(self.f1_ci_95[0], 4), round(self.f1_ci_95[1], 4)],
        }


def compute_confusion_matrix(
    y_true: Sequence[GradeLabel],
    y_pred: Sequence[GradeLabel],
) -> ConfusionMatrix:
    """
    Computes confusion matrix where:
        CORRECT = positive class (answer is faithful / accurate)
        WRONG = negative class (answer is incorrect / hallucinated)
        ABSTAIN / ERROR = tracked separately in abstain count
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    tp = fp = tn = fn = abstain = 0

    for gt, pred in zip(y_true, y_pred):
        if pred in (GradeLabel.ABSTAIN, GradeLabel.ERROR):
            abstain += 1
            continue

        if gt == GradeLabel.CORRECT:
            if pred == GradeLabel.CORRECT:
                tp += 1
            elif pred == GradeLabel.WRONG:
                fn += 1
        elif gt == GradeLabel.WRONG:
            if pred == GradeLabel.CORRECT:
                fp += 1
            elif pred == GradeLabel.WRONG:
                tn += 1

    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn, abstain=abstain)


def compute_classification_metrics(cm: ConfusionMatrix) -> Dict[str, float]:
    """
    Calculates Accuracy, Precision, Recall, F1, and Specificity.
    Total denominator includes abstain as non-decisions (abstain penalizes total accuracy).
    """
    total = cm.total
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "specificity": 0.0}

    accuracy = (cm.tp + cm.tn) / total

    # Precision = TP / (TP + FP)
    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0

    # F1 = harmonic mean
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Specificity = TN / (TN + FP) — ability to correctly catch wrong answers
    specificity = cm.tn / (cm.tn + cm.fp) if (cm.tn + cm.fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def compute_cohens_kappa(
    y_true: Sequence[GradeLabel],
    y_pred: Sequence[GradeLabel],
) -> float:
    """
    Computes Cohen's Kappa between ground truth and grader predictions.
    Only decided pairs (CORRECT, WRONG) are counted; abstains are excluded from agreement.
    Kappa = (Po - Pe) / (1 - Pe)
    """
    pairs = [
        (gt, pred) for gt, pred in zip(y_true, y_pred)
        if pred in (GradeLabel.CORRECT, GradeLabel.WRONG) and gt in (GradeLabel.CORRECT, GradeLabel.WRONG)
    ]
    n = len(pairs)
    if n == 0:
        return 0.0

    # Agreement table: rows = true, cols = pred
    # [0] = CORRECT, [1] = WRONG
    c_00 = sum(1 for gt, pred in pairs if gt == GradeLabel.CORRECT and pred == GradeLabel.CORRECT)
    c_01 = sum(1 for gt, pred in pairs if gt == GradeLabel.CORRECT and pred == GradeLabel.WRONG)
    c_10 = sum(1 for gt, pred in pairs if gt == GradeLabel.WRONG and pred == GradeLabel.CORRECT)
    c_11 = sum(1 for gt, pred in pairs if gt == GradeLabel.WRONG and pred == GradeLabel.WRONG)

    p_observed = (c_00 + c_11) / n

    # Marginals
    p_true_correct = (c_00 + c_01) / n
    p_true_wrong = (c_10 + c_11) / n
    p_pred_correct = (c_00 + c_10) / n
    p_pred_wrong = (c_01 + c_11) / n

    p_expected = (p_true_correct * p_pred_correct) + (p_true_wrong * p_pred_wrong)

    if math.isclose(1.0 - p_expected, 0.0, abs_tol=1e-9):
        return 1.0 if math.isclose(p_observed, 1.0, abs_tol=1e-9) else 0.0

    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return max(-1.0, min(1.0, kappa))


def _rank_data(values: Sequence[float]) -> List[float]:
    """Computes fractional ranks with average tie handling."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def compute_spearman_rho(
    scores: Sequence[float],
    y_true: Sequence[GradeLabel],
) -> Optional[float]:
    """
    Computes Spearman rank correlation between continuous grader scores and binary labels.
    Label mapping: WRONG = 0.0, CORRECT = 1.0.
    Returns None if variance is zero or length < 2.
    """
    # Filter to pairs where label is binary
    valid = [
        (s, 1.0 if gt == GradeLabel.CORRECT else 0.0)
        for s, gt in zip(scores, y_true)
        if gt in (GradeLabel.CORRECT, GradeLabel.WRONG) and s is not None and not math.isnan(s)
    ]
    n = len(valid)
    if n < 2:
        return None

    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]

    # Check zero variance
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return 0.0

    rank_x = _rank_data(xs)
    rank_y = _rank_data(ys)

    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n

    cov = sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y))
    var_x = sum((rx - mean_rx) ** 2 for rx in rank_x)
    var_y = sum((ry - mean_ry) ** 2 for ry in rank_y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0

    return max(-1.0, min(1.0, cov / denom))


def compute_percentiles(values: Sequence[float]) -> Tuple[float, float, float]:
    """Returns (p50, p95, mean)."""
    if not values:
        return (0.0, 0.0, 0.0)
    sorted_v = sorted(values)
    n = len(sorted_v)
    mean_v = sum(sorted_v) / n

    def pctl(p: float) -> float:
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_v[int(k)]
        return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)

    return (pctl(50.0), pctl(95.0), mean_v)


def bootstrap_ci(
    y_true: Sequence[GradeLabel],
    y_pred: Sequence[GradeLabel],
    metric_name: str = "f1",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes percentile bootstrap confidence interval for a metric (f1 or accuracy).
    Returns (lower_ci, upper_ci).
    """
    import random
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)

    bootstrap_scores: List[float] = []
    pairs = list(zip(y_true, y_pred))

    for _ in range(n_bootstrap):
        sample = [rng.choice(pairs) for _ in range(n)]
        samp_true, samp_pred = zip(*sample)
        cm = compute_confusion_matrix(samp_true, samp_pred)
        metrics = compute_classification_metrics(cm)
        bootstrap_scores.append(metrics.get(metric_name, 0.0))

    bootstrap_scores.sort()
    alpha = 1.0 - confidence_level
    lower_idx = int((alpha / 2.0) * n_bootstrap)
    upper_idx = int((1.0 - alpha / 2.0) * n_bootstrap)
    upper_idx = min(upper_idx, n_bootstrap - 1)

    return (round(bootstrap_scores[lower_idx], 4), round(bootstrap_scores[upper_idx], 4))


def mcnemar_test(
    y_true: Sequence[GradeLabel],
    y_pred_a: Sequence[GradeLabel],
    y_pred_b: Sequence[GradeLabel],
) -> Dict[str, Any]:
    """
    McNemar's test for paired comparisons of two graders on the same dataset.

    Contingency table:
        b: Grader A correct, Grader B wrong
        c: Grader A wrong, Grader B correct
    Statistic: (|b - c| - 1)^2 / (b + c) with Edwards' continuity correction.
    p-value from chi-square distribution with 1 degree of freedom.
    """
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("Lengths of y_true, y_pred_a, and y_pred_b must match")

    b = 0  # A correct, B incorrect
    c = 0  # A incorrect, B correct
    a_and_b_correct = 0
    a_and_b_incorrect = 0

    for gt, pred_a, pred_b in zip(y_true, y_pred_a, y_pred_b):
        a_correct = (pred_a == gt)
        b_correct = (pred_b == gt)

        if a_correct and b_correct:
            a_and_b_correct += 1
        elif not a_correct and not b_correct:
            a_and_b_incorrect += 1
        elif a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    discordant = b + c
    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "b_grader_a_only": b,
            "c_grader_b_only": c,
            "note": "Zero discordant pairs (both graders agreed on all samples)",
        }

    # Chi-square with continuity correction: (|b - c| - 1)^2 / (b + c)
    num = max(0.0, abs(b - c) - 1.0) ** 2
    statistic = num / discordant

    # Approximation of p-value for 1 degree of freedom:
    # 1 - erf(sqrt(stat / 2))
    p_value = math.erfc(math.sqrt(statistic / 2.0))

    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 5),
        "significant_at_05": p_value < 0.05,
        "b_grader_a_only": b,
        "c_grader_b_only": c,
        "note": "Statistically significant difference (p < 0.05)" if p_value < 0.05 else "No statistically significant difference (p >= 0.05)",
    }
