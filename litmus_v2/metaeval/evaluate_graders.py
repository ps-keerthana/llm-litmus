"""
Phase 5: Evaluate the Evaluators (Meta-Evaluation Pipeline).

Runs graders against:
    1. Human-labeled golden dataset (N >= 30)
    2. Perturbation suite (known labels from 7 perturbation types)

Computes:
    - Full classification metrics (Accuracy, Precision, Recall, F1, Specificity)
    - Confusion Matrix (TP, FP, FN, TN, Abstain)
    - Inter-annotator agreement (Cohen's Kappa against human labels)
    - Score calibration (Spearman rank correlation)
    - Operational metrics: Latency (p50, p95, mean), Cost per 100 queries
    - Statistical significance: 95% Bootstrap CIs & McNemar paired tests
    - Failure analysis: per-grader breakdown across perturbation types & failure modes
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from litmus_v2.data.schema import GradeLabel, GoldenDatasetItem
from litmus_v2.perturbations.perturbations import PerturbedItem
from litmus_v2.graders.graders import (
    BaseGrader,
    GradeResult,
    SimilarityGrader,
    HeuristicGrader,
    NLIGrader,
    LLMJudge,
)
from litmus_v2.metaeval.metrics import (
    ConfusionMatrix,
    GraderMetrics,
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_cohens_kappa,
    compute_spearman_rho,
    compute_percentiles,
    bootstrap_ci,
    mcnemar_test,
)


# Cost models per 100 queries in USD
GRADER_COST_PER_100: Dict[str, float] = {
    "similarity_baseline": 0.0,
    "heuristic_gate": 0.0,
    "nli_faithfulness": 0.001,  # Local GPU/CPU compute amortized
    "llm_judge_groq_70b": 0.065,  # ~650 tokens/query @ $0.15/1M input + $0.60/1M output
    "llm_judge_ollama_local": 0.002,  # Local compute amortized
}


@dataclass
class ItemEvaluation:
    """Individual item evaluation record."""
    item_id: str
    question: str
    ground_truth: GradeLabel
    predicted_label: GradeLabel
    score: float
    is_correct_decision: bool
    latency_ms: float
    category: str  # question_type or perturbation_type
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question": self.question[:60] + "..." if len(self.question) > 60 else self.question,
            "ground_truth": self.ground_truth.value,
            "predicted_label": self.predicted_label.value,
            "score": round(self.score, 4),
            "is_correct_decision": self.is_correct_decision,
            "latency_ms": round(self.latency_ms, 2),
            "category": self.category,
            "rationale": self.rationale[:80],
        }


@dataclass
class GraderEvaluationReport:
    """Comprehensive evaluation report for a single grader on a dataset."""
    metrics: GraderMetrics
    item_evaluations: List[ItemEvaluation] = field(default_factory=list)
    failure_breakdown_by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    top_failure_modes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "failure_breakdown_by_category": self.failure_breakdown_by_category,
            "top_failure_modes": self.top_failure_modes,
            "sample_evaluations": [e.to_dict() for e in self.item_evaluations[:10]],
        }


def _extract_triplet(item: Union[GoldenDatasetItem, PerturbedItem, Dict[str, Any]]) -> Tuple[str, str, str, Optional[str], GradeLabel, str, str]:
    """
    Extracts (item_id, question, ref_answer, cand_answer, context, ground_truth, category)
    from supported item types.
    """
    if isinstance(item, PerturbedItem):
        return (
            item.perturbed_id,
            item.question,
            item.original_answer,
            item.candidate_answer,
            None,
            item.expected_label,
            item.perturbation_type.value,
        )
    elif isinstance(item, GoldenDatasetItem):
        # Ground truth: if labeled as reference_correct and supported, answer is CORRECT
        gt = GradeLabel.CORRECT if (item.reference_correct and item.supported_by_chunk) else GradeLabel.WRONG
        category = item.question_type.value if hasattr(item.question_type, "value") else str(item.question_type)
        return (
            item.query_id,
            item.question,
            item.ground_truth_answer,
            item.ground_truth_answer,  # Candidate is reference when evaluating on clean gold
            item.context,
            gt,
            category,
        )
    elif isinstance(item, dict):
        item_id = str(item.get("id") or item.get("query_id") or item.get("perturbed_id", "unknown"))
        q = str(item.get("question", ""))
        ref = str(item.get("reference_answer") or item.get("ground_truth_answer") or item.get("original_answer", ""))
        cand = str(item.get("candidate_answer", ref))
        ctx = item.get("context")
        raw_gt = str(item.get("ground_truth") or item.get("expected_label", "correct")).lower()
        gt = GradeLabel(raw_gt) if raw_gt in (g.value for g in GradeLabel) else GradeLabel.CORRECT
        cat = str(item.get("category") or item.get("perturbation_type") or item.get("question_type", "general"))
        return (item_id, q, ref, cand, ctx, gt, cat)
    else:
        raise TypeError(f"Unsupported item type: {type(item)}")


def evaluate_single_grader(
    grader: BaseGrader,
    items: Sequence[Union[GoldenDatasetItem, PerturbedItem, Dict[str, Any]]],
    dataset_name: str = "dataset",
    cost_key: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> GraderEvaluationReport:
    """
    Evaluates one grader against a collection of items with known ground truth.
    """
    y_true: List[GradeLabel] = []
    y_pred: List[GradeLabel] = []
    scores: List[float] = []
    latencies: List[float] = []
    evaluations: List[ItemEvaluation] = []
    category_counts: Dict[str, Dict[str, int]] = {}

    for item in items:
        item_id, q, ref, cand, ctx, gt, cat = _extract_triplet(item)

        grade_res = grader.grade(
            question=q,
            reference_answer=ref,
            candidate_answer=cand,
            context=ctx,
        )

        y_true.append(gt)
        y_pred.append(grade_res.label)
        scores.append(grade_res.score)
        latencies.append(grade_res.latency_ms)

        is_correct = (grade_res.label == gt)
        evaluations.append(ItemEvaluation(
            item_id=item_id,
            question=q,
            ground_truth=gt,
            predicted_label=grade_res.label,
            score=grade_res.score,
            is_correct_decision=is_correct,
            latency_ms=grade_res.latency_ms,
            category=cat,
            rationale=grade_res.rationale,
        ))

        # Category breakdown
        if cat not in category_counts:
            category_counts[cat] = {"total": 0, "correct_decisions": 0, "grader_predicted_correct": 0, "grader_predicted_wrong": 0}
        category_counts[cat]["total"] += 1
        if is_correct:
            category_counts[cat]["correct_decisions"] += 1
        if grade_res.label == GradeLabel.CORRECT:
            category_counts[cat]["grader_predicted_correct"] += 1
        elif grade_res.label == GradeLabel.WRONG:
            category_counts[cat]["grader_predicted_wrong"] += 1

    # Confusion matrix & metrics
    cm = compute_confusion_matrix(y_true, y_pred)
    class_metrics = compute_classification_metrics(cm)
    kappa = compute_cohens_kappa(y_true, y_pred)
    spearman = compute_spearman_rho(scores, y_true)
    p50, p95, mean_lat = compute_percentiles(latencies)

    # Bootstrap CI for F1
    ci_f1 = bootstrap_ci(y_true, y_pred, metric_name="f1", n_bootstrap=n_bootstrap, seed=seed)

    # Cost
    cost_lookup = cost_key or grader.name
    cost_per_100 = GRADER_COST_PER_100.get(cost_lookup, 0.0)

    # Failure breakdown by category
    failure_breakdown: Dict[str, Dict[str, Any]] = {}
    top_failures: List[str] = []

    for cat, counts in category_counts.items():
        tot = counts["total"]
        corr = counts["correct_decisions"]
        acc = corr / tot if tot > 0 else 0.0
        err_rate = 1.0 - acc
        failure_breakdown[cat] = {
            "total_items": tot,
            "correct_decisions": corr,
            "accuracy": round(acc, 4),
            "error_rate": round(err_rate, 4),
        }
        if err_rate >= 0.3 and tot >= 3:
            top_failures.append(f"{grader.name} fails {err_rate*100:.1f}% on category '{cat}' ({tot - corr}/{tot} errors)")

    if not top_failures:
        top_failures.append(f"{grader.name} maintained robust performance across all tested categories (<30% error rate).")

    abstain_rate = cm.abstain / cm.total if cm.total > 0 else 0.0

    metrics = GraderMetrics(
        grader_name=grader.name,
        dataset_name=dataset_name,
        sample_size=len(items),
        confusion_matrix=cm,
        accuracy=class_metrics["accuracy"],
        precision=class_metrics["precision"],
        recall=class_metrics["recall"],
        f1=class_metrics["f1"],
        specificity=class_metrics["specificity"],
        cohens_kappa=kappa,
        spearman_rho=spearman,
        abstain_rate=abstain_rate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_mean_ms=mean_lat,
        cost_per_100_queries_usd=cost_per_100,
        f1_ci_95=ci_f1,
    )

    return GraderEvaluationReport(
        metrics=metrics,
        item_evaluations=evaluations,
        failure_breakdown_by_category=failure_breakdown,
        top_failure_modes=top_failures,
    )


@dataclass
class MetaEvalSuiteResult:
    """Full suite result comparing multiple graders."""
    dataset_name: str
    sample_size: int
    grader_reports: Dict[str, GraderEvaluationReport]
    pairwise_mcnemar: Dict[str, Dict[str, Any]]
    ranking_by_f1: List[str]
    ranking_by_kappa: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "sample_size": self.sample_size,
            "grader_reports": {k: v.to_dict() for k, v in self.grader_reports.items()},
            "pairwise_mcnemar": self.pairwise_mcnemar,
            "ranking_by_f1": self.ranking_by_f1,
            "ranking_by_kappa": self.ranking_by_kappa,
        }


def run_meta_evaluation(
    graders: Sequence[BaseGrader],
    items: Sequence[Union[GoldenDatasetItem, PerturbedItem, Dict[str, Any]]],
    dataset_name: str = "dataset",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> MetaEvalSuiteResult:
    """
    Runs all graders on items, computes metrics, failure analysis, and pairwise McNemar tests.
    """
    reports: Dict[str, GraderEvaluationReport] = {}

    for grader in graders:
        reports[grader.name] = evaluate_single_grader(
            grader=grader,
            items=items,
            dataset_name=dataset_name,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    # Pairwise McNemar tests
    pairwise: Dict[str, Dict[str, Any]] = {}
    grader_names = list(reports.keys())

    for i in range(len(grader_names)):
        for j in range(i + 1, len(grader_names)):
            g_a = grader_names[i]
            g_b = grader_names[j]
            rep_a = reports[g_a]
            rep_b = reports[g_b]

            y_true = [e.ground_truth for e in rep_a.item_evaluations]
            y_pred_a = [e.predicted_label for e in rep_a.item_evaluations]
            y_pred_b = [e.predicted_label for e in rep_b.item_evaluations]

            res = mcnemar_test(y_true, y_pred_a, y_pred_b)
            pair_key = f"{g_a}__vs__{g_b}"
            pairwise[pair_key] = res

    # Rankings
    f1_ranked = sorted(grader_names, key=lambda n: reports[n].metrics.f1, reverse=True)
    kappa_ranked = sorted(grader_names, key=lambda n: reports[n].metrics.cohens_kappa, reverse=True)

    return MetaEvalSuiteResult(
        dataset_name=dataset_name,
        sample_size=len(items),
        grader_reports=reports,
        pairwise_mcnemar=pairwise,
        ranking_by_f1=f1_ranked,
        ranking_by_kappa=kappa_ranked,
    )
