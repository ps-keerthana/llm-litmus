"""
Core Failure Attribution Module (core/attributor.py)
Deterministically categorizes failed query results and diagnoses RAG quality boundaries.
Integrates a counterfactual diagnoser to isolate retrieval vs. generation failures.
"""

import os
from typing import Dict, Any, Tuple, List
import config
from core.utils import logger


def load_ground_truth_context(expected_sources_str: str) -> List[str]:
    """Loads the raw content of the expected source files from the docs/ directory."""
    expected_sources = []
    if expected_sources_str and expected_sources_str != "N/A":
        expected_sources = [s.strip() for s in expected_sources_str.split(";")]
        
    chunks = []
    for src in expected_sources:
        doc_path = os.path.join(config.DOCS_FOLDER, src)
        if os.path.exists(doc_path):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    chunks.append(f.read())
            except Exception as e:
                logger.warning(f"[Diagnoser] Failed to read doc '{src}': {e}")
    return chunks


def run_counterfactual_diagnosis(result: Dict[str, Any]) -> Tuple[str, str, str, float]:
    """
    Executes a counterfactual sandbox run for a failed query:
    1. Loads raw ground-truth contexts directly from docs/.
    2. Swaps retrieved contexts with the ground-truth contexts.
    3. Re-runs answer generation.
    4. Computes semantic similarity of the counterfactual answer against the ground truth.
    
    If the counterfactual answer passes, the issue is isolated to the Retriever (Retrieval Failure).
    If it still fails, the issue is isolated to the Generator (Generation Failure).
    
    Returns:
        (failure_category, diagnosis_reason, counterfactual_answer, counterfactual_similarity)
    """
    question = result.get("question", "")
    ground_truth = result.get("ground_truth", "")
    expected_sources_str = result.get("expected_sources", "")
    
    # 1. Check for Knowledge Base Gap
    expected_sources = []
    if expected_sources_str and expected_sources_str != "N/A":
        expected_sources = [s.strip() for s in expected_sources_str.split(";")]
        
    missing_docs = []
    for src in expected_sources:
        doc_path = os.path.join(config.DOCS_FOLDER, src)
        if not os.path.exists(doc_path):
            missing_docs.append(src)
            
    if missing_docs:
        category = "Knowledge Base Gap"
        reason = f"Expected document(s) {missing_docs} are missing from docs directory '{config.DOCS_FOLDER}'."
        return category, reason, "", 0.0

    # 2. Load ground truth context chunks
    gt_chunks = load_ground_truth_context(expected_sources_str)
    if not gt_chunks:
        category = "Knowledge Base Gap"
        reason = "No valid expected source documents could be loaded from the docs directory."
        return category, reason, "", 0.0

    # 3. Run counterfactual generation
    from core.generator import generate_answer
    from core.metrics import compute_semantic_similarity
    
    logger.info(f"[Diagnoser] Running counterfactual sandbox generation for query: '{question[:40]}...'")
    try:
        cf_answer, _, _ = generate_answer(
            question=question,
            context_chunks=gt_chunks,
            temperature=0.0
        )
        cf_sim = compute_semantic_similarity(cf_answer, ground_truth)
        logger.info(f"[Diagnoser] Counterfactual similarity: {cf_sim:.3f}")
        
        # 4. Classify based on counterfactual similarity threshold.
        # Deliberately higher than the general MIN_SEMANTIC_SIM (0.65):
        # to call it a "Retrieval Failure" we need strong evidence that the
        # generator is definitively correct when given GT context.
        CF_ISOLATION_THRESHOLD = 0.75
        if cf_sim >= CF_ISOLATION_THRESHOLD:
            category = "Retrieval Failure"
            reason = (
                f"Isolator PASS: LLM generated a correct answer (sim={cf_sim:.3f}) when provided with raw ground-truth contexts. "
                "The generator prompt and model are correct; context was missed or ranked poorly by retriever."
            )
        else:
            category = "LLM Generation Failure"
            reason = (
                f"Isolator FAIL: LLM still generated an incorrect answer (sim={cf_sim:.3f} < {CF_ISOLATION_THRESHOLD}) "
                "even when provided with direct ground-truth context. The generator prompt or model reasoning capabilities are insufficient."
            )
        return category, reason, cf_answer, cf_sim

    except Exception as e:
        logger.error(f"[Diagnoser Error] Counterfactual sandbox generation failed: {e}")
        category = "Needs Manual Review"
        reason = f"Counterfactual execution failed with exception: {e}"
        return category, reason, "", 0.0


def attribute_failure(result: Dict[str, Any]) -> Tuple[str, str]:
    """
    Deterministically attributes a failed query to a root cause category.
    Returns a tuple: (failure_category, attribution_reason)
    
    If the status is 'FAIL', it executes the counterfactual diagnoser sandbox to isolate
    the root cause between Retrieval, Generation, and Knowledge Base gaps.
    """
    status = result.get("status", "PASS")
    if status == "PASS":
        return "N/A", "Query passed all quality checks."

    # Execute counterfactual diagnoser sandbox for failed queries
    category, reason, _, _ = run_counterfactual_diagnosis(result)
    return category, reason


def build_retrieval_diagnosis(result: Dict[str, Any]) -> Dict[str, bool]:
    """
    Analyzes retrieval and model context utilisation.
    Returns 4 boolean flags:
    - context_retrieved: True if hit_rate > 0
    - context_sufficient: True if context_recall >= ATTRIBUTION_RECALL_MIN
    - model_used_context: True if faithfulness is acceptable and model didn't refuse
    - model_hallucinated: True if hallucination rate is high
    """
    hit_rate = result.get("hit_rate", 0.0)
    context_recall = result.get("context_recall", 0.0)
    faithfulness = result.get("faithfulness", 1.0)
    hallucination = result.get("hallucination_rate", 0.0)
    answer = result.get("answer", "").lower()
    
    if isinstance(faithfulness, str):
        faithfulness = 1.0
    if isinstance(hallucination, str):
        hallucination = 0.0

    refusal_keywords = ["don't have information", "do not have information", "no information"]
    model_refused = any(kw in answer for kw in refusal_keywords)

    context_retrieved = hit_rate > 0.0
    context_sufficient = context_recall >= config.ATTRIBUTION_RECALL_MIN
    model_used_context = (not model_refused) and (faithfulness >= config.ATTRIBUTION_FAITHFULNESS_MIN)
    model_hallucinated = hallucination > config.ATTRIBUTION_HALLUCINATION_MAX

    return {
        "context_retrieved": context_retrieved,
        "context_sufficient": context_sufficient,
        "model_used_context": model_used_context,
        "model_hallucinated": model_hallucinated
    }
