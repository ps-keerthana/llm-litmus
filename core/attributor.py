"""
Core Failure Attribution Module (core/attributor.py)
Deterministically categorizes failed query results and diagnoses RAG quality boundaries.
"""

import os
from typing import Dict, Any, Tuple
import config

def attribute_failure(result: Dict[str, Any]) -> Tuple[str, str]:
    """
    Deterministically attributes a failed query to a root cause category.
    Returns a tuple: (failure_category, attribution_reason)
    
    Categories:
    - "Knowledge Base Gap": Expected source document is completely missing from corpus folder.
    - "Retrieval Failure": Expected source exists, but retriever failed to fetch/rank it.
    - "LLM Hallucination": LLM generated facts not supported by retrieved context.
    - "Evaluation False Negative": LLM judge deemed answer correct, but semantic similarity model flagged it.
    - "LLM Generation Failure": Context was successfully retrieved and sufficient, but LLM over-refused or answered incorrectly.
    - "Needs Manual Review": Any failure that doesn't trigger clear heuristic gates.
    """
    # 1. Fetch expected sources
    expected_sources_str = result.get("expected_sources", "")
    expected_sources = []
    if expected_sources_str and expected_sources_str != "N/A":
        expected_sources = [s.strip() for s in expected_sources_str.split(";")]
        
    # Check if any expected source file is actually missing from the docs directory
    missing_docs = []
    for src in expected_sources:
        doc_path = os.path.join(config.DOCS_FOLDER, src)
        if not os.path.exists(doc_path):
            missing_docs.append(src)
            
    is_missing_from_corpus = len(missing_docs) > 0

    # 2. Extract metrics
    hit_rate = result.get("hit_rate", 1.0)
    context_recall = result.get("context_recall", 1.0)
    semantic_sim = result.get("semantic_similarity", 0.0)
    
    # LLM judge metrics (might be "Not Evaluated" in smoke/no-judge mode)
    judge_correctness = result.get("correctness", 0.0)
    judge_faithfulness = result.get("faithfulness", 1.0)
    judge_hallucination = result.get("hallucination_rate", 0.0)
    
    # Convert judge metrics if they are strings (e.g. "Not Evaluated")
    is_judge_evaluated = True
    if isinstance(judge_correctness, str) or isinstance(judge_faithfulness, str):
        is_judge_evaluated = False
        judge_correctness = 0.0
        judge_faithfulness = 1.0
        judge_hallucination = 0.0

    # 3. Decision tree for attribution
    if is_missing_from_corpus:
        category = "Knowledge Base Gap"
        reason = f"Expected document(s) {missing_docs} are missing from corpus directory '{config.DOCS_FOLDER}'."
        return category, reason

    if hit_rate == 0.0 or context_recall < config.ATTRIBUTION_RECALL_MIN:
        category = "Retrieval Failure"
        reason = (
            f"Retrieval quality is low (hit_rate={hit_rate}, context_recall={context_recall:.2f} < "
            f"threshold {config.ATTRIBUTION_RECALL_MIN}). Expected source exists in corpus but was not ranked in top-k."
        )
        return category, reason

    if is_judge_evaluated and (judge_faithfulness < config.ATTRIBUTION_FAITHFULNESS_MIN or judge_hallucination > config.ATTRIBUTION_HALLUCINATION_MAX):
        category = "LLM Hallucination"
        reason = (
            f"LLM hallucinated unsupported claims (faithfulness={judge_faithfulness:.2f} < "
            f"{config.ATTRIBUTION_FAITHFULNESS_MIN} or hallucination={judge_hallucination:.2f} > "
            f"{config.ATTRIBUTION_HALLUCINATION_MAX})."
        )
        return category, reason

    if is_judge_evaluated and (judge_correctness >= config.ATTRIBUTION_JUDGE_CORRECTNESS_MIN and semantic_sim < config.ATTRIBUTION_SIM_MIN):
        category = "Evaluation False Negative"
        reason = (
            f"Semantic similarity is low ({semantic_sim:.3f} < {config.ATTRIBUTION_SIM_MIN}) but LLM judge "
            f"confirmed answer is correct ({judge_correctness:.2f}). Typically caused by length differences or structural mismatch."
        )
        return category, reason

    if is_judge_evaluated and (judge_correctness < config.ATTRIBUTION_JUDGE_CORRECTNESS_MIN and context_recall >= config.ATTRIBUTION_RECALL_MIN):
        category = "LLM Generation Failure"
        reason = (
            f"Retrieved context is sufficient (recall={context_recall:.2f}), but LLM failed to answer correctly "
            f"(correctness={judge_correctness:.2f} < {config.ATTRIBUTION_JUDGE_CORRECTNESS_MIN}). Over-refusal or inference gap."
        )
        return category, reason

    # Fallback gate
    category = "Needs Manual Review"
    reason = (
        f"Query failed absolute thresholds. Metrics: sim={semantic_sim:.3f}, correctness={judge_correctness}, "
        f"recall={context_recall:.2f}, hit_rate={hit_rate}."
    )
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
    
    # Check if they are strings (e.g. "Not Evaluated")
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
