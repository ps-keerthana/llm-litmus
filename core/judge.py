"""
Core LLM Judge Module (core/judge.py)
Employs an LLM-as-a-judge to evaluate generated answers across multiple dimensions
(correctness, faithfulness, completeness, hallucination, confidence) in a single JSON API call.
Implements oracle-efficient routing to bypass judge calls when similarity is unambiguous.
"""

import json
from typing import List, Dict, Any, Tuple
from core.generator import call_groq_with_retry
from config import ORACLE_AUTO_PASS_THRESHOLD, ORACLE_AUTO_FAIL_THRESHOLD

def llm_judge_evaluate(
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: List[str]
) -> Tuple[Dict[str, Any], int, int]:
    """
    Grades RAG generation quality using LLM-as-a-judge.
    Returns a tuple of (metrics_dict, prompt_tokens, completion_tokens).
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an expert evaluator for an Indian income tax Q&A system. Evaluate the generated answer against the ground truth and retrieved context.

Inputs:
- Question: {question}
- Ground Truth: {ground_truth}
- Retrieved Context:
{context}
- Generated Answer: {answer}

Definitions & Scoring Rules:
1. correctness (0.0 to 1.0):
   - Rate 1.0 if the answer is factually correct relative to the ground truth.
   - If the ground truth expects a refusal (e.g. "I don't have information about that"), and the answer correctly refuses, score 1.0.

2. faithfulness (0.0 to 1.0):
   - Rate 1.0 if every single claim in the answer is directly supported by the retrieved context.
   - Do NOT penalize faithfulness for brief answers or omissions of secondary context details. Only penalize if the answer states a fact that is not present in or is contradicted by the context.

3. completeness (0.0 to 1.0):
   - Rate 1.0 if the answer covers all the necessary details present in the ground truth.
   - Penalize if it leaves out vital requirements (e.g. key limits, age constraints) that are specified in the ground truth.

4. hallucination (0.0 to 1.0):
   - Rate 0.0 if there is no hallucinated or unsupported information.
   - Rate high (e.g. 0.8) if the answer fabricates details, numbers, or rules not in the context. Typically, hallucination = 1.0 - faithfulness.

5. confidence (0.0 to 1.0):
   - Rate your confidence level in this grading assessment.

Provide your evaluation as a JSON object with these exact keys:
- "correctness": float (0.0 to 1.0)
- "faithfulness": float (0.0 to 1.0)
- "completeness": float (0.0 to 1.0)
- "hallucination": float (0.0 to 1.0)
- "confidence": float (0.0 to 1.0)
- "reasoning": a brief, clear explanation of the grading assessment

Return ONLY a valid JSON object. Do not include markdown formatting or wrapping."""

    # Check cache first to bypass provider calls for identical outputs
    from core.cache import get_cache_key, lookup_cache, update_cache
    from core.providers import get_provider_client

    provider = get_provider_client()
    model_name = provider.get_model_name()

    cache_payload = f"JUDGE|{question}|{answer}|{ground_truth}|{context}"
    cache_key = get_cache_key(f"judge-{model_name}", cache_payload, 0.0)
    cached = lookup_cache(cache_key)
    if cached is not None:
        return (
            cached["metrics"],
            cached["prompt_tokens"],
            cached["completion_tokens"]
        )

    try:
        result = provider.complete(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        data = json.loads(result.text.strip())
        
        metrics = {
            "correctness": float(data.get("correctness", 0.0)),
            "faithfulness": float(data.get("faithfulness", 1.0)),
            "completeness": float(data.get("completeness", 1.0)),
            "hallucination": float(data.get("hallucination", 0.0)),
            "confidence": float(data.get("confidence", 1.0)),
            "reasoning": data.get("reasoning", "Grading succeeded.")
        }
        
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        
        # Save to cache
        update_cache(cache_key, {
            "metrics": metrics,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        })
        
        return metrics, prompt_tokens, completion_tokens
    except Exception as e:
        print(f"  [Error] LLM judge failed: {e}")
        # Robust fallback values
        fallback_metrics = {
            "correctness": 0.0,
            "faithfulness": 1.0,
            "completeness": 1.0,
            "hallucination": 0.0,
            "confidence": 0.0,
            "reasoning": f"Fallback applied due to evaluation error: {e}"
        }
        return fallback_metrics, 0, 0


def evaluate_with_oracle_routing(
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: List[str],
    semantic_sim: float,
    no_judge: bool = False,
    is_refusal: bool = False
) -> Tuple[Dict[str, Any], int, int, bool]:
    """
    Evaluates answer quality, using oracle-efficient routing to bypass expensive
    LLM judge calls when semantic similarity is unambiguous.
    
    Returns:
        (metrics_dict, prompt_tokens, completion_tokens, judge_called)
    """
    # Case 1: Legacy --no-judge CLI mode
    if no_judge:
        correctness_proxy = min(1.0, semantic_sim / 0.75)
        metrics = {
            "correctness": round(correctness_proxy, 3),
            "faithfulness": "Not Evaluated",
            "completeness": "Not Evaluated",
            "hallucination": "Not Evaluated",
            "confidence": round(semantic_sim, 3),
            "reasoning": f"Judge bypassed: --no-judge flag enabled. Proxy correctness from sim={semantic_sim}."
        }
        return metrics, 0, 0, False

    # Case 2: Handled refusal for out-of-scope query
    if is_refusal:
        metrics = {
            "correctness": 1.0,
            "faithfulness": 1.0,
            "completeness": 1.0,
            "hallucination": 0.0,
            "confidence": 1.0,
            "reasoning": "Bypassed judge: Correct refusal for out-of-scope question."
        }
        return metrics, 0, 0, False

    # Case 3: Oracle Auto-Pass (extremely high semantic similarity)
    if semantic_sim >= ORACLE_AUTO_PASS_THRESHOLD:
        metrics = {
            "correctness": 1.0,
            "faithfulness": 1.0,
            "completeness": 1.0,
            "hallucination": 0.0,
            "confidence": 1.0,
            "reasoning": f"Bypassed judge: High semantic similarity ({semantic_sim:.3f} >= {ORACLE_AUTO_PASS_THRESHOLD}) to ground truth."
        }
        return metrics, 0, 0, False

    # Case 4: Oracle Auto-Fail (extremely low semantic similarity)
    if semantic_sim <= ORACLE_AUTO_FAIL_THRESHOLD:
        metrics = {
            "correctness": 0.0,
            "faithfulness": 1.0,  # No hallucinated factual claims are asserted in garbage answers
            "completeness": 0.0,
            "hallucination": 0.0,
            "confidence": 1.0,
            "reasoning": f"Bypassed judge: Low semantic similarity ({semantic_sim:.3f} <= {ORACLE_AUTO_FAIL_THRESHOLD}) to ground truth."
        }
        return metrics, 0, 0, False

    # Case 5: Ambiguous region -> call the LLM judge
    metrics, p_tokens, c_tokens = llm_judge_evaluate(
        question, answer, ground_truth, context_chunks
    )
    return metrics, p_tokens, c_tokens, True
