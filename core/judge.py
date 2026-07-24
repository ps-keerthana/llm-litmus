"""
Core LLM Judge Module (core/judge.py)

Phase 1: Domain-agnostic judge prompt sourced from config.DOMAIN_DESCRIPTION.
Phase 3: 4-signal composite oracle — semantic + token F1 + numeric + negation —
         replaces the single-threshold embedding gate that failed on similar-vocabulary
         but factually wrong answers (e.g. '₹1.5L' vs '₹2L').
Phase 4: Opt-in ensemble judging — runs multiple judge models, detects disagreement,
         returns weighted-average scores. Enabled with JUDGE_ENSEMBLE=true.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import config
from config import (
    ORACLE_AUTO_PASS_THRESHOLD,
    ORACLE_AUTO_FAIL_THRESHOLD,
    MULTI_SIGNAL_TOKEN_F1_MIN,
    MULTI_SIGNAL_NUMBERS_CHECK,
    MULTI_SIGNAL_NEGATION_CHECK,
    JUDGE_ENSEMBLE_ENABLED,
    JUDGE_DISAGREEMENT_THRESHOLD,
)
from core.metrics import multi_signal_auto_pass
from core.generator import call_groq_with_retry


# ── Phase 4: Judge Configuration ─────────────────────────────────────────

@dataclass
class JudgeConfig:
    """Configuration for a single judge model in the ensemble."""
    provider: str       # "groq", "openai", "anthropic"
    model: str          # model name
    weight: float       # contribution weight (all weights should sum to ~1.0)
    temperature: float = 0.0


# Default ensemble: two Groq models with different sizes to reduce systematic bias.
# Weights sum to 1.0. The larger 70B model gets 60% weight.
DEFAULT_JUDGE_ENSEMBLE: List[JudgeConfig] = [
    JudgeConfig("groq", "llama-3.3-70b-versatile", weight=0.6),
    JudgeConfig("groq", "llama-3.1-8b-instant",    weight=0.4),
]


# ── Judge Prompt Builder ──────────────────────────────────────────────────

def _build_judge_prompt(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
) -> str:
    """
    Phase 1: Uses config.DOMAIN_DESCRIPTION so the judge is domain-aware.
    Previously hard-coded 'Indian income tax Q&A system'.
    """
    return f"""You are an expert evaluator for a {config.DOMAIN_DESCRIPTION} RAG system. \
Evaluate the generated answer against the ground truth and retrieved context.

Inputs:
- Question: {question}
- Ground Truth: {ground_truth}
- Retrieved Context:
{context}
- Generated Answer: {answer}

Definitions & Scoring Rules:
1. correctness (0.0 to 1.0):
   Rate 1.0 if factually correct relative to the ground truth.
   If the ground truth expects a refusal and the answer correctly refuses, score 1.0.

2. faithfulness (0.0 to 1.0):
   Rate 1.0 if every claim in the answer is directly supported by the retrieved context.
   Only penalise if the answer states a fact NOT present in or contradicted by the context.

3. completeness (0.0 to 1.0):
   Rate 1.0 if the answer covers all necessary details present in the ground truth.
   Penalise if it omits vital requirements, limits, or constraints.

4. hallucination (0.0 to 1.0):
   Rate 0.0 if there is no hallucinated or unsupported information.
   Rate high (0.8+) if the answer fabricates details, numbers, or rules not in the context.
   Typically hallucination = 1.0 - faithfulness.

5. confidence (0.0 to 1.0):
   Your confidence in this grading assessment.

Return ONLY a valid JSON object with these exact keys:
"correctness", "faithfulness", "completeness", "hallucination", "confidence", "reasoning"
Do not include markdown formatting or wrapping."""


# ── Single Judge Call ─────────────────────────────────────────────────────

def llm_judge_evaluate(
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: List[str],
    provider: Optional[str] = None,
) -> Tuple[Dict[str, Any], int, int]:
    """
    Grades RAG generation quality using a single LLM-as-a-judge.
    Returns (metrics_dict, prompt_tokens, completion_tokens).
    """
    context = "\n\n".join(context_chunks)
    prompt = _build_judge_prompt(question, answer, ground_truth, context)

    from core.cache import get_cache_key, lookup_cache, update_cache
    from core.providers import get_provider_client

    provider_client = get_provider_client(provider)
    model_name = provider_client.get_model_name()

    cache_payload = f"JUDGE|{question}|{answer}|{ground_truth}|{context}"
    cache_key = get_cache_key(f"judge-{model_name}", cache_payload, 0.0)
    cached = lookup_cache(cache_key)
    if cached is not None:
        return cached["metrics"], cached["prompt_tokens"], cached["completion_tokens"]

    try:
        result = provider_client.complete(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(result.text.strip())
        metrics = {
            "correctness":   float(data.get("correctness",   0.0)),
            "faithfulness":  float(data.get("faithfulness",  1.0)),
            "completeness":  float(data.get("completeness",  1.0)),
            "hallucination": float(data.get("hallucination", 0.0)),
            "confidence":    float(data.get("confidence",    1.0)),
            "reasoning":     data.get("reasoning", "Grading succeeded."),
        }
        update_cache(cache_key, {
            "metrics": metrics,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        })
        return metrics, result.prompt_tokens, result.completion_tokens

    except Exception as e:
        print(f"  [Judge] LLM judge failed: {e}")
        return {
            "correctness":   0.0,
            "faithfulness":  1.0,
            "completeness":  1.0,
            "hallucination": 0.0,
            "confidence":    0.0,
            "reasoning":     f"Fallback applied due to evaluation error: {e}",
        }, 0, 0


# ── Phase 4: Ensemble Judge ───────────────────────────────────────────────

def _run_single_judge(
    cfg: JudgeConfig,
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: List[str],
) -> Tuple[Dict[str, Any], int, int]:
    """Calls a single judge from the ensemble configuration."""
    context = "\n\n".join(context_chunks)
    prompt = _build_judge_prompt(question, answer, ground_truth, context)

    from core.providers import get_provider_client
    provider_client = get_provider_client(cfg.provider)

    try:
        result = provider_client.complete(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=cfg.temperature,
        )
        data = json.loads(result.text.strip())
        metrics = {
            "correctness":   float(data.get("correctness",   0.0)),
            "faithfulness":  float(data.get("faithfulness",  1.0)),
            "completeness":  float(data.get("completeness",  1.0)),
            "hallucination": float(data.get("hallucination", 0.0)),
            "confidence":    float(data.get("confidence",    1.0)),
            "reasoning":     data.get("reasoning", ""),
        }
        return metrics, result.prompt_tokens, result.completion_tokens
    except Exception as e:
        return {
            "correctness": 0.0, "faithfulness": 1.0, "completeness": 1.0,
            "hallucination": 0.0, "confidence": 0.0,
            "reasoning": f"Judge error ({cfg.model}): {e}",
        }, 0, 0


def ensemble_judge_evaluate(
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: List[str],
    ensemble: List[JudgeConfig] = None,
) -> Tuple[Dict[str, Any], int, int, bool]:
    """
    Phase 4: Runs multiple judge models and returns weighted-average scores.
    Also detects judge disagreement when any two judges differ by > JUDGE_DISAGREEMENT_THRESHOLD.

    Returns: (metrics, total_prompt_tokens, total_completion_tokens, judge_disagreement_detected)
    """
    if ensemble is None:
        ensemble = DEFAULT_JUDGE_ENSEMBLE

    all_scores: List[Tuple[Dict, float]] = []  # (metrics, weight)
    total_p, total_c = 0, 0

    for cfg in ensemble:
        scores, p, c = _run_single_judge(cfg, question, answer, ground_truth, context_chunks)
        all_scores.append((scores, cfg.weight))
        total_p += p
        total_c += c

    # Weighted average of numeric fields
    numeric_keys = ["correctness", "faithfulness", "completeness", "hallucination", "confidence"]
    total_weight = sum(w for _, w in all_scores)
    blended: Dict[str, Any] = {}

    for key in numeric_keys:
        blended[key] = round(
            sum(s[key] * w for s, w in all_scores) / total_weight, 3
        )

    # Use reasoning from the highest-weight judge
    blended["reasoning"] = all_scores[0][0].get("reasoning", "Ensemble judge.")

    # Disagreement detection: flag if any two judges differ by > threshold on correctness
    disagreement = False
    if len(all_scores) >= 2:
        for key in ["correctness", "hallucination"]:
            values = [s[key] for s, _ in all_scores]
            if max(values) - min(values) > JUDGE_DISAGREEMENT_THRESHOLD:
                disagreement = True
                break

    if disagreement:
        blended["reasoning"] += (
            f" [ENSEMBLE DISAGREEMENT DETECTED — individual scores: "
            + ", ".join(f"{cfg.model}={s['correctness']:.2f}" for (s, _), cfg in zip(all_scores, ensemble))
            + "]"
        )

    return blended, total_p, total_c, disagreement


# ── Phase 3: Multi-Signal Oracle Routing ─────────────────────────────────

def evaluate_with_oracle_routing(
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: List[str],
    semantic_sim: float,
    no_judge: bool = False,
    is_refusal: bool = False,
    provider: Optional[str] = None,
) -> Tuple[Dict[str, Any], int, int, bool]:
    """
    Evaluates answer quality using the 4-signal composite oracle gate (Phase 3).

    Decision tree:
      1. --no-judge  → skip all judge calls, proxy from semantic sim only
      2. Refusal     → auto-pass for correct out-of-scope handling
      3. Oracle gate → multi-signal composite: sem + token F1 + numbers + negation
                       Auto-PASS only if ALL signals agree
                       Auto-FAIL if semantic sim is extremely low
      4. Ambiguous   → call LLM judge (single or ensemble per config)

    Returns: (metrics_dict, prompt_tokens, completion_tokens, judge_called)
    """
    # ── Case 1: Judge disabled via CLI ────────────────────────────────────
    if no_judge:
        from core.metrics import compute_token_f1
        token_f1 = compute_token_f1(answer, ground_truth)
        correctness_proxy = min(1.0, semantic_sim / 0.75)
        return {
            "correctness":   round(correctness_proxy, 3),
            "faithfulness":  "Not Evaluated",
            "completeness":  "Not Evaluated",
            "hallucination": "Not Evaluated",
            "confidence":    round(semantic_sim, 3),
            "reasoning":     f"Judge bypassed (--no-judge). Proxy correctness from sim={semantic_sim}.",
            "token_f1":      token_f1,
            "judge_disagreement": False,
        }, 0, 0, False

    # ── Case 2: Correct refusal for out-of-scope query ────────────────────
    if is_refusal:
        return {
            "correctness":   1.0,
            "faithfulness":  1.0,
            "completeness":  1.0,
            "hallucination": 0.0,
            "confidence":    1.0,
            "reasoning":     "Bypassed judge: Correct refusal for out-of-scope question.",
            "token_f1":      1.0,
            "judge_disagreement": False,
        }, 0, 0, False

    # ── Case 3: Hard auto-fail (extremely low similarity) ─────────────────
    if semantic_sim <= ORACLE_AUTO_FAIL_THRESHOLD:
        # An answer with near-zero semantic similarity is almost certainly
        # hallucinated or completely off-topic — faithfulness should be 0.0
        # and hallucination should be 1.0, not the other way around.
        return {
            "correctness":   0.0,
            "faithfulness":  0.0,
            "completeness":  0.0,
            "hallucination": 1.0,
            "confidence":    1.0,
            "reasoning":     f"Auto-FAIL: semantic sim {semantic_sim:.3f} <= {ORACLE_AUTO_FAIL_THRESHOLD}. Answer is off-topic or hallucinated.",
            "token_f1":      0.0,
            "judge_disagreement": False,
        }, 0, 0, False

    # ── Case 4: Multi-signal composite auto-pass gate (Phase 3) ──────────
    auto_pass, signals = multi_signal_auto_pass(
        answer=answer,
        ground_truth=ground_truth,
        semantic_sim=semantic_sim,
        semantic_threshold=ORACLE_AUTO_PASS_THRESHOLD,
        token_f1_min=MULTI_SIGNAL_TOKEN_F1_MIN,
        check_numbers=MULTI_SIGNAL_NUMBERS_CHECK,
        check_negation=MULTI_SIGNAL_NEGATION_CHECK,
    )

    if auto_pass:
        return {
            "correctness":   1.0,
            "faithfulness":  1.0,
            "completeness":  1.0,
            "hallucination": 0.0,
            "confidence":    1.0,
            "reasoning": (
                f"Auto-PASS: all signals agree — "
                f"sem={semantic_sim:.3f}, f1={signals['token_f1']:.3f}, "
                f"numbers={'ok' if signals['numbers_ok'] else 'fail'}, "
                f"negation={'ok' if signals['negation_ok'] else 'fail'}."
            ),
            "token_f1": signals["token_f1"],
            "judge_disagreement": False,
        }, 0, 0, False

    # ── Case 5: Ambiguous region — call LLM judge ─────────────────────────
    if JUDGE_ENSEMBLE_ENABLED:
        metrics, p, c, disagreement = ensemble_judge_evaluate(
            question, answer, ground_truth, context_chunks
        )
        metrics["token_f1"] = signals["token_f1"]
        metrics["judge_disagreement"] = disagreement
        return metrics, p, c, True
    else:
        metrics, p, c = llm_judge_evaluate(
            question, answer, ground_truth, context_chunks, provider=provider
        )
        metrics["token_f1"] = signals["token_f1"]
        metrics.setdefault("judge_disagreement", False)
        return metrics, p, c, True
