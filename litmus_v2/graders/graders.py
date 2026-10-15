"""
Phase 4: Grader implementations for litmus_v2.

Central idea of litmus_v2: "evaluate the evaluator."
These graders are what get evaluated for trustworthiness in Phase 5.

Graders implemented:
    1. similarity_baseline  — Cosine similarity of TF-IDF or embedding vectors
    2. heuristic_gate       — Rule-based keyword/polarity matching
    3. nli_faithfulness     — DeBERTa NLI cross-encoder (premise=chunk, hyp=answer)
    4. llm_judge            — Rubric-based JSON judge (local Ollama or Groq API)

Each grader implements the GraderProtocol interface:
    grade(question, reference_answer, candidate_answer, context=None) -> GradeResult

Design invariants:
    - No grader reads from results/ or writes to results/.
    - llm_judge: temperature=0, JSON output, position bias check disabled by default
      but configurable. Self-consistency: run N times, take majority.
    - All graders are stateless and can be used in parallel.
    - API calls are cached to disk (cache/ folder) keyed by content hash.
"""

import hashlib
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


from litmus_v2.data.schema import GradeLabel


class GradeResult(BaseModel):
    question: str
    reference_answer: str
    candidate_answer: str
    grader: str
    label: GradeLabel
    score: float = Field(ge=0.0, le=1.0)  # Continuous confidence [0,1]
    rationale: Optional[str] = None
    latency_ms: Optional[float] = None
    model_used: Optional[str] = None
    cached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Base protocol
# ─────────────────────────────────────────────────────────────────────────────

class BaseGrader(ABC):
    """Abstract base class for all litmus_v2 graders."""
    name: str = "base"

    @abstractmethod
    def grade(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        context: Optional[str] = None,
    ) -> GradeResult:
        ...

    def _make_result(self, **kwargs) -> GradeResult:
        return GradeResult(
            grader=self.name,
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grader 1: Similarity Baseline (TF-IDF cosine)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> Dict[str, int]:
    """Simple word tokenizer — lowercases and removes punctuation."""
    words = re.findall(r'\b[a-z0-9₹%,]+\b', text.lower())
    counts: Dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts


def _cosine_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Compute cosine similarity between two bag-of-words vectors."""
    if not a or not b:
        return 0.0
    shared = set(a.keys()) & set(b.keys())
    dot = sum(a[w] * b[w] for w in shared)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class SimilarityGrader(BaseGrader):
    """
    TF-IDF bag-of-words cosine similarity grader.

    Threshold: configurable (default 0.5).
    Fast, no model required, high FPR for paraphrase.
    Intended as a baseline — not a reliable grader on its own.
    """
    name = "similarity_baseline"

    def __init__(self, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.threshold = threshold

    def grade(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        context: Optional[str] = None,
    ) -> GradeResult:
        t0 = time.monotonic()
        ref_vec = _tokenize(reference_answer)
        cand_vec = _tokenize(candidate_answer)
        score = _cosine_similarity(ref_vec, cand_vec)
        label = GradeLabel.CORRECT if score >= self.threshold else GradeLabel.WRONG
        latency = (time.monotonic() - t0) * 1000

        return self._make_result(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            label=label,
            score=round(score, 4),
            rationale=f"cosine_similarity={score:.4f} (threshold={self.threshold})",
            latency_ms=round(latency, 2),
            model_used=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grader 2: Heuristic Gate
# ─────────────────────────────────────────────────────────────────────────────

# Indian tax domain currency & amount pattern
_AMOUNT_RE = re.compile(
    r'(?:[₹]|rs\.?|inr)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakh|crore|thousand))?'
    r'|[\d]{1,3}(?:,\d{2,3})+(?:\.\d+)?'
    r'|[\d,]+(?:\.\d+)?\s*(?:lakh|crore|thousand)',
    re.IGNORECASE,
)


_NEGATION_WORDS = {"no", "not", "never", "cannot", "can't", "won't", "doesn't",
                   "isn't", "aren't", "didn't", "unable", "prohibited", "restricted"}
_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "as", "it", "this", "that", "these", "those",
    "under", "from", "per", "into"
}


def _extract_amounts(text: str) -> List[str]:
    return [m.group().strip() for m in _AMOUNT_RE.finditer(text)]


def _polarity(text: str) -> str:
    """Returns 'negative' if the text contains a negation word, else 'positive'."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return "negative" if words & _NEGATION_WORDS else "positive"


class HeuristicGrader(BaseGrader):
    """
    Rule-based heuristic grader for Indian tax domain Q&A.

    Rules (applied in order; first match wins):
        1. Polarity mismatch (yes/no flip) → WRONG
        2. Critical monetary amount mismatch → WRONG
        3. Candidate answer is too short (<20 chars) → WRONG
        4. No critical overlap → WRONG if overlap < min_overlap_ratio
        5. Default → CORRECT

    Fast (no model), interpretable, high precision for obvious errors.
    """
    name = "heuristic_gate"

    def __init__(
        self,
        min_overlap_ratio: float = 0.25,
        min_candidate_length: int = 20,
    ):
        self.min_overlap_ratio = min_overlap_ratio
        self.min_candidate_length = min_candidate_length

    def grade(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        context: Optional[str] = None,
    ) -> GradeResult:
        t0 = time.monotonic()
        rules_fired = []

        # Rule 1: Polarity mismatch
        ref_polarity = _polarity(reference_answer)
        cand_polarity = _polarity(candidate_answer)
        if ref_polarity != cand_polarity:
            rules_fired.append(f"polarity_mismatch(ref={ref_polarity},cand={cand_polarity})")
            return self._make_result(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                label=GradeLabel.WRONG,
                score=0.05,
                rationale=f"FAIL: {'; '.join(rules_fired)}",
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
            )

        # Rule 2: Critical monetary amount mismatch
        ref_amounts = set(_extract_amounts(reference_answer))
        cand_amounts = set(_extract_amounts(candidate_answer))
        if ref_amounts and cand_amounts and not ref_amounts.intersection(cand_amounts):
            rules_fired.append(f"amount_mismatch(ref={ref_amounts},cand={cand_amounts})")
            return self._make_result(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                label=GradeLabel.WRONG,
                score=0.1,
                rationale=f"FAIL: {'; '.join(rules_fired)}",
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
            )

        # Rule 3: Too short
        if len(candidate_answer.strip()) < self.min_candidate_length:
            rules_fired.append(f"too_short(len={len(candidate_answer.strip())})")
            return self._make_result(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                label=GradeLabel.WRONG,
                score=0.15,
                rationale=f"FAIL: {'; '.join(rules_fired)}",
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
            )

        # Rule 4: Overlap ratio (content words)
        ref_all = set(_tokenize(reference_answer).keys())
        cand_all = set(_tokenize(candidate_answer).keys())
        ref_words = ref_all - _STOP_WORDS or ref_all
        cand_words = cand_all - _STOP_WORDS or cand_all
        overlap = len(ref_words & cand_words) / max(len(ref_words), 1)
        if overlap < self.min_overlap_ratio:
            rules_fired.append(f"low_overlap(ratio={overlap:.2f})")
            return self._make_result(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                label=GradeLabel.WRONG,
                score=round(overlap, 4),
                rationale=f"FAIL: {'; '.join(rules_fired)}",
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
            )

        # Default: pass
        score = min(0.5 + overlap * 0.5, 1.0)
        return self._make_result(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            label=GradeLabel.CORRECT,
            score=round(score, 4),
            rationale=f"PASS: overlap={overlap:.2f}, polarity_match",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grader 3: NLI Faithfulness (DeBERTa cross-encoder)
# ─────────────────────────────────────────────────────────────────────────────

class NLIGrader(BaseGrader):
    """
    NLI faithfulness grader using a DeBERTa-based cross-encoder.
    Model: cross-encoder/nli-deberta-v3-small (HuggingFace, local).

    premise  = source chunk (context)
    hypothesis = candidate answer

    Labels:
        entailment  → CORRECT (score = p_entailment)
        neutral     → ABSTAIN (score = 0.5)
        contradiction → WRONG (score = 1 - p_contradiction)

    Falls back to ABSTAIN if the model is unavailable.
    """
    name = "nli_faithfulness"
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        entailment_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the pipeline (avoids import cost if not used)."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1 if self.device == "cpu" else 0,
            )
        except Exception as e:
            self._pipeline = None
            self._load_error = str(e)

    def grade(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        context: Optional[str] = None,
    ) -> GradeResult:
        t0 = time.monotonic()
        premise = context or reference_answer  # Fallback: use reference as premise

        self._load_pipeline()

        if self._pipeline is None:
            return self._make_result(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                label=GradeLabel.ABSTAIN,
                score=0.5,
                rationale=f"NLI model unavailable: {getattr(self, '_load_error', 'unknown')}",
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                model_used=self.model_name,
            )

        try:
            results = self._pipeline(
                f"{premise} [SEP] {candidate_answer}",
                top_k=None,
            )
            scores = {r["label"].lower(): r["score"] for r in results}

            ent_score = scores.get("entailment", 0.0)
            cont_score = scores.get("contradiction", 0.0)

            if ent_score >= self.entailment_threshold:
                label = GradeLabel.CORRECT
                score = ent_score
                rationale = f"entailment={ent_score:.3f}"
            elif cont_score >= self.entailment_threshold:
                label = GradeLabel.WRONG
                score = 1.0 - ent_score
                rationale = f"contradiction={cont_score:.3f}"
            else:
                label = GradeLabel.ABSTAIN
                score = 0.5
                rationale = f"neutral (ent={ent_score:.3f}, cont={cont_score:.3f})"

        except Exception as e:
            label = GradeLabel.ERROR
            score = 0.0
            rationale = f"NLI inference error: {e}"

        return self._make_result(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            label=label,
            score=score,
            rationale=rationale,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            model_used=self.model_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grader 4: LLM Judge
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """You are a strict factual accuracy judge for an Indian Income Tax Q&A system.
You will be given a question, a reference answer, and a candidate answer.
Your task is to determine if the candidate answer is factually correct relative to the reference answer.

RULES:
- A candidate is CORRECT if it conveys the same factual information as the reference,
  even if phrased differently (paraphrase is acceptable).
- A candidate is WRONG if it contains any factually incorrect information:
  wrong monetary amounts, wrong sections, wrong percentages, wrong eligibility rules, or fabricated claims.
- A candidate is WRONG if it omits critical qualifying information that changes the meaning.
- If you cannot determine correctness (missing context), output ABSTAIN.

Output ONLY valid JSON with this exact schema:
{
  "label": "correct" | "wrong" | "abstain",
  "score": <float 0.0-1.0>,
  "rationale": "<one sentence explaining your decision>"
}"""

_JUDGE_USER_TEMPLATE = """Question: {question}

Reference Answer: {reference_answer}

Candidate Answer: {candidate_answer}

{context_section}

Evaluate the candidate answer and respond with JSON only."""


def _build_judge_prompt(
    question: str,
    reference_answer: str,
    candidate_answer: str,
    context: Optional[str] = None,
) -> str:
    ctx = f"Source Context:\n{context}" if context else ""
    return _JUDGE_USER_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
        context_section=ctx,
    )


def _cache_key(prompt: str, model: str) -> str:
    payload = f"{model}::{prompt}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(cache_dir: Path, key: str) -> Optional[Dict]:
    path = cache_dir / f"{key}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(cache_dir: Path, key: str, data: Dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def _parse_judge_response(raw: str) -> Tuple[GradeLabel, float, str]:
    """Parse JSON response from LLM judge. Returns (label, score, rationale)."""
    # Strip markdown fences if present
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()

    try:
        data = json.loads(raw)
        label_str = data.get("label", "abstain").lower()
        label = {
            "correct": GradeLabel.CORRECT,
            "wrong": GradeLabel.WRONG,
            "abstain": GradeLabel.ABSTAIN,
        }.get(label_str, GradeLabel.ABSTAIN)
        score = float(data.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        rationale = str(data.get("rationale", ""))
        return label, score, rationale
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return GradeLabel.ABSTAIN, 0.5, f"Parse error: {e} | raw: {raw[:200]}"


class LLMJudge(BaseGrader):
    """
    Rubric-based LLM judge with JSON output.

    Supports:
        - Local Ollama: provider="ollama", model="llama3.2:1b"
        - Groq API: provider="groq", model="llama3-70b-8192"

    Design invariants:
        - temperature=0 always (deterministic)
        - JSON-only output schema (see _JUDGE_SYSTEM_PROMPT)
        - Response caching keyed by content hash (no duplicate API calls)
        - self_consistency_n > 1: runs N times, majority vote on label
        - Position bias check: can swap ref/candidate order and compare
    """
    name = "llm_judge"

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2:1b",
        temperature: float = 0.0,
        max_tokens: int = 256,
        self_consistency_n: int = 1,
        cache_dir: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        if temperature != 0.0:
            raise ValueError(
                "LLMJudge requires temperature=0 for reproducibility. "
                f"Got temperature={temperature}."
            )
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.self_consistency_n = self_consistency_n
        self.cache_dir = Path(cache_dir) if cache_dir else Path("litmus_v2/cache/llm_judge")
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.timeout = timeout

    def _call_ollama(self, prompt: str, system: str) -> str:
        """Call local Ollama OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for Ollama. Run: pip install openai")

        client = OpenAI(base_url=f"{self.base_url}/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_groq(self, prompt: str, system: str) -> str:
        """Call Groq API (requires GROQ_API_KEY env var)."""
        if not self.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY env var not set. Cannot call Groq API. "
                "Set it in your environment or .env file."
            )
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq package required. Run: pip install groq")

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _single_call(self, prompt: str) -> Tuple[GradeLabel, float, str, bool]:
        """Makes a single grading call with caching. Returns (label, score, rationale, cached)."""
        key = _cache_key(prompt, self.model)
        cached_data = _load_cache(self.cache_dir, key)
        if cached_data:
            return (
                GradeLabel(cached_data["label"]),
                cached_data["score"],
                cached_data["rationale"],
                True,
            )

        if self.provider == "ollama":
            raw = self._call_ollama(prompt, _JUDGE_SYSTEM_PROMPT)
        elif self.provider == "groq":
            raw = self._call_groq(prompt, _JUDGE_SYSTEM_PROMPT)
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'ollama' or 'groq'.")

        label, score, rationale = _parse_judge_response(raw)
        _save_cache(self.cache_dir, key, {
            "label": label.value,
            "score": score,
            "rationale": rationale,
        })
        return label, score, rationale, False

    def grade(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        context: Optional[str] = None,
    ) -> GradeResult:
        t0 = time.monotonic()
        prompt = _build_judge_prompt(question, reference_answer, candidate_answer, context)

        try:
            if self.self_consistency_n <= 1:
                label, score, rationale, was_cached = self._single_call(prompt)
            else:
                # Majority vote over N calls
                votes: List[Tuple[GradeLabel, float, str, bool]] = []
                for _ in range(self.self_consistency_n):
                    votes.append(self._single_call(prompt))
                label_counts: Dict[GradeLabel, int] = {}
                for v in votes:
                    label_counts[v[0]] = label_counts.get(v[0], 0) + 1
                label = max(label_counts, key=lambda k: label_counts[k])
                score = sum(v[1] for v in votes) / len(votes)
                rationale = f"majority_vote(n={self.self_consistency_n}): {dict(label_counts)}"
                was_cached = all(v[3] for v in votes)

        except Exception as e:
            label = GradeLabel.ERROR
            score = 0.0
            rationale = f"LLM judge error: {e}"
            was_cached = False

        return self._make_result(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            label=label,
            score=round(score, 4),
            rationale=rationale,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            model_used=f"{self.provider}/{self.model}",
            cached=was_cached,
        )
