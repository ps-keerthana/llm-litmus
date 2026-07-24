"""
Core Metrics Module
Phase 1: Semantic cosine similarity (original).
Phase 3: Multi-signal evaluation — token F1, numeric claim consistency, negation detection.
         These signals together fix the embedding-only weakness where numerically similar
         but factually wrong answers (e.g. '₹1.5L' vs '₹2L') could pass auto-scoring.
"""

import re
import string
from typing import Set, Tuple

import numpy as np
from core.retrieval import embedder


# ── Phase 1: Semantic Similarity ─────────────────────────────────────────

def compute_semantic_similarity(answer: str, ground_truth: str) -> float:
    """
    Cosine similarity between generated answer and ground truth
    using SentenceTransformer embeddings.
    """
    embeddings = embedder.encode([answer, ground_truth])
    vec1, vec2 = embeddings[0], embeddings[1]
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(float(np.dot(vec1, vec2) / (norm1 * norm2)), 3)


# ── Phase 3: Lexical Overlap (Token F1) ──────────────────────────────────

def _tokenize(text: str) -> Set[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score between prediction and ground truth.

    Why this matters: Embedding similarity can be high when two sentences share
    the same vocabulary but different numbers (e.g. ₹1.5L vs ₹2L). Token F1
    measures exact word overlap, catching vocabulary substitutions.

    Returns a float in [0.0, 1.0].
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 3)


# ── Phase 3: Numeric Claim Consistency ───────────────────────────────────

# Matches numbers like: 1.5, 1,50,000, 50000, ₹1.5, 2%, 80C (section numbers excluded)
_NUMBER_RE = re.compile(
    r"(?<![a-zA-Z])"            # not preceded by a letter (excludes 80C, 194J etc.)
    r"(?:₹\s*)?"                # optional rupee sign
    r"\d[\d,\.]*"               # the number itself
    r"(?:\s*(?:lakh|crore|%))?" # optional unit
)


def extract_numbers(text: str) -> Set[str]:
    """
    Extracts all numeric values from text, normalised to remove commas and spaces.

    Examples:
        "₹1,50,000" → {"150000"}
        "1.5 lakh"  → {"1.5lakh"}
        "80C"       → {} (section numbers excluded)
    """
    matches = _NUMBER_RE.findall(text.lower())
    normalised = set()
    for m in matches:
        # Remove spaces and commas for comparison
        n = m.replace(",", "").replace(" ", "").strip()
        if n:
            normalised.add(n)
    return normalised


def numbers_consistent(answer: str, ground_truth: str) -> bool:
    """
    Returns True if all numeric values in ground_truth appear in answer,
    or if ground_truth contains no numbers (nothing to check).

    This catches '₹2 lakh' vs '₹1.5 lakh' mismatches that fool embedding similarity.
    """
    gt_numbers = extract_numbers(ground_truth)
    if not gt_numbers:
        return True  # no numbers to verify — skip this signal

    answer_numbers = extract_numbers(answer)
    # Every number from ground truth must appear somewhere in the answer
    return gt_numbers.issubset(answer_numbers)


# ── Phase 3: Negation / Contradiction Detection ───────────────────────────

_NEGATION_PATTERNS = [
    # Captures "not available", "not allowed", "cannot", "no deduction"
    re.compile(r"\bnot\s+\w+", re.IGNORECASE),
    re.compile(r"\bcannot\b|\bcan't\b|\bcannot\b", re.IGNORECASE),
    re.compile(r"\bno\s+\w+\s+(?:is|are|can)\b", re.IGNORECASE),
    re.compile(r"\bis\s+not\b|\bare\s+not\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\b|\bdoes\s+not\b|\bdid\s+not\b", re.IGNORECASE),
]

_AFFIRMATIVE_PHRASES = [
    "is available", "can be claimed", "is allowed", "is eligible",
    "is applicable", "you can", "are entitled",
]
_NEGATIVE_PHRASES = [
    "is not available", "cannot be claimed", "is not allowed", "is not eligible",
    "is not applicable", "you cannot", "are not entitled",
]


def is_contradicting(answer: str, ground_truth: str) -> bool:
    """
    Returns True if the answer's polarity on key claims appears to contradict
    the ground truth.

    Strategy: check if the ground truth contains a strong affirmative or negative
    phrase and the answer contains its opposite.

    This is a lightweight heuristic — not perfect, but catches the common
    "the answer says YES but ground truth says NO" failure mode.
    """
    ans_lower = answer.lower()
    gt_lower = ground_truth.lower()

    for pos_phrase, neg_phrase in zip(_AFFIRMATIVE_PHRASES, _NEGATIVE_PHRASES):
        gt_positive = pos_phrase in gt_lower
        gt_negative = neg_phrase in gt_lower

        if gt_positive and neg_phrase in ans_lower:
            return True  # GT says "is available", answer says "is not available"
        if gt_negative and pos_phrase in ans_lower and neg_phrase not in ans_lower:
            return True  # GT says "cannot be claimed", answer says "can be claimed"

    return False


# ── Phase 3: Composite Multi-Signal Auto-Pass Gate ───────────────────────

def multi_signal_auto_pass(
    answer: str,
    ground_truth: str,
    semantic_sim: float,
    semantic_threshold: float = 0.85,
    token_f1_min: float = 0.60,
    check_numbers: bool = True,
    check_negation: bool = True,
) -> Tuple[bool, dict]:
    """
    Returns (should_auto_pass, signal_details).

    Auto-pass only when ALL enabled signals agree:
      1. Semantic similarity >= threshold
      2. Token F1 >= token_f1_min
      3. Numeric values in ground truth appear in answer (if check_numbers)
      4. Answer does not contradict ground truth polarity (if check_negation)

    Returns a dict of individual signal results for transparency in run records.
    """
    token_f1 = compute_token_f1(answer, ground_truth)
    num_ok = numbers_consistent(answer, ground_truth) if check_numbers else True
    neg_ok = (not is_contradicting(answer, ground_truth)) if check_negation else True

    sem_ok = semantic_sim >= semantic_threshold
    lex_ok = token_f1 >= token_f1_min

    auto_pass = sem_ok and lex_ok and num_ok and neg_ok

    signals = {
        "semantic_ok": sem_ok,
        "semantic_sim": semantic_sim,
        "token_f1": token_f1,
        "lexical_ok": lex_ok,
        "numbers_ok": num_ok,
        "negation_ok": neg_ok,
        "auto_pass": auto_pass,
    }
    return auto_pass, signals
