"""
Core Metrics Module
Phase 1: Semantic cosine similarity (original).
Phase 3: Multi-signal evaluation — token F1, numeric claim consistency, negation detection.
         These signals together fix the embedding-only weakness where numerically similar
         but factually wrong answers (e.g. '₹1.5L' vs '₹2L') could pass auto-scoring.

Changes vs original:
  - Numeric extraction now normalises lakh/crore to integers and handles Rs./Rs/INR prefixes.
  - Dead _NEGATION_PATTERNS list removed (was defined but never called anywhere).
  - Negation detection extended with ineligible/eligible, exempt/not exempt, nil synonyms.
  - compute_semantic_similarity loads the embedder lazily so tests start fast.
"""

import re
import string
from typing import Set, Tuple, Optional

import numpy as np


# ── Phase 1: Semantic Similarity ─────────────────────────────────────────────

# Lazy-loaded embedder: avoids importing sentence-transformers (and downloading the
# model) at import time. This makes test collection fast and avoids the ChromaDB
# import chain in modules that only need token F1 / numeric checks.
_embedder: Optional[object] = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from core.retrieval import embedder  # noqa: PLC0415
        _embedder = embedder
    return _embedder


def compute_semantic_similarity(answer: str, ground_truth: str) -> float:
    """
    Cosine similarity between generated answer and ground truth
    using SentenceTransformer embeddings (lazy-loaded).
    """
    emb = _get_embedder()
    embeddings = emb.encode([answer, ground_truth])
    vec1, vec2 = embeddings[0], embeddings[1]
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(float(np.dot(vec1, vec2) / (norm1 * norm2)), 3)


# ── Phase 3: Lexical Overlap (Token F1) ──────────────────────────────────────

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


# ── Phase 3: Numeric Claim Consistency ───────────────────────────────────────

# Currency prefix: ₹, Rs., Rs, INR (case-insensitive)
_CURRENCY_PREFIX_RE = re.compile(r"(?:₹|rs\.?\s*|inr\s*)", re.IGNORECASE)

# Lakh / crore expressions: optional currency prefix + number + unit
# Captures (value_str, unit) groups.
_LAKH_CRORE_RE = re.compile(
    r"(?:₹|rs\.?\s*|inr\s*)?"   # optional currency prefix
    r"(\d[\d,\.]*)"              # numeric value (group 1)
    r"\s*(lakh|crore)\b",        # unit (group 2)
    re.IGNORECASE,
)

# Bare number: integer or decimal in Indian comma format (e.g. 1,50,000)
# Not preceded by a letter (excludes 80C, 194J)
# Not followed by a word char (excludes section codes like 24b)
_BARE_NUMBER_RE = re.compile(
    r"(?<![a-zA-Z])"    # not preceded by a letter
    r"\d[\d,\.]*"       # digits, commas, dots
    r"(?!\w)"           # not followed by a word char
)


def _lakh_crore_to_int(value_str: str, unit: str) -> str:
    """Converts '1.5 lakh' -> '150000', '2 crore' -> '20000000'."""
    try:
        v = float(value_str.replace(",", ""))
        if unit.lower() == "lakh":
            return str(int(round(v * 100_000)))
        if unit.lower() == "crore":
            return str(int(round(v * 10_000_000)))
    except (ValueError, AttributeError):
        pass
    return f"{value_str}{unit}".lower()


def extract_numbers(text: str) -> Set[str]:
    """
    Extracts all numeric values from text, normalised for comparison.

    Handles:
      - ₹1,50,000        → {"150000"}
      - Rs. 1,50,000      → {"150000"}
      - INR 1.5 lakh      → {"150000"}
      - 1.5 lakh          → {"150000"}
      - 2 crore           → {"20000000"}
      - 10%               → {"10%"}
      - 50000             → {"50000"}
      - Section 80C       → {} (section numbers excluded)
      - 194J              → {} (excluded)
    """
    results: Set[str] = set()
    lower = text.lower()

    # Priority 1: lakh / crore expressions → normalise to integer
    for m in _LAKH_CRORE_RE.finditer(lower):
        value_part = m.group(1).strip()
        unit = m.group(2).strip()
        results.add(_lakh_crore_to_int(value_part, unit))

    # Priority 2: bare numbers not already captured as lakh/crore
    cleaned = _LAKH_CRORE_RE.sub(" __REMOVED__ ", lower)
    cleaned = _CURRENCY_PREFIX_RE.sub("", cleaned)

    for m in _BARE_NUMBER_RE.finditer(cleaned):
        raw = m.group(0).strip()
        start = m.start()
        # Skip if immediately preceded by a letter (e.g. inside "194J" after sub)
        if start > 0 and cleaned[start - 1].isalpha():
            continue
        n = raw.replace(",", "").strip()
        # Attach % if immediately followed by it
        end = m.end()
        if end < len(cleaned) and cleaned[end] == "%":
            n += "%"
        if n:
            results.add(n)

    return results


def numbers_consistent(answer: str, ground_truth: str) -> bool:
    """
    Returns True if all numeric values in ground_truth appear in answer,
    or if ground_truth contains no numbers (nothing to check).

    Normalises lakh/crore to integers so:
      "₹1.5 lakh" == "₹1,50,000" == "150000"

    This catches '₹2 lakh' vs '₹1.5 lakh' mismatches that fool embedding similarity.
    """
    gt_numbers = extract_numbers(ground_truth)
    if not gt_numbers:
        return True  # no numbers to verify — skip this signal

    answer_numbers = extract_numbers(answer)
    return gt_numbers.issubset(answer_numbers)


# ── Phase 3: Negation / Contradiction Detection ───────────────────────────────
# Note: _NEGATION_PATTERNS was removed — it was defined but never called.
# The active detection uses the phrase-pair and synonym-pair mechanisms below.

_AFFIRMATIVE_PHRASES = [
    "is available",
    "can be claimed",
    "is allowed",
    "is eligible",
    "is applicable",
    "you can",
    "are entitled",
    "is exempt",
    "is deductible",
]
_NEGATIVE_PHRASES = [
    "is not available",
    "cannot be claimed",
    "is not allowed",
    "is not eligible",
    "is not applicable",
    "you cannot",
    "are not entitled",
    "is not exempt",
    "is not deductible",
]

# Synonym pairs: (positive_surface, negative_surface) for common phrasing variations.
# Each pair catches a specific "yes/no" mismatch that the phrase list doesn't cover.
_SYNONYM_PAIRS: list = [
    ("eligible", "ineligible"),
    ("exempt", "not exempt"),
    ("deductible", "not deductible"),
    ("deduction is allowed", "no deduction"),
    ("deduction is available", "deduction is nil"),
    ("deduction is available", "zero deduction"),
]


def is_contradicting(answer: str, ground_truth: str) -> bool:
    """
    Returns True if the answer's polarity on key claims appears to contradict
    the ground truth.

    Checks two mechanisms:
    1. Phrase-pair polarity flip (affirmative vs. negative phrases).
    2. Synonym-pair detection for common Indian tax phrasing
       (ineligible, nil, zero deduction, etc.).

    This is a lightweight heuristic — not perfect, but catches the most common
    "the answer says YES but ground truth says NO" failure mode.
    """
    ans_lower = answer.lower()
    gt_lower = ground_truth.lower()

    # Mechanism 1: phrase-pair flip
    for pos_phrase, neg_phrase in zip(_AFFIRMATIVE_PHRASES, _NEGATIVE_PHRASES):
        gt_positive = pos_phrase in gt_lower
        gt_negative = neg_phrase in gt_lower

        if gt_positive and neg_phrase in ans_lower:
            return True
        if gt_negative and pos_phrase in ans_lower and neg_phrase not in ans_lower:
            return True

    # Mechanism 2: synonym pairs
    for pos_form, neg_form in _SYNONYM_PAIRS:
        gt_pos = pos_form in gt_lower
        gt_neg = neg_form in gt_lower

        if gt_pos and neg_form in ans_lower and pos_form not in ans_lower:
            return True
        if gt_neg and pos_form in ans_lower and neg_form not in ans_lower:
            return True

    return False


# ── Phase 3: Composite Multi-Signal Auto-Pass Gate ────────────────────────────

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
