"""
Phase 3: Perturbation Suite for litmus_v2.

Programmatically generates wrong answers with known labels from verified
reference answers. Each perturbation type has a known ground-truth label
(reference_correct=False, perturbation_type=<type>) so graders can be
evaluated for their catch rate.

Perturbation types implemented:
    1. number_swap      — Replace monetary/numeric values with wrong ones
    2. negation_flip    — Insert "not" to flip polarity
    3. entity_swap      — Swap a named entity with a plausible wrong entity
    4. unsupported_claim_insertion — Inject a fabricated claim not in the chunk
    5. truncation       — Cut the answer before the critical information
    6. off_topic        — Replace answer with plausible-sounding but unrelated text
    7. paraphrase       — Paraphrase without changing meaning (label: correct=True,
                          used to test false-positive rate of graders)

All perturbations are:
- Seeded (reproducible)
- Config-driven (rules in perturbation_rules.yaml)
- Labeled with explicit ground truth
- Saved as PerturbedItem objects (never written to results/)
"""

import random
import re
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from litmus_v2.data.schema import CandidateQA, QuestionType


class PerturbationType(str, Enum):
    NUMBER_SWAP = "number_swap"
    NEGATION_FLIP = "negation_flip"
    ENTITY_SWAP = "entity_swap"
    UNSUPPORTED_CLAIM = "unsupported_claim_insertion"
    TRUNCATION = "truncation"
    OFF_TOPIC = "off_topic"
    PARAPHRASE = "paraphrase"


class PerturbedItem(BaseModel):
    """A single perturbed answer with its known ground-truth label."""
    id: str                          # e.g. LITMUS_QA_001_numswap
    source_id: str                   # Original CandidateQA id
    question: str
    reference_answer: str            # Original correct answer
    perturbed_answer: str            # The modified (wrong or paraphrased) answer
    perturbation_type: PerturbationType
    expected_correct: bool           # Ground truth: paraphrase=True, all others=False
    source_chunk_text: Optional[str] = None
    seed: int = 42
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Indian tax domain entity tables
# ─────────────────────────────────────────────────────────────────────────────

# Wrong monetary amounts to substitute (when original is found in text)
_WRONG_AMOUNTS = {
    "1,50,000": ["1,00,000", "2,00,000", "75,000", "1,25,000"],
    "50,000": ["25,000", "75,000", "1,00,000", "30,000"],
    "5,000": ["2,500", "10,000", "7,500", "3,000"],
    "30,000": ["20,000", "50,000", "40,000", "15,000"],
    "25,000": ["20,000", "30,000", "15,000", "40,000"],
    "75,000": ["50,000", "1,00,000", "60,000", "80,000"],
    "2,00,000": ["1,50,000", "2,50,000", "1,75,000", "1,00,000"],
}

# Wrong numeric thresholds / periods
_WRONG_NUMBERS = {
    "3 years": ["5 years", "1 year", "2 years", "7 years"],
    "24 months": ["12 months", "36 months", "18 months", "48 months"],
    "40%": ["50%", "30%", "45%", "60%"],
    "50%": ["40%", "60%", "45%", "70%"],
    "60 years": ["55 years", "65 years", "58 years", "62 years"],
}

# Entity swaps: section/provision replacements
_ENTITY_SWAPS = {
    "section 80c": "Section 80D",
    "section 80d": "Section 80C",
    "section 80ccd(1b)": "Section 80CCD(1)",
    "section 192": "Section 194J",
    "section 194j": "Section 192",
    "elss": "PPF",
    "ppf": "ELSS",
    "nps": "ELSS",
    "tier-1": "Tier-2",
    "long-term": "short-term",
    "short-term": "long-term",
    "non-metro": "metro",
    "metro": "non-metro",
    "senior citizen": "non-senior citizen",
    "two children": "three children",
    "employer": "employee",
}

# Unsupported claim injections (fabricated but plausible-sounding)
_UNSUPPORTED_CLAIMS = [
    " Additionally, a surcharge of 10% applies for incomes above ₹50 lakh.",
    " This benefit is available only if the taxpayer files ITR-2.",
    " The deduction is capped at ₹2,00,000 under the Finance Act 2023 amendment.",
    " Payments must be made via NEFT/RTGS only, as per CBDT circular 21/2022.",
    " The benefit is restricted to government employees under Rule 3A.",
    " However, the deduction is not available to NRIs.",
]

# Off-topic replacements
_OFF_TOPIC_ANSWERS = [
    "Under the GST framework, composite supply is taxed at the rate of the principal supply.",
    "The applicant must file Form 26AS to reconcile TDS credits before the due date.",
    "Companies are required to file their annual returns with the MCA within 60 days of the AGM.",
    "The Securities and Exchange Board of India regulates mutual fund NAV disclosure norms.",
    "FEMA regulations require prior RBI approval for outward remittances above certain thresholds.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Individual perturbation functions
# ─────────────────────────────────────────────────────────────────────────────

def _number_swap(answer: str, rng: random.Random) -> Optional[str]:
    """Replace a monetary/numeric value with a wrong one."""
    # Try amounts first
    for original, alternatives in _WRONG_AMOUNTS.items():
        if original in answer:
            replacement = rng.choice(alternatives)
            return answer.replace(original, replacement, 1)
    # Try numeric periods
    for original, alternatives in _WRONG_NUMBERS.items():
        if original.lower() in answer.lower():
            replacement = rng.choice(alternatives)
            return re.sub(re.escape(original), replacement, answer, count=1, flags=re.IGNORECASE)
    return None


def _negation_flip(answer: str, rng: random.Random) -> Optional[str]:
    """Flip the polarity of the answer by inserting/removing 'not'."""
    # Remove existing negation (double-flip)
    for neg in ("cannot be claimed", "not available", "not eligible", "is not"):
        if neg in answer.lower():
            return re.sub(re.escape(neg), neg.replace("not ", "").replace("cannot", "can"), answer, count=1, flags=re.IGNORECASE)

    # Insert negation after common verbs
    for pattern, replacement in [
        (r"\bcan be claimed\b", "cannot be claimed"),
        (r"\bis available\b", "is not available"),
        (r"\bare eligible\b", "are not eligible"),
        (r"\bcan be set off\b", "cannot be set off"),
        (r"\bmust be held\b", "need not be held"),
        (r"\bcan claim\b", "cannot claim"),
        (r"\bis allowed\b", "is not allowed"),
        (r"\bqualify\b", "do not qualify"),
    ]:
        if re.search(pattern, answer, re.IGNORECASE):
            return re.sub(pattern, replacement, answer, count=1, flags=re.IGNORECASE)

    # Fallback: prepend "Not correct. " to the answer
    return f"Not correct. {answer}"


def _entity_swap(answer: str, rng: random.Random) -> Optional[str]:
    """Swap a named entity (section, product) with a wrong one."""
    # Try domain swaps first
    for entity, wrong_entity in _ENTITY_SWAPS.items():
        if entity.lower() in answer.lower():
            return re.sub(re.escape(entity), wrong_entity, answer, count=1, flags=re.IGNORECASE)

    return None


def _unsupported_claim(answer: str, rng: random.Random) -> str:
    """Inject a fabricated claim not present in the source chunk."""
    injection = rng.choice(_UNSUPPORTED_CLAIMS)
    # Append after the first sentence
    sentences = re.split(r'(?<=[.!?])\s+', answer, maxsplit=1)
    if len(sentences) > 1:
        return sentences[0] + injection + " " + sentences[1]
    return answer.rstrip(".") + "." + injection


def _truncation(answer: str, rng: random.Random) -> Optional[str]:
    """Cut the answer at a sentence boundary, removing the critical tail."""
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) >= 2:
        # Keep only the first sentence (which usually lacks the key number/condition)
        return sentences[0]
    # Fallback: keep only first half
    mid = max(1, len(answer) // 2)
    return answer[:mid].rstrip()


def _off_topic(answer: str, rng: random.Random) -> str:
    """Replace entire answer with an off-topic but plausible-sounding statement."""
    return rng.choice(_OFF_TOPIC_ANSWERS)


def _paraphrase(answer: str, rng: random.Random) -> str:
    """
    Surface-level paraphrase: reorder / substitute synonyms.
    Keeps semantics intact — ground truth remains correct=True.
    """
    replacements = [
        ("per financial year", "in a financial year"),
        ("is ₹", "amounts to ₹"),
        ("up to", "not exceeding"),
        ("at least", "a minimum of"),
        ("is mandatory", "is compulsory"),
        ("can be claimed", "is deductible"),
        ("are eligible", "qualify"),
        ("must be held", "should be held"),
        ("over and above", "in addition to"),
        ("The maximum", "A maximum"),
    ]
    result = answer
    changed = False
    rng.shuffle(replacements)
    for original, repl in replacements:
        if original.lower() in result.lower():
            result = re.sub(re.escape(original), repl, result, count=1, flags=re.IGNORECASE)
            changed = True
            break
    if not changed:
        # Minor word order / phrasing change
        result = result.replace("The ", "The overall ", 1) if "The " in result else result + " (paraphrased)"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def perturb_item(
    qa: CandidateQA,
    perturbation_type: PerturbationType,
    chunk_text: Optional[str] = None,
    seed: int = 42,
) -> Optional[PerturbedItem]:
    """
    Generates a single perturbed answer for a CandidateQA item.
    Returns None if the perturbation is not applicable (e.g., no number found for number_swap).
    """
    rng = random.Random(seed)
    answer = qa.reference_answer
    perturbed: Optional[str] = None

    if perturbation_type == PerturbationType.NUMBER_SWAP:
        perturbed = _number_swap(answer, rng)
    elif perturbation_type == PerturbationType.NEGATION_FLIP:
        perturbed = _negation_flip(answer, rng)
    elif perturbation_type == PerturbationType.ENTITY_SWAP:
        perturbed = _entity_swap(answer, rng)
    elif perturbation_type == PerturbationType.UNSUPPORTED_CLAIM:
        perturbed = _unsupported_claim(answer, rng)
    elif perturbation_type == PerturbationType.TRUNCATION:
        perturbed = _truncation(answer, rng)
    elif perturbation_type == PerturbationType.OFF_TOPIC:
        perturbed = _off_topic(answer, rng)
    elif perturbation_type == PerturbationType.PARAPHRASE:
        perturbed = _paraphrase(answer, rng)

    if perturbed is None:
        return None

    # Paraphrase is semantically correct; all others are wrong
    expected_correct = (perturbation_type == PerturbationType.PARAPHRASE)

    item_id = f"{qa.id}_{perturbation_type.value.replace('_', '')[:8]}"

    return PerturbedItem(
        id=item_id,
        source_id=qa.id,
        question=qa.question,
        reference_answer=qa.reference_answer,
        perturbed_answer=perturbed,
        perturbation_type=perturbation_type,
        expected_correct=expected_correct,
        source_chunk_text=chunk_text,
        seed=seed,
        metadata={
            "question_type": qa.question_type.value,
            "doc_group": qa.doc_group,
        },
    )


def generate_perturbation_suite(
    candidates: List[CandidateQA],
    chunk_lookup: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    types: Optional[List[PerturbationType]] = None,
) -> List[PerturbedItem]:
    """
    Generates the full perturbation suite from a list of candidate Q/A items.

    For each item, attempts all 7 perturbation types (or the specified subset).
    Items where the perturbation is not applicable are skipped.

    Args:
        candidates: Verified CandidateQA items (from human-labeled gold set ideally).
        chunk_lookup: Optional dict mapping chunk_id -> Chunk for source text.
        seed: Random seed for reproducibility.
        types: Which perturbation types to apply (default: all 7).

    Returns:
        List of PerturbedItem objects sorted by source_id.
    """
    if types is None:
        types = list(PerturbationType)

    chunk_lookup = chunk_lookup or {}
    results: List[PerturbedItem] = []

    for idx, qa in enumerate(candidates):
        chunk_text = None
        if qa.primary_chunk_id and qa.primary_chunk_id in chunk_lookup:
            chunk = chunk_lookup[qa.primary_chunk_id]
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)

        for ptype in types:
            item_seed = seed + idx * 100 + list(PerturbationType).index(ptype)
            item = perturb_item(qa, ptype, chunk_text=chunk_text, seed=item_seed)
            if item is not None:
                results.append(item)

    results.sort(key=lambda x: x.source_id)
    return results


def perturbation_suite_summary(items: List[PerturbedItem]) -> Dict[str, Any]:
    """Returns a summary dict of the perturbation suite for logging."""
    from collections import Counter
    type_counts = Counter(item.perturbation_type.value for item in items)
    correct_count = sum(1 for item in items if item.expected_correct)
    wrong_count = sum(1 for item in items if not item.expected_correct)
    return {
        "total": len(items),
        "by_type": dict(type_counts),
        "expected_correct": correct_count,     # paraphrases
        "expected_wrong": wrong_count,          # all real perturbations
    }
