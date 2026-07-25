"""
tests/test_metrics.py
=====================
Unit tests for core/metrics.py

Covers:
  - compute_token_f1: perfect match, zero overlap, partial, empty strings
  - extract_numbers: Indian currency formats, percentages, section exclusions
  - numbers_consistent: missing numbers, subset match, no numbers in GT
  - is_contradicting: polarity flip detection, no-contradiction passthrough
  - multi_signal_auto_pass: all-pass, semantic-fail, f1-fail, number-fail

These tests are pure logic tests — no LLM calls, no network, no embedder.
The embedder import in core.retrieval is lazy-loaded inside functions, so
importing core.metrics directly does NOT trigger sentence-transformer loading.
"""

import os
import sys
import unittest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.metrics import (
    compute_token_f1,
    extract_numbers,
    numbers_consistent,
    is_contradicting,
    multi_signal_auto_pass,
)


# ─── Token F1 Tests ───────────────────────────────────────────────────────────

class TestComputeTokenF1(unittest.TestCase):
    """Tests for compute_token_f1(prediction, ground_truth)."""

    def test_perfect_match(self):
        """Identical strings should return F1 = 1.0."""
        f1 = compute_token_f1("the limit is one lakh fifty thousand", "the limit is one lakh fifty thousand")
        self.assertAlmostEqual(f1, 1.0, places=2)

    def test_zero_overlap(self):
        """Completely different vocabulary should return F1 = 0.0."""
        f1 = compute_token_f1("apples oranges bananas", "clouds thunder lightning")
        self.assertAlmostEqual(f1, 0.0, places=2)

    def test_partial_overlap(self):
        """Partial overlap should return F1 between 0 and 1."""
        f1 = compute_token_f1("the deduction limit is one lakh", "the limit is one lakh fifty thousand rupees")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_both_empty(self):
        """Both empty strings should return 1.0 (trivially identical)."""
        f1 = compute_token_f1("", "")
        self.assertAlmostEqual(f1, 1.0, places=2)

    def test_one_empty(self):
        """One empty string should return 0.0."""
        self.assertAlmostEqual(compute_token_f1("some content", ""), 0.0, places=2)
        self.assertAlmostEqual(compute_token_f1("", "some content"), 0.0, places=2)

    def test_case_insensitive(self):
        """Token F1 should be case-insensitive."""
        f1 = compute_token_f1("Section 80C Deduction Limit", "section 80c deduction limit")
        self.assertAlmostEqual(f1, 1.0, places=2)

    def test_punctuation_stripped(self):
        """Punctuation should be stripped before comparison."""
        f1 = compute_token_f1("limit is 1.5 lakh.", "limit is 1.5 lakh")
        self.assertAlmostEqual(f1, 1.0, places=2)

    def test_result_bounded(self):
        """F1 must always be in [0.0, 1.0]."""
        for pred, gt in [
            ("a b c", "d e f"),
            ("x y z", "x y z"),
            ("one two", "one two three four"),
        ]:
            f1 = compute_token_f1(pred, gt)
            self.assertGreaterEqual(f1, 0.0)
            self.assertLessEqual(f1, 1.0)


# ─── Extract Numbers Tests ────────────────────────────────────────────────────

class TestExtractNumbers(unittest.TestCase):
    """Tests for extract_numbers(text) — Indian currency, percent, plain integers."""

    def test_plain_integer(self):
        nums = extract_numbers("The penalty is 50000 rupees.")
        self.assertIn("50000", nums)

    def test_rupee_symbol(self):
        """Rupee sign should be stripped from extracted values."""
        nums = extract_numbers("The deduction limit is ₹1,50,000.")
        # Normalised: commas removed → "150000"
        self.assertTrue(any("150000" in n for n in nums))

    def test_lakh_suffix(self):
        """Numbers with lakh suffix should be extracted as a unit."""
        nums = extract_numbers("Deduction of ₹1.5 lakh is allowed.")
        self.assertTrue(any("lakh" in n for n in nums))

    def test_percentage(self):
        """Percentages like 10% should be captured."""
        nums = extract_numbers("LTCG is taxed at 10%.")
        self.assertTrue(any("10" in n for n in nums))

    def test_section_numbers_excluded(self):
        """Section identifiers like 80C or 194J must NOT be extracted as numbers."""
        nums = extract_numbers("Under Section 80C, the limit is ₹1.5 lakh.")
        # "80" must NOT appear as a standalone numeric match
        self.assertFalse(any(n == "80" for n in nums))

    def test_empty_string(self):
        """Empty input should return an empty set."""
        self.assertEqual(extract_numbers(""), set())

    def test_no_numbers(self):
        """Text with no numeric content should return an empty set."""
        self.assertEqual(extract_numbers("The sky is blue and the grass is green."), set())


# ─── Numbers Consistent Tests ─────────────────────────────────────────────────

class TestNumbersConsistent(unittest.TestCase):
    """Tests for numbers_consistent(answer, ground_truth)."""

    def test_exact_match(self):
        """Same numbers in answer and ground truth → True."""
        self.assertTrue(numbers_consistent(
            "The limit is ₹1,50,000 per year.",
            "The deduction cap is ₹1,50,000 annually."
        ))

    def test_missing_number_in_answer(self):
        """Ground truth has 1.5 lakh but answer says 2 lakh → False."""
        self.assertFalse(numbers_consistent(
            "The deduction limit is ₹2 lakh.",
            "The deduction limit is ₹1.5 lakh."
        ))

    def test_no_numbers_in_ground_truth(self):
        """If ground truth has no numbers, any answer passes (nothing to verify)."""
        self.assertTrue(numbers_consistent(
            "Any answer text here.",
            "The section applies to eligible taxpayers only."
        ))

    def test_superset_in_answer(self):
        """Answer with extra numbers beyond GT is still consistent."""
        self.assertTrue(numbers_consistent(
            "Section 80C limit is ₹1.5 lakh; 80D limit is ₹25,000.",
            "The 80C deduction limit is ₹1.5 lakh."
        ))

    def test_partial_numbers_missing(self):
        """If even one required number from GT is missing from answer → False."""
        self.assertFalse(numbers_consistent(
            "The basic exemption is ₹2.5 lakh.",
            "Basic exemption is ₹2.5 lakh and 80C limit is ₹1.5 lakh."
        ))


# ─── Is Contradicting Tests ───────────────────────────────────────────────────

class TestIsContradicting(unittest.TestCase):
    """Tests for is_contradicting(answer, ground_truth)."""

    def test_no_contradiction(self):
        """Both agreeing on a positive claim → no contradiction."""
        self.assertFalse(is_contradicting(
            "HRA exemption can be claimed under Section 10(13A).",
            "HRA exemption is available under Section 10(13A)."
        ))

    def test_gt_positive_answer_negative(self):
        """GT says 'can be claimed'; answer says 'cannot be claimed' → contradiction."""
        self.assertTrue(is_contradicting(
            "Home loan interest deduction cannot be claimed under Section 24b.",
            "Home loan interest deduction can be claimed under Section 24b."
        ))

    def test_gt_negative_answer_positive(self):
        """GT says 'not eligible'; answer says 'eligible' → contradiction."""
        self.assertTrue(is_contradicting(
            "NPS employer contribution is eligible under 80CCD(2).",
            "NPS employer contribution is not eligible under 80CCD(2)."
        ))

    def test_no_polarity_phrases(self):
        """Neither text contains polarity phrases → no contradiction detected."""
        self.assertFalse(is_contradicting(
            "The tax rate is 10%.",
            "LTCG tax rate under Section 112A is 10%."
        ))


# ─── Multi-Signal Auto-Pass Tests ─────────────────────────────────────────────

class TestMultiSignalAutoPass(unittest.TestCase):
    """Tests for multi_signal_auto_pass(...) composite gate."""

    # High-quality answer that should pass all signals
    GOOD_ANSWER = "The aggregate deduction limit under Section 80C is one lakh fifty thousand rupees per financial year."
    GOOD_GT = "The aggregate deduction limit under Section 80C is one lakh fifty thousand rupees per year."

    def test_all_signals_pass(self):
        """Perfect answer with high semantic sim should auto-pass."""
        passed, signals = multi_signal_auto_pass(
            answer=self.GOOD_ANSWER,
            ground_truth=self.GOOD_GT,
            semantic_sim=0.92,
            semantic_threshold=0.85,
            token_f1_min=0.60,
        )
        self.assertTrue(passed, f"Expected auto-pass but got signals: {signals}")
        self.assertTrue(signals["semantic_ok"])
        self.assertTrue(signals["lexical_ok"])

    def test_semantic_below_threshold(self):
        """Low semantic similarity should prevent auto-pass even with good F1."""
        passed, signals = multi_signal_auto_pass(
            answer=self.GOOD_ANSWER,
            ground_truth=self.GOOD_GT,
            semantic_sim=0.70,   # Below default threshold of 0.85
            semantic_threshold=0.85,
            token_f1_min=0.60,
        )
        self.assertFalse(passed)
        self.assertFalse(signals["semantic_ok"])

    def test_f1_below_threshold(self):
        """Good semantic sim but low F1 should prevent auto-pass."""
        passed, signals = multi_signal_auto_pass(
            answer="Taxpayers can benefit from various provisions.",  # Low F1 vs GT
            ground_truth=self.GOOD_GT,
            semantic_sim=0.88,
            semantic_threshold=0.85,
            token_f1_min=0.60,
        )
        self.assertFalse(passed)
        self.assertFalse(signals["lexical_ok"])

    def test_signals_dict_has_required_keys(self):
        """Returned signals dict must always contain the documented keys."""
        _, signals = multi_signal_auto_pass(
            answer=self.GOOD_ANSWER,
            ground_truth=self.GOOD_GT,
            semantic_sim=0.90,
        )
        required_keys = {"semantic_ok", "semantic_sim", "token_f1", "lexical_ok",
                         "numbers_ok", "negation_ok", "auto_pass"}
        self.assertTrue(required_keys.issubset(signals.keys()),
                        f"Missing keys: {required_keys - signals.keys()}")

    def test_auto_pass_key_matches_return_value(self):
        """signals['auto_pass'] must match the boolean return value."""
        passed, signals = multi_signal_auto_pass(
            answer=self.GOOD_ANSWER,
            ground_truth=self.GOOD_GT,
            semantic_sim=0.92,
        )
        self.assertEqual(passed, signals["auto_pass"])

    def test_number_check_disabled(self):
        """With check_numbers=False, number mismatch should not block auto-pass."""
        passed, signals = multi_signal_auto_pass(
            answer="The limit is two lakh rupees.",   # Wrong number but check disabled
            ground_truth="The limit is one lakh fifty thousand rupees.",
            semantic_sim=0.90,
            semantic_threshold=0.85,
            token_f1_min=0.50,
            check_numbers=False,
        )
        # numbers_ok must be True because the check was disabled
        self.assertTrue(signals["numbers_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
