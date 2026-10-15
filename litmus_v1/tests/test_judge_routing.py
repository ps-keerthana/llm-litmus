"""
tests/test_judge_routing.py
============================
Unit tests for the oracle routing logic in core/judge.py

Tests the evaluate_with_oracle_routing() decision tree without making
any real LLM calls. The function is tested at the routing boundaries:

  1. --no-judge path: faithfulness/hallucination are "Not Evaluated"
  2. is_refusal path: all scores auto-pass at 1.0
  3. auto-fail path: semantic_sim <= 0.25 → faithfulness=0.0, hallucination=1.0
  4. auto-pass path: all multi-signal checks agree at high similarity

The LLM judge calls (Case 5) are NOT tested here as they require
a live API key. Integration tests for the judge belong in a
separate test_judge_integration.py (not committed).
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch the embedder before importing anything from core.*
# This prevents sentence-transformers from downloading models during testing.
_mock_embedder = MagicMock()
_mock_embedder.encode.return_value = [[0.1] * 384, [0.1] * 384]  # fixed identical vectors

import core.retrieval as _retrieval_mod  # noqa: E402
_retrieval_mod.embedder = _mock_embedder  # inject mock before judge imports it

from core.judge import evaluate_with_oracle_routing  # noqa: E402


class TestNoJudgeRouting(unittest.TestCase):
    """When no_judge=True, all metrics must return without calling the LLM."""

    def test_faithfulness_is_not_evaluated(self):
        result, p_tokens, c_tokens, judge_called = evaluate_with_oracle_routing(
            question="What is the 80C limit?",
            answer="The limit is ₹1.5 lakh.",
            ground_truth="The deduction limit under Section 80C is ₹1.5 lakh.",
            context_chunks=["Section 80C limit is ₹1.5 lakh per year."],
            semantic_sim=0.88,
            no_judge=True,
        )
        self.assertEqual(result["faithfulness"], "Not Evaluated",
                         "Faithfulness must be 'Not Evaluated' in --no-judge mode.")

    def test_hallucination_is_not_evaluated(self):
        result, _, _, _ = evaluate_with_oracle_routing(
            question="What is the 80C limit?",
            answer="The limit is ₹1.5 lakh.",
            ground_truth="The deduction limit under Section 80C is ₹1.5 lakh.",
            context_chunks=["Section 80C limit is ₹1.5 lakh per year."],
            semantic_sim=0.88,
            no_judge=True,
        )
        self.assertEqual(result["hallucination"], "Not Evaluated",
                         "Hallucination must be 'Not Evaluated' in --no-judge mode.")

    def test_completeness_is_not_evaluated(self):
        result, _, _, _ = evaluate_with_oracle_routing(
            question="What is the 80C limit?",
            answer="The limit is ₹1.5 lakh.",
            ground_truth="The deduction limit under Section 80C is ₹1.5 lakh.",
            context_chunks=[],
            semantic_sim=0.80,
            no_judge=True,
        )
        self.assertEqual(result["completeness"], "Not Evaluated")

    def test_no_judge_does_not_call_llm(self):
        """No network calls should be made when no_judge=True."""
        with patch("core.judge.llm_judge_evaluate") as mock_judge, \
             patch("core.judge.ensemble_judge_evaluate") as mock_ensemble:
            evaluate_with_oracle_routing(
                question="q", answer="a", ground_truth="gt",
                context_chunks=[], semantic_sim=0.70, no_judge=True,
            )
            mock_judge.assert_not_called()
            mock_ensemble.assert_not_called()

    def test_no_judge_returns_proxy_correctness(self):
        """Correctness in no-judge mode must be a float proxy, not 'Not Evaluated'."""
        result, _, _, _ = evaluate_with_oracle_routing(
            question="q", answer="a", ground_truth="gt",
            context_chunks=[], semantic_sim=0.60, no_judge=True,
        )
        self.assertIsInstance(result["correctness"], float)

    def test_no_judge_returns_token_f1(self):
        """Token F1 must be computed and included even in --no-judge mode."""
        result, _, _, _ = evaluate_with_oracle_routing(
            question="q", answer="same words here", ground_truth="same words here",
            context_chunks=[], semantic_sim=0.75, no_judge=True,
        )
        self.assertIn("token_f1", result)
        self.assertIsInstance(result["token_f1"], float)


class TestRefusalRouting(unittest.TestCase):
    """When is_refusal=True, answer correctly declines out-of-scope → auto-pass."""

    def test_refusal_correctness_is_1(self):
        result, _, _, _ = evaluate_with_oracle_routing(
            question="What is the meaning of life?",
            answer="I can only answer questions related to Indian income tax.",
            ground_truth="I can only answer questions related to Indian income tax.",
            context_chunks=[],
            semantic_sim=0.95,
            is_refusal=True,
        )
        self.assertEqual(result["correctness"], 1.0)

    def test_refusal_faithfulness_is_1(self):
        result, _, _, _ = evaluate_with_oracle_routing(
            question="q", answer="a", ground_truth="gt",
            context_chunks=[], semantic_sim=0.85, is_refusal=True,
        )
        self.assertEqual(result["faithfulness"], 1.0)

    def test_refusal_hallucination_is_0(self):
        result, _, _, _ = evaluate_with_oracle_routing(
            question="q", answer="a", ground_truth="gt",
            context_chunks=[], semantic_sim=0.85, is_refusal=True,
        )
        self.assertEqual(result["hallucination"], 0.0)

    def test_refusal_does_not_call_llm(self):
        with patch("core.judge.llm_judge_evaluate") as mock_judge, \
             patch("core.judge.ensemble_judge_evaluate") as mock_ensemble:
            evaluate_with_oracle_routing(
                question="q", answer="a", ground_truth="gt",
                context_chunks=[], semantic_sim=0.90, is_refusal=True,
            )
            mock_judge.assert_not_called()
            mock_ensemble.assert_not_called()

    def test_judge_not_called_flag(self):
        _, _, _, judge_called = evaluate_with_oracle_routing(
            question="q", answer="a", ground_truth="gt",
            context_chunks=[], semantic_sim=0.90, is_refusal=True,
        )
        self.assertFalse(judge_called)


class TestAutoFailRouting(unittest.TestCase):
    """When semantic_sim <= ORACLE_AUTO_FAIL_THRESHOLD (0.25), auto-fail fires."""

    def _run_auto_fail(self, sim=0.15):
        return evaluate_with_oracle_routing(
            question="What is the 80C limit?",
            answer="The capital of France is Paris.",   # Completely wrong answer
            ground_truth="The deduction limit under Section 80C is ₹1.5 lakh.",
            context_chunks=["Section 80C: ₹1.5 lakh annual limit."],
            semantic_sim=sim,
            no_judge=False,
        )

    def test_auto_fail_correctness_is_0(self):
        result, _, _, _ = self._run_auto_fail()
        self.assertEqual(result["correctness"], 0.0)

    def test_auto_fail_faithfulness_is_0(self):
        """Auto-fail must set faithfulness=0.0, not 1.0 (previous bug)."""
        result, _, _, _ = self._run_auto_fail()
        self.assertEqual(result["faithfulness"], 0.0,
                         "Auto-fail faithfulness must be 0.0 — previously a bug where it was 1.0.")

    def test_auto_fail_hallucination_is_1(self):
        """Auto-fail must set hallucination=1.0."""
        result, _, _, _ = self._run_auto_fail()
        self.assertEqual(result["hallucination"], 1.0,
                         "Auto-fail hallucination must be 1.0 — previously a bug where it was 0.0.")

    def test_auto_fail_no_tokens_consumed(self):
        """Auto-fail must return 0 tokens — no LLM call."""
        _, p_tokens, c_tokens, _ = self._run_auto_fail()
        self.assertEqual(p_tokens, 0)
        self.assertEqual(c_tokens, 0)

    def test_auto_fail_does_not_call_llm(self):
        with patch("core.judge.llm_judge_evaluate") as mock_judge, \
             patch("core.judge.ensemble_judge_evaluate") as mock_ensemble:
            self._run_auto_fail()
            mock_judge.assert_not_called()
            mock_ensemble.assert_not_called()

    def test_auto_fail_at_exact_threshold(self):
        """Semantic sim exactly == 0.25 should also auto-fail."""
        result, _, _, _ = self._run_auto_fail(sim=0.25)
        self.assertEqual(result["faithfulness"], 0.0)
        self.assertEqual(result["hallucination"], 1.0)

    def test_above_threshold_does_not_auto_fail(self):
        """Sim just above 0.25 should NOT trigger the auto-fail branch."""
        mock_judge_result = ({"correctness": 0.5, "faithfulness": 0.8, "hallucination": 0.2, "confidence": 1.0, "reasoning": "Evaluated by mock judge"}, 10, 10)
        with patch("core.judge.llm_judge_evaluate", return_value=mock_judge_result), \
             patch("core.judge.ensemble_judge_evaluate", return_value=mock_judge_result + (False,)):
            result, _, _, _ = self._run_auto_fail(sim=0.26)
            # Should NOT have auto-fail reasoning
            self.assertNotIn("Auto-FAIL: semantic sim", result.get("reasoning", ""))


class TestAutoPassRouting(unittest.TestCase):
    """High-quality answers that pass all 4 signals should auto-pass without LLM."""

    def test_auto_pass_correctness_is_1(self):
        result, _, _, judge_called = evaluate_with_oracle_routing(
            question="What is the 80C deduction limit?",
            answer="The deduction limit under Section 80C is one lakh fifty thousand rupees per year.",
            ground_truth="The deduction limit under Section 80C is one lakh fifty thousand rupees per year.",
            context_chunks=["Section 80C: ₹1.5 lakh annual limit."],
            semantic_sim=0.92,
            no_judge=False,
        )
        self.assertEqual(result["correctness"], 1.0)
        self.assertFalse(judge_called, "LLM judge must NOT be called on auto-pass.")

    def test_auto_pass_hallucination_is_0(self):
        result, _, _, _ = evaluate_with_oracle_routing(
            question="q",
            answer="The deduction limit under Section 80C is one lakh fifty thousand per year.",
            ground_truth="The deduction limit under Section 80C is one lakh fifty thousand per year.",
            context_chunks=["Context chunk."],
            semantic_sim=0.95,
            no_judge=False,
        )
        self.assertEqual(result["hallucination"], 0.0)

    def test_auto_pass_zero_tokens(self):
        """Auto-pass must not consume any API tokens."""
        _, p_tokens, c_tokens, _ = evaluate_with_oracle_routing(
            question="q",
            answer="The deduction limit under Section 80C is one lakh fifty thousand per year.",
            ground_truth="The deduction limit under Section 80C is one lakh fifty thousand per year.",
            context_chunks=["ctx"],
            semantic_sim=0.95,
            no_judge=False,
        )
        self.assertEqual(p_tokens, 0)
        self.assertEqual(c_tokens, 0)


class TestRoutingReturnShape(unittest.TestCase):
    """Routing always returns a 4-tuple with the correct types regardless of path."""

    CASES = [
        {"no_judge": True,  "semantic_sim": 0.80, "label": "no_judge"},
        {"is_refusal": True, "semantic_sim": 0.90, "label": "refusal"},
        {"no_judge": False, "semantic_sim": 0.10, "label": "auto_fail"},
    ]

    def _check_shape(self, **kwargs):
        label = kwargs.pop("label", "")
        result, p, c, judge_called = evaluate_with_oracle_routing(
            question="q", answer="a", ground_truth="gt",
            context_chunks=[], **kwargs,
        )
        self.assertIsInstance(result, dict, f"[{label}] result must be a dict")
        self.assertIsInstance(p, int, f"[{label}] prompt_tokens must be int")
        self.assertIsInstance(c, int, f"[{label}] completion_tokens must be int")
        self.assertIsInstance(judge_called, bool, f"[{label}] judge_called must be bool")
        for key in ("correctness", "faithfulness", "hallucination", "confidence", "reasoning"):
            self.assertIn(key, result, f"[{label}] missing key: {key}")

    def test_no_judge_shape(self):
        self._check_shape(no_judge=True, semantic_sim=0.80, label="no_judge")

    def test_refusal_shape(self):
        self._check_shape(is_refusal=True, semantic_sim=0.90, label="refusal")

    def test_auto_fail_shape(self):
        self._check_shape(no_judge=False, semantic_sim=0.10, label="auto_fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
