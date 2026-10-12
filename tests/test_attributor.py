"""
tests/test_attributor.py
========================
Unit tests for core/attributor.py:
  - build_retrieval_diagnosis
  - attribute_failure on pass
  - run_counterfactual_diagnosis with mocked generation
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.attributor import build_retrieval_diagnosis, attribute_failure, run_counterfactual_diagnosis


class TestAttributor(unittest.TestCase):
    """Tests for core/attributor.py failure attribution logic."""

    def test_build_retrieval_diagnosis_good_run(self):
        result = {
            "hit_rate": 1.0,
            "context_recall": 0.8,
            "faithfulness": 0.9,
            "hallucination_rate": 0.0,
            "answer": "Section 80C deduction limit is ₹1.5 lakh.",
        }
        diag = build_retrieval_diagnosis(result)
        self.assertTrue(diag["context_retrieved"])
        self.assertTrue(diag["context_sufficient"])
        self.assertTrue(diag["model_used_context"])
        self.assertFalse(diag["model_hallucinated"])

    def test_build_retrieval_diagnosis_refusal(self):
        result = {
            "hit_rate": 1.0,
            "context_recall": 0.8,
            "faithfulness": 1.0,
            "hallucination_rate": 0.0,
            "answer": "I do not have information to answer this question.",
        }
        diag = build_retrieval_diagnosis(result)
        self.assertFalse(diag["model_used_context"])

    def test_attribute_failure_pass_query(self):
        result = {"status": "PASS"}
        cat, reason = attribute_failure(result)
        self.assertEqual(cat, "N/A")
        self.assertIn("passed", reason)
        self.assertIsNone(result.get("diagnostic_report"))

    @patch("core.attributor.load_ground_truth_context")
    @patch("core.generator.generate_answer")
    @patch("core.metrics.compute_semantic_similarity")
    def test_counterfactual_retrieval_failure(self, mock_sim, mock_gen, mock_load):
        mock_load.return_value = ["Context content from Section 80C."]
        from core.generator import GenerationResult
        mock_gen.return_value = GenerationResult(
            answer="Deduction is ₹1.5 lakh.", prompt_tokens=10, completion_tokens=5
        )
        mock_sim.return_value = 0.85  # >= 0.75 CF_ISOLATION_THRESHOLD

        result = {
            "status": "FAIL",
            "question": "What is 80C limit?",
            "ground_truth": "Limit is ₹1.5 lakh.",
            "expected_sources": "section_80c_deductions.txt",
        }
        cat, reason, cf_ans, cf_sim = run_counterfactual_diagnosis(result)
        self.assertEqual(cat, "Retrieval Failure")
        self.assertIn("Isolator PASS", reason)
        self.assertGreaterEqual(cf_sim, 0.75)


if __name__ == "__main__":
    unittest.main()
