"""
tests/test_retrieval_metrics.py
================================
Unit tests for retrieval metric computations in core/retrieval.py.

Tests hit_rate, MRR, nDCG@K, context precision, and coverage
against hand-computed expected values using a mocked ChromaDB collection.
No real ChromaDB or embedding model is used.
"""

import os
import sys
import math
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch the embedder before anything in core.retrieval runs
_mock_embedder = MagicMock()
_mock_embedder.encode.return_value = [[0.1] * 384]

import core.retrieval as _retrieval_mod  # noqa: E402
_retrieval_mod.embedder = _mock_embedder


class TestRetrievalMetrics(unittest.TestCase):
    """
    Tests evaluate_retrieval() against hand-computed expected values.

    Setup: expected_source = "doc_a.txt", retrieved 3 chunks from
    [doc_a.txt, doc_b.txt, doc_a.txt]. Chunk 0 and 2 are relevant.
    """

    def _run_metrics(self, retrieved_sources, expected_source, similarities=None):
        from core.retrieval import evaluate_retrieval
        if similarities is None:
            similarities = [0.9, 0.7, 0.6][:len(retrieved_sources)]
        return evaluate_retrieval(
            retrieved_sources=retrieved_sources,
            expected_sources_str=expected_source,
            retrieved_similarities=similarities,
            question="What is the deduction under section 80C?",
            retrieved_chunks=["The deduction limit under Section 80C is ₹1.5 lakh."] * len(retrieved_sources),
            k=len(retrieved_sources),
        )

    def test_perfect_hit_rate(self):
        """All chunks from the expected source → hit_rate = 1.0."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_a.txt", "doc_a.txt"],
            expected_source="doc_a.txt",
        )
        self.assertEqual(metrics["hit_rate"], 1.0)

    def test_zero_hit_rate(self):
        """No chunks from the expected source → hit_rate = 0.0."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_b.txt", "doc_c.txt"],
            expected_source="doc_a.txt",
        )
        self.assertEqual(metrics["hit_rate"], 0.0)

    def test_partial_hit_rate(self):
        """1 of 2 chunks from expected source → hit_rate = 1.0 (any hit = success)."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_a.txt", "doc_b.txt"],
            expected_source="doc_a.txt",
        )
        # hit_rate is binary: did we retrieve at least one relevant chunk?
        self.assertEqual(metrics["hit_rate"], 1.0)

    def test_mrr_first_position(self):
        """Relevant chunk at rank 1 → MRR = 1.0."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_a.txt", "doc_b.txt"],
            expected_source="doc_a.txt",
        )
        self.assertEqual(metrics["mrr"], 1.0)

    def test_mrr_second_position(self):
        """Relevant chunk at rank 2 → MRR = 0.5."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_b.txt", "doc_a.txt"],
            expected_source="doc_a.txt",
        )
        self.assertAlmostEqual(metrics["mrr"], 0.5, places=2)

    def test_mrr_no_relevant(self):
        """No relevant chunk → MRR = 0.0."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_b.txt", "doc_c.txt"],
            expected_source="doc_a.txt",
        )
        self.assertEqual(metrics["mrr"], 0.0)

    def test_metrics_dict_has_required_keys(self):
        """evaluate_retrieval must always return the documented keys."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_a.txt"],
            expected_source="doc_a.txt",
        )
        required_keys = {"hit_rate", "mrr", "context_precision", "context_recall"}
        self.assertTrue(
            required_keys.issubset(metrics.keys()),
            f"Missing keys: {required_keys - metrics.keys()}"
        )

    def test_context_precision_all_relevant(self):
        """All retrieved chunks from expected source → precision = 1.0."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_a.txt", "doc_a.txt"],
            expected_source="doc_a.txt",
        )
        self.assertEqual(metrics["context_precision"], 1.0)

    def test_context_precision_none_relevant(self):
        """No retrieved chunks from expected source and low similarities → precision = 0.0."""
        metrics = self._run_metrics(
            retrieved_sources=["doc_b.txt", "doc_b.txt"],
            expected_source="doc_a.txt",
            similarities=[0.1, 0.1],
        )
        self.assertEqual(metrics["context_precision"], 0.0)

    def test_ndcg_at_k_hand_calculated(self):
        """
        nDCG@K hand-computed:
        DCG = sum(rel_i / log2(i + 1)) for i = 1..k
        Case: retrieved = [doc_b, doc_a], expected = [doc_a]
        relevances = [0, 1]
        DCG = 0/log2(2) + 1/log2(3) = 1 / 1.5849625 ≈ 0.6309
        Ideal: [1, 0] -> IDCG = 1/log2(2) = 1.0
        nDCG = 0.6309 / 1.0 = 0.631
        """
        metrics = self._run_metrics(
            retrieved_sources=["doc_b.txt", "doc_a.txt"],
            expected_source="doc_a.txt",
        )
        self.assertEqual(metrics["ndcg_at_k"], 0.631)

    def test_map_score_hand_calculated(self):
        """
        MAP (Average Precision) hand-computed:
        Case: retrieved = [doc_a, doc_x, doc_b], expected = 'doc_a.txt; doc_b.txt'
        Rank 1 (doc_a): Precision@1 = 1/1 = 1.0
        Rank 2 (doc_x): irrelevant
        Rank 3 (doc_b): Precision@3 = 2/3 ≈ 0.6667
        AP = (1.0 + 0.6667) / 2 = 1.6667 / 2 ≈ 0.833
        """
        metrics = self._run_metrics(
            retrieved_sources=["doc_a.txt", "doc_x.txt", "doc_b.txt"],
            expected_source="doc_a.txt; doc_b.txt",
        )
        self.assertEqual(metrics["map_score"], 0.833)

    def test_no_expected_source_returns_defaults(self):
        """Empty expected_source should not crash — return safe defaults."""
        from core.retrieval import evaluate_retrieval
        metrics = evaluate_retrieval(
            retrieved_sources=["doc_a.txt"],
            expected_sources_str="",
            retrieved_similarities=[0.9],
            question="What is the deduction?",
            retrieved_chunks=["Chunk text."],
            k=1,
        )
        self.assertIsInstance(metrics, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
