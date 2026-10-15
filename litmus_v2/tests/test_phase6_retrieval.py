"""
Tests for Phase 6: Retrieval & RAG Pipeline.
Verifies Hit@K, MRR, nDCG@K, MAP with exact hand-calculated mathematical expectations.
Verifies BM25Retriever indexing, query matching, and RAGPipeline latency tracking.
"""

import math
from typing import List, Set, Sequence
import pytest
from litmus_v2.data.schema import Chunk
from litmus_v2.retrieval.metrics import (
    hit_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision_at_k,
    compute_retrieval_metrics,
)
from litmus_v2.retrieval.retriever import (
    BM25Retriever,
    RAGPipeline,
    _tokenize,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-Verifiable IR Metric Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHitAtK:
    def test_hit_at_rank_1(self):
        assert hit_at_k(["c1", "c2", "c3"], {"c1"}, k=1) == 1.0

    def test_hit_at_rank_2_misses_k1_hits_k2(self):
        assert hit_at_k(["c1", "c2", "c3"], {"c2"}, k=1) == 0.0
        assert hit_at_k(["c1", "c2", "c3"], {"c2"}, k=2) == 1.0
        assert hit_at_k(["c1", "c2", "c3"], {"c2"}, k=5) == 1.0

    def test_complete_miss(self):
        assert hit_at_k(["c1", "c2"], {"c9"}, k=5) == 0.0

    def test_empty_relevant_gives_zero(self):
        assert hit_at_k(["c1"], set(), k=5) == 0.0

    def test_k_zero_gives_zero(self):
        assert hit_at_k(["c1"], {"c1"}, k=0) == 0.0


class TestReciprocalRank:
    def test_rank_1_gives_1(self):
        assert reciprocal_rank(["c1", "c2"], {"c1"}) == 1.0

    def test_rank_2_gives_half(self):
        assert reciprocal_rank(["c1", "c2", "c3"], {"c2"}) == 0.5

    def test_rank_4_gives_quarter(self):
        assert reciprocal_rank(["c1", "c2", "c3", "c4"], {"c4"}) == 0.25

    def test_not_found_gives_zero(self):
        assert reciprocal_rank(["c1", "c2"], {"c9"}) == 0.0


class TestNDCGAtKHandCalculated:
    """
    Hand calculations:
        log2(2) = 1.0
        log2(3) = 1.5849625
        log2(4) = 2.0
        log2(5) = 2.32192809
    """

    def test_perfect_single_hit_at_rank_1(self):
        # DCG@1 = 1 / log2(2) = 1.0
        # IDCG@1 = 1.0
        # nDCG@1 = 1.0
        val = ndcg_at_k(["c1", "c2"], {"c1"}, k=1)
        assert math.isclose(val, 1.0, abs_tol=1e-5)

    def test_single_hit_at_rank_2(self):
        # DCG@2 = 0 + 1 / log2(3) = 1 / 1.5849625 = 0.63092975
        # IDCG@2 = 1 / log2(2) = 1.0 (since only 1 relevant exists)
        # nDCG@2 = 0.63092975
        expected = 1.0 / math.log2(3)
        val = ndcg_at_k(["c2", "c1"], {"c1"}, k=2)
        assert math.isclose(val, expected, abs_tol=1e-5)
        assert math.isclose(val, 0.63093, abs_tol=1e-4)

    def test_two_relevant_both_top_two(self):
        # DCG@2 = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
        # IDCG@2 = 1/log2(2) + 1/log2(3) = 1.63093
        # nDCG@2 = 1.0
        val = ndcg_at_k(["c1", "c2", "c3"], {"c1", "c2"}, k=2)
        assert math.isclose(val, 1.0, abs_tol=1e-5)

    def test_two_relevant_at_ranks_2_and_4(self):
        # Retrieved: [c9, c1, c8, c2]
        # DCG@4 = 0 + 1/log2(3) + 0 + 1/log2(5)
        # IDCG@4 = 1/log2(2) + 1/log2(3) (min(4, 2) = 2 ideal hits)
        dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
        idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
        expected = dcg / idcg
        val = ndcg_at_k(["c9", "c1", "c8", "c2"], {"c1", "c2"}, k=4)
        assert math.isclose(val, expected, abs_tol=1e-5)


class TestAveragePrecisionHandCalculated:
    def test_single_relevant_at_rank_1(self):
        # P(1) = 1/1 = 1.0. Denominator = 1. AP = 1.0
        assert average_precision_at_k(["c1", "c2"], {"c1"}, k=1) == 1.0
        assert average_precision_at_k(["c1", "c2"], {"c1"}, k=5) == 1.0

    def test_single_relevant_at_rank_2(self):
        # P(1) = 0. P(2) = 1/2 = 0.5. Denominator = min(2, 1) = 1. AP = 0.5
        assert average_precision_at_k(["c2", "c1"], {"c1"}, k=2) == 0.5

    def test_two_relevant_at_ranks_1_and_3(self):
        # Retrieved: [c1, c9, c2]
        # P(1) = 1/1 = 1.0 (hit)
        # P(2) = 1/2 (miss)
        # P(3) = 2/3 = 0.66667 (hit)
        # Sum of precisions at hits = 1.0 + (2/3) = 1.66667
        # Denominator = min(3, 2) = 2
        # AP = (1.0 + 2/3) / 2 = 5/6 = 0.83333
        expected = (1.0 + 2.0 / 3.0) / 2.0
        val = average_precision_at_k(["c1", "c9", "c2"], {"c1", "c2"}, k=3)
        assert math.isclose(val, expected, abs_tol=1e-5)


class TestComputeRetrievalMetricsSummary:
    def test_summary_aggregates_across_queries(self):
        retrieved = [
            ["c1", "c2", "c3"],  # Q1: hit at 1
            ["c2", "c1", "c3"],  # Q2: hit at 2
        ]
        relevant = [
            {"c1"},
            {"c1"},
        ]
        metrics = compute_retrieval_metrics(retrieved, relevant, k_values=(1, 2, 5))
        # Q1 RR=1.0, Q2 RR=0.5 -> MRR = 0.75
        assert metrics["mrr"] == 0.75
        # Q1 Hit@1=1.0, Q2 Hit@1=0.0 -> Mean Hit@1 = 0.5
        assert metrics["hit_at_1"] == 0.5
        # Both hit at 2 -> Mean Hit@2 = 1.0
        assert metrics["hit_at_2"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# BM25Retriever & RAGPipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25Retriever:
    @pytest.fixture
    def tax_chunks(self) -> List[Chunk]:
        return [
            Chunk(
                chunk_id="chunk_80c",
                doc_name="section_80c.txt",
                doc_title="Section 80C Deductions",
                text="Under Section 80C, deduction limit is ₹1,50,000 for PPF, ELSS, and life insurance.",
                chunk_index=0,
                start_char=0,
                end_char=80,
                token_count=15,
                content_hash="hash80c",
            ),
            Chunk(
                chunk_id="chunk_80d",
                doc_name="section_80d.txt",
                doc_title="Section 80D Health Insurance",
                text="Section 80D provides tax deduction up to ₹25,000 for medical insurance premiums.",
                chunk_index=0,
                start_char=0,
                end_char=80,
                token_count=14,
                content_hash="hash80d",
            ),
            Chunk(
                chunk_id="chunk_hra",
                doc_name="hra_rules.txt",
                doc_title="House Rent Allowance Exemption",
                text="HRA exemption under Section 10(13A) is calculated based on actual rent paid and salary.",
                chunk_index=0,
                start_char=0,
                end_char=85,
                token_count=15,
                content_hash="hashhra",
            ),
        ]

    def test_retrieves_exact_matching_chunk_at_rank_1(self, tax_chunks):
        retriever = BM25Retriever()
        retriever.index_chunks(tax_chunks)
        results = retriever.retrieve("What is the deduction limit under Section 80C?", top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_80c"
        assert results[0].rank == 1
        assert results[0].score > results[1].score

    def test_retrieves_medical_insurance_for_80d(self, tax_chunks):
        retriever = BM25Retriever()
        retriever.index_chunks(tax_chunks)
        results = retriever.retrieve("medical insurance premiums deduction", top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_80d"

    def test_tokenizer_preserves_tax_symbols(self):
        tokens = _tokenize("Section 80C limit is ₹1,50,000 & Section 10(13A)")
        assert "80c" in tokens
        assert "150000" in tokens
        assert "1013a" in tokens


class TestRAGPipeline:
    def test_end_to_end_rag_latency_and_output(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_name="doc1.txt",
                doc_title="Title 1",
                text="The HRA exemption requires rent receipts.",
                chunk_index=0,
                start_char=0,
                end_char=40,
                token_count=8,
                content_hash="h1",
            )
        ]
        retriever = BM25Retriever()
        retriever.index_chunks(chunks)

        def mock_llm(q: str, ctx: str) -> str:
            return f"Answer based on: {ctx}"

        rag = RAGPipeline(retriever=retriever, generator_fn=mock_llm, model_name="test_llm")
        answer = rag.answer_query("Does HRA require rent receipts?")

        assert answer.question == "Does HRA require rent receipts?"
        assert "Answer based on" in answer.generated_answer
        assert len(answer.retrieved_chunks) == 1
        assert answer.retrieval_latency_ms >= 0.0
        assert answer.total_latency_ms >= answer.retrieval_latency_ms
        assert answer.model_name == "test_llm"
