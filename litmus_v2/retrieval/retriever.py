"""
Phase 6: Retrieval & RAG Pipeline.

Implements:
    - BM25Retriever: Inverted index and BM25 scoring over Indian tax document chunks.
    - RAGPipeline: End-to-end retrieval-augmented generation with latency tracking and metadata.
"""

from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from litmus_v2.data.schema import Chunk


def _tokenize(text: str) -> List[str]:
    """Tokenizes text preserving alphanumeric tax terms (e.g., 80c, 80tta, 10(13a), ₹)."""
    # Remove commas between digits (e.g., 1,50,000 -> 150000)
    cleaned = re.sub(r'(?<=\d),(?=\d)', '', text.lower())
    cleaned = re.sub(r'[\(\)]', '', cleaned)
    tokens = re.findall(r'[a-z0-9₹%]+', cleaned)
    res: List[str] = []
    for t in tokens:
        res.append(t)
        if t.startswith('₹') and len(t) > 1:
            res.append(t[1:])
    return res


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    doc_name: str
    rank: int


class BM25Retriever:
    """
    In-memory BM25 index over document chunks.
    Formula: Okapi BM25 with k1=1.5, b=0.75.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: Dict[str, Chunk] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.inverted_index: Dict[str, Dict[str, int]] = {}  # term -> {chunk_id: term_freq}
        self.doc_freqs: Dict[str, int] = {}  # term -> count of chunks containing term
        self.corpus_size: int = 0

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Indexes a collection of Chunk objects."""
        self.chunks = {c.chunk_id: c for c in chunks}
        self.corpus_size = len(chunks)
        self.inverted_index.clear()
        self.doc_freqs.clear()
        self.doc_lengths.clear()

        total_length = 0
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            length = len(tokens)
            self.doc_lengths[chunk.chunk_id] = length
            total_length += length

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for t, count in tf.items():
                if t not in self.inverted_index:
                    self.inverted_index[t] = {}
                    self.doc_freqs[t] = 0
                self.inverted_index[t][chunk.chunk_id] = count
                self.doc_freqs[t] += 1

        self.avg_doc_length = total_length / max(self.corpus_size, 1)

    def _idf(self, term: str) -> float:
        """Standard BM25 IDF with floor to avoid negative weights."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # idf = ln(1 + (N - df + 0.5) / (df + 0.5))
        val = (self.corpus_size - df + 0.5) / (df + 0.5)
        return math.log(1.0 + max(val, 0.0))

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieves top_k relevant chunks for a query using BM25.
        Returns ranked list of SearchResult.
        """
        query_tokens = _tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: Dict[str, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf = self._idf(token)
            postings = self.inverted_index[token]

            for chunk_id, tf in postings.items():
                doc_len = self.doc_lengths[chunk_id]
                # BM25 tf component
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_length, 1.0)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / denom)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        # Sort by score descending
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, start=1):
            chunk = self.chunks[chunk_id]
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=round(score, 4),
                text=chunk.text,
                doc_name=chunk.doc_name,
                rank=rank,
            ))

        return results


@dataclass
class RAGAnswer:
    question: str
    retrieved_chunks: List[SearchResult]
    generated_answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model_name: str
    context_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "retrieved_chunk_ids": [r.chunk_id for r in self.retrieved_chunks],
            "generated_answer": self.generated_answer,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "model_name": self.model_name,
            "num_chunks_retrieved": len(self.retrieved_chunks),
        }


class RAGPipeline:
    """
    Complete RAG Pipeline connecting BM25Retriever with generation.
    Supports pluggable generator callable for testing, Ollama, or Groq.
    """
    def __init__(
        self,
        retriever: BM25Retriever,
        generator_fn: Optional[Callable[[str, str], str]] = None,
        model_name: str = "bm25_rag",
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.generator_fn = generator_fn or self._default_mock_generator
        self.model_name = model_name
        self.top_k = top_k

    @staticmethod
    def _default_mock_generator(question: str, context: str) -> str:
        """Fallback extraction generator if no LLM is provided."""
        lines = [line.strip() for line in context.split("\n") if line.strip()]
        return lines[0] if lines else "Information not available."

    def answer_query(self, question: str) -> RAGAnswer:
        """Executes retrieval and generation for a single question."""
        t0 = time.monotonic()
        results = self.retriever.retrieve(question, top_k=self.top_k)
        t_retrieval = (time.monotonic() - t0) * 1000

        # Construct context
        context_parts = [f"[{r.doc_name} / {r.chunk_id}]: {r.text}" for r in results]
        combined_context = "\n\n".join(context_parts)

        # Generate answer
        t_gen_start = time.monotonic()
        generated = self.generator_fn(question, combined_context)
        t_gen = (time.monotonic() - t_gen_start) * 1000
        total_latency = (time.monotonic() - t0) * 1000

        return RAGAnswer(
            question=question,
            retrieved_chunks=results,
            generated_answer=generated,
            retrieval_latency_ms=t_retrieval,
            generation_latency_ms=t_gen,
            total_latency_ms=total_latency,
            model_name=self.model_name,
            context_used=combined_context,
        )
