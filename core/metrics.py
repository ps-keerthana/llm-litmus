"""
Core Metrics Module
Provides local metrics evaluations such as embedding-based semantic similarity comparisons.
"""

import numpy as np
from core.retrieval import embedder


def compute_semantic_similarity(answer: str, ground_truth: str) -> float:
    """
    Computes semantic cosine similarity between the generated answer and ground truth
    using SentenceTransformer embeddings.
    """
    embeddings = embedder.encode([answer, ground_truth])
    vec1, vec2 = embeddings[0], embeddings[1]
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(float(np.dot(vec1, vec2) / (norm1 * norm2)), 3)
