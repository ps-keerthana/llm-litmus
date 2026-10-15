"""
Core Retrieval Module
Handles document loading, chunk embedding, vector store ingestion,
and retrieval evaluation metrics.

Phase 1: Domain-agnostic — no hard-coded tax assumptions; source tracked at load time (O(n) not O(n×m))
Phase 2: Extended metrics — nDCG@K, Precision@K, MAP, Coverage added alongside existing ones
Phase 6: Scalable ingestion — persistent ChromaDB, incremental indexing, batched embedding,
         configurable chunking strategies (paragraph | sentence | fixed_size)
"""

import hashlib
import os
import math
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    DOCS_FOLDER, EMBEDDING_MODEL_NAME, DEFAULT_TOP_K, COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE, CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_PERSIST_PATH, CHROMA_USE_PERSISTENT,
)

# ── Embedding model (shared across retrieval + metrics) ──────────────────
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ── ChromaDB client (in-memory by default; persistent if CHROMA_PERSISTENT=true) ──
def _make_chroma_client() -> chromadb.ClientAPI:
    if CHROMA_USE_PERSISTENT:
        os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    return chromadb.Client()

chroma_client = _make_chroma_client()


# ── Chunking Strategies ──────────────────────────────────────────────────

def _chunk_paragraph(text: str) -> List[str]:
    """Split on double newline — default strategy."""
    return [c.strip() for c in text.strip().split("\n\n") if c.strip()]


def _chunk_sentence(text: str) -> List[str]:
    """Split on sentence boundaries (period/question-mark/exclamation)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _chunk_fixed_size(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split into fixed token-count windows with overlap (approximated by words)."""
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + size])
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + size >= len(words):
            break
    return chunks


def _apply_chunk_strategy(text: str) -> List[str]:
    if CHUNK_STRATEGY == "sentence":
        return _chunk_sentence(text)
    if CHUNK_STRATEGY == "fixed_size":
        return _chunk_fixed_size(text)
    return _chunk_paragraph(text)  # default: paragraph


def _file_hash(path: str) -> str:
    """SHA-256 hash of a file's content for incremental indexing."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ── Document Loading ─────────────────────────────────────────────────────

def load_docs(folder: str = DOCS_FOLDER) -> List[Dict[str, Any]]:
    """
    Reads all .txt and .md files from the target directory and chunks them.
    Phase 1: Tracks source at load time — O(n) reads, not O(n×m).
    Phase 6: Applies the configured CHUNK_STRATEGY.

    Returns a list of dicts:
        {text, source, chunk_index, doc_title, char_count, file_hash}
    """
    chunk_dicts: List[Dict[str, Any]] = []

    if not os.path.exists(folder):
        print(f"[Warning] Docs folder '{folder}' not found.")
        return chunk_dicts

    supported_extensions = (".txt", ".md")

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(supported_extensions):
            continue

        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[Warning] Could not read '{filename}': {e}")
            continue

        doc_title = os.path.splitext(filename)[0].replace("_", " ").title()
        f_hash = _file_hash(filepath)
        raw_chunks = _apply_chunk_strategy(text)

        for i, chunk in enumerate(raw_chunks):
            if chunk:
                chunk_dicts.append({
                    "text": chunk,
                    "source": filename,
                    "chunk_index": i,
                    "doc_title": doc_title,
                    "char_count": len(chunk),
                    "file_hash": f_hash,
                })

    return chunk_dicts


# ── Vector Store ─────────────────────────────────────────────────────────

def build_vector_store(
    chunk_dicts: List[Dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
) -> Any:
    """
    Populates ChromaDB collection with vector embeddings of text chunks.
    Phase 1: Uses config.COLLECTION_NAME (no hard-coded 'tax_eval').
    Phase 6: Batched embedding; incremental skip for unchanged docs when persistent.
    """
    if CHROMA_USE_PERSISTENT:
        return _build_vector_store_incremental(chunk_dicts, collection_name)
    return _build_vector_store_inmemory(chunk_dicts, collection_name)


def _build_vector_store_inmemory(
    chunk_dicts: List[Dict[str, Any]],
    collection_name: str,
) -> Any:
    """Rebuild the full in-memory collection on every run."""
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(collection_name)

    if not chunk_dicts:
        return collection

    texts = [c["text"] for c in chunk_dicts]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "doc_title": c["doc_title"],
            "char_count": c["char_count"],
            "file_hash": c["file_hash"],
        }
        for c in chunk_dicts
    ]
    ids = [f"chunk_{i}" for i in range(len(chunk_dicts))]

    # Batch embedding to avoid OOM on large corpora
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(embedder.encode(batch).tolist())

    collection.add(
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return collection


def _build_vector_store_incremental(
    chunk_dicts: List[Dict[str, Any]],
    collection_name: str,
) -> Any:
    """
    Persistent incremental indexing: only re-embeds documents whose file hash changed.
    Phase 6 scalable ingestion.
    """
    collection = chroma_client.get_or_create_collection(collection_name)

    # Determine which file hashes are already indexed
    existing = collection.get(include=["metadatas"])
    indexed_hashes: set = set()
    if existing and existing.get("metadatas"):
        for m in existing["metadatas"]:
            if m and m.get("file_hash"):
                indexed_hashes.add(m["file_hash"])

    # Remove chunks belonging to files that no longer exist
    current_sources = {c["source"] for c in chunk_dicts}
    if existing and existing.get("ids"):
        stale_ids = [
            id_ for id_, meta in zip(existing["ids"], existing.get("metadatas") or [])
            if meta and meta.get("source") not in current_sources
        ]
        if stale_ids:
            collection.delete(ids=stale_ids)
            print(f"  [Retrieval] Removed {len(stale_ids)} stale chunks from index.")

    # Only embed chunks from files with new/changed hashes
    new_chunks = [c for c in chunk_dicts if c["file_hash"] not in indexed_hashes]

    if not new_chunks:
        print(f"  [Retrieval] Incremental index: all {len(chunk_dicts)} chunks already up-to-date.")
        return collection

    print(f"  [Retrieval] Incremental index: embedding {len(new_chunks)} new/changed chunks.")
    texts = [c["text"] for c in new_chunks]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "doc_title": c["doc_title"],
            "char_count": c["char_count"],
            "file_hash": c["file_hash"],
        }
        for c in new_chunks
    ]
    # Use a stable ID: source + chunk_index avoids duplicates on re-run
    ids = [f"{c['source']}::chunk_{c['chunk_index']}" for c in new_chunks]

    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(embedder.encode(batch).tolist())

    collection.upsert(
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return collection


# ── Retrieval ────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    collection: Any,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[List[str], List[float], List[str]]:
    """
    Queries vector store for top_k nearest chunks.
    Returns (retrieved_documents, similarity_scores, source_filenames).
    """
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count() or top_k),
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Cosine similarity = 1.0 - Cosine distance
    similarity_scores = [round(float(1.0 - d), 4) for d in distances]
    sources = [m.get("source", "unknown") for m in metadatas]

    return documents, similarity_scores, sources


# ── Retrieval Evaluation Metrics ─────────────────────────────────────────

def _dcg(relevances: List[float]) -> float:
    """Discounted Cumulative Gain."""
    return sum(
        rel / math.log2(idx + 2)
        for idx, rel in enumerate(relevances)
    )


def _ndcg_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    A relevant result at rank 1 counts more than at rank 5.
    nDCG = 1.0 means perfect ranking.
    """
    relevances = [1.0 if src in expected_sources else 0.0 for src in retrieved_sources[:k]]
    dcg = _dcg(relevances)
    # Ideal: all relevant results at top positions
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = _dcg(ideal_relevances)
    return round(dcg / idcg, 3) if idcg > 0 else 0.0


def _precision_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int) -> float:
    """Precision@K: fraction of top-K retrieved that are relevant."""
    top_k = retrieved_sources[:k]
    hits = sum(1 for src in top_k if src in expected_sources)
    return round(hits / k, 3) if k > 0 else 0.0


def _average_precision(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """
    Average Precision (AP): averages precision at every rank where a new relevant
    source document appears. Mean over queries gives MAP (range 0.0 to 1.0).
    """
    if not expected_sources:
        return 1.0
    seen_sources = set()
    num_relevant_found = 0
    ap = 0.0
    for idx, src in enumerate(retrieved_sources):
        if src in expected_sources and src not in seen_sources:
            seen_sources.add(src)
            num_relevant_found += 1
            ap += num_relevant_found / (idx + 1)
    return round(ap / len(expected_sources), 3)


def _coverage(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """
    Coverage: fraction of all expected source documents that appear
    anywhere in the retrieved set. Critical when multiple docs are needed.
    """
    if not expected_sources:
        return 1.0
    covered = sum(1 for src in expected_sources if src in retrieved_sources)
    return round(covered / len(expected_sources), 3)


def evaluate_retrieval(
    retrieved_sources: List[str],
    expected_sources_str: str,
    retrieved_similarities: List[float],
    question: str,
    retrieved_chunks: List[str],
    expected_citations_str: str = "",
    k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Computes all retrieval evaluation metrics.

    Phase 1 (original): Hit Rate, Recall@K, MRR, Context Precision, Context Recall
    Phase 2 (new):      nDCG@K, Precision@K, MAP, Coverage
    """
    if not expected_sources_str or expected_sources_str == "N/A":
        return {
            "hit_rate": 1.0,
            "recall_k": 1.0,
            "mrr": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            # Phase 2 additions
            "ndcg_at_k": 1.0,
            "precision_at_k": 1.0,
            "map_score": 1.0,
            "coverage": 1.0,
        }

    expected_sources = [s.strip() for s in expected_sources_str.split(";")]

    # ── Hit Rate ──────────────────────────────────────────────────────────
    hit = 1.0 if any(src in retrieved_sources for src in expected_sources) else 0.0

    # ── Recall@K ──────────────────────────────────────────────────────────
    matched_expected = [src for src in expected_sources if src in retrieved_sources]
    recall = round(len(matched_expected) / len(expected_sources), 3)

    # ── MRR ───────────────────────────────────────────────────────────────
    mrr = 0.0
    for idx, src in enumerate(retrieved_sources):
        if src in expected_sources:
            mrr = round(1.0 / (idx + 1), 3)
            break

    # ── Context Precision ─────────────────────────────────────────────────
    precision_hits = []
    running_hits = 0.0
    for idx, src in enumerate(retrieved_sources):
        if src in expected_sources or retrieved_similarities[idx] >= 0.45:
            running_hits += 1.0
            precision_hits.append(running_hits / (idx + 1))
        else:
            precision_hits.append(0.0)
    context_precision = round(sum(precision_hits) / len(precision_hits), 3) if precision_hits else 0.0

    # ── Context Recall ────────────────────────────────────────────────────
    if expected_citations_str:
        citation_clean = expected_citations_str.strip().lower()
        full_retrieved_context = "\n".join(retrieved_chunks).lower()
        if citation_clean in full_retrieved_context:
            context_recall = 1.0
        else:
            words = citation_clean.split()
            matches = sum(1 for w in words if w in full_retrieved_context)
            context_recall = round(matches / len(words), 3) if words else 0.0
    else:
        context_recall = recall

    # ── Phase 2 Metrics ───────────────────────────────────────────────────
    ndcg = _ndcg_at_k(retrieved_sources, expected_sources, k)
    prec_k = _precision_at_k(retrieved_sources, expected_sources, k)
    ap = _average_precision(retrieved_sources, expected_sources)
    cov = _coverage(retrieved_sources, expected_sources)

    return {
        "hit_rate": hit,
        "recall_k": recall,
        "mrr": mrr,
        "context_precision": context_precision,
        "context_recall": context_recall,
        # Phase 2 additions
        "ndcg_at_k": ndcg,
        "precision_at_k": prec_k,
        "map_score": ap,
        "coverage": cov,
    }
