"""
Core Retrieval Module
Handles document loading, chunk embedding, vector store ingestion,
and retrieval diagnostic metrics (Hit Rate, Recall@K, MRR, Context Precision/Recall).
"""

import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from config import DOCS_FOLDER, EMBEDDING_MODEL_NAME, DEFAULT_TOP_K

# Initialize embedding and vector database clients
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.Client()


def load_docs(folder: str = DOCS_FOLDER) -> List[str]:
    """
    Reads all .txt files from the target directory and chunks them by double newline.
    """
    chunks = []
    if not os.path.exists(folder):
        print(f"[Warning] Docs folder '{folder}' not found.")
        return chunks
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in text.strip().split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
    return chunks


def build_vector_store(chunks: List[str], collection_name: str = "tax_eval") -> Any:
    """
    Populates ChromaDB collection with vector embeddings of text chunks.
    """
    collection = chroma_client.get_or_create_collection(collection_name)
    if collection.count() > 0:
        chroma_client.delete_collection(collection_name)
        collection = chroma_client.get_or_create_collection(collection_name)

    embeddings = embedder.encode(chunks).tolist()
    # Tag chunks with metadata mapping them back to their source document
    metadata_list = []
    
    # We infer document source from chunk content mapping or order (in production, loaded per file)
    # Since load_docs reads files in sorted order, we match chunk contents back to files
    doc_files = sorted([f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt")])
    for chunk in chunks:
        matched_file = "income_tax_basics.txt" # fallback default
        for f_name in doc_files:
            # Check if chunk text exists in original document (or part of it)
            with open(os.path.join(DOCS_FOLDER, f_name), "r", encoding="utf-8") as f:
                full_text = f.read()
            if chunk[:80] in full_text:
                matched_file = f_name
                break
        metadata_list.append({"source": matched_file})

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata_list,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection


def retrieve(question: str, collection: Any, top_k: int = DEFAULT_TOP_K) -> Tuple[List[str], List[float], List[str]]:
    """
    Queries vector store for top_k nearest chunks.
    Returns a tuple of (retrieved_documents, similarity_scores, source_filenames).
    """
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []
    
    # Cosine similarity = 1.0 - Cosine distance
    similarity_scores = [round(float(1.0 - d), 4) for d in distances]
    sources = [m.get("source", "unknown") for m in metadatas]
    
    return documents, similarity_scores, sources


def evaluate_retrieval(
    retrieved_sources: List[str],
    expected_sources_str: str,
    retrieved_similarities: List[float],
    question: str,
    retrieved_chunks: List[str],
    expected_citations_str: str = ""
) -> Dict[str, Any]:
    """
    Computes diagnostic retrieval metrics: Hit Rate, Recall@K, MRR, Context Precision, and Context Recall.
    """
    if not expected_sources_str or expected_sources_str == "N/A":
        return {
            "hit_rate": 1.0,
            "recall_k": 1.0,
            "mrr": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0
        }
        
    expected_sources = [s.strip() for s in expected_sources_str.split(";")]
    
    # ── Hit Rate ──────────────────────────────────────────
    hit = 1.0 if any(src in retrieved_sources for src in expected_sources) else 0.0
    
    # ── Recall@K ──────────────────────────────────────────
    matched_expected = [src for src in expected_sources if src in retrieved_sources]
    recall = round(len(matched_expected) / len(expected_sources), 3)
    
    # ── MRR (Mean Reciprocal Rank) ────────────────────────
    mrr = 0.0
    for idx, src in enumerate(retrieved_sources):
        if src in expected_sources:
            mrr = round(1.0 / (idx + 1), 3)
            break
            
    # ── Context Precision ─────────────────────────────────
    # Average precision of retrieved documents.
    # Relevant chunks are those matching expected sources or having high query similarity
    precision_hits = []
    running_hits = 0.0
    for idx, src in enumerate(retrieved_sources):
        if src in expected_sources or retrieved_similarities[idx] >= 0.45:
            running_hits += 1.0
            precision_hits.append(running_hits / (idx + 1))
        else:
            precision_hits.append(0.0)
            
    context_precision = round(sum(precision_hits) / len(precision_hits), 3) if precision_hits else 0.0
    
    # ── Context Recall ────────────────────────────────────
    # Check if the expected citation string is contained in any retrieved chunks
    context_recall = 0.0
    if expected_citations_str:
        citation_clean = expected_citations_str.strip().lower()
        full_retrieved_context = "\n".join(retrieved_chunks).lower()
        if citation_clean in full_retrieved_context:
            context_recall = 1.0
        else:
            # Soft fallback: match keywords
            words = citation_clean.split()
            matches = sum(1 for w in words if w in full_retrieved_context)
            context_recall = round(matches / len(words), 3) if words else 0.0
    else:
        # If no citation is specified, fall back to matching the sources
        context_recall = recall
        
    return {
        "hit_rate": hit,
        "recall_k": recall,
        "mrr": mrr,
        "context_precision": context_precision,
        "context_recall": context_recall
    }
