"""
Dataset Generator (generate_dataset.py)

Phase 1: Domain-agnostic — uses config.DOMAIN_DESCRIPTION in the generation prompt.
         Works for any corpus, not just Indian income tax.
Phase 9: Improved benchmark generation:
  - Diversity scoring / deduplication (rejects new questions with embedding sim > 0.90 to any existing)
  - Category and difficulty balancing via CLI flags
  - Dataset validation report after generation
  - Docs support for .txt and .md files (Phase 1)
"""

import os
import csv
import json
import time
import uuid
import argparse
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import config

_groq_client = None

def _get_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=api_key, timeout=30.0)
    return _groq_client


# ── Full dataset schema ────────────────────────────────────────────────────

FIELDNAMES = [
    "unique_id", "question", "ground_truth", "category", "difficulty",
    "tags", "expected_sources", "expected_citations", "reasoning_type",
    "version", "evaluation_notes",
]

VALID_CATEGORIES   = {"factual", "reasoning", "multi_hop", "adversarial", "edge_case", "out_of_scope"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_REASONING    = {"direct_lookup", "multi_step", "comparative", "negation", "numerical"}


# ── Document Loading (Phase 1: .txt + .md) ────────────────────────────────

def load_docs(folder: Optional[str] = None) -> List[Dict[str, str]]:
    """Loads text chunks from all .txt and .md files in the docs folder."""
    folder = folder or config.DOCS_FOLDER
    chunks = []
    if not os.path.exists(folder):
        print(f"[WARNING] Docs folder '{folder}' not found.")
        return chunks
    for filename in sorted(os.listdir(folder)):
        if filename.endswith((".txt", ".md")):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in text.strip().split("\n\n"):
                chunk = chunk.strip()
                if len(chunk) > 40:
                    chunks.append({"source": filename, "text": chunk})
    return chunks


def load_existing_dataset(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Phase 9: Deduplication ────────────────────────────────────────────────

def _embed_questions(questions: List[str]):
    """Embed a batch of question strings using the same model as the eval pipeline."""
    from core.retrieval import embedder
    return embedder.encode(questions)


def is_duplicate(new_question: str, existing_embeddings, threshold: float = 0.90) -> bool:
    """Returns True if new_question is too similar to any existing question."""
    if existing_embeddings is None or len(existing_embeddings) == 0:
        return False
    import numpy as np
    new_emb = _embed_questions([new_question])[0]
    norms = [np.linalg.norm(e) for e in existing_embeddings]
    new_norm = np.linalg.norm(new_emb)
    for emb, n in zip(existing_embeddings, norms):
        if n > 0 and new_norm > 0:
            sim = float(np.dot(new_emb, emb) / (new_norm * n))
            if sim >= threshold:
                return True
    return False


# ── Phase 9: Question Generation (Domain-Agnostic) ───────────────────────

def generate_questions_for_chunk(
    chunk: Dict[str, str],
    count: int = 5,
    target_category: Optional[str] = None,
    target_difficulty: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 1 + 9: Uses config.DOMAIN_DESCRIPTION — works for any domain.
    Accepts optional category/difficulty constraints for balancing.
    """
    category_hint = f"Generate questions of category: {target_category}." if target_category else \
        "Mix of factual, reasoning, multi_hop, edge_case, and out_of_scope."
    difficulty_hint = f"All questions should be difficulty: {target_difficulty}." if target_difficulty else \
        "Mix of easy, medium, and hard."

    prompt = f"""You are a benchmark dataset synthesizer for a {config.DOMAIN_DESCRIPTION} RAG system.

Generate {count} diverse Question-Answer pairs from the context below.

Context (Source: {chunk['source']}):
\"\"\"
{chunk['text']}
\"\"\"

Category guidance: {category_hint}
Difficulty guidance: {difficulty_hint}

Requirements:
- category: one of factual | reasoning | multi_hop | edge_case | out_of_scope
- difficulty: one of easy | medium | hard
- reasoning_type: one of direct_lookup | multi_step | comparative | negation | numerical
- For out_of_scope questions, the ground_truth must clearly state the document does not cover this.
- tags: 2-3 relevant comma-separated keywords.
- expected_citations: a short verbatim phrase from the context that supports the answer.

Return ONLY a valid JSON object with a single key "questions":
{{
  "questions": [
    {{
      "question": "...",
      "ground_truth": "...",
      "category": "...",
      "difficulty": "...",
      "tags": "...",
      "reasoning_type": "...",
      "expected_citations": "..."
    }}
  ]
}}"""

    try:
        response = _get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(response.choices[0].message.content.strip())
        questions = data.get("questions", [])

        enriched = []
        for q in questions:
            cat = q.get("category", "factual").lower().replace(" ", "_")
            diff = q.get("difficulty", "medium").lower()
            rt = q.get("reasoning_type", "direct_lookup").lower().replace(" ", "_")

            enriched.append({
                "unique_id":          f"Q{str(uuid.uuid4())[:8].upper()}",
                "question":           q.get("question", ""),
                "ground_truth":       q.get("ground_truth", ""),
                "category":           cat if cat in VALID_CATEGORIES else "factual",
                "difficulty":         diff if diff in VALID_DIFFICULTIES else "medium",
                "tags":               q.get("tags", ""),
                "expected_sources":   chunk["source"],
                "expected_citations": q.get("expected_citations", ""),
                "reasoning_type":     rt if rt in VALID_REASONING else "direct_lookup",
                "version":            "2.0",
                "evaluation_notes":   "",
            })
        return enriched

    except Exception as e:
        print(f"  [Error] Generation failed for chunk: {e}")
        return []


# ── Phase 9: Validation Report ────────────────────────────────────────────

def print_validation_report(questions: List[Dict[str, Any]], duplicates_removed: int) -> None:
    """Print a summary of the dataset quality after generation."""
    from collections import Counter

    cats   = Counter(q.get("category", "?") for q in questions)
    diffs  = Counter(q.get("difficulty", "?") for q in questions)
    rts    = Counter(q.get("reasoning_type", "?") for q in questions)
    srcs   = Counter(q.get("expected_sources", "?") for q in questions)

    print("\n" + "=" * 55)
    print("  Dataset Validation Report")
    print("=" * 55)
    print(f"  Total questions:      {len(questions)}")
    print(f"  Duplicates removed:   {duplicates_removed}")
    print(f"\n  Category distribution:")
    for cat, cnt in sorted(cats.items()):
        print(f"    {cat:<20} {cnt:>4}  ({cnt/len(questions)*100:.1f}%)")
    print(f"\n  Difficulty distribution:")
    for diff, cnt in sorted(diffs.items()):
        print(f"    {diff:<20} {cnt:>4}  ({cnt/len(questions)*100:.1f}%)")
    print(f"\n  Reasoning type distribution:")
    for rt, cnt in sorted(rts.items()):
        print(f"    {rt:<25} {cnt:>4}")
    print(f"\n  Coverage per source:")
    for src, cnt in sorted(srcs.items()):
        print(f"    {src:<35} {cnt:>4}")

    # Diversity score: average pairwise embedding similarity (lower = more diverse)
    # Only compute on a sample to keep it fast
    sample = [q["question"] for q in questions[:50]]
    if len(sample) >= 2:
        try:
            import numpy as np
            embs = _embed_questions(sample)
            sims = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    n1, n2 = np.linalg.norm(embs[i]), np.linalg.norm(embs[j])
                    if n1 > 0 and n2 > 0:
                        sims.append(float(np.dot(embs[i], embs[j]) / (n1 * n2)))
            avg_sim = round(sum(sims) / len(sims), 3) if sims else 0.0
            print(f"\n  Avg pairwise question similarity (sample of {len(sample)}): {avg_sim}")
            print(f"  Diversity score (1 - avg_sim): {round(1 - avg_sim, 3)}")
        except Exception:
            pass
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-Litmus Dataset Generator — synthesize evaluation questions from any corpus."
    )
    parser.add_argument("--output", default=config.DATASET_PATH,
        help="Output CSV path. Default: config.DATASET_PATH.")
    parser.add_argument("--docs-folder", default=config.DOCS_FOLDER, dest="docs_folder",
        help="Docs folder to generate from. Default: config.DOCS_FOLDER.")
    parser.add_argument("--target-total", type=int, default=210, dest="target_total",
        help="Total questions to reach in the output dataset.")
    parser.add_argument("--questions-per-chunk", type=int, default=5, dest="qpc",
        help="Questions to generate per document chunk.")
    parser.add_argument("--dedup-threshold", type=float, default=0.90, dest="dedup",
        help="Embedding similarity threshold for deduplication (0–1). Default: 0.90.")
    parser.add_argument("--target-category", default=None, dest="category",
        help="Force all generated questions to this category (e.g. 'reasoning').")
    parser.add_argument("--target-difficulty", default=None, dest="difficulty",
        help="Force all generated questions to this difficulty (e.g. 'hard').")
    args = parser.parse_args()

    print("=" * 55)
    print("  LLM-Litmus Dataset Generator")
    print(f"  Domain: {config.DOMAIN_DESCRIPTION}")
    print("=" * 55)

    chunks = load_docs(args.docs_folder)
    if not chunks:
        print("No documents found. Add .txt or .md files to the docs folder.")
        return

    existing = load_existing_dataset(args.output)
    print(f"Existing questions:  {len(existing)}")
    print(f"Target total:        {args.target_total}")

    needed = args.target_total - len(existing)
    if needed <= 0:
        print(f"Dataset already at target. Running validation report only...")
        print_validation_report(existing, duplicates_removed=0)
        return

    print(f"Need to generate:    ~{needed} questions from {len(chunks)} chunks")

    # Build embeddings of existing questions for deduplication
    print("Computing deduplication embeddings for existing questions...")
    existing_embeddings = []
    if existing:
        try:
            existing_embeddings = list(_embed_questions([q["question"] for q in existing]))
        except Exception as e:
            print(f"  [Warning] Could not build dedup embeddings: {e}")

    generated: List[Dict[str, Any]] = []
    duplicates_removed = 0

    for i, chunk in enumerate(chunks):
        if len(existing) + len(generated) >= args.target_total:
            break
        print(f"  Chunk {i+1:02d}/{len(chunks)} [{chunk['source']}]...", end=" ", flush=True)

        new_qs = generate_questions_for_chunk(
            chunk, count=args.qpc,
            target_category=args.category,
            target_difficulty=args.difficulty,
        )

        accepted = []
        for q in new_qs:
            question_text = q["question"]
            if not question_text:
                continue
            if is_duplicate(question_text, existing_embeddings, threshold=args.dedup):
                duplicates_removed += 1
                continue
            # Add to dedup pool
            try:
                new_emb = list(_embed_questions([question_text])[0])
                existing_embeddings.append(new_emb)
            except Exception:
                pass
            accepted.append(q)

        generated.extend(accepted)
        print(f"+{len(accepted)} accepted  ({len(new_qs) - len(accepted)} duplicates removed)")
        time.sleep(1.5)  # Rate limit buffer

    all_rows = existing + generated
    final_rows = all_rows[:args.target_total]

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nWrote {len(final_rows)} questions to {args.output}")
    print_validation_report(final_rows, duplicates_removed)


if __name__ == "__main__":
    main()
