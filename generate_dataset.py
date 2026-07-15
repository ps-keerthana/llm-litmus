"""
Dataset Generator (generate_dataset.py)
Uses Groq LLM to synthesize new Question-Answer pairs from the docs/ corpus
and appends them to golden_dataset.csv with the full metadata schema.

Run this script manually when expanding the benchmark suite:
    python generate_dataset.py
"""

import os
import csv
import json
import time
import uuid
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Lazy-initialize the Groq client to avoid crashing on import without API key
_groq_client = None

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=api_key, timeout=20.0)
    return _groq_client


DATASET_PATH = "golden_dataset.csv"
DOCS_FOLDER = "docs"
TARGET_TOTAL = 210
QUESTIONS_PER_CHUNK = 5

# Full schema matching golden_dataset.csv
FIELDNAMES = [
    "unique_id", "question", "ground_truth", "category", "difficulty",
    "tags", "expected_sources", "expected_citations", "reasoning_type",
    "version", "evaluation_notes"
]


def load_docs(folder: str = DOCS_FOLDER) -> List[Dict[str, str]]:
    """Loads text chunks from all .txt files in the docs folder."""
    chunks = []
    if not os.path.exists(folder):
        print(f"[WARNING] Docs folder '{folder}' not found.")
        return chunks
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in text.strip().split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    chunks.append({"source": filename, "text": chunk})
    return chunks


def load_existing_dataset(filepath: str = DATASET_PATH) -> List[Dict[str, Any]]:
    """Returns all rows currently in the golden dataset CSV."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_questions_for_chunk(chunk: Dict[str, str], count: int = QUESTIONS_PER_CHUNK) -> List[Dict[str, Any]]:
    """
    Calls Groq to generate structured Q&A pairs from a single context chunk.
    Returns a list of question dicts conforming to the full metadata schema.
    """
    prompt = f"""You are a benchmark dataset synthesizer for an Indian income tax RAG system.

Generate {count} diverse Question-Answer pairs from the context below.

Context (Source: {chunk['source']}):
\"\"\"
{chunk['text']}
\"\"\"

Requirements:
- Include a mix of "factual", "edge_case", and "out_of_scope" categories.
- Assign difficulty: "easy", "medium", or "hard".
- Assign reasoning_type: one of "direct_lookup", "multi_step", "comparative", "negation", "numerical".
- For "out_of_scope" questions, the ground_truth must clearly state the document does not contain this.
- tags: a comma-separated string of 2-3 relevant keywords.

Return ONLY a valid JSON object with a single key "questions":
{{
  "questions": [
    {{
      "question": "...",
      "ground_truth": "...",
      "category": "factual|edge_case|out_of_scope",
      "difficulty": "easy|medium|hard",
      "tags": "tag1, tag2",
      "reasoning_type": "direct_lookup|multi_step|comparative|negation|numerical"
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
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        questions = data.get("questions", [])
        
        # Enrich with full metadata schema
        enriched = []
        for q in questions:
            enriched.append({
                "unique_id": f"Q{str(uuid.uuid4())[:8].upper()}",
                "question": q.get("question", ""),
                "ground_truth": q.get("ground_truth", ""),
                "category": q.get("category", "factual").lower(),
                "difficulty": q.get("difficulty", "easy").lower(),
                "tags": q.get("tags", ""),
                "expected_sources": chunk["source"],
                "expected_citations": "",
                "reasoning_type": q.get("reasoning_type", "direct_lookup").lower(),
                "version": "2.0",
                "evaluation_notes": ""
            })
        return enriched

    except Exception as e:
        print(f"  [Error] Failed to generate questions for chunk: {e}")
        return []


def main() -> None:
    print("=" * 55)
    print("  Golden Dataset Generator")
    print("=" * 55)
    
    chunks = load_docs()
    if not chunks:
        print("No documents found. Add .txt files to the docs/ folder.")
        return

    existing = load_existing_dataset()
    current_count = len(existing)
    print(f"Existing questions: {current_count}")
    print(f"Target total:       {TARGET_TOTAL}")

    needed = TARGET_TOTAL - current_count
    if needed <= 0:
        print(f"Dataset already at target ({current_count} >= {TARGET_TOTAL}). No generation needed.")
        return

    print(f"Generating:         ~{needed} new questions from {len(chunks)} chunks\n")
    
    generated = []
    for i, chunk in enumerate(chunks):
        if current_count + len(generated) >= TARGET_TOTAL:
            break
        print(f"  Chunk {i+1:02d}/{len(chunks)} [{chunk['source']}]...", end=" ", flush=True)
        new_qs = generate_questions_for_chunk(chunk)
        generated.extend(new_qs)
        print(f"+{len(new_qs)} questions")
        time.sleep(1.5)  # Rate limit buffer

    all_rows = existing + generated
    final_rows = all_rows[:TARGET_TOTAL]

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nDone. Wrote {len(final_rows)} total questions to {DATASET_PATH}.")


if __name__ == "__main__":
    main()
