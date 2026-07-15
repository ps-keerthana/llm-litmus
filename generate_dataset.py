import os
import csv
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=20.0)

# ── Load documents ──────────────────────────────────────
def load_docs(folder="docs"):
    chunks = []
    if not os.path.exists(folder):
        print(f"Docs folder '{folder}' not found.")
        return chunks
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in text.strip().split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
    return chunks

# ── Load existing golden dataset ──────────────────────────
def load_existing_dataset(filepath="golden_dataset.csv"):
    existing_questions = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_questions.append(row)
    return existing_questions

# ── Generate QA pairs using Groq ──────────────────────────
def generate_questions_for_chunk(chunk, count=5):
    prompt = f"""You are an AI data synthesizer. Your task is to generate high-quality, realistic Question-Answer pairs based on the context chunk below.

Context Chunk:
\"\"\"
{chunk}
\"\"\"

Please generate {count} diverse questions.
Requirements:
1. Categorize them into one of the following:
   - "factual": Direct answers from the context.
   - "edge_case": Complex questions, hypothetical scenarios, or slight variations that test constraints mentioned in the context.
   - "out_of_scope": Questions that sound related to the context topic (e.g., delivery, cancellation, pricing) but cannot be answered using the provided text. The expected ground_truth for out_of_scope should clearly state that the document does not contain this information.
2. Assign a difficulty: "easy", "medium", or "hard".
3. Return ONLY a valid JSON object with a single key "questions" containing a list of objects. Do not include any markdown styling, conversational filler, or wrap the JSON in markdown blocks.

JSON Schema:
{{
  "questions": [
    {{
      "question": "...",
      "ground_truth": "...",
      "category": "factual" | "edge_case" | "out_of_scope",
      "difficulty": "easy" | "medium" | "hard"
    }}
  ]
}}
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return data.get("questions", [])
    except Exception as e:
        print(f"Error generating questions for chunk: {e}")
        return []

def main():
    print("Starting golden dataset generator...")
    chunks = load_docs()
    if not chunks:
        print("No documents found to generate questions from.")
        return

    existing = load_existing_dataset()
    print(f"Found {len(existing)} existing questions.")

    target_total = 105
    needed = target_total - len(existing)
    
    if needed <= 0:
        print(f"Dataset already has {len(existing)} questions, which is >= target of {target_total}.")
        return

    print(f"Need to generate {needed} new questions to reach target of {target_total}.")
    
    # Calculate questions per chunk
    num_chunks = len(chunks)
    # Generate around needed+10 to be safe
    needed_buffer = needed + 10
    questions_per_chunk = max(1, -(-needed_buffer // num_chunks)) # ceiling division
    
    generated_rows = []
    
    for i, chunk in enumerate(chunks):
        if len(existing) + len(generated_rows) >= target_total:
            break
            
        print(f"Generating questions for chunk {i+1}/{num_chunks}...")
        questions = generate_questions_for_chunk(chunk, count=questions_per_chunk)
        for q in questions:
            # Clean categories & difficulties
            q["category"] = q.get("category", "factual").lower()
            q["difficulty"] = q.get("difficulty", "easy").lower()
            generated_rows.append(q)
            if len(existing) + len(generated_rows) >= target_total:
                break
        time.sleep(1) # buffer for rate limits

    # Append to existing
    all_rows = existing + generated_rows
    
    with open("golden_dataset.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question", "ground_truth", "category", "difficulty"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows[:target_total])

    print(f"Successfully generated {len(generated_rows)} new questions!")
    print(f"Total dataset size is now {len(all_rows[:target_total])} questions (saved to golden_dataset.csv).")

if __name__ == "__main__":
    main()
