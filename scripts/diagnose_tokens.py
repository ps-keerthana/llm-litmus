"""Diagnostic: measure actual token budget of the first smoke query prompt."""
import os, sys, csv, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.retrieval import load_docs, build_vector_store, retrieve

chunks = load_docs()
collection = build_vector_store(chunks)

# Replicate exact smoke selection logic from evaluate.py
random.seed(42)
questions = []
with open("golden_dataset.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        questions.append(row)

categories = {}
for q in questions:
    cat = q["category"]
    categories.setdefault(cat, []).append(q)

smoke_subset = []
for cat, q_list in sorted(categories.items()):
    sample_size = min(len(q_list), 2)
    smoke_subset.extend(random.sample(q_list, sample_size))
smoke_subset = smoke_subset[:10]

print("=== SMOKE QUERY SELECTION (seed=42) ===")
for i, q in enumerate(smoke_subset):
    print(f"  [{i+1}] [{q['category']} | {q['difficulty']}] {q['question'][:80]}")
print()

# Measure the first query's full prompt
q = smoke_subset[0]
print(f"First query: {q['question']}")
retrieved_chunks, sims, sources = retrieve(q["question"], collection, top_k=3)

print(f"\nRetrieved {len(retrieved_chunks)} chunks:")
context_parts = []
for i, (chunk, sim, src) in enumerate(zip(retrieved_chunks, sims, sources)):
    words = len(chunk.split())
    est_tok = int(words / 0.75)
    print(f"  Chunk {i+1}: {words} words | ~{est_tok} tokens | sim={sim:.3f} | src={src}")
    print(f"    {chunk[:100]}...")
    context_parts.append(chunk)

context = "\n\n".join(context_parts)
system_prompt = (
    "You are a helpful Indian income tax assistant. "
    "Answer the question using ONLY the context below.\n"
    "If the answer is not in the context, say \"I don't have information about that.\""
)
full_prompt = f"""{system_prompt}

Context:
{context}

Question: {q['question']}

Answer:"""

prompt_words = len(full_prompt.split())
est_input_tokens = int(prompt_words / 0.75)
est_output_tokens = 120  # typical answer length
est_total = est_input_tokens + est_output_tokens

print(f"\n=== TOKEN BUDGET (first query) ===")
print(f"Prompt words:           {prompt_words}")
print(f"Est. input tokens:      ~{est_input_tokens}")
print(f"Est. output tokens:     ~{est_output_tokens}  (typical answer)")
print(f"Est. TOTAL tokens/call: ~{est_total}")
print(f"Prompt chars:           {len(full_prompt)}")
print()

# Estimate for all 10 smoke queries
avg_per_query = est_total
total_10 = avg_per_query * 10
print(f"=== SMOKE RUN TOKEN PROJECTION (10 queries, no-judge) ===")
print(f"Per-query tokens:  ~{avg_per_query}")
print(f"10-query total:    ~{total_10}")
print(f"Groq free TPM:      6,000")
print()

if total_10 < 6000:
    print(">>> Total tokens fit in a single Groq minute window.")
elif total_10 < 12000:
    print(">>> Requires ~2 Groq minutes. Sequential with 60s gap would work.")
else:
    print(">>> Token budget too high. Context truncation needed.")

print()
# Also check: does ONE request exceed TPM if sent all at once vs. one at a time?
print(f"=== SINGLE REQUEST vs TPM ===")
print(f"One request uses ~{est_total} tokens.")
print(f"Groq TPM limit:    6,000 tokens/min")
print(f"After one request: {6000 - est_total} tokens remaining in that minute window.")
print(f"Rate limit headroom: {'OK - should not 429 on first request' if est_total < 6000 else 'EXCEEDS TPM LIMIT'}")
