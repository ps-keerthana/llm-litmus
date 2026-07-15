"""
RAG Pipeline Module
Handles document loading, embedding, vector store creation, retrieval, and LLM answer generation.
Designed to be imported by evaluate_dataset.py, dashboard.py, and other scripts.
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# ── Shared Clients ────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=20.0)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

MODEL_NAME = "llama-3.1-8b-instant"

# ── Groq API Pricing (per 1M tokens) ─────────────────────
PRICE_INPUT_1M = 0.05
PRICE_OUTPUT_1M = 0.08

# ── Robust API Call with Retry Logic ──────────────────────
LAST_API_SLEEP_TIME = 0.0

def call_groq_with_retry(messages, response_format=None, temperature=0, max_retries=5):
    """Call Groq API with exponential backoff and rate limit handling."""
    from groq import RateLimitError
    global LAST_API_SLEEP_TIME
    LAST_API_SLEEP_TIME = 0.0
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = groq_client.chat.completions.create(**kwargs)
            return response
        except RateLimitError as e:
            print(f"  [Rate Limit] Groq TPM/RPM limit hit on attempt {attempt+1}/{max_retries}. Sleeping 12s...")
            time.sleep(12.0)
            LAST_API_SLEEP_TIME += 12.0
        except Exception as e:
            print(f"  [Warning] Groq API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff)
            LAST_API_SLEEP_TIME += backoff
            backoff *= 2.0

# ── Step 1: Load and chunk documents ─────────────────────
def load_docs(folder="docs"):
    """Read all .txt files from a folder, split by double newline into chunks."""
    chunks = []
    if not os.path.exists(folder):
        print(f"Docs folder '{folder}' not found.")
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

# ── Step 2: Embed chunks and store in ChromaDB ───────────
def build_vector_store(chunks, collection_name="tax_eval"):
    """Build (or rebuild) a ChromaDB collection from document chunks."""
    collection = chroma_client.get_or_create_collection(collection_name)
    if collection.count() > 0:
        chroma_client.delete_collection(collection_name)
        collection = chroma_client.get_or_create_collection(collection_name)

    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

# ── Step 3: Retrieve relevant chunks for a question ──────
def retrieve(question, collection, top_k=3):
    """Retrieve the top-k most relevant document chunks for a question."""
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    return results["documents"][0]

# ── Step 4: Generate answer using Groq + retrieved context ─
def generate_answer(question, context_chunks, system_prompt=None, temperature=0):
    """Generate an answer using the RAG pipeline. Returns (answer, prompt_tokens, completion_tokens)."""
    context = "\n\n".join(context_chunks)

    if system_prompt is None:
        system_prompt = (
            'You are a helpful Indian income tax assistant. '
            'Answer the question using ONLY the context below.\n'
            'If the answer is not in the context, say "I don\'t have information about that."'
        )

    prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = call_groq_with_retry(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        answer = response.choices[0].message.content.strip()

        usage = getattr(response, "usage", None)
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return answer, prompt_tokens, completion_tokens
    except Exception as e:
        print(f"  [Error] Failed to generate answer after retries: {e}")
        return "I don't have information about that. (Error: LLM generation failed)", 0, 0

# ── Full pipeline: question → answer ─────────────────────
def ask(question, collection, top_k=3, system_prompt=None, temperature=0):
    """Run the full RAG pipeline: retrieve → generate. Returns (answer, chunks, prompt_tokens, completion_tokens)."""
    chunks = retrieve(question, collection, top_k=top_k)
    answer, p_tokens, c_tokens = generate_answer(question, chunks, system_prompt, temperature)
    return answer, chunks, p_tokens, c_tokens

def calculate_cost(prompt_tokens, completion_tokens):
    """Calculate cost in USD based on Groq pricing."""
    return round((prompt_tokens * PRICE_INPUT_1M + completion_tokens * PRICE_OUTPUT_1M) / 1_000_000, 6)


# ── Run standalone demo ──────────────────────────────────
if __name__ == "__main__":
    print("Loading documents...")
    chunks = load_docs()
    print(f"  {len(chunks)} chunks loaded")

    print("Building vector store...")
    collection = build_vector_store(chunks)
    print("  Done\n")

    test_questions = [
        "What is the maximum deduction under Section 80C?",
        "Is HRA exemption available under the new tax regime?",
        "What is the TDS rate on FD interest?",
        "Can I claim 80C deduction for my grandchild's tuition?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        answer, sources, _, _ = ask(q, collection)
        print(f"A: {answer}")
        print(f"Sources: {sources[0][:80]}...")
        print()
        time.sleep(1)