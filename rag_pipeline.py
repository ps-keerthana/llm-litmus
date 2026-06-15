import os
import time
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# ── clients ──────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

# ── step 1: load and chunk your documents ────────────────
def load_docs(folder="docs"):
    chunks = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
            # split by double newline = one FAQ block per chunk
            for chunk in text.strip().split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
    return chunks

# ── step 2: embed chunks and store in chromadb ───────────
def build_vector_store(chunks):
    collection = chroma_client.get_or_create_collection("faq")
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

# ── step 3: retrieve relevant chunks for a question ──────
def retrieve(question, collection, top_k=3):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    return results["documents"][0]  # list of matching chunks

# ── step 4: generate answer using groq + retrieved chunks ─
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ── full pipeline: question → answer ─────────────────────
def ask(question, collection):
    chunks = retrieve(question, collection)
    answer = generate_answer(question, chunks)
    return answer, chunks

# ── run it ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading documents...")
    chunks = load_docs()
    print(f"  {len(chunks)} chunks loaded")

    print("Building vector store...")
    collection = build_vector_store(chunks)
    print("  Done\n")

    test_questions = [
        "What is the return policy?",
        "Do you deliver on Sundays?",
        "Can I pay with UPI?",
        "What happens if I want to cancel after 3 hours?"
    ]

    for q in test_questions:
        print(f"Q: {q}")
        answer, sources = ask(q, collection)
        print(f"A: {answer}")
        print(f"Sources used: {sources[0][:80]}...")
        print()
        time.sleep(1)  # respect groq rate limits