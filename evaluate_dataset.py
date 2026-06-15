import os
import csv
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

# ── reuse your rag pipeline functions ────────────────────
def load_docs(folder="docs"):
    chunks = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in text.strip().split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
    return chunks

def build_vector_store(chunks):
    collection = chroma_client.get_or_create_collection("faq_eval")
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def retrieve(question, collection, top_k=3):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    return results["documents"][0]

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

# ── simple scorer: word overlap between answer and ground truth ──
def score_answer(answer, ground_truth):
    answer_words = set(answer.lower().split())
    truth_words = set(ground_truth.lower().split())
    if not truth_words:
        return 0.0
    overlap = answer_words.intersection(truth_words)
    return round(len(overlap) / len(truth_words), 2)

# ── hallucination check: did the answer use words NOT in context? ─
def check_hallucination(answer, context_chunks):
    # safe refusal phrases — these are honest, not hallucinations
    refusal_phrases = [
        "i don't have information",
        "i do not have information",
        "not mentioned",
        "not specified",
        "no information",
        "cannot find",
        "not available in"
    ]

    answer_lower = answer.lower()

    # if the model is honestly saying it doesn't know — not a hallucination
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return 0.0, []

    context_text = " ".join(context_chunks).lower()
    answer_words = answer_lower.split()

    # only flag content words — ignore stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "it",
                 "in", "on", "at", "to", "for", "of", "and", "or",
                 "but", "not", "with", "this", "that", "you", "we",
                 "our", "your", "be", "can", "will", "have", "has",
                 "if", "after", "before", "within", "than", "more"}

    hallucinated = [
        w for w in answer_words
        if w not in context_text
        and w not in stopwords
        and len(w) > 3
        and w.isalpha()
    ]

    hallucination_rate = round(len(hallucinated) / max(len(answer_words), 1), 2)
    return hallucination_rate, hallucinated
# ── main eval loop ────────────────────────────────────────
def run_eval():
    print("Setting up RAG pipeline...")
    chunks = load_docs()
    collection = build_vector_store(chunks)

    results = []
    passed = 0
    failed = 0

    with open("golden_dataset.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    print(f"Running eval on {len(questions)} questions...\n")

    for i, row in enumerate(questions):
        question = row["question"]
        ground_truth = row["ground_truth"]
        category = row["category"]
        difficulty = row["difficulty"]

        start_time = time.time()
        context_chunks = retrieve(question, collection)
        answer = generate_answer(question, context_chunks)
        latency = round(time.time() - start_time, 2)

        relevancy_score = score_answer(answer, ground_truth)
        hallucination_rate, hallucinated_words = check_hallucination(
            answer, context_chunks
        )

        # pass/fail decision
        status = "PASS" if relevancy_score >= 0.3 and hallucination_rate <= 0.2 else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "category": category,
            "difficulty": difficulty,
            "relevancy_score": relevancy_score,
            "hallucination_rate": hallucination_rate,
            "hallucinated_words": hallucinated_words,
            "latency_sec": latency,
            "status": status
        })

        print(f"[{i+1}/{len(questions)}] {status} | "
              f"relevancy={relevancy_score} | "
              f"hallucination={hallucination_rate} | "
              f"latency={latency}s")
        print(f"  Q: {question}")
        print(f"  A: {answer[:80]}...")
        print()

        time.sleep(1)  # groq rate limit buffer

    # ── save results ──────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "run_timestamp": timestamp,
        "total_questions": len(questions),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(questions) * 100, 1),
        "hallucination_rate_avg": round(
            sum(r["hallucination_rate"] for r in results) / len(results), 3
        ),
        "avg_latency_sec": round(
            sum(r["latency_sec"] for r in results) / len(results), 2
        ),
        "results": results
    }

    os.makedirs("eval_results", exist_ok=True)
    output_path = f"eval_results/run_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 50)
    print(f"EVAL COMPLETE")
    print(f"  Pass rate:         {output['pass_rate']}%")
    print(f"  Avg hallucination: {output['hallucination_rate_avg']}")
    print(f"  Avg latency:       {output['avg_latency_sec']}s")
    print(f"  Results saved to:  {output_path}")
    print("=" * 50)

    return output

if __name__ == "__main__":
    run_eval()