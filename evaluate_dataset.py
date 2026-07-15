"""
Evaluation Engine
Runs the full golden dataset through the RAG pipeline and scores each answer using:
  1. Semantic cosine similarity (local embedding comparison)
  2. LLM-as-a-judge (combined relevancy + faithfulness evaluation)
Tracks latency percentiles, token costs, and appends results to metrics_history.json.
"""

import os
import csv
import json
import time
import subprocess
from datetime import datetime
import numpy as np

from rag_pipeline import (
    embedder, load_docs, build_vector_store, retrieve,
    generate_answer, call_groq_with_retry, calculate_cost
)

# ── Metric 1: Semantic Cosine Similarity ────────────────
def compute_semantic_similarity(answer, ground_truth):
    """Compute cosine similarity between answer and ground truth embeddings."""
    embeddings = embedder.encode([answer, ground_truth])
    vec1, vec2 = embeddings[0], embeddings[1]
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(float(np.dot(vec1, vec2) / (norm1 * norm2)), 3)

# ── Metric 2 & 3: Combined LLM Judge ────────────────────
def llm_judge_evaluate(question, answer, ground_truth, context_chunks):
    """Use LLM-as-a-judge to evaluate relevancy and faithfulness in a single API call."""
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an expert evaluator for an Indian income tax Q&A system. Evaluate the generated answer against the ground truth and retrieved context.

Inputs:
- Question: {question}
- Ground Truth: {ground_truth}
- Retrieved Context:
{context}
- Generated Answer: {answer}

Evaluate and return a JSON object with these keys:
1. "relevancy_score": float 0.0-1.0 (1.0 = answer matches ground truth meaning perfectly. If ground truth expects a refusal like "document does not mention" and the answer correctly refuses, score 1.0)
2. "faithfulness_score": float 0.0-1.0 (1.0 = every claim is supported by context. Correct refusals when context lacks info = 1.0)
3. "relevancy_reasoning": brief explanation
4. "faithfulness_reasoning": brief explanation identifying any hallucinated claims

Return ONLY valid JSON."""

    try:
        response = call_groq_with_retry(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content.strip())

        usage = getattr(response, "usage", None)
        return (
            float(data.get("relevancy_score", 0.0)),
            float(data.get("faithfulness_score", 1.0)),
            data.get("relevancy_reasoning", ""),
            data.get("faithfulness_reasoning", ""),
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0
        )
    except Exception as e:
        print(f"  [Error] LLM judge failed: {e}")
        return 0.0, 1.0, f"Error: {e}", f"Error: {e}", 0, 0

# ── Git Commit SHA ────────────────────────────────────────
def get_git_sha():
    """Get the current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

# ── Metrics History ───────────────────────────────────────
def append_to_metrics_history(summary):
    """Append a run summary to metrics_history.json."""
    history_file = "metrics_history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(summary)

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Appended to {history_file} ({len(history)} total runs)")

# ── Main Evaluation Loop ──────────────────────────────────
def run_eval():
    print("Setting up RAG pipeline...")
    chunks = load_docs()
    print(f"  {len(chunks)} chunks loaded")
    collection = build_vector_store(chunks)
    print("  Vector store ready\n")

    if not os.path.exists("golden_dataset.csv"):
        print("ERROR: golden_dataset.csv not found.")
        return

    with open("golden_dataset.csv", "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    print(f"Running eval on {len(questions)} questions...\n")

    # Thresholds
    MIN_SEMANTIC_SIM = 0.65
    MIN_LLM_RELEVANCY = 0.75
    MAX_HALLUCINATION = 0.05

    results = []
    passed = 0
    failed = 0

    for i, row in enumerate(questions):
        question = row["question"]
        ground_truth = row["ground_truth"]
        category = row["category"]
        difficulty = row["difficulty"]

        # Generate answer
        start_time = time.time()
        context_chunks = retrieve(question, collection)
        answer, gen_p, gen_c = generate_answer(question, context_chunks)
        latency = round(time.time() - start_time, 2)

        # Metric 1: Semantic similarity (local, no API call)
        semantic_sim = compute_semantic_similarity(answer, ground_truth)

        # Metric 2 & 3: LLM judge (single API call)
        llm_rel, faith, rel_reason, faith_reason, judge_p, judge_c = llm_judge_evaluate(
            question, answer, ground_truth, context_chunks
        )
        hallucination = round(1.0 - faith, 2)

        # Cost
        total_p = gen_p + judge_p
        total_c = gen_c + judge_c
        cost = calculate_cost(total_p, total_c)

        # Pass/Fail
        is_relevant = semantic_sim >= MIN_SEMANTIC_SIM or llm_rel >= MIN_LLM_RELEVANCY
        is_faithful = hallucination <= MAX_HALLUCINATION
        status = "PASS" if is_relevant and is_faithful else "FAIL"

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
            "semantic_similarity": semantic_sim,
            "llm_relevancy": llm_rel,
            "relevancy_reasoning": rel_reason,
            "faithfulness": faith,
            "hallucination_rate": hallucination,
            "faithfulness_reasoning": faith_reason,
            "latency_sec": latency,
            "prompt_tokens": total_p,
            "completion_tokens": total_c,
            "cost_usd": cost,
            "status": status
        })

        print(f"[{i+1}/{len(questions)}] {status} | "
              f"sim={semantic_sim} | rel={llm_rel} | "
              f"hall={hallucination} | {latency}s | ${cost:.6f}")
        print(f"  Q: {question}")
        print(f"  A: {answer[:80]}...")
        if status == "FAIL":
            print(f"  -> {faith_reason}")
        print()

        time.sleep(1.8)  # Respect Groq rate limits

    # ── Aggregates ────────────────────────────────────────
    latencies = [r["latency_sec"] for r in results]
    p50 = round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0
    p95 = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    avg_hall = round(sum(r["hallucination_rate"] for r in results) / len(results), 3)
    avg_faith = round(sum(r["faithfulness"] for r in results) / len(results), 3)
    avg_sim = round(sum(r["semantic_similarity"] for r in results) / len(results), 3)
    avg_rel = round(sum(r["llm_relevancy"] for r in results) / len(results), 3)
    total_cost = round(sum(r["cost_usd"] for r in results), 6)
    avg_cost = round(total_cost / len(results), 6)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    commit_sha = get_git_sha()

    output = {
        "run_timestamp": timestamp,
        "commit_sha": commit_sha,
        "total_questions": len(questions),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(questions) * 100, 1),
        "hallucination_rate_avg": avg_hall,
        "avg_faithfulness": avg_faith,
        "avg_semantic_similarity": avg_sim,
        "avg_relevancy": avg_rel,
        "avg_latency_sec": avg_lat,
        "p50_latency_sec": p50,
        "p95_latency_sec": p95,
        "total_cost_usd": total_cost,
        "avg_cost_usd": avg_cost,
        "results": results
    }

    # Save detailed results
    os.makedirs("eval_results", exist_ok=True)
    output_path = f"eval_results/run_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Append summary to metrics history (git-tracked)
    history_entry = {
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "pass_rate": output["pass_rate"],
        "avg_faithfulness": avg_faith,
        "avg_semantic_similarity": avg_sim,
        "avg_relevancy": avg_rel,
        "hallucination_rate_avg": avg_hall,
        "p50_latency_sec": p50,
        "p95_latency_sec": p95,
        "total_cost_usd": total_cost,
        "avg_cost_usd": avg_cost,
        "total_questions": len(questions),
        "passed": passed,
        "failed": failed
    }
    append_to_metrics_history(history_entry)

    print("=" * 50)
    print("EVAL COMPLETE")
    print(f"  Pass rate:          {output['pass_rate']}%")
    print(f"  Avg semantic sim:   {avg_sim}")
    print(f"  Avg faithfulness:   {avg_faith}")
    print(f"  Avg hallucination:  {avg_hall}")
    print(f"  Latency (avg/p50/p95): {avg_lat}s / {p50}s / {p95}s")
    print(f"  Total cost:         ${total_cost:.6f}")
    print(f"  Commit:             {commit_sha}")
    print(f"  Saved to:           {output_path}")
    print("=" * 50)

    return output


if __name__ == "__main__":
    run_eval()