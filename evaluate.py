"""
Evaluation Suite Runner (evaluate.py)
Executes the benchmark evaluation on the tax Q&A dataset.
Supports full audits and fast CI smoke tests, calculating retrieval metrics,
and invoking JSON LLM-as-a-judge grates. Logs reproducible outputs.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import csv
import json
import time
import argparse
import random
from datetime import datetime
from typing import Dict, Any
import numpy as np

from config import (
    DATASET_PATH, EVAL_RESULTS_DIR,
    VERSION_DATASET, VERSION_PROMPT, VERSION_RETRIEVER, VERSION_EMBEDDING, VERSION_LLM,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, THRESHOLD_P95_LATENCY
)
from core.retrieval import load_docs, build_vector_store, retrieve, evaluate_retrieval
from core.generator import generate_answer
from core.metrics import compute_semantic_similarity
from core.judge import llm_judge_evaluate
from core.utils import get_git_sha, get_git_branch, calculate_cost, logger


def run_evaluation(smoke: bool = False, no_judge: bool = False) -> Dict[str, Any]:
    logger.info("Setting up RAG tax knowledge base index...")
    chunks = load_docs()
    logger.info(f"Loaded {len(chunks)} text chunks from documents.")
    
    collection = build_vector_store(chunks)
    logger.info("Ingestion complete. Vector store ready.")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Golden dataset not found at {DATASET_PATH}")

    # Read the rich golden dataset
    questions = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    if smoke:
        # Deterministic, balanced subset for fast CI checks.
        # Hard cap: 2 queries per category, max 10 total.
        # This keeps total Groq calls (generation only) well within free-tier TPM.
        SMOKE_PER_CATEGORY = 2
        random.seed(42)
        categories: Dict[str, list] = {}
        for q in questions:
            cat = q["category"]
            categories.setdefault(cat, []).append(q)

        smoke_subset = []
        for cat, q_list in sorted(categories.items()):  # sorted = deterministic order
            sample_size = min(len(q_list), SMOKE_PER_CATEGORY)
            smoke_subset.extend(random.sample(q_list, sample_size))

        questions = smoke_subset[:10]  # hard safety cap
        logger.info(
            f"Running SMOKE evaluation: {len(questions)} queries "
            f"({SMOKE_PER_CATEGORY}/category, seed=42)."
        )
    else:
        logger.info(f"Running FULL evaluation: testing all {len(questions)} queries.")

    if no_judge:
        logger.info(
            "LLM judge DISABLED (--no-judge). "
            "Scoring via local semantic similarity only — zero additional API calls."
        )

    results = []
    passed = 0
    failed = 0

    # Minimum similarity threshold to skip LLM judge calls and prevent rate limits
    BYPASS_SIM_THRESHOLD = 0.90
    
    # QA thresholds for pass/fail classification
    MIN_SEMANTIC_SIM = 0.65
    MIN_LLM_CORRECTNESS = 0.75
    MAX_HALLUCINATION = 0.05

    # Minimum interval between sequential Groq API calls.
    # Groq free tier: 30 RPM = 2.0s minimum per request.
    # This prevents rate-limit bursting when all 10 smoke queries
    # are sent in the same second.
    MIN_REQUEST_INTERVAL_SEC = 2.1  # slight buffer above 2.0s minimum

    for idx, row in enumerate(questions):
        q_id = row["unique_id"]
        question = row["question"]
        ground_truth = row["ground_truth"]
        category = row["category"]
        difficulty = row["difficulty"]
        expected_sources = row["expected_sources"]
        expected_citations = row["expected_citations"]
        reasoning_type = row["reasoning_type"]

        logger.info(f"[{idx+1}/{len(questions)}] Processing {q_id} ({category} | {difficulty})...")

        # ── 1. Retrieval Step ─────────────────────────────────
        ret_start = time.time()
        retrieved_chunks, similarities, sources = retrieve(question, collection, top_k=DEFAULT_TOP_K)
        ret_latency = round(time.time() - ret_start, 3)

        # Calculate retrieval metrics
        ret_metrics = evaluate_retrieval(
            retrieved_sources=sources,
            expected_sources_str=expected_sources,
            retrieved_similarities=similarities,
            question=question,
            retrieved_chunks=retrieved_chunks,
            expected_citations_str=expected_citations
        )

        # ── 2. Generation Step ───────────────────────────────
        gen_start = time.time()
        answer, gen_p, gen_c = generate_answer(question, retrieved_chunks, temperature=DEFAULT_TEMPERATURE)
        elapsed = time.time() - gen_start

        # Enforce minimum inter-request interval to stay within RPM limits.
        # Skip this delay if the call was loaded from cache (no API call made).
        import core.generator as _gen_mod
        if not _gen_mod.WAS_LAST_CALL_CACHED:
            already_waited = _gen_mod.LAST_API_SLEEP_TIME
            effective_elapsed = elapsed - already_waited
            remaining_wait = max(0.0, MIN_REQUEST_INTERVAL_SEC - effective_elapsed)
            if remaining_wait > 0 and idx < len(questions) - 1:  # no need to wait after last query
                time.sleep(remaining_wait)
        
        # Subtract any rate-limiting sleep duration to get the true execution latency
        import core.generator
        latency = round(max(0.0, elapsed - core.generator.LAST_API_SLEEP_TIME), 2)

        # ── 3. Evaluation & LLM Judge Step ────────────────────
        semantic_sim = compute_semantic_similarity(answer, ground_truth)

        # Dynamic out-of-scope bypass check
        is_refusal = False
        if category == "out_of_scope":
            refusal_keywords = [
                "don't have information", "do not have information", 
                "no information", "not mentioned", "not specified", 
                "not cover", "does not contain", "cannot find",
                "doesn't provide", "don't provide", "do not provide"
            ]
            answer_lower = answer.lower()
            if any(kw in answer_lower for kw in refusal_keywords):
                is_refusal = True

        bypass_judge = False
        if no_judge:
            # CI mode: skip all LLM judge API calls.
            # Use semantic similarity as a proxy for correctness.
            correctness_proxy = min(1.0, semantic_sim / 0.75)  # scaled to [0, 1]
            judge_metrics = {
                "correctness": round(correctness_proxy, 3),
                "faithfulness": 1.0,
                "completeness": round(correctness_proxy, 3),
                "hallucination": 0.0,
                "confidence": round(semantic_sim, 3),
                "reasoning": f"Judge skipped (--no-judge). Proxy correctness from sim={semantic_sim}."
            }
            judge_p, judge_c = 0, 0
            bypass_judge = True
        elif is_refusal:
            judge_metrics = {
                "correctness": 1.0, "faithfulness": 1.0, "completeness": 1.0,
                "hallucination": 0.0, "confidence": 1.0,
                "reasoning": "Bypassed judge: Correct refusal for out-of-scope question."
            }
            judge_p, judge_c = 0, 0
            bypass_judge = True
        elif semantic_sim >= BYPASS_SIM_THRESHOLD:
            judge_metrics = {
                "correctness": 1.0, "faithfulness": 1.0, "completeness": 1.0,
                "hallucination": 0.0, "confidence": 1.0,
                "reasoning": f"Bypassed judge: High semantic similarity ({semantic_sim}) to ground truth."
            }
            judge_p, judge_c = 0, 0
            bypass_judge = True

        if not bypass_judge:
            judge_metrics, judge_p, judge_c = llm_judge_evaluate(
                question, answer, ground_truth, retrieved_chunks
            )

        # Total tokens and costs
        total_p = gen_p + judge_p
        total_c = gen_c + judge_c
        cost = calculate_cost(total_p, total_c)

        # Pass/Fail Classification
        is_correct = semantic_sim >= MIN_SEMANTIC_SIM or judge_metrics["correctness"] >= MIN_LLM_CORRECTNESS
        is_faithful = judge_metrics["hallucination"] <= MAX_HALLUCINATION
        status = "PASS" if is_correct and is_faithful else "FAIL"

        # Failure Diagnosis Classification (to help developers trace retrieval vs generation bugs)
        failure_category = "N/A"
        if status == "FAIL":
            if ret_metrics["hit_rate"] == 0.0:
                failure_category = "Retrieval Failure (Missing Source)"
            elif ret_metrics["context_recall"] < 0.5:
                failure_category = "Retrieval Failure (Low Context Recall)"
            elif judge_metrics["hallucination"] > MAX_HALLUCINATION:
                failure_category = "LLM Hallucination"
            elif judge_metrics["completeness"] < 0.70:
                failure_category = "LLM Incomplete Answer"
            elif latency > THRESHOLD_P95_LATENCY:
                failure_category = "SLA Latency Violation"
            else:
                failure_category = "LLM Generation Mismatch"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        # Save query record for complete failure reproducibility
        results.append({
            "unique_id": q_id,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "category": category,
            "difficulty": difficulty,
            "reasoning_type": reasoning_type,
            "tags": row["tags"],
            "expected_sources": expected_sources,
            "retrieved_sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_similarities": similarities,
            "semantic_similarity": semantic_sim,
            "correctness": judge_metrics["correctness"],
            "faithfulness": judge_metrics["faithfulness"],
            "completeness": judge_metrics["completeness"],
            "hallucination_rate": judge_metrics["hallucination"],
            "judge_confidence": judge_metrics["confidence"],
            "judge_reasoning": judge_metrics["reasoning"],
            "hit_rate": ret_metrics["hit_rate"],
            "recall_k": ret_metrics["recall_k"],
            "mrr": ret_metrics["mrr"],
            "context_precision": ret_metrics["context_precision"],
            "context_recall": ret_metrics["context_recall"],
            "latency_sec": latency,
            "retrieval_latency_sec": ret_latency,
            "prompt_tokens": total_p,
            "completion_tokens": total_c,
            "cost_usd": cost,
            "status": status,
            "failure_category": failure_category,
            "prompt_used": (
                "Indian tax assistant context system prompt plus top 3 document chunks context prompt."
            )
        })

        time.sleep(1.8)  # Rate limit safety sleep

    # ── Aggregate Calculations ─────────────────────────────
    latencies = [r["latency_sec"] for r in results]
    p50 = round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0
    p95 = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    p99 = round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    avg_hall = round(sum(r["hallucination_rate"] for r in results) / len(results), 3) if results else 0.0
    avg_faith = round(sum(r["faithfulness"] for r in results) / len(results), 3) if results else 0.0
    avg_sim = round(sum(r["semantic_similarity"] for r in results) / len(results), 3) if results else 0.0
    avg_corr = round(sum(r["correctness"] for r in results) / len(results), 3) if results else 0.0
    avg_comp = round(sum(r["completeness"] for r in results) / len(results), 3) if results else 0.0
    avg_hit = round(sum(r["hit_rate"] for r in results) / len(results), 3) if results else 0.0
    avg_mrr = round(sum(r["mrr"] for r in results) / len(results), 3) if results else 0.0
    avg_prec = round(sum(r["context_precision"] for r in results) / len(results), 3) if results else 0.0
    avg_recall = round(sum(r["context_recall"] for r in results) / len(results), 3) if results else 0.0

    total_cost = round(sum(r["cost_usd"] for r in results), 6)
    avg_cost = round(total_cost / len(results), 6) if results else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    commit_sha = get_git_sha()
    branch_name = get_git_branch()

    summary_entry = {
        "timestamp": timestamp,
        "git_commit_hash": commit_sha,
        "branch": branch_name,
        "dataset_version": VERSION_DATASET,
        "prompt_version": VERSION_PROMPT,
        "retriever_version": VERSION_RETRIEVER,
        "embedding_model": VERSION_EMBEDDING,
        "llm_model": VERSION_LLM,
        "pass_rate": round(passed / len(questions) * 100, 1) if questions else 0.0,
        "hallucination_rate_avg": avg_hall,
        "avg_faithfulness": avg_faith,
        "avg_semantic_similarity": avg_sim,
        "avg_correctness": avg_corr,
        "avg_completeness": avg_comp,
        "avg_retrieval_hit_rate": avg_hit,
        "avg_retrieval_mrr": avg_mrr,
        "avg_context_precision": avg_prec,
        "avg_context_recall": avg_recall,
        "avg_latency_sec": avg_lat,
        "p50_latency_sec": p50,
        "p95_latency_sec": p95,
        "p99_latency_sec": p99,
        "total_cost_usd": total_cost,
        "avg_cost_usd": avg_cost,
        "total_questions": len(questions),
        "passed": passed,
        "failed": failed,
        "mode": "smoke" if smoke else "full",
        "judge_enabled": not no_judge
    }

    # Save detailed runs containing all diagnostics
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    detailed_run_path = os.path.join(EVAL_RESULTS_DIR, f"run_{timestamp}.json")
    
    full_output = {**summary_entry, "results": results}
    with open(detailed_run_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # Log metrics to centralized history tracking module
    from core.reporter import append_to_metrics_history
    append_to_metrics_history(summary_entry)

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE COMPLETED")
    logger.info(f"Mode:               {summary_entry['mode'].upper()}")
    logger.info(f"Pass Rate:          {summary_entry['pass_rate']}% ({passed}/{len(questions)})")
    logger.info(f"Retrieval Hit Rate: {round(avg_hit * 100, 1)}%")
    logger.info(f"Avg Faithfulness:   {avg_faith} (Hallucination: {avg_hall})")
    logger.info(f"Latency p50/p95:    {p50}s / {p95}s (SLA target: < 3.5s)")
    logger.info(f"Total API Cost:     ${total_cost:.5f}")
    logger.info(f"Saved Report:       {detailed_run_path}")
    logger.info("=" * 60)

    return summary_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tax RAG Eval Execution Platform.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast balanced subset (10 queries, 2/category). Use for CI checks."
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        dest="no_judge",
        help="Skip the LLM-as-a-judge step (zero extra API calls). "
             "Scoring via local semantic similarity only. Recommended for CI."
    )
    args = parser.parse_args()

    try:
        run_evaluation(smoke=args.smoke, no_judge=args.no_judge)
    except Exception as exc:
        logger.error(f"Fatal error in evaluation runner: {exc}")
        exit(1)
