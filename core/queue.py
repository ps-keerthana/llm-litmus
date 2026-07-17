import os
import csv
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple

from config import (
    DATASET_PATH, EVAL_RESULTS_DIR,
    VERSION_DATASET, VERSION_PROMPT, VERSION_RETRIEVER, VERSION_EMBEDDING, VERSION_LLM,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DB_PATH
)
from db.connection import get_db_connection, db_transaction
from core.retrieval import load_docs, build_vector_store, retrieve, evaluate_retrieval
from core.generator import generate_answer
from core.metrics import compute_semantic_similarity
from core.judge import llm_judge_evaluate
from core.utils import get_git_sha, get_git_branch, calculate_cost, logger
from core.attributor import attribute_failure, build_retrieval_diagnosis
import core.generator as _generator_mod


def enqueue_run(mode: str) -> str:
    """
    Reads the golden dataset, extracts questions (filtering deterministic smoke subset if mode is 'smoke'),
    initializes the run metadata, and inserts the queued tasks into the SQLite database.
    
    Returns the unique run_id.
    """
    # 1. Read dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Golden dataset not found at {DATASET_PATH}")

    questions = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    # 2. Filter if smoke mode
    if mode == "smoke":
        SMOKE_PER_CATEGORY = 2
        random.seed(42)
        categories: Dict[str, list] = {}
        for q in questions:
            cat = q["category"]
            categories.setdefault(cat, []).append(q)

        smoke_subset = []
        for cat, q_list in sorted(categories.items()):
            sample_size = min(len(q_list), SMOKE_PER_CATEGORY)
            smoke_subset.extend(random.sample(q_list, sample_size))
        questions = smoke_subset[:10]  # hard safety cap

    # 3. Create run registry
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    commit_sha = get_git_sha()
    branch_name = get_git_branch()

    with db_transaction() as conn:
        # Create run record
        conn.execute(
            """
            INSERT INTO eval_runs (run_id, status, mode, commit_sha, branch, metadata)
            VALUES (?, 'running', ?, ?, ?, ?);
            """,
            (run_id, mode, commit_sha, branch_name, json.dumps({}))
        )

        # Enqueue questions
        for q in questions:
            conn.execute(
                """
                INSERT INTO eval_queue (run_id, status, unique_id, question, ground_truth, category, difficulty, expected_sources)
                VALUES (?, 'pending', ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    q["unique_id"],
                    q["question"],
                    q["ground_truth"],
                    q["category"],
                    q["difficulty"],
                    q.get("expected_sources", "")
                )
            )

    logger.info(f"Enqueued {len(questions)} queries for run {run_id} ({mode.upper()} mode).")
    return run_id


def process_queue(run_id: str, no_judge: bool = False) -> Dict[str, Any]:
    """
    Background worker loop that fetches and executes pending queue tasks for a run.
    Re-runs RAG pipeline logic, records outputs/metrics, and updates database state.
    """
    # 1. Fetch pending items
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM eval_queue WHERE run_id = ? AND status = 'pending';",
        (run_id,)
    )
    tasks = cursor.fetchall()
    conn.close()

    if not tasks:
        logger.info(f"No pending tasks found for run {run_id}.")
        return {}

    # 2. Ingest Vector Store Index (once per run loop)
    logger.info("Initializing vector store collections index for queue processing...")
    chunks = load_docs()
    collection = build_vector_store(chunks)
    logger.info("Vector store ready. Beginning execution loops...")

    # Load golden dataset mapping for full metadata row values
    dataset_rows = {}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_rows[row["unique_id"]] = row

    results = []
    passed = 0
    failed = 0
    cached_hits = 0

    # Verification threshold settings
    MIN_SEMANTIC_SIM = 0.65
    MIN_LLM_CORRECTNESS = 0.75
    MAX_HALLUCINATION = 0.05

    for idx, task in enumerate(tasks):
        task_id = task["id"]
        q_id = task["unique_id"]
        question = task["question"]
        ground_truth = task["ground_truth"]
        category = task["category"]
        difficulty = task["difficulty"]
        
        row_meta = dataset_rows.get(q_id, {})
        expected_sources = task["expected_sources"] or ""
        expected_citations = row_meta.get("expected_citations", "")
        reasoning_type = row_meta.get("reasoning_type", "")
        tags = row_meta.get("tags", "")

        logger.info(f"[{idx+1}/{len(tasks)}] Processing {q_id} in queue...")

        # Update task status to processing
        with db_transaction() as conn:
            conn.execute(
                "UPDATE eval_queue SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE id = ?;",
                (task_id,)
            )

        try:
            # ── 1. Retrieval Step ─────────────────────────────────
            ret_start = time.time()
            retrieved_chunks, similarities, sources = retrieve(question, collection, top_k=DEFAULT_TOP_K)
            ret_latency = round(time.time() - ret_start, 3)

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

            if _generator_mod.WAS_LAST_CALL_CACHED:
                cached_hits += 1

            latency = round(max(0.0, elapsed - _generator_mod.LAST_API_SLEEP_TIME), 2)

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

            # Centralized Oracle Routing
            from core.judge import evaluate_with_oracle_routing
            judge_metrics, judge_p, judge_c, judge_called = evaluate_with_oracle_routing(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                context_chunks=retrieved_chunks,
                semantic_sim=semantic_sim,
                no_judge=no_judge,
                is_refusal=is_refusal
            )

            total_p = gen_p + judge_p
            total_c = gen_c + judge_c
            cost = calculate_cost(total_p, total_c)

            # Pass/Fail logic
            is_correct = semantic_sim >= 0.65 or (
                judge_metrics["correctness"] >= MIN_LLM_CORRECTNESS if isinstance(judge_metrics["correctness"], (int, float)) else False
            )
            is_faithful = (
                judge_metrics["hallucination"] <= MAX_HALLUCINATION if isinstance(judge_metrics["hallucination"], (int, float)) else True
            )
            status = "PASS" if is_correct and is_faithful else "FAIL"

            if status == "PASS":
                passed += 1
            else:
                failed += 1

            record = {
                "unique_id": q_id,
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "category": category,
                "difficulty": difficulty,
                "expected_sources": expected_sources,
                "retrieved_sources": json.dumps(sources),
                "retrieved_chunks": json.dumps(retrieved_chunks),
                "retrieved_similarities": json.dumps(similarities),
                "semantic_similarity": semantic_sim,
                "correctness": judge_metrics["correctness"],
                "faithfulness": judge_metrics["faithfulness"],
                "completeness": judge_metrics["completeness"],
                "hallucination_rate": judge_metrics["hallucination"],
                "judge_confidence": judge_metrics["confidence"],
                "judge_reasoning": judge_metrics["reasoning"],
                "hit_rate": ret_metrics["hit_rate"],
                "mrr": ret_metrics["mrr"],
                "context_precision": ret_metrics["context_precision"],
                "context_recall": ret_metrics["context_recall"],
                "latency_sec": latency,
                "cost_usd": cost,
                "status": status,
                "judge_enabled": not no_judge,
                "cached": _generator_mod.WAS_LAST_CALL_CACHED,
                "prompt_used": f"System prompt version {VERSION_PROMPT} plus top {DEFAULT_TOP_K} documents context."
            }

            # Failure attribution & diagnosis
            failure_category, attribution_reason = attribute_failure(record)
            retrieval_diagnosis = build_retrieval_diagnosis(record)

            record["failure_category"] = failure_category
            record["attribution_reason"] = attribution_reason
            record["retrieval_diagnosis"] = retrieval_diagnosis

            results.append(record)

            # Insert detailed results to SQLite
            with db_transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO eval_results (
                        run_id, unique_id, question, ground_truth, answer, category, difficulty, expected_sources,
                        retrieved_sources, retrieved_chunks, retrieved_similarities, semantic_similarity,
                        correctness, faithfulness, completeness, hallucination_rate, judge_confidence, judge_reasoning,
                        hit_rate, mrr, context_precision, context_recall, latency_sec, cost_usd, status,
                        failure_category, attribution_reason, retrieval_diagnosis, judge_enabled, cached, prompt_used,
                        diagnostic_report
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    );
                    """,
                    (
                        run_id, q_id, question, ground_truth, answer, category, difficulty, expected_sources,
                        json.dumps(sources), json.dumps(retrieved_chunks), json.dumps(similarities), semantic_sim,
                        record["correctness"], record["faithfulness"], record["completeness"], record["hallucination_rate"],
                        record["judge_confidence"], record["judge_reasoning"], record["hit_rate"], record["mrr"],
                        record["context_precision"], record["context_recall"], latency, cost, status,
                        failure_category, attribution_reason, json.dumps(retrieval_diagnosis),
                        1 if not no_judge else 0, 1 if record["cached"] else 0, record["prompt_used"],
                        json.dumps(record.get("diagnostic_report"))
                    )
                )

            # Update queue status
            with db_transaction() as conn:
                conn.execute(
                    "UPDATE eval_queue SET status = 'completed', updated_at = CURRENT_TIMESTAMP WHERE id = ?;",
                    (task_id,)
                )

        except Exception as e:
            logger.error(f"[Queue Execution Error] Question {q_id} failed: {e}")
            with db_transaction() as conn:
                conn.execute(
                    "UPDATE eval_queue SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?;",
                    (str(e), task_id)
                )

    # 3. Aggregate Calculations
    latencies = [r["latency_sec"] for r in results]
    p50 = round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0
    p95 = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    p99 = round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    def _safe_avg(results, key):
        nums = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if not nums:
            return "Not Evaluated"
        return round(sum(nums) / len(nums), 3)

    avg_hall = _safe_avg(results, "hallucination_rate")
    avg_faith = _safe_avg(results, "faithfulness")
    avg_sim = _safe_avg(results, "semantic_similarity")
    avg_corr = _safe_avg(results, "correctness")
    avg_comp = _safe_avg(results, "completeness")
    avg_hit = round(sum(r["hit_rate"] for r in results) / len(results), 3) if results else 0.0
    avg_mrr = round(sum(r["mrr"] for r in results) / len(results), 3) if results else 0.0
    avg_prec = round(sum(r["context_precision"] for r in results) / len(results), 3) if results else 0.0
    avg_recall = round(sum(r["context_recall"] for r in results) / len(results), 3) if results else 0.0

    total_cost = round(sum(r["cost_usd"] for r in results), 6)
    avg_cost = round(total_cost / len(results), 6) if results else 0.0

    commit_sha = get_git_sha()
    branch_name = get_git_branch()
    timestamp = run_id.replace("run_", "")
    cache_hit_rate = round(cached_hits / len(results) * 100, 1) if results else 0.0

    summary_entry = {
        "timestamp": timestamp,
        "git_commit_hash": commit_sha,
        "branch": branch_name,
        "dataset_version": VERSION_DATASET,
        "prompt_version": VERSION_PROMPT,
        "retriever_version": VERSION_RETRIEVER,
        "embedding_model": VERSION_EMBEDDING,
        "llm_model": VERSION_LLM,
        "pass_rate": round(passed / len(results) * 100, 1) if results else 0.0,
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
        "total_questions": len(results),
        "passed": passed,
        "failed": failed,
        "mode": "smoke" if len(results) <= 10 else "full",
        "judge_enabled": not no_judge,
        "cached_queries_count": cached_hits,
        "cache_hit_rate": cache_hit_rate,
        "evaluation_metadata": {
            "git_commit": commit_sha,
            "evaluation_timestamp": timestamp,
            "run_mode": "smoke" if len(results) <= 10 else "full",
            "model_name": VERSION_LLM,
            "prompt_version": VERSION_PROMPT,
            "dataset_version": VERSION_DATASET,
            "embedding_model": VERSION_EMBEDDING,
            "retriever_name": "ChromaDB vector store",
            "chunking_strategy": "Double newline delimiter",
            "judge_enabled": not no_judge,
            "cache_enabled": True,
            "cache_hit_rate": cache_hit_rate,
            "cached_queries_count": cached_hits
        }
    }

    # Save detailed run report back to files for legacy dashboard compat
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    detailed_run_path = os.path.join(EVAL_RESULTS_DIR, f"{run_id}.json")
    full_output = {**summary_entry, "results": results}
    with open(detailed_run_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # 4. Save summary metadata to eval_runs
    with db_transaction() as conn:
        conn.execute(
            """
            UPDATE eval_runs 
            SET status = 'completed', metadata = ?, completed_at = CURRENT_TIMESTAMP
            WHERE run_id = ?;
            """,
            (json.dumps(summary_entry), run_id)
        )

    # Appending history metrics
    from core.reporter import append_to_metrics_history
    append_to_metrics_history(summary_entry)

    logger.info("=" * 60)
    logger.info("EVALUATION INTELLIGENCE QUEUE PROCESSING COMPLETE")
    logger.info(f"Run ID:             {run_id}")
    logger.info(f"Pass Rate:          {summary_entry['pass_rate']}% ({passed}/{len(results)})")
    logger.info(f"Total API Cost:     ${total_cost:.5f}")
    logger.info("=" * 60)

    return summary_entry
