"""
Populate SQLite database from JSON files in eval_results/ directory.
This ensures the API serves all real historic runs, instead of only the mocked test run.
"""
import os
import sys
import glob
import json
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db.connection import get_db_connection, init_db

def populate():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get JSON files
    json_files = glob.glob(os.path.join("eval_results", "run_*.json"))
    print(f"Found {len(json_files)} run files in eval_results/")
    
    for f in json_files:
        run_id = os.path.splitext(os.path.basename(f))[0]
        
        # Check if already exists in DB
        cursor.execute("SELECT run_id FROM eval_runs WHERE run_id = ?;", (run_id,))
        if cursor.fetchone():
            print(f"Run {run_id} already exists in DB. Skipping.")
            continue
            
        print(f"Importing {run_id} into DB...")
        try:
            with open(f, "r", encoding="utf-8") as fp:
                run_data = json.load(fp)
        except Exception as exc:
            print(f"Error reading {f}: {exc}")
            continue
            
        # Build metadata dict
        meta_keys = [
            "timestamp", "git_commit_hash", "branch", "dataset_version",
            "prompt_version", "retriever_version", "embedding_model", "llm_model",
            "pass_rate", "hallucination_rate_avg", "avg_faithfulness",
            "avg_semantic_similarity", "avg_correctness", "avg_completeness",
            "avg_retrieval_hit_rate", "avg_retrieval_mrr", "avg_context_precision",
            "avg_context_recall", "avg_latency_sec", "p50_latency_sec",
            "p95_latency_sec", "p99_latency_sec", "total_cost_usd", "avg_cost_usd",
            "total_questions", "passed", "failed", "mode", "judge_enabled",
            "cached_queries_count", "cache_hit_rate", "evaluation_metadata"
        ]
        metadata = {}
        for k in meta_keys:
            if k in run_data:
                metadata[k] = run_data[k]
                
        # Insert run
        cursor.execute(
            """
            INSERT INTO eval_runs (run_id, status, mode, commit_sha, branch, metadata, created_at, completed_at)
            VALUES (?, 'completed', ?, ?, ?, ?, datetime('now'), datetime('now'));
            """,
            (
                run_id,
                run_data.get("mode", "smoke"),
                run_data.get("git_commit_hash", "unknown"),
                run_data.get("branch", "unknown"),
                json.dumps(metadata)
            )
        )
        
        # Insert results
        results = run_data.get("results", [])
        for idx, r in enumerate(results):
            # Normalize fields to handle legacy files
            unique_id = r.get("unique_id", f"Q{idx+1:03d}")
            question = r.get("question", "")
            ground_truth = r.get("ground_truth", "")
            answer = r.get("answer", "")
            category = r.get("category", "factual")
            difficulty = r.get("difficulty", "easy")
            expected_sources = r.get("expected_sources", "unknown")
            
            retrieved_sources = r.get("retrieved_sources", [])
            if isinstance(retrieved_sources, list):
                retrieved_sources = json.dumps(retrieved_sources)
            elif retrieved_sources is None:
                retrieved_sources = "[]"
                
            retrieved_chunks = r.get("retrieved_chunks", [])
            if isinstance(retrieved_chunks, list):
                retrieved_chunks = json.dumps(retrieved_chunks)
            elif retrieved_chunks is None:
                retrieved_chunks = "[]"
                
            retrieved_similarities = r.get("retrieved_similarities", [])
            if isinstance(retrieved_similarities, list):
                retrieved_similarities = json.dumps(retrieved_similarities)
            elif retrieved_similarities is None:
                retrieved_similarities = "[]"
                
            retrieval_diagnosis = r.get("retrieval_diagnosis", {})
            if isinstance(retrieval_diagnosis, dict):
                retrieval_diagnosis = json.dumps(retrieval_diagnosis)
            elif retrieval_diagnosis is None:
                retrieval_diagnosis = "{}"
                
            diagnostic_report = r.get("diagnostic_report")
            if isinstance(diagnostic_report, dict):
                diagnostic_report = json.dumps(diagnostic_report)
            else:
                diagnostic_report = None
                
            semantic_similarity = r.get("semantic_similarity", 0.0)
            correctness = r.get("correctness", r.get("semantic_similarity", 1.0))
            faithfulness = r.get("faithfulness", 1.0)
            completeness = r.get("completeness", 1.0)
            hallucination_rate = r.get("hallucination_rate", r.get("hallucination", 0.0))
            judge_confidence = r.get("judge_confidence", r.get("confidence", 1.0))
            judge_reasoning = r.get("judge_reasoning", r.get("relevancy_reasoning", "No detail."))
            
            hit_rate = r.get("hit_rate", r.get("llm_relevancy", 1.0))
            mrr = r.get("mrr", 1.0)
            context_precision = r.get("context_precision", 1.0)
            context_recall = r.get("context_recall", 1.0)
            
            latency_sec = r.get("latency_sec", 0.0)
            cost_usd = r.get("cost_usd", 0.0)
            status = r.get("status", "PASS")
            failure_category = r.get("failure_category", "N/A")
            attribution_reason = r.get("attribution_reason", "")
            
            judge_enabled = 1 if r.get("judge_enabled", False) else 0
            cached = 1 if r.get("cached", False) else 0
            prompt_used = r.get("prompt_used", "")
            
            cursor.execute(
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
                    run_id,
                    unique_id,
                    question,
                    ground_truth,
                    answer,
                    category,
                    difficulty,
                    expected_sources,
                    retrieved_sources,
                    retrieved_chunks,
                    retrieved_similarities,
                    semantic_similarity,
                    correctness if correctness != "Not Evaluated" else None,
                    faithfulness if faithfulness != "Not Evaluated" else None,
                    completeness if completeness != "Not Evaluated" else None,
                    hallucination_rate if hallucination_rate != "Not Evaluated" else None,
                    judge_confidence if judge_confidence != "Not Evaluated" else None,
                    judge_reasoning,
                    hit_rate,
                    mrr,
                    context_precision,
                    context_recall,
                    latency_sec,
                    cost_usd,
                    status,
                    failure_category,
                    attribution_reason,
                    retrieval_diagnosis,
                    judge_enabled,
                    cached,
                    prompt_used,
                    diagnostic_report
                )
            )
            
    conn.commit()
    conn.close()
    print("Populate database completed.")

if __name__ == "__main__":
    populate()
