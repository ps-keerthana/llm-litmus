"""
Evaluation Suite Runner (evaluate.py)

Phase 1: Domain-agnostic — no hard-coded tax strings; supports --docs-folder and --dataset flags.
Phase 2: Expanded retrieval metrics (nDCG@K, Precision@K, MAP, Coverage) flow through to results.
Phase 3: Multi-signal evaluation signals (token_f1, numbers_ok, judge_disagreement) in records.
Phase 5: --adversarial flag runs the robustness test suite against adversarial_dataset.csv.
Phase 10: Telemetry spans record per-phase latency for waterfall dashboard view.
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
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

import config
from config import (
    DATASET_PATH, EVAL_RESULTS_DIR, ADVERSARIAL_DATASET_PATH,
    VERSION_DATASET, VERSION_PROMPT, VERSION_RETRIEVER, VERSION_EMBEDDING, VERSION_LLM,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, THRESHOLD_P95_LATENCY,
)
from core.retrieval import load_docs, build_vector_store, retrieve, evaluate_retrieval
from core.generator import generate_answer
from core.metrics import compute_semantic_similarity
from core.judge import llm_judge_evaluate, evaluate_with_oracle_routing
from core.utils import get_git_sha, get_git_branch, calculate_cost, logger
from core.attributor import attribute_failure, build_retrieval_diagnosis
from core.telemetry import Tracer
import core.generator as _generator_mod
from db.connection import init_db
from core.providers import get_provider_client


def _load_dataset(path: str) -> List[Dict]:
    """Load a CSV evaluation dataset from the given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_evaluation(
    smoke: bool = False,
    no_judge: bool = False,
    no_diag: bool = False,
    provider: Optional[str] = None,
    adversarial: bool = False,
    docs_folder: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Dict[str, Any]:

    init_db()

    provider_client = get_provider_client(provider)
    active_provider_name = provider if provider else config.LLM_PROVIDER
    active_model_name = provider_client.get_model_name()

    logger.info("Using LLM Provider: '%s' (Model: '%s')", active_provider_name, active_model_name)
    logger.info("Domain: '%s' — %s", config.DOMAIN, config.DOMAIN_DESCRIPTION)

    # ── Knowledge Base Ingestion ─────────────────────────────────────────
    effective_docs_folder = docs_folder or config.DOCS_FOLDER
    logger.info("Ingesting knowledge base from '%s'...", effective_docs_folder)

    chunk_dicts = load_docs(effective_docs_folder)
    logger.info("Loaded %d chunks from documents.", len(chunk_dicts))

    collection = build_vector_store(chunk_dicts)
    logger.info("Vector store ready (%d chunks indexed).", collection.count())

    # ── Dataset Selection ─────────────────────────────────────────────────
    if adversarial:
        effective_dataset = dataset_path or ADVERSARIAL_DATASET_PATH
        logger.info("Running ADVERSARIAL robustness test suite: %s", effective_dataset)
    else:
        effective_dataset = dataset_path or DATASET_PATH

    questions = _load_dataset(effective_dataset)

    if smoke and not adversarial:
        SMOKE_PER_CATEGORY = 2
        random.seed(42)
        categories: Dict[str, list] = {}
        for q in questions:
            cat = q.get("category", "general")
            categories.setdefault(cat, []).append(q)
        smoke_subset = []
        for cat, q_list in sorted(categories.items()):
            smoke_subset.extend(random.sample(q_list, min(len(q_list), SMOKE_PER_CATEGORY)))
        questions = smoke_subset[:10]
        logger.info("SMOKE mode: %d queries (2/category, seed=42).", len(questions))
    elif smoke and adversarial:
        questions = questions[:5]
        logger.info("SMOKE + ADVERSARIAL mode: %d queries.", len(questions))
    else:
        logger.info("FULL mode: %d queries.", len(questions))

    if no_judge:
        logger.info("LLM judge DISABLED (--no-judge). Scoring via local signals only.")
    if no_diag:
        logger.info("Counterfactual diagnoser DISABLED (--no-diag).")

    # ── Telemetry Setup ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracer = Tracer(run_id=timestamp)

    # ── Thresholds ───────────────────────────────────────────────────────
    MIN_SEMANTIC_SIM = 0.65
    MIN_LLM_CORRECTNESS = 0.75
    MAX_HALLUCINATION = 0.05

    results = []
    passed = 0
    failed = 0
    cached_hits = 0

    for idx, row in enumerate(questions):
        q_id = row.get("unique_id", f"Q{idx+1:03d}")
        question = row.get("question", "")
        ground_truth = row.get("ground_truth", "")
        category = row.get("category", "general")
        difficulty = row.get("difficulty", "medium")
        expected_sources = row.get("expected_sources", "N/A")
        expected_citations = row.get("expected_citations", "")
        reasoning_type = row.get("reasoning_type", "direct_lookup")
        adversarial_category = row.get("adversarial_category", "") if adversarial else ""

        logger.info("[%d/%d] %s (%s | %s)...", idx + 1, len(questions), q_id, category, difficulty)

        # ── 1. Retrieval ─────────────────────────────────────────────────
        with tracer.span("retrieval", query_id=q_id) as ret_span:
            ret_start = time.time()
            retrieved_chunks, similarities, sources = retrieve(question, collection, top_k=DEFAULT_TOP_K)
            ret_latency = round(time.time() - ret_start, 3)
            ret_span.set("chunks_retrieved", len(retrieved_chunks))
            ret_span.set("top_similarity", similarities[0] if similarities else 0)
            ret_span.set("retrieval_latency_sec", ret_latency)

        ret_metrics = evaluate_retrieval(
            retrieved_sources=sources,
            expected_sources_str=expected_sources,
            retrieved_similarities=similarities,
            question=question,
            retrieved_chunks=retrieved_chunks,
            expected_citations_str=expected_citations,
            k=DEFAULT_TOP_K,
        )

        # ── 2. Generation ────────────────────────────────────────────────
        with tracer.span("generation", query_id=q_id) as gen_span:
            gen_start = time.time()
            answer, gen_p, gen_c = generate_answer(
                question, retrieved_chunks, temperature=DEFAULT_TEMPERATURE,
                provider_name=provider,
            )
            elapsed = time.time() - gen_start
            gen_span.set("prompt_tokens", gen_p)
            gen_span.set("completion_tokens", gen_c)
            gen_span.set("cached", _generator_mod.WAS_LAST_CALL_CACHED)

        if _generator_mod.WAS_LAST_CALL_CACHED:
            cached_hits += 1

        latency = round(max(0.0, elapsed - _generator_mod.LAST_API_SLEEP_TIME), 2)

        # ── 3. Evaluation ────────────────────────────────────────────────
        with tracer.span("evaluation", query_id=q_id) as eval_span:
            semantic_sim = compute_semantic_similarity(answer, ground_truth)
            eval_span.set("semantic_sim", semantic_sim)

            is_refusal = False
            if category == "out_of_scope":
                refusal_keywords = [
                    "don't have information", "do not have information",
                    "no information", "not mentioned", "not specified",
                    "not cover", "does not contain", "cannot find",
                    "doesn't provide", "don't provide", "do not provide",
                ]
                if any(kw in answer.lower() for kw in refusal_keywords):
                    is_refusal = True

            judge_metrics, judge_p, judge_c, judge_called = evaluate_with_oracle_routing(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                context_chunks=retrieved_chunks,
                semantic_sim=semantic_sim,
                no_judge=no_judge,
                is_refusal=is_refusal,
                provider=provider,
            )
            eval_span.set("judge_called", judge_called)
            eval_span.set("token_f1", judge_metrics.get("token_f1", "N/A"))

        total_p = gen_p + judge_p
        total_c = gen_c + judge_c
        cost = calculate_cost(total_p, total_c)

        # ── Pass/Fail Classification ──────────────────────────────────────
        is_correct = semantic_sim >= MIN_SEMANTIC_SIM or (
            isinstance(judge_metrics.get("correctness"), (int, float))
            and judge_metrics["correctness"] >= MIN_LLM_CORRECTNESS
        )
        is_faithful = (
            not isinstance(judge_metrics.get("hallucination"), (int, float))
            or judge_metrics["hallucination"] <= MAX_HALLUCINATION
        )
        status = "PASS" if is_correct and is_faithful else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        # ── Build Full Record ─────────────────────────────────────────────
        record: Dict[str, Any] = {
            "unique_id":             q_id,
            "question":              question,
            "ground_truth":          ground_truth,
            "answer":                answer,
            "category":              category,
            "difficulty":            difficulty,
            "reasoning_type":        reasoning_type,
            "tags":                  row.get("tags", ""),
            "expected_sources":      expected_sources,
            "retrieved_sources":     sources,
            "retrieved_chunks":      retrieved_chunks,
            "retrieved_similarities":similarities,
            "semantic_similarity":   semantic_sim,
            # Judge scores
            "correctness":           judge_metrics.get("correctness"),
            "faithfulness":          judge_metrics.get("faithfulness"),
            "completeness":          judge_metrics.get("completeness"),
            "hallucination_rate":    judge_metrics.get("hallucination"),
            "judge_confidence":      judge_metrics.get("confidence"),
            "judge_reasoning":       judge_metrics.get("reasoning"),
            # Phase 3: multi-signal signals
            "token_f1":              judge_metrics.get("token_f1"),
            "judge_disagreement":    judge_metrics.get("judge_disagreement", False),
            # Phase 1 retrieval metrics
            "hit_rate":              ret_metrics["hit_rate"],
            "recall_k":              ret_metrics["recall_k"],
            "mrr":                   ret_metrics["mrr"],
            "context_precision":     ret_metrics["context_precision"],
            "context_recall":        ret_metrics["context_recall"],
            # Phase 2 retrieval metrics
            "ndcg_at_k":             ret_metrics["ndcg_at_k"],
            "precision_at_k":        ret_metrics["precision_at_k"],
            "map_score":             ret_metrics["map_score"],
            "coverage":              ret_metrics["coverage"],
            # Latency / cost
            "latency_sec":           latency,
            "retrieval_latency_sec": ret_latency,
            "prompt_tokens":         total_p,
            "completion_tokens":     total_c,
            "cost_usd":              cost,
            "status":                status,
            "judge_enabled":         judge_called,
            "cached":                _generator_mod.WAS_LAST_CALL_CACHED,
            "prompt_used":           f"System prompt v{VERSION_PROMPT}, top-{DEFAULT_TOP_K} chunks.",
        }

        # Phase 5: adversarial metadata
        if adversarial and adversarial_category:
            record["adversarial_category"] = adversarial_category

        # ── Failure Attribution ───────────────────────────────────────────
        with tracer.span("attribution", query_id=q_id) as attr_span:
            if no_diag:
                failure_category = "N/A" if status == "PASS" else "Undiagnosed (--no-diag)"
                attribution_reason = (
                    "Query passed all quality checks." if status == "PASS"
                    else "Counterfactual diagnoser disabled."
                )
            else:
                failure_category, attribution_reason = attribute_failure(record)
            attr_span.set("failure_category", failure_category)

        retrieval_diagnosis = build_retrieval_diagnosis(record)
        record["failure_category"] = failure_category
        record["attribution_reason"] = attribution_reason
        record["retrieval_diagnosis"] = retrieval_diagnosis

        results.append(record)

    # ── Aggregates ─────────────────────────────────────────────────────────────
    # Only include real (uncached) latencies in percentile calculations.
    # Cache hits resolve in ~0ms (in-process dict lookup) and skew p50/p95/p99
    # to 0.0s, making the numbers meaningless as API latency indicators.
    real_latencies = [r["latency_sec"] for r in results if not r.get("cached", False)]
    latencies = real_latencies if real_latencies else [r["latency_sec"] for r in results]
    p50  = round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0
    p95  = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    p99  = round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    def _safe_avg(key, default=0.0):
        nums = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return "Not Evaluated" if not nums else round(sum(nums) / len(nums), 3)

    avg_hall  = _safe_avg("hallucination_rate")
    avg_faith = _safe_avg("faithfulness")
    avg_sim   = _safe_avg("semantic_similarity")
    avg_corr  = _safe_avg("correctness")
    avg_comp  = _safe_avg("completeness")
    avg_f1    = _safe_avg("token_f1")

    def _list_avg(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    avg_hit      = _list_avg("hit_rate")
    avg_mrr      = _list_avg("mrr")
    avg_prec     = _list_avg("context_precision")
    avg_recall   = _list_avg("context_recall")
    avg_ndcg     = _list_avg("ndcg_at_k")
    avg_prec_k   = _list_avg("precision_at_k")
    avg_map      = _list_avg("map_score")
    avg_coverage = _list_avg("coverage")

    total_cost = round(sum(r["cost_usd"] for r in results), 6)
    avg_cost   = round(total_cost / len(results), 6) if results else 0.0
    cache_hit_rate = round(cached_hits / len(results) * 100, 1) if results else 0.0

    commit_sha  = get_git_sha()
    branch_name = get_git_branch()

    summary_entry = {
        "timestamp":               timestamp,
        "git_commit_hash":         commit_sha,
        "branch":                  branch_name,
        "domain":                  config.DOMAIN,
        "domain_description":      config.DOMAIN_DESCRIPTION,
        "dataset_version":         VERSION_DATASET,
        "prompt_version":          VERSION_PROMPT,
        "retriever_version":       VERSION_RETRIEVER,
        "embedding_model":         VERSION_EMBEDDING,
        "llm_model":               active_model_name,
        "provider":                active_provider_name,
        "pass_rate":               round(passed / len(questions) * 100, 1) if questions else 0.0,
        "hallucination_rate_avg":  avg_hall,
        "avg_faithfulness":        avg_faith,
        "avg_semantic_similarity": avg_sim,
        "avg_correctness":         avg_corr,
        "avg_completeness":        avg_comp,
        "avg_token_f1":            avg_f1,
        # Phase 1 retrieval
        "avg_retrieval_hit_rate":  avg_hit,
        "avg_retrieval_mrr":       avg_mrr,
        "avg_context_precision":   avg_prec,
        "avg_context_recall":      avg_recall,
        # Phase 2 retrieval
        "avg_ndcg_at_k":           avg_ndcg,
        "avg_precision_at_k":      avg_prec_k,
        "avg_map_score":           avg_map,
        "avg_coverage":            avg_coverage,
        # Latency
        "avg_latency_sec":         avg_lat,
        "p50_latency_sec":         p50,
        "p95_latency_sec":         p95,
        "p99_latency_sec":         p99,
        # Cost
        "total_cost_usd":          total_cost,
        "avg_cost_usd":            avg_cost,
        # Run metadata
        "total_questions":         len(questions),
        "passed":                  passed,
        "failed":                  failed,
        "mode":                    "smoke" if smoke else ("adversarial" if adversarial else "full"),
        "judge_enabled":           not no_judge,
        "diag_enabled":            not no_diag,
        "cached_queries_count":    cached_hits,
        "cache_hit_rate":          cache_hit_rate,
        "trace_path":              tracer.get_trace_path(),
        "evaluation_metadata": {
            "git_commit":              commit_sha,
            "evaluation_timestamp":    timestamp,
            "run_mode":                "smoke" if smoke else ("adversarial" if adversarial else "full"),
            "provider":                active_provider_name,
            "model_name":              active_model_name,
            "prompt_version":          VERSION_PROMPT,
            "dataset_version":         VERSION_DATASET,
            "embedding_model":         VERSION_EMBEDDING,
            "retriever_name":          "ChromaDB vector store",
            "chunking_strategy":       config.CHUNK_STRATEGY,
            "top_k":                   DEFAULT_TOP_K,
            "judge_enabled":           not no_judge,
            "ensemble_judge_enabled":  config.JUDGE_ENSEMBLE_ENABLED,
            "cache_enabled":           True,
            "cache_hit_rate":          cache_hit_rate,
            "adversarial":             adversarial,
        },
    }

    # ── Save Results ──────────────────────────────────────────────────────
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    run_path = os.path.join(EVAL_RESULTS_DIR, f"run_{timestamp}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump({**summary_entry, "results": results}, f, indent=2)

    from core.reporter import append_to_metrics_history
    append_to_metrics_history(summary_entry)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("Mode:           %s | Domain: %s", summary_entry["mode"].upper(), config.DOMAIN)
    logger.info("Provider/Model: %s / %s", active_provider_name.upper(), active_model_name)
    logger.info("Pass Rate:      %.1f%% (%d/%d)", summary_entry["pass_rate"], passed, len(questions))
    logger.info("Hit Rate:       %.1f%% | nDCG@K: %.3f | MAP: %.3f", avg_hit * 100, avg_ndcg, avg_map)
    logger.info("Faithfulness:   %s  (Hallucination: %s)", avg_faith, avg_hall)
    logger.info("Token F1 avg:   %s", avg_f1)
    logger.info("Latency p50/p95:%ss / %ss", p50, p95)
    logger.info("Total Cost:     $%.5f", total_cost)
    logger.info("Trace:          %s", tracer.get_trace_path())
    logger.info("Report:         %s", run_path)
    logger.info("=" * 60)

    return summary_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-Litmus — RAG Evaluation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --smoke                         # quick CI smoke test
  python evaluate.py --provider groq                 # full run with Groq
  python evaluate.py --adversarial                   # robustness test suite
  python evaluate.py --docs-folder ./legal_docs \\
    --dataset ./legal_dataset.csv                    # evaluate a legal RAG system
  JUDGE_ENSEMBLE=true python evaluate.py --smoke     # run ensemble judging
        """,
    )
    parser.add_argument("--smoke", action="store_true",
        help="Run a fast balanced subset (10 queries, 2/category, seed=42). Use for CI.")
    parser.add_argument("--no-judge", action="store_true", dest="no_judge",
        help="Skip LLM-as-a-judge (zero extra API calls). Scoring via local signals only.")
    parser.add_argument("--no-diag", action="store_true", dest="no_diag",
        help="Skip counterfactual diagnoser for failed queries.")
    parser.add_argument("--provider", type=str, default=None,
        help="LLM provider: groq | ollama | openai | anthropic. Default: config.LLM_PROVIDER.")
    parser.add_argument("--adversarial", action="store_true",
        help="Run adversarial robustness test suite instead of standard benchmark.")
    parser.add_argument("--docs-folder", type=str, default=None, dest="docs_folder",
        help="Path to the knowledge base documents folder. Default: config.DOCS_FOLDER.")
    parser.add_argument("--dataset", type=str, default=None,
        help="Path to the evaluation dataset CSV. Default: config.DATASET_PATH.")

    args = parser.parse_args()

    try:
        run_evaluation(
            smoke=args.smoke,
            no_judge=args.no_judge,
            no_diag=args.no_diag,
            provider=args.provider,
            adversarial=args.adversarial,
            docs_folder=args.docs_folder,
            dataset_path=args.dataset,
        )
    except Exception as exc:
        logger.error("Fatal error in evaluation runner: %s", exc)
        raise SystemExit(1) from exc
