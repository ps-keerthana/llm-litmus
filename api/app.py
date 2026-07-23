"""
FastAPI Backend Service (api/app.py)
Exposes the LLM evaluation platform data over a REST API.
Reads from SQLite (eval_platform.db) via db/connection.py and serves
typed JSON responses consumed by the Streamlit dashboard and future clients.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
load_dotenv()

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import (
    DB_PATH,
    METRICS_HISTORY_PATH,
    VERSION_DATASET,
    VERSION_EMBEDDING,
    VERSION_LLM,
    VERSION_PROMPT,
    VERSION_RETRIEVER,
)
from db.connection import db_transaction, get_db_connection, init_db

# ── App Bootstrap ──────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Eval Platform API",
    description="REST API for the Tax RAG evaluation pipeline.",
    version="1.0.0",
)

# Allow Streamlit and any browser-based client to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Ensure the database schema is initialized on server start."""
    init_db()


# ── Pydantic Response Models ────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    db_path: str
    dataset_version: str
    prompt_version: str
    retriever_version: str
    embedding_model: str
    llm_model: str


class RunSummary(BaseModel):
    run_id: str
    status: str
    mode: str
    commit_sha: Optional[str]
    branch: Optional[str]
    created_at: Optional[str]
    completed_at: Optional[str]
    metadata: Optional[Dict[str, Any]]


class QueryResult(BaseModel):
    """Per-query result record. Judge-optional fields may be 'Not Evaluated' strings."""

    id: int
    run_id: str
    unique_id: str
    question: str
    ground_truth: str
    answer: Optional[str]
    category: str
    difficulty: str
    expected_sources: Optional[str]
    retrieved_sources: Optional[Any]
    retrieved_chunks: Optional[Any]
    retrieved_similarities: Optional[Any]
    semantic_similarity: Optional[float]
    # These fields may be float OR the string 'Not Evaluated' when judge is disabled
    correctness: Optional[Union[float, str]]
    faithfulness: Optional[Union[float, str]]
    completeness: Optional[Union[float, str]]
    hallucination_rate: Optional[Union[float, str]]
    judge_confidence: Optional[Union[float, str]]
    judge_reasoning: Optional[str]
    hit_rate: Optional[float]
    mrr: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    latency_sec: Optional[float]
    cost_usd: Optional[float]
    status: str
    failure_category: Optional[str]
    attribution_reason: Optional[str]
    retrieval_diagnosis: Optional[Any]
    diagnostic_report: Optional[Any]
    judge_enabled: Optional[int]
    cached: Optional[int]
    prompt_used: Optional[str]
    created_at: Optional[str]

    @field_validator(
        "correctness", "faithfulness", "completeness",
        "hallucination_rate", "judge_confidence",
        mode="before",
    )
    @classmethod
    def _coerce_not_evaluated(cls, v: Any) -> Optional[Union[float, str]]:
        """Pass through floats; keep 'Not Evaluated' strings as-is; convert None."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # Keep string values like 'Not Evaluated' so callers can display them
        return v


class RunDetail(BaseModel):
    run_id: str
    status: str
    mode: str
    commit_sha: Optional[str]
    branch: Optional[str]
    created_at: Optional[str]
    completed_at: Optional[str]
    metadata: Optional[Dict[str, Any]]
    results: List[QueryResult]


class EnqueueRequest(BaseModel):
    mode: str = "smoke"
    no_judge: bool = True
    provider: Optional[str] = None  # 'groq' | 'ollama' | 'openai' | 'anthropic'; None = config default


class EnqueueResponse(BaseModel):
    run_id: str
    message: str
    mode: str


class QueueTaskStatus(BaseModel):
    id: int
    run_id: str
    status: str
    unique_id: str
    question: str
    category: str
    difficulty: str
    created_at: Optional[str]
    updated_at: Optional[str]
    error_message: Optional[str]


# ── Helper Functions ────────────────────────────────────────────────────────


def _parse_json_field(value: Any) -> Any:
    """Safely parse a JSON string field into a Python object."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _row_to_run_summary(row) -> RunSummary:
    return RunSummary(
        run_id=row["run_id"],
        status=row["status"],
        mode=row["mode"],
        commit_sha=row["commit_sha"],
        branch=row["branch"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        metadata=_parse_json_field(row["metadata"]),
    )


def _row_to_query_result(row) -> QueryResult:
    return QueryResult(
        id=row["id"],
        run_id=row["run_id"],
        unique_id=row["unique_id"],
        question=row["question"],
        ground_truth=row["ground_truth"],
        answer=row["answer"],
        category=row["category"],
        difficulty=row["difficulty"],
        expected_sources=row["expected_sources"],
        retrieved_sources=_parse_json_field(row["retrieved_sources"]),
        retrieved_chunks=_parse_json_field(row["retrieved_chunks"]),
        retrieved_similarities=_parse_json_field(row["retrieved_similarities"]),
        semantic_similarity=row["semantic_similarity"],
        correctness=row["correctness"],
        faithfulness=row["faithfulness"],
        completeness=row["completeness"],
        hallucination_rate=row["hallucination_rate"],
        judge_confidence=row["judge_confidence"],
        judge_reasoning=row["judge_reasoning"],
        hit_rate=row["hit_rate"],
        mrr=row["mrr"],
        context_precision=row["context_precision"],
        context_recall=row["context_recall"],
        latency_sec=row["latency_sec"],
        cost_usd=row["cost_usd"],
        status=row["status"],
        failure_category=row["failure_category"],
        attribution_reason=row["attribution_reason"],
        retrieval_diagnosis=_parse_json_field(row["retrieval_diagnosis"]),
        diagnostic_report=_parse_json_field(row["diagnostic_report"]),
        judge_enabled=row["judge_enabled"],
        cached=row["cached"],
        prompt_used=row["prompt_used"],
        created_at=row["created_at"],
    )


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Platform"])
def health_check() -> HealthResponse:
    """
    Returns service liveness status and platform version metadata.
    """
    return HealthResponse(
        status="ok",
        db_path=DB_PATH,
        dataset_version=VERSION_DATASET,
        prompt_version=VERSION_PROMPT,
        retriever_version=VERSION_RETRIEVER,
        embedding_model=VERSION_EMBEDDING,
        llm_model=VERSION_LLM,
    )


@app.get("/runs", response_model=List[RunSummary], tags=["Runs"])
def list_runs() -> List[RunSummary]:
    """
    Returns a list of all evaluation runs ordered by creation time (newest first).
    Includes only summary-level metadata — no per-query results.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC;"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [_row_to_run_summary(r) for r in rows]


@app.get("/runs/latest", response_model=RunDetail, tags=["Runs"])
def get_latest_run() -> RunDetail:
    """
    Returns the most recently completed evaluation run with all query results.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM eval_runs WHERE status = 'completed' ORDER BY completed_at DESC LIMIT 1;"
        )
        run_row = cursor.fetchone()
        if not run_row:
            raise HTTPException(status_code=404, detail="No completed runs found.")

        cursor.execute(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id;",
            (run_row["run_id"],),
        )
        result_rows = cursor.fetchall()
    finally:
        conn.close()

    return RunDetail(
        run_id=run_row["run_id"],
        status=run_row["status"],
        mode=run_row["mode"],
        commit_sha=run_row["commit_sha"],
        branch=run_row["branch"],
        created_at=run_row["created_at"],
        completed_at=run_row["completed_at"],
        metadata=_parse_json_field(run_row["metadata"]),
        results=[_row_to_query_result(r) for r in result_rows],
    )


@app.get("/runs/{run_id}", response_model=RunDetail, tags=["Runs"])
def get_run(run_id: str) -> RunDetail:
    """
    Returns full details for a specific run by run_id, including all query results.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM eval_runs WHERE run_id = ?;", (run_id,))
        run_row = cursor.fetchone()
        if not run_row:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        cursor.execute(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id;",
            (run_id,),
        )
        result_rows = cursor.fetchall()
    finally:
        conn.close()

    return RunDetail(
        run_id=run_row["run_id"],
        status=run_row["status"],
        mode=run_row["mode"],
        commit_sha=run_row["commit_sha"],
        branch=run_row["branch"],
        created_at=run_row["created_at"],
        completed_at=run_row["completed_at"],
        metadata=_parse_json_field(run_row["metadata"]),
        results=[_row_to_query_result(r) for r in result_rows],
    )


@app.get("/runs/{run_id}/results", response_model=List[QueryResult], tags=["Results"])
def get_run_results(run_id: str) -> List[QueryResult]:
    """
    Returns only the per-query results for a specific run (no run metadata envelope).
    Useful for lightweight data fetching when the summary is already cached.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Verify the run exists
        cursor.execute("SELECT run_id FROM eval_runs WHERE run_id = ?;", (run_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        cursor.execute(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id;",
            (run_id,),
        )
        result_rows = cursor.fetchall()
    finally:
        conn.close()

    return [_row_to_query_result(r) for r in result_rows]


@app.get("/history", response_model=List[Dict[str, Any]], tags=["History"])
def get_history() -> List[Dict[str, Any]]:
    """
    Returns the metrics history log for trend charts.
    Reads from metrics_history.json (maintained by core/reporter.py).
    Falls back to an empty list if the file does not exist.
    """
    history_path = METRICS_HISTORY_PATH
    if not os.path.exists(history_path):
        # Try from the parent directory (when run from inside api/)
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), METRICS_HISTORY_PATH)
        if os.path.exists(alt_path):
            history_path = alt_path
        else:
            return []

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


@app.get("/queue/{run_id}", response_model=List[QueueTaskStatus], tags=["Queue"])
def get_queue_status(run_id: str) -> List[QueueTaskStatus]:
    """
    Returns the queue task statuses for all items belonging to a specific run.
    Useful for monitoring in-progress runs.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM eval_queue WHERE run_id = ? ORDER BY id;",
            (run_id,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        QueueTaskStatus(
            id=r["id"],
            run_id=r["run_id"],
            status=r["status"],
            unique_id=r["unique_id"],
            question=r["question"],
            category=r["category"],
            difficulty=r["difficulty"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            error_message=r["error_message"],
        )
        for r in rows
    ]


@app.post("/runs/enqueue", response_model=EnqueueResponse, tags=["Runs"])
def enqueue_run_endpoint(
    request: EnqueueRequest, background_tasks: BackgroundTasks
) -> EnqueueResponse:
    """
    Triggers a new evaluation run asynchronously.
    Enqueues questions from the golden dataset into eval_queue,
    then processes the queue in a FastAPI BackgroundTask (non-blocking).

    Body:
        mode: "smoke" | "full"  (default: "smoke")
        no_judge: bool          (default: True — skips LLM judge for speed)
    """
    if request.mode not in ("smoke", "full"):
        raise HTTPException(
            status_code=422,
            detail="mode must be 'smoke' or 'full'.",
        )

    from core.queue import enqueue_run, process_queue

    try:
        run_id = enqueue_run(request.mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Run the worker in a background task so the HTTP response returns immediately.
    background_tasks.add_task(process_queue, run_id, request.no_judge, request.provider)

    return EnqueueResponse(
        run_id=run_id,
        message=f"Run '{run_id}' enqueued and processing started in background.",
        mode=request.mode,
    )
