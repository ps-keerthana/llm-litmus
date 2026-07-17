import os
import sqlite3
from typing import Generator
from contextlib import contextmanager

from config import DB_PATH

def get_db_connection() -> sqlite3.Connection:
    """
    Returns a connection to the SQLite database.
    Enables foreign keys and sets row_factory to sqlite3.Row for dict-like access.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_transaction() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for safe database transactions, automatically committing or rolling back."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    """
    Initializes the database schema if the tables do not already exist.
    """
    with db_transaction() as conn:
        # Cache Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_cache (
                cache_key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_version TEXT,
                dataset_version TEXT,
                commit_sha TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Evaluation Runs Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT CHECK(status IN ('running', 'completed', 'failed')) DEFAULT 'running',
                mode TEXT NOT NULL,
                commit_sha TEXT,
                branch TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)

        # Evaluation Queue Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
                unique_id TEXT NOT NULL,
                question TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                category TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                expected_sources TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY(run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
            );
        """)

        # Evaluation Results Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                unique_id TEXT NOT NULL,
                question TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                answer TEXT,
                category TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                expected_sources TEXT,
                retrieved_sources TEXT,
                retrieved_chunks TEXT,
                retrieved_similarities TEXT,
                semantic_similarity REAL,
                correctness REAL,
                faithfulness REAL,
                completeness REAL,
                hallucination_rate REAL,
                judge_confidence REAL,
                judge_reasoning TEXT,
                hit_rate REAL,
                mrr REAL,
                context_precision REAL,
                context_recall REAL,
                latency_sec REAL,
                cost_usd REAL,
                status TEXT,
                failure_category TEXT,
                attribution_reason TEXT,
                retrieval_diagnosis TEXT,
                diagnostic_report TEXT,
                judge_enabled INTEGER,
                cached INTEGER,
                prompt_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
            );
        """)

        # Add index for speed on key lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_queue_run ON eval_queue(run_id);")

        # Scheduler: dual token-bucket state shared across all processes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limit_buckets (
                bucket_id          TEXT PRIMARY KEY,
                last_refill_time   REAL NOT NULL,     -- Unix wall-clock timestamp
                rpm_remaining      REAL NOT NULL,     -- Fractional request slots remaining
                tpm_remaining      REAL NOT NULL      -- Fractional token slots remaining
            );
        """)

        # ── Graceful migrations for existing databases ────────────────────────
        # Each ALTER TABLE is wrapped in try/except; already-present columns are silently ignored.

        # Step 5: diagnostic_report column
        try:
            conn.execute("ALTER TABLE eval_results ADD COLUMN diagnostic_report TEXT;")
        except sqlite3.OperationalError:
            pass  # Column already exists
