"""
Proactive Request Scheduler (core/scheduler.py)

Implements a dual token-bucket rate limiter backed by SQLite so that the
bucket state is shared across every process (CLI runner, FastAPI background
worker, counterfactual diagnoser) that touches the same eval_platform.db.

Design principles:
- Proactive: blocks BEFORE the API call, never reacts to 429s.
- Cross-process: SQLite BEGIN IMMEDIATE serialises all bucket writers.
- Dual-axis: enforces both Requests-Per-Minute (RPM) and Tokens-Per-Minute (TPM).
- Self-healing: after each call the actual token count refunds any over-estimate.
- No negative state: bucket values are floor-clamped at 0.

Public API (called only from core/generator.py):
    wait_s = acquire(estimated_tokens: int) -> float
    refund(estimated_tokens: int, actual_tokens: int) -> None
"""

import time
import sqlite3
import logging
from typing import Tuple

from config import (
    DB_PATH,
    SCHEDULER_BUCKET_ID,
    SCHEDULER_MAX_RPM,
    SCHEDULER_MAX_TPM,
    SCHEDULER_ESTIMATED_OUTPUT_TOKENS,
)

logger = logging.getLogger("LLMEvalPipeline")

# ── Internal constants ────────────────────────────────────────────────────────
_WINDOW_SEC: float = 60.0           # Refill window in seconds
_LOCK_RETRY_ATTEMPTS: int = 6       # Max retries when SQLite is locked
_LOCK_RETRY_SLEEP_SEC: float = 0.10 # Sleep between lock retries

# RPM/TPM refill rates (tokens or requests per second)
_RPM_PER_SEC: float = SCHEDULER_MAX_RPM / _WINDOW_SEC
_TPM_PER_SEC: float = SCHEDULER_MAX_TPM / _WINDOW_SEC


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _open_conn() -> sqlite3.Connection:
    """Opens a direct connection without row_factory (we read by index here)."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")   # Allows concurrent readers
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _ensure_bucket(conn: sqlite3.Connection) -> None:
    """
    Inserts the bucket row with full capacity if it does not already exist.
    Must be called inside an open IMMEDIATE transaction.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO rate_limit_buckets
            (bucket_id, last_refill_time, rpm_remaining, tpm_remaining, last_request_time)
        VALUES (?, ?, ?, ?, ?);
        """,
        (SCHEDULER_BUCKET_ID, time.time(), float(SCHEDULER_MAX_RPM), float(SCHEDULER_MAX_TPM), 0.0),
    )


def _read_and_refill(conn: sqlite3.Connection) -> Tuple[float, float, float, float]:
    """
    Reads the current bucket row, computes refill based on elapsed time, and
    returns (rpm_remaining, tpm_remaining, last_request_time, now).

    The refilled values are NOT written back here — the caller writes them
    atomically alongside the debit so there is exactly one write per acquire.
    """
    row = conn.execute(
        "SELECT last_refill_time, rpm_remaining, tpm_remaining, last_request_time "
        "FROM rate_limit_buckets WHERE bucket_id = ?;",
        (SCHEDULER_BUCKET_ID,),
    ).fetchone()

    now = time.time()
    if row is None:
        # Race: another process inserted between our INSERT OR IGNORE and this
        # SELECT — shouldn't happen inside IMMEDIATE, but handle defensively.
        return float(SCHEDULER_MAX_RPM), float(SCHEDULER_MAX_TPM), 0.0, now

    last_refill, rpm_rem, tpm_rem, last_req = row
    if last_req is None:
        last_req = 0.0
    elapsed = max(0.0, now - last_refill)

    rpm_rem = min(float(SCHEDULER_MAX_RPM), rpm_rem + elapsed * _RPM_PER_SEC)
    tpm_rem = min(float(SCHEDULER_MAX_TPM), tpm_rem + elapsed * _TPM_PER_SEC)

    return rpm_rem, tpm_rem, last_req, now


def _write_bucket(conn: sqlite3.Connection, rpm_rem: float, tpm_rem: float, last_req: float, now: float) -> None:
    """Persists the updated bucket state."""
    conn.execute(
        """
        UPDATE rate_limit_buckets
           SET last_refill_time = ?,
               rpm_remaining    = ?,
               tpm_remaining    = ?,
               last_request_time = ?
         WHERE bucket_id = ?;
        """,
        (now, max(0.0, rpm_rem), max(0.0, tpm_rem), last_req, SCHEDULER_BUCKET_ID),
    )


def _locked_transaction(fn):
    """
    Decorator-like helper: executes fn(conn) inside a BEGIN IMMEDIATE transaction
    with automatic retry on SQLite lock contention.
    Returns whatever fn returns.
    """
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        conn = _open_conn()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            _ensure_bucket(conn)
            result = fn(conn)
            conn.commit()
            return result
        except sqlite3.OperationalError as e:
            conn.rollback()
            if "locked" in str(e).lower() and attempt < _LOCK_RETRY_ATTEMPTS - 1:
                time.sleep(_LOCK_RETRY_SLEEP_SEC * (attempt + 1))
            else:
                raise
        finally:
            conn.close()


# ── Public API ─────────────────────────────────────────────────────────────────

def acquire(estimated_prompt_tokens: int) -> float:
    """
    Blocks proactively until the global dual token-bucket has capacity for one
    request consuming approximately ``estimated_prompt_tokens`` + the configured
    estimated output tokens.

    Cross-process safe: every acquire serialises through SQLite BEGIN IMMEDIATE.

    Returns
    -------
    float
        Total seconds spent waiting inside this call (for latency accounting).
    """
    estimated_total = min(SCHEDULER_MAX_TPM, estimated_prompt_tokens + SCHEDULER_ESTIMATED_OUTPUT_TOKENS)
    total_waited = 0.0

    while True:
        wait_needed = _compute_wait(estimated_total)

        if wait_needed <= 0.0:
            # Capacity confirmed — debit and return.
            _debit(estimated_total)
            return total_waited

        # Sleep outside the transaction to release the SQLite lock during the wait.
        logger.info(
            "[SCHEDULER] waiting %.1fs  bucket_remaining_rpm=%.1f  "
            "bucket_remaining_tpm=%.0f  estimated_tokens=%d",
            wait_needed,
            _peek_rpm(),
            _peek_tpm(),
            estimated_total,
        )
        time.sleep(wait_needed)
        total_waited += wait_needed


def refund(estimated_tokens: int, actual_tokens: int) -> None:
    """
    Returns the over-estimated token reservation to the bucket after the actual
    token count is known from the API response.

    If actual_tokens >= estimated_tokens (rare with conservative estimates) this
    is a no-op — we do NOT drain the bucket below zero retroactively.

    Parameters
    ----------
    estimated_tokens : int
        The total token estimate passed to acquire() (prompt + output estimate).
    actual_tokens : int
        The actual prompt + completion tokens reported by the provider response.
    """
    delta = estimated_tokens - actual_tokens
    if delta <= 0:
        return  # No refund needed

    def _do_refund(conn: sqlite3.Connection) -> None:
        rpm_rem, tpm_rem, last_req, now = _read_and_refill(conn)
        tpm_rem = min(float(SCHEDULER_MAX_TPM), tpm_rem + delta)
        _write_bucket(conn, rpm_rem, tpm_rem, last_req, now)

    try:
        _locked_transaction(_do_refund)
        logger.debug(
            "[SCHEDULER] refund  estimated_tokens=%d  actual_tokens=%d  refund=%d",
            estimated_tokens,
            actual_tokens,
            delta,
        )
    except Exception as exc:  # noqa: BLE001
        # Refund is best-effort — never crash a call because of a refund failure.
        logger.warning("[SCHEDULER] refund failed (non-fatal): %s", exc)


def drain() -> None:
    """
    Empties the token bucket to zero immediately.

    Call this after a reactive 429 to prevent the "bucket refills during
    Groq's retry-after sleep then fires immediately" failure mode.

    When a 429 occurs, we sleep for the provider's Retry-After duration.
    During that sleep (often 400–900 seconds), the SQLite bucket naturally
    refills to full capacity (since elapsed > 60s window). Without draining,
    the next acquire() returns 0 wait and fires immediately — triggering
    another 429 from the same depleted provider quota.

    After drain(), the next acquire() will wait for the bucket to refill
    at the normal 1/RPM rate (5–10s per request slot), giving Groq's
    actual quota window time to recover.
    """
    def _do_drain(conn: sqlite3.Connection) -> None:
        now = time.time()
        conn.execute(
            """
            UPDATE rate_limit_buckets
               SET rpm_remaining  = 0.0,
                   tpm_remaining  = 0.0,
                   last_refill_time = ?,
                   last_request_time = ?
             WHERE bucket_id = ?;
            """,
            (now, now, SCHEDULER_BUCKET_ID),
        )

    try:
        _locked_transaction(_do_drain)
        logger.info("[SCHEDULER] bucket drained after reactive 429 — next acquire() will pace normally")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SCHEDULER] drain failed (non-fatal): %s", exc)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _compute_wait(estimated_total_tokens: int) -> float:
    """
    Returns the number of seconds to wait before the bucket will have capacity,
    or 0.0 if capacity is already available.
    Runs inside a BEGIN IMMEDIATE transaction so the read is consistent.
    """
    from config import SCHEDULER_MIN_SPACING_SEC
    def _check(conn: sqlite3.Connection) -> float:
        rpm_rem, tpm_rem, last_req, now = _read_and_refill(conn)

        rpm_wait = 0.0
        tpm_wait = 0.0
        spacing_wait = 0.0

        if rpm_rem < 1.0:
            rpm_wait = (1.0 - rpm_rem) / _RPM_PER_SEC

        if tpm_rem < estimated_total_tokens:
            tpm_wait = (estimated_total_tokens - tpm_rem) / _TPM_PER_SEC

        if last_req > 0.0:
            elapsed = now - last_req
            if elapsed < SCHEDULER_MIN_SPACING_SEC:
                spacing_wait = SCHEDULER_MIN_SPACING_SEC - elapsed

        return max(rpm_wait, tpm_wait, spacing_wait)

    return _locked_transaction(_check)


def _debit(estimated_total_tokens: int) -> None:
    """Atomically debits 1 request slot and estimated_total_tokens from the bucket."""
    def _do_debit(conn: sqlite3.Connection) -> None:
        rpm_rem, tpm_rem, last_req, now = _read_and_refill(conn)
        rpm_rem = max(0.0, rpm_rem - 1.0)
        tpm_rem = max(0.0, tpm_rem - estimated_total_tokens)
        _write_bucket(conn, rpm_rem, tpm_rem, now, now)

    _locked_transaction(_do_debit)


def _peek_rpm() -> float:
    """Non-blocking read of current RPM remaining (for logging only)."""
    try:
        conn = _open_conn()
        try:
            row = conn.execute(
                "SELECT last_refill_time, rpm_remaining "
                "FROM rate_limit_buckets WHERE bucket_id = ?;",
                (SCHEDULER_BUCKET_ID,),
            ).fetchone()
            if row is None:
                return float(SCHEDULER_MAX_RPM)
            last_refill, rpm_rem = row
            elapsed = max(0.0, time.time() - last_refill)
            return min(float(SCHEDULER_MAX_RPM), rpm_rem + elapsed * _RPM_PER_SEC)
        finally:
            conn.close()
    except Exception:
        return float(SCHEDULER_MAX_RPM)


def _peek_tpm() -> float:
    """Non-blocking read of current TPM remaining (for logging only)."""
    try:
        conn = _open_conn()
        try:
            row = conn.execute(
                "SELECT last_refill_time, tpm_remaining "
                "FROM rate_limit_buckets WHERE bucket_id = ?;",
                (SCHEDULER_BUCKET_ID,),
            ).fetchone()
            if row is None:
                return float(SCHEDULER_MAX_TPM)
            last_refill, tpm_rem = row
            elapsed = max(0.0, time.time() - last_refill)
            return min(float(SCHEDULER_MAX_TPM), tpm_rem + elapsed * _TPM_PER_SEC)
        finally:
            conn.close()
    except Exception:
        return float(SCHEDULER_MAX_TPM)
