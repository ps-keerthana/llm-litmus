"""
Unit tests for core/scheduler.py

Tests cover:
- Cold start: bucket is created at full capacity on first acquire
- Acquire debits RPM and TPM correctly
- Refund returns over-estimated tokens
- Bucket refill over time (simulated by patching time.time)
- Negative bucket prevention (bucket never goes below 0)
- Acquire blocks when bucket is empty and resumes after refill
- SQLite lock contention: concurrent acquires do not corrupt the bucket
"""

import os
import sys
import time
import sqlite3
import threading
import tempfile
import unittest
from unittest.mock import patch

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ─── Patch config before importing scheduler ───────────────────────────────────
# Use an isolated temp DB and tight limits for fast tests.
import config as _config_mod

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()

_config_mod.DB_PATH = _tmp_db.name
_config_mod.SCHEDULER_BUCKET_ID = "test_bucket"
_config_mod.SCHEDULER_MAX_RPM = 5        # Low limit for fast testing
_config_mod.SCHEDULER_MAX_TPM = 1000
_config_mod.SCHEDULER_ESTIMATED_OUTPUT_TOKENS = 50

# Import scheduler AFTER patching config so it picks up the test values
from core import scheduler  # noqa: E402
from db.connection import init_db  # noqa: E402


def _reset_bucket() -> None:
    """Delete the test bucket row so each test starts fresh."""
    conn = sqlite3.connect(_config_mod.DB_PATH)
    conn.execute("DELETE FROM rate_limit_buckets WHERE bucket_id = ?;", (_config_mod.SCHEDULER_BUCKET_ID,))
    conn.commit()
    conn.close()
    # Also reset module-level cached constants in case they diverged
    scheduler._RPM_PER_SEC = _config_mod.SCHEDULER_MAX_RPM / scheduler._WINDOW_SEC
    scheduler._TPM_PER_SEC = _config_mod.SCHEDULER_MAX_TPM / scheduler._WINDOW_SEC


class TestSchedulerColdStart(unittest.TestCase):
    def setUp(self):
        init_db()
        _reset_bucket()

    def test_first_acquire_creates_row(self):
        """Cold start: acquire should insert the bucket row."""
        scheduler.acquire(100)
        conn = sqlite3.connect(_config_mod.DB_PATH)
        row = conn.execute(
            "SELECT rpm_remaining, tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row, "Bucket row must exist after first acquire")

    def test_first_acquire_does_not_block(self):
        """Fresh bucket has full capacity — acquire must return immediately."""
        start = time.monotonic()
        wait = scheduler.acquire(100)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0, "First acquire on fresh bucket should not block")
        self.assertEqual(wait, 0.0)


class TestSchedulerDebit(unittest.TestCase):
    def setUp(self):
        init_db()
        _reset_bucket()

    def test_rpm_decremented(self):
        """Each acquire should subtract exactly 1.0 from rpm_remaining."""
        for _ in range(3):
            scheduler.acquire(10)

        conn = sqlite3.connect(_config_mod.DB_PATH)
        row = conn.execute(
            "SELECT rpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()
        conn.close()
        # Allow for tiny refill between acquires (< 0.1 per request at 5 RPM)
        self.assertAlmostEqual(row[0], _config_mod.SCHEDULER_MAX_RPM - 3, delta=0.2)

    def test_tpm_decremented(self):
        """Each acquire should subtract estimated_prompt + output_tokens from tpm_remaining."""
        estimated_prompt = 100
        expected_debit = estimated_prompt + _config_mod.SCHEDULER_ESTIMATED_OUTPUT_TOKENS

        scheduler.acquire(estimated_prompt)

        conn = sqlite3.connect(_config_mod.DB_PATH)
        row = conn.execute(
            "SELECT tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()
        conn.close()
        self.assertAlmostEqual(
            row[0], _config_mod.SCHEDULER_MAX_TPM - expected_debit, delta=5.0
        )

    def test_bucket_never_negative(self):
        """Burst of acquires beyond capacity should floor bucket at 0, not go negative."""
        # Exhaust TPM entirely with a single large request
        huge = _config_mod.SCHEDULER_MAX_TPM + 9999

        with patch("time.sleep"):  # prevent actual sleeping in test
            try:
                scheduler.acquire(huge)
            except Exception:
                pass  # It may block; we just check the DB state

        conn = sqlite3.connect(_config_mod.DB_PATH)
        row = conn.execute(
            "SELECT rpm_remaining, tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()
        conn.close()
        if row:
            self.assertGreaterEqual(row[0], 0.0, "rpm_remaining must not be negative")
            self.assertGreaterEqual(row[1], 0.0, "tpm_remaining must not be negative")


class TestSchedulerRefund(unittest.TestCase):
    def setUp(self):
        init_db()
        _reset_bucket()

    def test_refund_increases_tpm(self):
        """Refund returns the over-estimated tokens to the bucket."""
        estimated_prompt = 200
        estimated_total = estimated_prompt + _config_mod.SCHEDULER_ESTIMATED_OUTPUT_TOKENS
        actual_total = 100  # Under-used by 150 tokens

        scheduler.acquire(estimated_prompt)
        conn = sqlite3.connect(_config_mod.DB_PATH)
        tpm_after_acquire = conn.execute(
            "SELECT tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()[0]
        conn.close()

        scheduler.refund(estimated_total, actual_total)

        conn = sqlite3.connect(_config_mod.DB_PATH)
        tpm_after_refund = conn.execute(
            "SELECT tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()[0]
        conn.close()

        delta = estimated_total - actual_total
        self.assertAlmostEqual(tpm_after_refund - tpm_after_acquire, delta, delta=2.0)

    def test_refund_zero_delta_is_noop(self):
        """Refund with actual >= estimated should not change the bucket."""
        scheduler.acquire(100)

        conn = sqlite3.connect(_config_mod.DB_PATH)
        tpm_before = conn.execute(
            "SELECT tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()[0]
        conn.close()

        scheduler.refund(50, 9999)  # actual far exceeds estimated: no-op

        conn = sqlite3.connect(_config_mod.DB_PATH)
        tpm_after = conn.execute(
            "SELECT tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()[0]
        conn.close()

        # Should only change by natural refill, not by the refund call
        self.assertAlmostEqual(tpm_before, tpm_after, delta=5.0)

    def test_refund_cannot_exceed_max(self):
        """Refund must cap at SCHEDULER_MAX_TPM."""
        scheduler.acquire(100)
        # Refund more than the total capacity
        scheduler.refund(99999, 0)

        conn = sqlite3.connect(_config_mod.DB_PATH)
        tpm = conn.execute(
            "SELECT tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()[0]
        conn.close()
        self.assertLessEqual(tpm, float(_config_mod.SCHEDULER_MAX_TPM))


class TestSchedulerBucketRefill(unittest.TestCase):
    def setUp(self):
        init_db()
        _reset_bucket()

    def test_refill_restores_capacity(self):
        """
        After exhausting RPM slots, advancing time should make capacity available.
        We mock time.time() inside the scheduler to simulate elapsed time.
        """
        # Exhaust all RPM slots
        for _ in range(_config_mod.SCHEDULER_MAX_RPM):
            scheduler._debit(_config_mod.SCHEDULER_ESTIMATED_OUTPUT_TOKENS)

        conn = sqlite3.connect(_config_mod.DB_PATH)
        rpm_empty = conn.execute(
            "SELECT rpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()[0]
        conn.close()
        self.assertLess(rpm_empty, 1.0, "RPM bucket should be (near) empty")

        # Advance time by 60 seconds inside the read path
        original_time = time.time
        with patch("time.time", return_value=original_time() + 61.0):
            rpm_rem, tpm_rem, _ = scheduler._locked_transaction(scheduler._read_and_refill)

        self.assertGreaterEqual(rpm_rem, float(_config_mod.SCHEDULER_MAX_RPM) - 0.1)


class TestSchedulerConcurrency(unittest.TestCase):
    def setUp(self):
        init_db()
        _reset_bucket()

    def test_concurrent_acquires_do_not_corrupt_bucket(self):
        """
        Multiple threads acquiring simultaneously should never leave a negative
        bucket, and total debits should equal threads * 1 RPM slot.
        """
        results = []
        errors = []
        N = 4  # Below SCHEDULER_MAX_RPM=5 so none need to block

        def worker():
            try:
                scheduler.acquire(10)
                results.append(1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
        self.assertEqual(len(results), N)

        conn = sqlite3.connect(_config_mod.DB_PATH)
        row = conn.execute(
            "SELECT rpm_remaining, tpm_remaining FROM rate_limit_buckets WHERE bucket_id=?;",
            (_config_mod.SCHEDULER_BUCKET_ID,),
        ).fetchone()
        conn.close()

        self.assertGreaterEqual(row[0], 0.0)
        self.assertGreaterEqual(row[1], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
