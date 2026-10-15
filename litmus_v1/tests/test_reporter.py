"""
tests/test_reporter.py
======================
Unit tests for core/reporter.py::check_regressions()

Tests all 5 regression conditions plus the AND->OR fix for latency.
All tests are pure logic — no LLM calls, no file I/O.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.reporter import check_regressions

# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_run(**overrides):
    """Return a minimal valid base-run dict."""
    base = {
        "pass_rate": 88.0,
        "hallucination_rate_avg": 0.02,
        "p95_latency_sec": 2.0,
        "avg_cost_usd": 0.00005,
        "avg_retrieval_hit_rate": 0.97,
        "provider": "groq",
    }
    base.update(overrides)
    return base


def _latest_run(**overrides):
    """Return a minimal valid latest-run dict."""
    run = _base_run()
    run.update(overrides)
    return run


class TestAbsoluteThresholds(unittest.TestCase):
    """Absolute threshold checks that block regardless of previous run."""

    def test_pass_rate_below_threshold_fails(self):
        latest = _latest_run(pass_rate=65.0)
        failures = check_regressions(latest, _base_run())
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(any("Pass rate" in f for f in hard))

    def test_pass_rate_at_threshold_passes(self):
        # Use a baseline also at 70.0 so the regression check doesn't fire.
        # At exactly the threshold (not below it), the absolute gate should pass.
        base = _base_run(pass_rate=70.0)
        latest = _latest_run(pass_rate=70.0)
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertFalse(any("below absolute threshold" in f for f in hard))

    def test_hallucination_above_threshold_fails(self):
        latest = _latest_run(hallucination_rate_avg=0.06)
        failures = check_regressions(latest, _base_run())
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(any("hallucination" in f.lower() for f in hard))

    def test_hallucination_not_evaluated_adds_warning(self):
        """When judge is disabled, hallucination is 'Not Evaluated' -> SKIPPED warning."""
        latest = _latest_run(hallucination_rate_avg="Not Evaluated")
        failures = check_regressions(latest, _base_run())
        skipped = [f for f in failures if "SKIPPED" in f]
        self.assertTrue(len(skipped) > 0, "Expected a SKIPPED warning for disabled judge")

    def test_hallucination_not_evaluated_does_not_hard_fail(self):
        """A SKIPPED warning must not be a hard gate failure."""
        latest = _latest_run(hallucination_rate_avg="Not Evaluated")
        failures = check_regressions(latest, _base_run())
        hard = [f for f in failures if "SKIPPED" not in f]
        # Only a SKIPPED warning, not a hard hallucination failure
        self.assertFalse(
            any("hallucination rate" in f.lower() and "exceeds" in f.lower() for f in hard)
        )

    def test_hit_rate_below_threshold_fails(self):
        latest = _latest_run(avg_retrieval_hit_rate=0.75)
        failures = check_regressions(latest, _base_run())
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(any("retrieval" in f.lower() or "hit rate" in f.lower() for f in hard))


class TestLatencyRegressionOrLogic(unittest.TestCase):
    """
    The AND->OR fix: latency regression should fire when EITHER
    the % limit OR the abs limit is exceeded.
    """

    def test_large_pct_small_abs_triggers(self):
        """50% slowdown on a fast baseline (delta < 0.3s abs) should still fire (OR logic)."""
        base = _base_run(p95_latency_sec=0.4)   # fast baseline
        latest = _latest_run(p95_latency_sec=0.6, provider="groq")  # 50% up, +0.2s abs
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(
            any("latency" in f.lower() for f in hard),
            f"Expected latency regression but got: {hard}"
        )

    def test_small_pct_large_abs_triggers(self):
        """10% slowdown on a slow baseline (abs > 0.3s) should fire (OR logic)."""
        base = _base_run(p95_latency_sec=4.0)
        latest = _latest_run(p95_latency_sec=4.45, provider="groq")  # ~11% up, +0.45s abs
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(
            any("latency" in f.lower() for f in hard),
            f"Expected latency regression but got: {hard}"
        )

    def test_small_pct_small_abs_does_not_trigger(self):
        """5% slowdown, +0.1s — should not fire."""
        base = _base_run(p95_latency_sec=2.0)
        latest = _latest_run(p95_latency_sec=2.1, provider="groq")
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertFalse(any("latency" in f.lower() for f in hard))

    def test_latency_not_checked_for_ollama(self):
        """Ollama is a local provider — latency SLA should not apply."""
        base = _base_run(p95_latency_sec=1.0, provider="ollama")
        latest = _latest_run(p95_latency_sec=30.0, provider="ollama")
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertFalse(
            any("latency" in f.lower() for f in hard),
            "Latency gate must not fire for Ollama provider"
        )


class TestPassRateRegression(unittest.TestCase):
    """Pass rate relative regression (drop > 5pp)."""

    def test_regression_over_5pp_fails(self):
        base = _base_run(pass_rate=88.0)
        latest = _latest_run(pass_rate=82.0)  # -6pp
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(any("pass rate" in f.lower() or "regression" in f.lower() for f in hard))

    def test_regression_exactly_5pp_does_not_fail(self):
        base = _base_run(pass_rate=88.0)
        latest = _latest_run(pass_rate=83.0)  # exactly -5pp (at limit, not over)
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertFalse(
            any("regression" in f.lower() and "pass rate" in f.lower() for f in hard),
            f"Exactly at limit should not trigger regression: {hard}"
        )


class TestCostRegression(unittest.TestCase):
    """Cost relative regression (increase > 20%)."""

    def test_cost_above_20pct_fails(self):
        base = _base_run(avg_cost_usd=0.00005)
        latest = _latest_run(avg_cost_usd=0.00007)  # +40%
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(any("cost" in f.lower() for f in hard))

    def test_cost_under_20pct_passes(self):
        base = _base_run(avg_cost_usd=0.00005)
        latest = _latest_run(avg_cost_usd=0.000055)  # +10%
        failures = check_regressions(latest, base)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertFalse(any("cost" in f.lower() for f in hard))


class TestNoBaselineRun(unittest.TestCase):
    """When no baseline is available, only absolute thresholds should apply."""

    def test_none_baseline_does_not_crash(self):
        latest = _latest_run(pass_rate=85.0)
        # Should not raise
        failures = check_regressions(latest, None)
        self.assertIsInstance(failures, list)

    def test_none_baseline_still_catches_absolute_failure(self):
        latest = _latest_run(pass_rate=60.0)
        failures = check_regressions(latest, None)
        hard = [f for f in failures if "SKIPPED" not in f]
        self.assertTrue(any("pass rate" in f.lower() for f in hard))


if __name__ == "__main__":
    unittest.main(verbosity=2)
