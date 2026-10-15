"""
tests/test_ci_gate.py
=====================
Unit tests for ci_gate.py:
  - Exit code 0 on passing run
  - Exit code 1 on hard failure
  - SKIPPED warning does NOT trigger exit code 1
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ci_gate


class TestCIGateExitCodes(unittest.TestCase):
    """Tests for ci_gate.main() exit codes."""

    @patch("ci_gate.load_latest_runs")
    @patch("ci_gate.write_markdown_summary")
    @patch("argparse.ArgumentParser.parse_args")
    def test_gate_passes_exit_code_zero(self, mock_args, mock_summary, mock_runs):
        mock_args.return_value = MagicMock(mode="full", run_id=None, compare=None)
        latest = {
            "pass_rate": 85.0,
            "avg_retrieval_hit_rate": 0.95,
            "hallucination_rate_avg": 0.02,
            "cost_total": 0.05,
            "p95_latency_sec": 1.2,
        }
        baseline = {
            "pass_rate": 84.0,
            "avg_retrieval_hit_rate": 0.94,
            "hallucination_rate_avg": 0.02,
            "cost_total": 0.05,
            "p95_latency_sec": 1.2,
        }
        mock_runs.return_value = (latest, baseline)

        with patch("ci_gate.check_regressions", return_value=[]):
            with self.assertRaises(SystemExit) as cm:
                ci_gate.main()
            self.assertEqual(cm.exception.code, 0)

    @patch("ci_gate.load_latest_runs")
    @patch("ci_gate.write_markdown_summary")
    @patch("argparse.ArgumentParser.parse_args")
    def test_gate_fails_exit_code_one(self, mock_args, mock_summary, mock_runs):
        mock_args.return_value = MagicMock(mode="full", run_id=None, compare=None)
        mock_runs.return_value = ({}, {})

        with patch("ci_gate.check_regressions", return_value=["Pass rate 50.0% < threshold 70.0%"]):
            with self.assertRaises(SystemExit) as cm:
                ci_gate.main()
            self.assertEqual(cm.exception.code, 1)

    @patch("ci_gate.load_latest_runs")
    @patch("ci_gate.write_markdown_summary")
    @patch("argparse.ArgumentParser.parse_args")
    def test_skipped_warning_does_not_fail_gate(self, mock_args, mock_summary, mock_runs):
        mock_args.return_value = MagicMock(mode="full", run_id=None, compare=None)
        mock_runs.return_value = ({}, {})

        # Warning starts with ⚠️  SKIPPED
        with patch("ci_gate.check_regressions", return_value=["⚠️  SKIPPED: Hallucination rate not evaluated"]):
            with self.assertRaises(SystemExit) as cm:
                ci_gate.main()
            self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
