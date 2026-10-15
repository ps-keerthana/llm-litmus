"""
Tests for Phase 8: HTML Report & Dashboard Generation.
Verifies leaderboard generation, failure gallery rendering, and HTML formatting.
"""

from pathlib import Path
import pytest
from litmus_v2.report.generate_html_report import generate_html_report


class TestHTMLReportGeneration:
    @pytest.fixture
    def mock_suite_result(self):
        return {
            "dataset_name": "tax_benchmark_dev",
            "sample_size": 50,
            "grader_reports": {
                "similarity_baseline": {
                    "metrics": {
                        "accuracy": 0.65,
                        "f1": 0.62,
                        "cohens_kappa": 0.35,
                        "specificity": 0.40,
                        "latency_p95_ms": 15.0,
                        "cost_per_100_queries_usd": 0.0,
                    },
                    "sample_evaluations": [
                        {
                            "item_id": "item_1",
                            "question": "What is the 80C deduction limit?",
                            "ground_truth": "wrong",
                            "predicted_label": "correct",
                            "score": 0.92,
                            "is_correct_decision": False,
                            "latency_ms": 12.0,
                            "category": "number_swap",
                            "rationale": "High token overlap",
                        }
                    ],
                },
                "heuristic_gate": {
                    "metrics": {
                        "accuracy": 0.85,
                        "f1": 0.83,
                        "cohens_kappa": 0.72,
                        "specificity": 0.88,
                        "latency_p95_ms": 2.5,
                        "cost_per_100_queries_usd": 0.0,
                    },
                    "sample_evaluations": [],
                },
            },
            "pairwise_mcnemar": {
                "similarity_baseline__vs__heuristic_gate": {
                    "statistic": 8.52,
                    "p_value": 0.0035,
                    "significant_at_05": True,
                    "b_grader_a_only": 2,
                    "c_grader_b_only": 14,
                }
            },
            "ranking_by_f1": ["heuristic_gate", "similarity_baseline"],
        }

    def test_generates_valid_html_with_sections(self, mock_suite_result, tmp_path):
        out_file = tmp_path / "dashboard.html"
        html = generate_html_report(mock_suite_result, output_path=out_file)

        assert out_file.exists()
        assert "<!DOCTYPE html>" in html
        assert "litmus_v2" in html
        assert "Grader Trustworthiness Leaderboard" in html
        assert "heuristic_gate" in html
        assert "similarity_baseline" in html
        assert "McNemar" in html
        assert "Significant (p&lt;0.05)" in html

    def test_renders_failure_gallery_item(self, mock_suite_result):
        html = generate_html_report(mock_suite_result)
        assert "number_swap" in html
        assert "What is the 80C deduction limit?" in html
        assert "WRONG" in html
        assert "CORRECT" in html
