"""
Tests for Phase 4: Graders.
Tests SimilarityGrader, HeuristicGrader, NLIGrader (mocked), and LLMJudge (mocked).
All tests are unit tests — no real model or API calls.
"""

import json
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from litmus_v2.graders.graders import (
    GradeLabel,
    GradeResult,
    SimilarityGrader,
    HeuristicGrader,
    NLIGrader,
    LLMJudge,
    _cosine_similarity,
    _tokenize,
    _polarity,
    _extract_amounts,
    _parse_judge_response,
    _build_judge_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE = "The maximum aggregate deduction under Section 80C is ₹1,50,000 per financial year."
PARAPHRASE = "A maximum deduction of ₹1,50,000 is available in a financial year under Section 80C."
NUMBER_SWAPPED = "The maximum aggregate deduction under Section 80C is ₹1,00,000 per financial year."
NEGATED = "The maximum aggregate deduction under Section 80C is NOT ₹1,50,000 per financial year."
OFF_TOPIC = "Under the GST framework, composite supply is taxed at the principal supply rate."
TRUNCATED = "The maximum aggregate deduction under Section 80C is"


# ─────────────────────────────────────────────────────────────────────────────
# Utility function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_tokenize_lowercases(self):
        tokens = _tokenize("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_handles_rupee(self):
        tokens = _tokenize("₹1,50,000")
        assert any("₹" in t or "1" in t for t in tokens)

    def test_cosine_identical(self):
        v = {"a": 1, "b": 2}
        assert math.isclose(_cosine_similarity(v, v), 1.0)

    def test_cosine_disjoint(self):
        assert _cosine_similarity({"a": 1}, {"b": 1}) == 0.0

    def test_cosine_empty(self):
        assert _cosine_similarity({}, {"a": 1}) == 0.0

    def test_polarity_positive(self):
        assert _polarity("You can claim this deduction.") == "positive"

    def test_polarity_negative(self):
        assert _polarity("You cannot claim this deduction.") == "negative"

    def test_extract_amounts_basic(self):
        amounts = _extract_amounts("The limit is ₹1,50,000 per year.")
        assert any("1,50,000" in a for a in amounts)

    def test_parse_judge_response_correct(self):
        raw = '{"label": "correct", "score": 0.9, "rationale": "Same meaning."}'
        label, score, rationale = _parse_judge_response(raw)
        assert label == GradeLabel.CORRECT
        assert score == 0.9

    def test_parse_judge_response_wrong(self):
        raw = '{"label": "wrong", "score": 0.1, "rationale": "Wrong amount."}'
        label, score, rationale = _parse_judge_response(raw)
        assert label == GradeLabel.WRONG

    def test_parse_judge_response_markdown_fences(self):
        raw = "```json\n{\"label\": \"correct\", \"score\": 0.85, \"rationale\": \"ok\"}\n```"
        label, score, rationale = _parse_judge_response(raw)
        assert label == GradeLabel.CORRECT

    def test_parse_judge_response_malformed(self):
        label, score, rationale = _parse_judge_response("not json at all")
        assert label == GradeLabel.ABSTAIN
        assert "Parse error" in rationale

    def test_parse_judge_clamps_score(self):
        raw = '{"label": "correct", "score": 1.5, "rationale": "ok"}'
        _, score, _ = _parse_judge_response(raw)
        assert score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SimilarityGrader
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilarityGrader:
    @pytest.fixture
    def grader(self):
        return SimilarityGrader(threshold=0.5)

    def test_identical_answer_is_correct(self, grader):
        result = grader.grade("Q?", REFERENCE, REFERENCE)
        assert result.label == GradeLabel.CORRECT
        assert result.score > 0.9

    def test_paraphrase_score(self, grader):
        result = grader.grade("Q?", REFERENCE, PARAPHRASE)
        # Paraphrase shares many key words — score should be reasonably high
        assert result.score > 0.3
        assert result.grader == "similarity_baseline"

    def test_off_topic_is_wrong(self, grader):
        result = grader.grade("Q?", REFERENCE, OFF_TOPIC)
        assert result.label == GradeLabel.WRONG

    def test_returns_grade_result(self, grader):
        result = grader.grade("Q?", REFERENCE, PARAPHRASE)
        assert isinstance(result, GradeResult)
        assert result.grader == "similarity_baseline"
        assert result.latency_ms is not None
        assert result.latency_ms >= 0.0

    def test_score_in_range(self, grader):
        for cand in [REFERENCE, PARAPHRASE, NUMBER_SWAPPED, OFF_TOPIC]:
            result = grader.grade("Q?", REFERENCE, cand)
            assert 0.0 <= result.score <= 1.0

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            SimilarityGrader(threshold=1.5)

    def test_low_threshold_more_correct(self):
        grader_strict = SimilarityGrader(threshold=0.9)
        grader_lenient = SimilarityGrader(threshold=0.1)
        result_strict = grader_strict.grade("Q?", REFERENCE, PARAPHRASE)
        result_lenient = grader_lenient.grade("Q?", REFERENCE, PARAPHRASE)
        # Lenient should be CORRECT, strict may be WRONG
        assert result_lenient.label == GradeLabel.CORRECT


# ─────────────────────────────────────────────────────────────────────────────
# HeuristicGrader
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicGrader:
    @pytest.fixture
    def grader(self):
        return HeuristicGrader()

    def test_exact_match_is_correct(self, grader):
        result = grader.grade("Q?", REFERENCE, REFERENCE)
        assert result.label == GradeLabel.CORRECT

    def test_number_swap_is_wrong(self, grader):
        result = grader.grade("Q?", REFERENCE, NUMBER_SWAPPED)
        assert result.label == GradeLabel.WRONG
        assert "amount_mismatch" in result.rationale

    def test_negation_flip_is_wrong(self, grader):
        result = grader.grade(
            "Q?",
            "You can claim this deduction.",
            "You cannot claim this deduction.",
        )
        assert result.label == GradeLabel.WRONG
        assert "polarity_mismatch" in result.rationale

    def test_off_topic_is_wrong(self, grader):
        result = grader.grade("Q?", REFERENCE, OFF_TOPIC)
        assert result.label == GradeLabel.WRONG

    def test_too_short_is_wrong(self, grader):
        result = grader.grade("Q?", REFERENCE, "Yes.")
        assert result.label == GradeLabel.WRONG
        assert "too_short" in result.rationale

    def test_paraphrase_may_pass(self, grader):
        """Paraphrase shares amounts and polarity — heuristic should pass it."""
        result = grader.grade("Q?", REFERENCE, PARAPHRASE)
        # With same amounts and positive polarity, should be CORRECT
        assert result.label == GradeLabel.CORRECT

    def test_returns_grade_result(self, grader):
        result = grader.grade("Q?", REFERENCE, REFERENCE)
        assert isinstance(result, GradeResult)
        assert result.grader == "heuristic_gate"

    def test_score_in_range(self, grader):
        for cand in [REFERENCE, PARAPHRASE, NUMBER_SWAPPED, OFF_TOPIC, "Yes."]:
            result = grader.grade("Q?", REFERENCE, cand)
            assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# NLIGrader (model mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestNLIGrader:
    def _make_grader(self):
        return NLIGrader()

    def test_abstain_when_model_unavailable(self):
        grader = self._make_grader()
        # Force pipeline to None (simulates missing transformers)
        grader._pipeline = None
        grader._load_error = "transformers not installed"
        # Patch _load_pipeline to not actually load
        grader._load_pipeline = lambda: None
        result = grader.grade("Q?", REFERENCE, PARAPHRASE, context="ctx")
        assert result.label == GradeLabel.ABSTAIN
        assert result.grader == "nli_faithfulness"

    def test_entailment_gives_correct(self):
        grader = self._make_grader()
        mock_pipeline = MagicMock(return_value=[
            {"label": "ENTAILMENT", "score": 0.92},
            {"label": "NEUTRAL", "score": 0.05},
            {"label": "CONTRADICTION", "score": 0.03},
        ])
        grader._pipeline = mock_pipeline
        result = grader.grade("Q?", REFERENCE, PARAPHRASE, context=REFERENCE)
        assert result.label == GradeLabel.CORRECT
        assert result.score > 0.5

    def test_contradiction_gives_wrong(self):
        grader = self._make_grader()
        mock_pipeline = MagicMock(return_value=[
            {"label": "ENTAILMENT", "score": 0.05},
            {"label": "NEUTRAL", "score": 0.05},
            {"label": "CONTRADICTION", "score": 0.90},
        ])
        grader._pipeline = mock_pipeline
        result = grader.grade("Q?", REFERENCE, NUMBER_SWAPPED, context=REFERENCE)
        assert result.label == GradeLabel.WRONG

    def test_neutral_gives_abstain(self):
        grader = self._make_grader()
        mock_pipeline = MagicMock(return_value=[
            {"label": "ENTAILMENT", "score": 0.3},
            {"label": "NEUTRAL", "score": 0.6},
            {"label": "CONTRADICTION", "score": 0.1},
        ])
        grader._pipeline = mock_pipeline
        result = grader.grade("Q?", REFERENCE, "Somewhat relevant answer.", context=REFERENCE)
        assert result.label == GradeLabel.ABSTAIN

    def test_score_in_range(self):
        grader = self._make_grader()
        grader._pipeline = None
        grader._load_pipeline = lambda: None
        result = grader.grade("Q?", REFERENCE, PARAPHRASE)
        assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LLMJudge (API mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMJudge:
    def test_temperature_zero_required(self):
        with pytest.raises(ValueError, match="temperature=0"):
            LLMJudge(temperature=0.5)

    def test_temperature_zero_allowed(self):
        judge = LLMJudge(temperature=0.0)
        assert judge.temperature == 0.0

    def test_correct_grade_from_mocked_ollama(self, tmp_path):
        judge = LLMJudge(
            provider="ollama",
            model="llama3.2:1b",
            cache_dir=str(tmp_path / "cache"),
        )
        mock_response = '{"label": "correct", "score": 0.9, "rationale": "Same factual content."}'
        judge._call_ollama = MagicMock(return_value=mock_response)

        result = judge.grade("Q?", REFERENCE, PARAPHRASE)
        assert result.label == GradeLabel.CORRECT
        assert result.score == 0.9
        assert result.grader == "llm_judge"

    def test_wrong_grade_from_mocked_ollama(self, tmp_path):
        judge = LLMJudge(
            provider="ollama",
            model="llama3.2:1b",
            cache_dir=str(tmp_path / "cache"),
        )
        mock_response = '{"label": "wrong", "score": 0.1, "rationale": "Wrong amount."}'
        judge._call_ollama = MagicMock(return_value=mock_response)

        result = judge.grade("Q?", REFERENCE, NUMBER_SWAPPED)
        assert result.label == GradeLabel.WRONG
        assert result.score == 0.1

    def test_response_is_cached(self, tmp_path):
        """Second call with same prompt should use cache (no API call)."""
        judge = LLMJudge(
            provider="ollama",
            model="llama3.2:1b",
            cache_dir=str(tmp_path / "cache"),
        )
        mock_response = '{"label": "correct", "score": 0.85, "rationale": "ok"}'
        judge._call_ollama = MagicMock(return_value=mock_response)

        result1 = judge.grade("Q?", REFERENCE, PARAPHRASE)
        result2 = judge.grade("Q?", REFERENCE, PARAPHRASE)

        # First call uses API, second uses cache
        assert judge._call_ollama.call_count == 1
        assert result2.cached is True

    def test_groq_missing_api_key_raises(self, tmp_path):
        judge = LLMJudge(
            provider="groq",
            model="llama3-70b-8192",
            api_key="",
            cache_dir=str(tmp_path / "cache"),
        )
        # Clear env var
        import os
        orig = os.environ.pop("GROQ_API_KEY", None)
        try:
            with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
                judge._call_groq("prompt", "system")
        finally:
            if orig:
                os.environ["GROQ_API_KEY"] = orig

    def test_error_handling_returns_error_grade(self, tmp_path):
        judge = LLMJudge(
            provider="ollama",
            model="llama3.2:1b",
            cache_dir=str(tmp_path / "cache"),
        )
        judge._call_ollama = MagicMock(side_effect=ConnectionError("Ollama not running"))
        result = judge.grade("Q?", REFERENCE, PARAPHRASE)
        assert result.label == GradeLabel.ERROR
        assert "error" in result.rationale.lower()

    def test_self_consistency_majority_vote(self, tmp_path):
        judge = LLMJudge(
            provider="ollama",
            model="llama3.2:1b",
            self_consistency_n=3,
            cache_dir=str(tmp_path / "cache"),
        )
        # First call correct, second call correct, third call wrong → majority=correct
        responses = [
            '{"label": "correct", "score": 0.9, "rationale": "r1"}',
            '{"label": "correct", "score": 0.85, "rationale": "r2"}',
            '{"label": "wrong", "score": 0.2, "rationale": "r3"}',
        ]
        judge._call_ollama = MagicMock(side_effect=responses * 10)
        result = judge.grade("Q?", REFERENCE, PARAPHRASE)
        assert result.label == GradeLabel.CORRECT

    def test_build_judge_prompt_contains_question(self):
        prompt = _build_judge_prompt("What is the limit?", REFERENCE, PARAPHRASE)
        assert "What is the limit?" in prompt
        assert REFERENCE in prompt
        assert PARAPHRASE in prompt

    def test_build_judge_prompt_with_context(self):
        prompt = _build_judge_prompt("Q?", REFERENCE, PARAPHRASE, context="Source chunk text.")
        assert "Source chunk text." in prompt
