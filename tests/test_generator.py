"""
tests/test_generator.py
========================
Unit tests for core/generator.py with a mocked provider.

Tests GenerationResult struct, cache hit/miss, error handling.
No real LLM calls or API keys needed.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestGenerationResult(unittest.TestCase):
    """Tests for the GenerationResult dataclass."""

    def test_as_tuple_returns_three_element_tuple(self):
        from core.generator import GenerationResult
        r = GenerationResult(answer="hello", prompt_tokens=10, completion_tokens=5)
        tup = r.as_tuple()
        self.assertEqual(tup, ("hello", 10, 5))

    def test_defaults(self):
        from core.generator import GenerationResult
        r = GenerationResult(answer="a", prompt_tokens=0, completion_tokens=0)
        self.assertEqual(r.sleep_time, 0.0)
        self.assertFalse(r.was_cached)

    def test_cached_flag(self):
        from core.generator import GenerationResult
        r = GenerationResult(answer="a", prompt_tokens=0, completion_tokens=0, was_cached=True)
        self.assertTrue(r.was_cached)


class TestGenerateAnswerWithMockedProvider(unittest.TestCase):
    """
    Tests generate_answer() with a fully mocked provider.
    No network, no API key, no LLM calls.
    """

    def _mock_provider(self, text="Mocked answer.", prompt_tokens=15, completion_tokens=8):
        """Build a mock provider client."""
        from core.providers import CompletionResult
        mock_result = MagicMock(spec=CompletionResult)
        mock_result.text = text
        mock_result.prompt_tokens = prompt_tokens
        mock_result.completion_tokens = completion_tokens

        mock_provider = MagicMock()
        mock_provider.complete.return_value = mock_result
        mock_provider.get_model_name.return_value = "mock-model"
        return mock_provider

    @patch("core.generator.get_provider_client")
    @patch("core.generator.lookup_cache", return_value=None)
    @patch("core.generator.update_cache")
    def test_returns_generation_result(self, mock_update, mock_lookup, mock_get_provider):
        mock_get_provider.return_value = self._mock_provider()
        from core.generator import generate_answer
        result = generate_answer(
            question="What is Section 80C limit?",
            context_chunks=["Section 80C limit is ₹1.5 lakh."],
        )
        self.assertEqual(result.answer, "Mocked answer.")
        self.assertEqual(result.prompt_tokens, 15)
        self.assertEqual(result.completion_tokens, 8)
        self.assertFalse(result.was_cached)

    @patch("core.generator.get_provider_client")
    @patch("core.generator.lookup_cache")
    @patch("core.generator.update_cache")
    def test_cache_hit_returns_cached_values(self, mock_update, mock_lookup, mock_get_provider):
        mock_lookup.return_value = {
            "answer": "Cached answer",
            "prompt_tokens": 20,
            "completion_tokens": 10,
        }
        mock_get_provider.return_value = self._mock_provider()
        from core.generator import generate_answer
        result = generate_answer("q", ["ctx"])
        self.assertEqual(result.answer, "Cached answer")
        self.assertTrue(result.was_cached)
        # Provider should NOT be called on a cache hit
        mock_get_provider.return_value.complete.assert_not_called()

    @patch("core.generator.get_provider_client")
    @patch("core.generator.lookup_cache", return_value=None)
    @patch("core.generator.update_cache")
    def test_provider_failure_returns_error_string(self, mock_update, mock_lookup, mock_get_provider):
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = RuntimeError("API timeout")
        mock_provider.get_model_name.return_value = "mock-model"
        mock_get_provider.return_value = mock_provider

        from core.generator import generate_answer
        result = generate_answer("q", ["ctx"])
        self.assertIn("Error", result.answer)
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)

    @patch("core.generator.get_provider_client")
    @patch("core.generator.lookup_cache", return_value=None)
    @patch("core.generator.update_cache")
    def test_as_tuple_unpacking(self, mock_update, mock_lookup, mock_get_provider):
        mock_get_provider.return_value = self._mock_provider(
            text="Answer text", prompt_tokens=5, completion_tokens=3
        )
        from core.generator import generate_answer
        result = generate_answer("q", ["ctx"])
        answer, p, c = result.as_tuple()
        self.assertEqual(answer, "Answer text")
        self.assertEqual(p, 5)
        self.assertEqual(c, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
