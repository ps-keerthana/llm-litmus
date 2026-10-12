"""
tests/test_generator.py
========================
Unit tests for core/generator.py with a mocked provider.

Tests GenerationResult struct, cache hit/miss, error handling.
No real LLM calls or API keys needed.
The groq/openai/anthropic packages are mocked at module level so this
test runs even without them installed.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Mock the external SDK packages before any core.* import ──────────────────
# This prevents "No module named 'groq'" on machines without the venv.
import types
import importlib.util

def _mock_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, None)
    return mod

if "groq" not in sys.modules:
    _groq_mock = _mock_module("groq")
    _groq_mock.Groq = MagicMock()
    _groq_mock.RateLimitError = Exception
    sys.modules["groq"] = _groq_mock

if "openai" not in sys.modules:
    sys.modules["openai"] = _mock_module("openai")

if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = _mock_module("anthropic")

# Prevent chromadb import chain
import core.retrieval as _retrieval_mod
_mock_embedder = MagicMock()
_retrieval_mod.embedder = _mock_embedder

# Now safe to import
from core.generator import GenerationResult, generate_answer  # noqa: E402


class TestGenerationResult(unittest.TestCase):
    """Tests for the GenerationResult dataclass."""

    def test_as_tuple_returns_three_element_tuple(self):
        r = GenerationResult(answer="hello", prompt_tokens=10, completion_tokens=5)
        tup = r.as_tuple()
        self.assertEqual(tup, ("hello", 10, 5))

    def test_defaults(self):
        r = GenerationResult(answer="a", prompt_tokens=0, completion_tokens=0)
        self.assertEqual(r.sleep_time, 0.0)
        self.assertFalse(r.was_cached)

    def test_cached_flag(self):
        r = GenerationResult(answer="a", prompt_tokens=0, completion_tokens=0, was_cached=True)
        self.assertTrue(r.was_cached)


class TestGenerateAnswerWithMockedProvider(unittest.TestCase):
    """
    Tests generate_answer() with a fully mocked provider.
    No network, no API key, no LLM calls.
    """

    def _mock_completion_result(self, text="Mocked answer.", prompt_tokens=15, completion_tokens=8):
        mock = MagicMock()
        mock.text = text
        mock.prompt_tokens = prompt_tokens
        mock.completion_tokens = completion_tokens
        return mock

    @patch("core.generator.get_provider_client")
    @patch("core.cache.lookup_cache", return_value=None)
    @patch("core.cache.update_cache")
    def test_returns_generation_result(self, mock_update, mock_lookup, mock_get_provider):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = self._mock_completion_result()
        mock_provider.get_model_name.return_value = "mock-model"
        mock_get_provider.return_value = mock_provider

        result = generate_answer(
            question="What is Section 80C limit?",
            context_chunks=["Section 80C limit is ₹1.5 lakh."],
        )
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.answer, "Mocked answer.")
        self.assertEqual(result.prompt_tokens, 15)
        self.assertEqual(result.completion_tokens, 8)
        self.assertFalse(result.was_cached)

    @patch("core.generator.get_provider_client")
    @patch("core.cache.lookup_cache")
    @patch("core.cache.update_cache")
    def test_cache_hit_returns_cached_values(self, mock_update, mock_lookup, mock_get_provider):
        mock_lookup.return_value = {
            "answer": "Cached answer",
            "prompt_tokens": 20,
            "completion_tokens": 10,
        }
        mock_provider = MagicMock()
        mock_provider.get_model_name.return_value = "mock-model"
        mock_get_provider.return_value = mock_provider

        result = generate_answer("q", ["ctx"])
        self.assertEqual(result.answer, "Cached answer")
        self.assertTrue(result.was_cached)
        # Provider.complete should NOT be called on a cache hit
        mock_provider.complete.assert_not_called()

    @patch("core.generator.get_provider_client")
    @patch("core.cache.lookup_cache", return_value=None)
    @patch("core.cache.update_cache")
    def test_provider_failure_returns_error_string(self, mock_update, mock_lookup, mock_get_provider):
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = RuntimeError("API timeout")
        mock_provider.get_model_name.return_value = "mock-model"
        mock_get_provider.return_value = mock_provider

        result = generate_answer("q", ["ctx"])
        self.assertIn("Error", result.answer)
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)

    @patch("core.generator.get_provider_client")
    @patch("core.cache.lookup_cache", return_value=None)
    @patch("core.cache.update_cache")
    def test_as_tuple_unpacking(self, mock_update, mock_lookup, mock_get_provider):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = self._mock_completion_result(
            text="Answer text", prompt_tokens=5, completion_tokens=3
        )
        mock_provider.get_model_name.return_value = "mock-model"
        mock_get_provider.return_value = mock_provider

        result = generate_answer("q", ["ctx"])
        answer, p, c = result.as_tuple()
        self.assertEqual(answer, "Answer text")
        self.assertEqual(p, 5)
        self.assertEqual(c, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
