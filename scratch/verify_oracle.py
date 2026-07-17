import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.judge import evaluate_with_oracle_routing

def test_oracle() -> None:
    # 1. High similarity test (Auto-Pass)
    print("Testing Oracle Auto-Pass (similarity = 0.95)...")
    metrics, p_tokens, c_tokens, judge_called = evaluate_with_oracle_routing(
        question="What is the standard deduction?",
        answer="Rs 50,000.",
        ground_truth="Rs 50,000.",
        context_chunks=["Standard deduction is Rs 50,000."],
        semantic_sim=0.95,
        no_judge=False
    )
    print("Metrics returned:", metrics)
    print(f"Judge called: {judge_called}")
    assert judge_called is False, "Judge should not be called for high similarity"
    assert metrics["correctness"] == 1.0, "Expected correctness = 1.0"
    assert "High semantic similarity" in metrics["reasoning"]

    # 2. Low similarity test (Auto-Fail)
    print("\nTesting Oracle Auto-Fail (similarity = 0.15)...")
    metrics, p_tokens, c_tokens, judge_called = evaluate_with_oracle_routing(
        question="What is the standard deduction?",
        answer="Apples and oranges.",
        ground_truth="Rs 50,000.",
        context_chunks=["Standard deduction is Rs 50,000."],
        semantic_sim=0.15,
        no_judge=False
    )
    print("Metrics returned:", metrics)
    print(f"Judge called: {judge_called}")
    assert judge_called is False, "Judge should not be called for low similarity"
    assert metrics["correctness"] == 0.0, "Expected correctness = 0.0"
    assert "Low semantic similarity" in metrics["reasoning"]

    # 3. Ambiguous region (requires LLM judge)
    # We mock llm_judge_evaluate during this test to avoid hitting Groq.
    print("\nTesting Oracle Routing on ambiguous region (similarity = 0.60)...")
    import core.judge as judge_mod
    original_judge = judge_mod.llm_judge_evaluate
    
    mock_called = False
    def mock_judge(q, a, g, c):
        nonlocal mock_called
        mock_called = True
        return {"correctness": 0.8, "faithfulness": 1.0, "completeness": 0.9, "hallucination": 0.0, "confidence": 0.9, "reasoning": "Mocked"}, 100, 50

    judge_mod.llm_judge_evaluate = mock_judge

    try:
        metrics, p_tokens, c_tokens, judge_called = evaluate_with_oracle_routing(
            question="What is the HRA limit?",
            answer="HRA is 40% of salary.",
            ground_truth="HRA is 50% for metro cities.",
            context_chunks=["HRA is 50% for metro, 40% non-metro."],
            semantic_sim=0.60,
            no_judge=False
        )
        print("Metrics returned:", metrics)
        print(f"Judge called: {judge_called}")
        assert judge_called is True, "Judge should be called for ambiguous similarity"
        assert mock_called is True, "Expected mock judge function to be called"
        assert metrics["correctness"] == 0.8, "Expected mock correctness score"
    finally:
        # Restore original judge function
        judge_mod.llm_judge_evaluate = original_judge

    print("\n[SUCCESS] Oracle-efficient routing Step 4 verified successfully!")

if __name__ == "__main__":
    test_oracle()
