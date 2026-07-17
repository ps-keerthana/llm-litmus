import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from core.attributor import attribute_failure, build_retrieval_diagnosis, run_counterfactual_diagnosis
import core.attributor as attr_mod

# ── Mock generate_answer to avoid real API calls ─────────────
import core.generator as gen_mod

original_generate_answer = gen_mod.generate_answer

def mock_generate_good(*args, **kwargs):
    """Simulates a good LLM answer when given ground-truth context."""
    return "Section 80C deduction limit is Rs 1.5 lakh per financial year.", 50, 20

def mock_generate_bad(*args, **kwargs):
    """Simulates a completely wrong LLM answer even with ground-truth context (model limitation)."""
    return "The stock market index closed higher yesterday amid global uncertainty.", 50, 20

gen_mod.generate_answer = mock_generate_good  # default: good generator

def test_pass_record():
    print("Test 1: PASS record should return N/A...")
    record = {"status": "PASS", "question": "What is 80C limit?", "ground_truth": "Rs 1.5 lakh", "expected_sources": "section_80c_deductions.txt"}
    category, reason = attribute_failure(record)
    assert category == "N/A", f"Expected N/A, got: {category}"
    print(f"  Result: {category} | Reason: {reason}")
    print("  PASSED.")

def test_kb_gap():
    print("\nTest 2: Missing document should return Knowledge Base Gap...")
    record = {
        "status": "FAIL", "question": "What is LTCG?",
        "ground_truth": "Long term capital gains taxed at 10% above Rs 1 lakh.",
        "expected_sources": "nonexistent_doc.txt",
        "hit_rate": 0.0, "context_recall": 0.0, "correctness": 0.0,
        "faithfulness": 0.5, "hallucination_rate": 0.0
    }
    category, reason = attribute_failure(record)
    assert category == "Knowledge Base Gap", f"Expected KB Gap, got: {category}"
    print(f"  Result: {category} | Reason: {reason}")
    print("  PASSED.")

def test_retrieval_failure(mock_answer_fn):
    print("\nTest 3: Retrieval Failure (good generator, bad retrieval)...")
    gen_mod.generate_answer = mock_answer_fn

    record = {
        "status": "FAIL", "question": "What is the 80C limit?",
        "ground_truth": "Section 80C deduction limit is Rs 1.5 lakh per financial year.",
        "expected_sources": "section_80c_deductions.txt",
        "hit_rate": 0.0, "context_recall": 0.2, "correctness": 0.0,
        "faithfulness": 1.0, "hallucination_rate": 0.0,
        "answer": "I don't know."
    }
    category, reason, cf_answer, cf_sim = run_counterfactual_diagnosis(record)
    print(f"  Result: {category}")
    print(f"  Counterfactual Answer: {cf_answer}")
    print(f"  Counterfactual Similarity: {cf_sim:.3f}")
    assert category == "Retrieval Failure", f"Expected Retrieval Failure, got: {category}"
    print("  PASSED.")

def test_generation_failure(mock_answer_fn):
    print("\nTest 4: Generation Failure (bad generator even with ground-truth context)...")
    gen_mod.generate_answer = mock_answer_fn

    record = {
        "status": "FAIL", "question": "What is the 80C limit?",
        "ground_truth": "Section 80C deduction limit is Rs 1.5 lakh per financial year.",
        "expected_sources": "section_80c_deductions.txt",
        "hit_rate": 1.0, "context_recall": 0.8, "correctness": 0.0,
        "faithfulness": 1.0, "hallucination_rate": 0.0,
        "answer": "I don't know."
    }
    category, reason, cf_answer, cf_sim = run_counterfactual_diagnosis(record)
    print(f"  Result: {category}")
    print(f"  Counterfactual Answer: {cf_answer}")
    print(f"  Counterfactual Similarity: {cf_sim:.3f}")
    assert category == "LLM Generation Failure", f"Expected LLM Generation Failure, got: {category}"
    print("  PASSED.")

def test_retrieval_diagnosis():
    print("\nTest 5: Retrieval diagnosis flags...")
    record_good = {
        "hit_rate": 1.0, "context_recall": 0.9, "faithfulness": 0.9,
        "hallucination_rate": 0.01, "answer": "Rs 1.5 lakh."
    }
    diag = build_retrieval_diagnosis(record_good)
    assert diag["context_retrieved"] is True
    assert diag["context_sufficient"] is True
    assert diag["model_used_context"] is True
    assert diag["model_hallucinated"] is False
    print(f"  Good record flags: {diag}")

    record_bad = {
        "hit_rate": 0.0, "context_recall": 0.0, "faithfulness": 0.3,
        "hallucination_rate": 0.6, "answer": "Something random."
    }
    diag_bad = build_retrieval_diagnosis(record_bad)
    assert diag_bad["context_retrieved"] is False
    assert diag_bad["context_sufficient"] is False
    assert diag_bad["model_hallucinated"] is True
    print(f"  Bad record flags:  {diag_bad}")
    print("  PASSED.")

if __name__ == "__main__":
    test_pass_record()
    test_kb_gap()
    test_retrieval_failure(mock_generate_good)
    test_generation_failure(mock_generate_bad)
    test_retrieval_diagnosis()

    # Restore
    gen_mod.generate_answer = original_generate_answer

    print("\n[SUCCESS] Counterfactual Diagnoser Step 5 verified successfully!")
