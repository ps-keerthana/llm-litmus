"""
scripts/ablation_study.py
=========================
Ablation Study: Semantic Similarity vs. 4-Signal Quality Gate.

Demonstrates why semantic embedding similarity alone is insufficient for RAG
quality gates, and how LLM-Litmus's 4-signal gate catches critical factual errors
(numeric substitutions and negation flips) that fool embedding cosine similarity.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.metrics import (
    compute_semantic_similarity,
    compute_token_f1,
    numbers_consistent,
    is_contradicting,
    multi_signal_auto_pass,
)


def run_ablation():
    test_cases = [
        {
            "id": "CASE_1_CORRECT",
            "name": "Factually Correct Answer",
            "ground_truth": "The maximum deduction limit under Section 80C is ₹1,50,000 per financial year.",
            "answer": "Under Section 80C, individual taxpayers can claim a deduction limit of up to ₹1.5 lakh annually.",
            "expected_verdict": "SHOULD_PASS",
        },
        {
            "id": "CASE_2_WRONG_NUMBER",
            "name": "Substituted Number (High Similarity Hallucination)",
            "ground_truth": "The maximum deduction limit under Section 80C is ₹1,50,000 per financial year.",
            "answer": "The maximum deduction limit under Section 80C is ₹5,00,000 per financial year.",
            "expected_verdict": "SHOULD_FAIL",
        },
        {
            "id": "CASE_3_FLIPPED_NEGATION",
            "name": "Flipped Polarity / Negation Contradiction",
            "ground_truth": "HRA exemption can be claimed by salaried employees paying rent.",
            "answer": "HRA exemption cannot be claimed by salaried employees paying rent.",
            "expected_verdict": "SHOULD_FAIL",
        },
        {
            "id": "CASE_4_WRONG_CLAIM_AMOUNT",
            "name": "Wrong Section 87A Rebate Amount",
            "ground_truth": "Under the new regime, Section 87A rebate is available for taxable income up to ₹7,00,000.",
            "answer": "Under the new regime, Section 87A rebate is available for taxable income up to ₹10,00,000.",
            "expected_verdict": "SHOULD_FAIL",
        },
        {
            "id": "CASE_5_ELIGIBILITY_FLIP",
            "name": "Eligibility Inversion",
            "ground_truth": "Salaried individuals who opt for the new regime are eligible for standard deduction of ₹75,000.",
            "answer": "Salaried individuals who opt for the new regime are ineligible for standard deduction of ₹75,000.",
            "expected_verdict": "SHOULD_FAIL",
        },
    ]

    print("=" * 80)
    print("ABLATION STUDY: SEMANTIC SIMILARITY ALONE vs. 4-SIGNAL QUALITY GATE")
    print("=" * 80)
    print(f"{'Case ID':<24} | {'Sim':<6} | {'Sim Only':<9} | {'4-Signal':<9} | {'Caught By':<22}")
    print("-" * 80)

    sim_only_correct_count = 0
    gate_correct_count = 0

    SEMANTIC_PASS_THRESHOLD = 0.85

    for case in test_cases:
        gt = case["ground_truth"]
        ans = case["answer"]

        sim = compute_semantic_similarity(ans, gt)
        f1 = compute_token_f1(ans, gt)
        num_ok = numbers_consistent(ans, gt)
        neg_ok = not is_contradicting(ans, gt)

        auto_pass, signals = multi_signal_auto_pass(ans, gt, semantic_sim=sim)

        sim_passed = sim >= SEMANTIC_PASS_THRESHOLD
        gate_passed = auto_pass

        # Check which signals failed
        caught_by = []
        if not signals["numbers_ok"]:
            caught_by.append("Numeric Check")
        if not signals["negation_ok"]:
            caught_by.append("Negation Check")
        if not signals["lexical_ok"]:
            caught_by.append(f"Token F1 ({f1:.2f})")
        if not signals["semantic_ok"]:
            caught_by.append(f"Semantic ({sim:.2f})")

        caught_str = ", ".join(caught_by) if caught_by else "All Signals Agree"

        print(f"{case['id']:<24} | {sim:0.3f}  | {'PASS' if sim_passed else 'FAIL':<9} | {'PASS' if gate_passed else 'FAIL':<9} | {caught_str:<22}")

        # Correct verdict assessment
        should_pass = case["expected_verdict"] == "SHOULD_PASS"
        if sim_passed == should_pass:
            sim_only_correct_count += 1
        if gate_passed == should_pass:
            gate_correct_count += 1

    total = len(test_cases)
    print("=" * 80)
    print(f"Accuracy across adversarial / factual sanity set ({total} cases):")
    print(f"  - Semantic Similarity Alone: {sim_only_correct_count}/{total} ({sim_only_correct_count/total*100:.0f}%)")
    print(f"  - 4-Signal Multi-Signal Gate: {gate_correct_count}/{total} ({gate_correct_count/total*100:.0f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation()
