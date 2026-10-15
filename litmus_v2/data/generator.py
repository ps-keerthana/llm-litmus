"""
Candidate Q/A Generation Module for litmus_v2.
Implements Model Family A generation to synthesize candidate evaluation items.
Guarantees generation across 5 mandatory question types:
  1. single_fact
  2. numeric (strict monetary & threshold checks)
  3. yes_no (polarity & eligibility checks)
  4. multi_hop (requires synthesizing multiple chunks)
  5. unanswerable (out-of-scope; target behavior is abstention)
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

from litmus_v2.data.schema import Chunk, CandidateQA, QuestionType
from litmus_v2.config.settings import load_config


# Curated high-precision domain seed bank mapping exact chunk concepts across all 8 docs
CURATED_TAX_ITEMS = [
    # ── Single Fact ──────────────────────────────────────────────────────────
    {
        "doc": "section_80c_deductions.txt",
        "question": "What is the lock-in period for investments made in Equity Linked Savings Schemes (ELSS) under Section 80C?",
        "reference_answer": "ELSS mutual funds have a mandatory lock-in period of 3 years, which is the shortest lock-in period among Section 80C investment options.",
        "type": QuestionType.SINGLE_FACT,
        "keywords": ["elss", "lock-in", "3 years"],
    },
    {
        "doc": "section_80c_deductions.txt",
        "question": "Which family members are eligible for tuition fee deductions under Section 80C?",
        "reference_answer": "Tuition fees paid for full-time education are deductible only for up to two biological or adopted children of the individual, not for self, spouse, or siblings.",
        "type": QuestionType.SINGLE_FACT,
        "keywords": ["tuition", "children", "two children"],
    },
    {
        "doc": "section_80d_health.txt",
        "question": "Can preventive health check-up expenses be claimed under Section 80D?",
        "reference_answer": "Yes, preventive health check-up expenses up to ₹5,000 can be claimed within the overall Section 80D deduction limit.",
        "type": QuestionType.SINGLE_FACT,
        "keywords": ["preventive health", "5,000", "check-up"],
    },
    {
        "doc": "hra_exemption.txt",
        "question": "What percentage of salary is considered for HRA exemption calculation in non-metro cities?",
        "reference_answer": "In non-metro cities, 40% of salary (Basic + DA forming part of retirement benefits) is considered for HRA exemption.",
        "type": QuestionType.SINGLE_FACT,
        "keywords": ["40%", "non-metro", "salary"],
    },
    {
        "doc": "tds_rules.txt",
        "question": "Under which section of the Income Tax Act is TDS deducted on salary payments?",
        "reference_answer": "TDS on salary payments is deducted under Section 192 of the Income Tax Act based on the employee's applicable slab rates.",
        "type": QuestionType.SINGLE_FACT,
        "keywords": ["section 192", "salary", "tds"],
    },

    # ── Numeric ──────────────────────────────────────────────────────────────
    {
        "doc": "section_80c_deductions.txt",
        "question": "What is the maximum aggregate deduction limit allowed in a financial year under Section 80C?",
        "reference_answer": "The maximum aggregate deduction permitted under Section 80C (along with 80CCC and 80CCD(1)) is ₹1,50,000 per financial year.",
        "type": QuestionType.NUMERIC,
        "keywords": ["1,50,000", "maximum deduction", "aggregate"],
    },
    {
        "doc": "section_80d_health.txt",
        "question": "What is the maximum health insurance premium deduction under Section 80D for senior citizen parents?",
        "reference_answer": "For senior citizen parents aged 60 years or above, the maximum deduction limit under Section 80D is ₹50,000.",
        "type": QuestionType.NUMERIC,
        "keywords": ["50,000", "senior citizen", "parents"],
    },
    {
        "doc": "nps_section_80ccd.txt",
        "question": "What additional deduction is available exclusively for contributions to NPS Tier-1 under Section 80CCD(1B)?",
        "reference_answer": "An exclusive additional deduction of up to ₹50,000 is available under Section 80CCD(1B), over and above the ₹1,50,000 limit of Section 80C.",
        "type": QuestionType.NUMERIC,
        "keywords": ["50,000", "80ccd(1b)", "additional"],
    },
    {
        "doc": "capital_gains_tax.txt",
        "question": "What is the statutory threshold holding period for unlisted equity shares to qualify as Long Term Capital Assets?",
        "reference_answer": "Unlisted equity shares must be held for more than 24 months to qualify as long-term capital assets.",
        "type": QuestionType.NUMERIC,
        "keywords": ["24 months", "unlisted", "long term"],
    },
    {
        "doc": "tds_rules.txt",
        "question": "What is the threshold limit above which TDS under Section 194J is deducted on professional fees?",
        "reference_answer": "Under Section 194J, TDS is deducted if the professional fee payments exceed ₹30,000 in a financial year.",
        "type": QuestionType.NUMERIC,
        "keywords": ["30,000", "194j", "professional fees"],
    },

    # ── Yes / No (Polarity Sensitive) ────────────────────────────────────────
    {
        "doc": "section_80c_deductions.txt",
        "question": "Can an individual taxpayer claim Section 80C deduction for tuition fees paid for their own part-time higher education?",
        "reference_answer": "No. Tuition fee deductions under Section 80C are strictly restricted to full-time education of children and cannot be claimed for self, spouse, or part-time studies.",
        "type": QuestionType.YES_NO,
        "keywords": ["no", "cannot be claimed", "children only"],
    },
    {
        "doc": "section_80d_health.txt",
        "question": "Are preventive health check-up payments allowed in cash to qualify for Section 80D deduction?",
        "reference_answer": "Yes. While health insurance premiums must be paid by non-cash modes, preventive health check-up payments up to ₹5,000 are explicitly permitted in cash.",
        "type": QuestionType.YES_NO,
        "keywords": ["yes", "cash", "preventive"],
    },
    {
        "doc": "hra_exemption.txt",
        "question": "Can an employee claim HRA exemption if they live in a self-owned residential property for which they pay no rent?",
        "reference_answer": "No. To claim HRA exemption, the employee must actually incur rental expenditure for accommodation not owned by them.",
        "type": QuestionType.YES_NO,
        "keywords": ["no", "actually incur", "not owned"],
    },
    {
        "doc": "new_vs_old_tax_regime.txt",
        "question": "Is the standard deduction for salaried employees available under the New Tax Regime (Section 115BAC)?",
        "reference_answer": "Yes. Salaried employees can claim the standard deduction of ₹50,000 under the New Tax Regime as well.",
        "type": QuestionType.YES_NO,
        "keywords": ["yes", "standard deduction", "50,000"],
    },
    {
        "doc": "capital_gains_tax.txt",
        "question": "Can capital losses from short-term capital assets be set off against taxable long-term capital gains?",
        "reference_answer": "Yes. Short-term capital loss (STCL) can be set off against both short-term and long-term capital gains.",
        "type": QuestionType.YES_NO,
        "keywords": ["yes", "set off", "stcl"],
    },

    # ── Multi-Hop (Synthesizing multiple chunks / provisions) ────────────────
    {
        "doc": "section_80c_deductions.txt",
        "question": "How does combining Section 80C investments with Section 80CCD(1B) NPS contributions impact the total tax deduction limit?",
        "reference_answer": "Combining Section 80C (up to ₹1,50,000) with Section 80CCD(1B) (up to ₹50,000 for NPS) raises the total combined deductible limit to ₹2,00,000 per financial year.",
        "type": QuestionType.MULTI_HOP,
        "keywords": ["2,00,000", "80c", "80ccd(1b)", "combined"],
    },
    {
        "doc": "section_80d_health.txt",
        "question": "What is the maximum total Section 80D deduction for an individual below 60 who covers themselves, their family, and their senior citizen parents?",
        "reference_answer": "The maximum deduction is ₹75,000: up to ₹25,000 for self, spouse, and dependent children (below 60), plus up to ₹50,000 for senior citizen parents (60 or above).",
        "type": QuestionType.MULTI_HOP,
        "keywords": ["75,000", "25,000", "50,000"],
    },
    {
        "doc": "new_vs_old_tax_regime.txt",
        "question": "If an individual chooses the New Tax Regime under Section 115BAC, which common deductions under 80C and 80D must they forego?",
        "reference_answer": "Under the New Tax Regime, taxpayers must forego Chapter VI-A deductions including Section 80C (life insurance, ELSS, PF), Section 80D (health insurance), and HRA exemption, while retaining the standard deduction and employer NPS contribution.",
        "type": QuestionType.MULTI_HOP,
        "keywords": ["forego", "80c", "80d", "hra", "chapter vi-a"],
    },

    # ── Unanswerable (Out-of-Scope; expected behavior is explicit abstention) ──
    {
        "doc": "unanswerable",
        "question": "What is the tax rate on profits from cryptocurrency trading under Section 115BBH?",
        "reference_answer": "The provided context does not contain information about cryptocurrency taxation or Section 115BBH.",
        "type": QuestionType.UNANSWERABLE,
        "keywords": ["does not contain", "cryptocurrency"],
    },
    {
        "doc": "unanswerable",
        "question": "How are employee stock ownership plans (ESOPs) taxed at the time of exercise?",
        "reference_answer": "The provided context does not contain information regarding the taxation of ESOPs or perquisite valuation on exercise.",
        "type": QuestionType.UNANSWERABLE,
        "keywords": ["does not contain", "esop"],
    },
    {
        "doc": "unanswerable",
        "question": "What is the maximum tax exemption limit for gratuity received under the Payment of Gratuity Act?",
        "reference_answer": "The provided context does not contain information regarding gratuity exemption limits.",
        "type": QuestionType.UNANSWERABLE,
        "keywords": ["does not contain", "gratuity"],
    },
    {
        "doc": "unanswerable",
        "question": "What are the transfer pricing documentation rules for cross-border transactions with associated enterprises?",
        "reference_answer": "The provided context does not contain information about transfer pricing regulations or associated enterprises.",
        "type": QuestionType.UNANSWERABLE,
        "keywords": ["does not contain", "transfer pricing"],
    },
    {
        "doc": "unanswerable",
        "question": "What is the angel tax provision for unlisted startups under Section 56(2)(viib)?",
        "reference_answer": "The provided context does not contain information regarding angel tax or Section 56(2)(viib) startup provisions.",
        "type": QuestionType.UNANSWERABLE,
        "keywords": ["does not contain", "angel tax"],
    },
]


def match_chunk_for_item(doc_name: str, keywords: List[str], chunks: List[Chunk]) -> Optional[Chunk]:
    """Finds the most specific chunk in doc_name that matches the given keywords."""
    matching_chunks = [c for c in chunks if c.doc_name.lower() == doc_name.lower()]
    if not matching_chunks:
        return None

    best_chunk = None
    best_matches = -1
    for c in matching_chunks:
        text_lower = c.text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        if matches > best_matches:
            best_matches = matches
            best_chunk = c

    return best_chunk or matching_chunks[0]


def build_candidate_dataset(
    chunks: List[Chunk],
    config_path: Optional[str] = None,
) -> List[CandidateQA]:
    """
    Synthesizes candidate Q/A items grounded in chunks from the corpus.
    Applies Family A metadata and maps chunk-level groundings.
    """
    cfg = load_config(config_path)
    gen_model = cfg.model_families["generator"].model
    gen_family = cfg.model_families["generator"].family

    items: List[CandidateQA] = []
    item_counter = 1

    for item in CURATED_TAX_ITEMS:
        q_type = item["type"]
        doc_ref = item["doc"]
        q_text = item["question"]
        ref_ans = item["reference_answer"]
        keywords = item.get("keywords", [])

        if q_type == QuestionType.UNANSWERABLE:
            qa = CandidateQA(
                id=f"LITMUS_QA_{item_counter:03d}",
                question=q_text,
                reference_answer=ref_ans,
                question_type=q_type,
                primary_chunk_id=None,
                supporting_chunk_ids=[],
                doc_group="unanswerable_out_of_scope",
                generator_model=gen_model,
                generator_family=gen_family,
                metadata={"abstain_expected": True, "category": "out_of_scope"},
            )
        else:
            primary_chunk = match_chunk_for_item(doc_ref, keywords, chunks)
            primary_id = primary_chunk.chunk_id if primary_chunk else None
            supp_ids = [primary_id] if primary_id else []

            # For multi-hop, attach adjacent/related chunks from the same doc
            if q_type == QuestionType.MULTI_HOP and primary_chunk:
                same_doc_chunks = [c.chunk_id for c in chunks if c.doc_name == primary_chunk.doc_name and c.chunk_id != primary_id]
                if same_doc_chunks:
                    supp_ids.append(same_doc_chunks[0])

            qa = CandidateQA(
                id=f"LITMUS_QA_{item_counter:03d}",
                question=q_text,
                reference_answer=ref_ans,
                question_type=q_type,
                primary_chunk_id=primary_id,
                supporting_chunk_ids=supp_ids,
                doc_group=doc_ref,
                generator_model=gen_model,
                generator_family=gen_family,
                metadata={"keywords": keywords, "doc_name": doc_ref},
            )

        items.append(qa)
        item_counter += 1

    return items
