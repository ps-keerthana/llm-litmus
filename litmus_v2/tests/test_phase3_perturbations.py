"""
Tests for Phase 3: Perturbation Suite.
Verifies all 7 perturbation types, known labels, reproducibility, and summary stats.
"""

import pytest
from litmus_v2.data.schema import QuestionType
from litmus_v2.perturbations.perturbations import (
    PerturbationType,
    PerturbedItem,
    perturb_item,
    generate_perturbation_suite,
    perturbation_suite_summary,
)
from litmus_v2.tests.fixtures.sample_chunks import SAMPLE_CHUNKS, make_candidate_qa


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def numeric_qa():
    return make_candidate_qa(
        id="LITMUS_QA_006",
        question="What is the maximum aggregate deduction limit under Section 80C?",
        reference_answer=(
            "The maximum aggregate deduction permitted under Section 80C "
            "(along with 80CCC and 80CCD(1)) is ₹1,50,000 per financial year."
        ),
        question_type=QuestionType.NUMERIC,
    )


@pytest.fixture
def yes_no_qa():
    return make_candidate_qa(
        id="LITMUS_QA_011",
        question="Can an individual claim tuition fee deduction for self-education under 80C?",
        reference_answer=(
            "No. Tuition fee deductions under Section 80C are strictly restricted to "
            "full-time education of children and cannot be claimed for self, spouse, "
            "or part-time studies."
        ),
        question_type=QuestionType.YES_NO,
    )


@pytest.fixture
def single_fact_qa():
    return make_candidate_qa(
        id="LITMUS_QA_001",
        question="What is the lock-in period for ELSS under Section 80C?",
        reference_answer=(
            "ELSS mutual funds have a mandatory lock-in period of 3 years, "
            "which is the shortest lock-in period among Section 80C investment options."
        ),
        question_type=QuestionType.SINGLE_FACT,
    )


@pytest.fixture
def multi_hop_qa():
    return make_candidate_qa(
        id="LITMUS_QA_016",
        question="How does combining 80C and 80CCD(1B) impact the total deduction?",
        reference_answer=(
            "Combining Section 80C (up to ₹1,50,000) with Section 80CCD(1B) "
            "(up to ₹50,000 for NPS) raises the total combined deductible limit "
            "to ₹2,00,000 per financial year."
        ),
        question_type=QuestionType.MULTI_HOP,
    )


@pytest.fixture
def chunk_lookup():
    return {c.chunk_id: c for c in SAMPLE_CHUNKS}


# ─────────────────────────────────────────────────────────────────────────────
# Individual perturbation type tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNumberSwap:
    def test_changes_the_answer(self, numeric_qa):
        item = perturb_item(numeric_qa, PerturbationType.NUMBER_SWAP, seed=42)
        assert item is not None
        assert item.perturbed_answer != item.reference_answer

    def test_is_labeled_wrong(self, numeric_qa):
        item = perturb_item(numeric_qa, PerturbationType.NUMBER_SWAP, seed=42)
        assert item.expected_correct is False

    def test_contains_different_number(self, numeric_qa):
        item = perturb_item(numeric_qa, PerturbationType.NUMBER_SWAP, seed=42)
        assert "1,50,000" not in item.perturbed_answer

    def test_seed_is_recorded(self, numeric_qa):
        item = perturb_item(numeric_qa, PerturbationType.NUMBER_SWAP, seed=77)
        assert item.seed == 77


class TestNegationFlip:
    def test_flips_yes_no_answer(self, yes_no_qa):
        item = perturb_item(yes_no_qa, PerturbationType.NEGATION_FLIP, seed=42)
        assert item is not None
        assert item.perturbed_answer != item.reference_answer

    def test_is_labeled_wrong(self, yes_no_qa):
        item = perturb_item(yes_no_qa, PerturbationType.NEGATION_FLIP, seed=42)
        assert item.expected_correct is False

    def test_flips_single_fact_answer(self, single_fact_qa):
        item = perturb_item(single_fact_qa, PerturbationType.NEGATION_FLIP, seed=42)
        assert item is not None
        assert item.expected_correct is False


class TestEntitySwap:
    def test_swaps_section_reference(self, single_fact_qa):
        item = perturb_item(single_fact_qa, PerturbationType.ENTITY_SWAP, seed=42)
        # If applicable: section or entity should differ
        if item is not None:
            assert item.perturbed_answer != item.reference_answer
            assert item.expected_correct is False

    def test_multi_hop_entity_swap(self, multi_hop_qa):
        item = perturb_item(multi_hop_qa, PerturbationType.ENTITY_SWAP, seed=42)
        if item is not None:
            assert item.expected_correct is False


class TestUnsupportedClaim:
    def test_adds_fabricated_claim(self, single_fact_qa):
        item = perturb_item(single_fact_qa, PerturbationType.UNSUPPORTED_CLAIM, seed=42)
        assert item is not None
        # Perturbed answer should be longer than original
        assert len(item.perturbed_answer) > len(item.reference_answer)

    def test_is_labeled_wrong(self, single_fact_qa):
        item = perturb_item(single_fact_qa, PerturbationType.UNSUPPORTED_CLAIM, seed=42)
        assert item.expected_correct is False


class TestTruncation:
    def test_is_shorter_than_original(self, multi_hop_qa):
        item = perturb_item(multi_hop_qa, PerturbationType.TRUNCATION, seed=42)
        assert item is not None
        assert len(item.perturbed_answer) < len(item.reference_answer)

    def test_is_labeled_wrong(self, multi_hop_qa):
        item = perturb_item(multi_hop_qa, PerturbationType.TRUNCATION, seed=42)
        assert item.expected_correct is False


class TestOffTopic:
    def test_replaces_with_different_content(self, single_fact_qa):
        item = perturb_item(single_fact_qa, PerturbationType.OFF_TOPIC, seed=42)
        assert item is not None
        assert item.perturbed_answer != item.reference_answer
        # Should not contain domain-specific text from the original
        assert "ELSS" not in item.perturbed_answer

    def test_is_labeled_wrong(self, single_fact_qa):
        item = perturb_item(single_fact_qa, PerturbationType.OFF_TOPIC, seed=42)
        assert item.expected_correct is False


class TestParaphrase:
    def test_produces_different_surface_form(self, numeric_qa):
        item = perturb_item(numeric_qa, PerturbationType.PARAPHRASE, seed=42)
        assert item is not None
        assert item.perturbed_answer != item.reference_answer

    def test_is_labeled_CORRECT(self, numeric_qa):
        """CRITICAL: Paraphrase is semantically correct — must have expected_correct=True."""
        item = perturb_item(numeric_qa, PerturbationType.PARAPHRASE, seed=42)
        assert item.expected_correct is True, (
            "Paraphrase perturbation must have expected_correct=True "
            "(it tests for false positives in graders)"
        )

    def test_single_fact_paraphrase_correct(self, single_fact_qa):
        item = perturb_item(single_fact_qa, PerturbationType.PARAPHRASE, seed=42)
        assert item.expected_correct is True


# ─────────────────────────────────────────────────────────────────────────────
# Suite generation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratePerturbationSuite:
    @pytest.fixture
    def small_candidates(self):
        return [
            make_candidate_qa(
                id="Q001",
                reference_answer="The maximum deduction under Section 80C is ₹1,50,000 per financial year.",
                question_type=QuestionType.NUMERIC,
            ),
            make_candidate_qa(
                id="Q002",
                reference_answer="ELSS funds have a mandatory lock-in period of 3 years.",
                question_type=QuestionType.SINGLE_FACT,
            ),
        ]

    def test_generates_items_for_each_type(self, small_candidates, chunk_lookup):
        items = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        types_present = {item.perturbation_type for item in items}
        # All 7 types should appear
        for ptype in PerturbationType:
            assert ptype in types_present, f"Missing perturbation type: {ptype}"

    def test_is_reproducible(self, small_candidates, chunk_lookup):
        items1 = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        items2 = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        assert [(i.id, i.perturbed_answer) for i in items1] == \
               [(i.id, i.perturbed_answer) for i in items2]

    def test_different_seeds_produce_different_items(self, small_candidates, chunk_lookup):
        items1 = generate_perturbation_suite(small_candidates, chunk_lookup, seed=1)
        items2 = generate_perturbation_suite(small_candidates, chunk_lookup, seed=2)
        answers1 = {(i.id, i.perturbed_answer) for i in items1}
        answers2 = {(i.id, i.perturbed_answer) for i in items2}
        # At least some items should differ across seeds
        assert answers1 != answers2 or len(items1) == 0

    def test_no_mock_markers_in_perturbed_answers(self, small_candidates, chunk_lookup):
        items = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        for item in items:
            assert "TODO" not in item.perturbed_answer
            assert "PLACEHOLDER" not in item.perturbed_answer.upper()
            assert item.perturbed_answer.strip(), f"{item.id} has empty perturbed_answer"

    def test_paraphrases_are_expected_correct(self, small_candidates, chunk_lookup):
        items = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        paraphrases = [i for i in items if i.perturbation_type == PerturbationType.PARAPHRASE]
        assert all(i.expected_correct is True for i in paraphrases), \
            "All paraphrase items must have expected_correct=True"

    def test_non_paraphrase_are_expected_wrong(self, small_candidates, chunk_lookup):
        items = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        non_para = [i for i in items if i.perturbation_type != PerturbationType.PARAPHRASE]
        assert all(i.expected_correct is False for i in non_para), \
            "All non-paraphrase perturbations must have expected_correct=False"

    def test_type_filter_works(self, small_candidates, chunk_lookup):
        items = generate_perturbation_suite(
            small_candidates, chunk_lookup, seed=42,
            types=[PerturbationType.NUMBER_SWAP, PerturbationType.TRUNCATION],
        )
        types_present = {item.perturbation_type for item in items}
        assert types_present.issubset({PerturbationType.NUMBER_SWAP, PerturbationType.TRUNCATION})

    def test_ids_are_unique(self, small_candidates, chunk_lookup):
        items = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        ids = [i.id for i in items]
        assert len(ids) == len(set(ids))

    def test_source_ids_traceable(self, small_candidates, chunk_lookup):
        items = generate_perturbation_suite(small_candidates, chunk_lookup, seed=42)
        source_ids = {i.source_id for i in items}
        candidate_ids = {qa.id for qa in small_candidates}
        assert source_ids.issubset(candidate_ids)


class TestPerturbationSuiteSummary:
    def test_summary_has_required_keys(self, chunk_lookup):
        candidates = [make_candidate_qa(
            reference_answer="The maximum deduction is ₹1,50,000 per financial year.",
            question_type=QuestionType.NUMERIC,
        )]
        items = generate_perturbation_suite(candidates, chunk_lookup, seed=42)
        summary = perturbation_suite_summary(items)
        assert "total" in summary
        assert "by_type" in summary
        assert "expected_correct" in summary
        assert "expected_wrong" in summary

    def test_summary_counts_add_up(self, chunk_lookup):
        candidates = [make_candidate_qa(
            reference_answer="The maximum deduction is ₹1,50,000 per financial year.",
            question_type=QuestionType.NUMERIC,
        )]
        items = generate_perturbation_suite(candidates, chunk_lookup, seed=42)
        summary = perturbation_suite_summary(items)
        assert summary["expected_correct"] + summary["expected_wrong"] == summary["total"]
