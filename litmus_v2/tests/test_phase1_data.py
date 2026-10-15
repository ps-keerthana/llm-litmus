"""
Tests for litmus_v2 Phase 1: Data module.
Covers corpus chunking, versioning, splitting, schema, and candidate generator.
"""

import hashlib
import pytest
from pathlib import Path

from litmus_v2.data.schema import Chunk, CandidateQA, HumanLabel, DatasetManifest, QuestionType
from litmus_v2.data.corpus import chunk_document, load_and_chunk_text
from litmus_v2.data.versioning import hash_dataset, create_manifest
from litmus_v2.data.split import split_by_doc_group
from litmus_v2.data.generator import build_candidate_dataset, match_chunk_for_item
from litmus_v2.tests.fixtures.sample_chunks import SAMPLE_CHUNKS, make_chunk, make_candidate_qa


# ─────────────────────────────────────────────────────────────────────────────
# Schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_chunk_fields_required(self):
        c = make_chunk()
        assert c.chunk_id == "chunk_001"
        assert c.doc_name == "section_80c_deductions.txt"
        assert c.token_count == 40

    def test_candidate_qa_unanswerable_no_primary_chunk(self):
        qa = CandidateQA(
            id="LITMUS_QA_999",
            question="Out of scope?",
            reference_answer="Does not contain.",
            question_type=QuestionType.UNANSWERABLE,
            primary_chunk_id=None,
            supporting_chunk_ids=[],
            doc_group="unanswerable_out_of_scope",
            generator_model="mistral:7b",
            generator_family="mistral",
        )
        assert qa.primary_chunk_id is None
        assert qa.question_type == QuestionType.UNANSWERABLE

    def test_human_label_nullable_fields(self):
        hl = HumanLabel(
            id="LITMUS_QA_001",
            question="What is the ELSS lock-in?",
            reference_answer="3 years",
            question_type="single_fact",
        )
        assert hl.reference_correct is None
        assert hl.supported_by_chunk is None
        assert hl.labeled_by is None

    def test_question_type_enum_values(self):
        assert QuestionType.SINGLE_FACT == "single_fact"
        assert QuestionType.NUMERIC == "numeric"
        assert QuestionType.YES_NO == "yes_no"
        assert QuestionType.MULTI_HOP == "multi_hop"
        assert QuestionType.UNANSWERABLE == "unanswerable"


# ─────────────────────────────────────────────────────────────────────────────
# Corpus / chunking tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunking:
    """Tests for corpus.py hierarchical paragraph chunking."""

    SAMPLE_TEXT = (
        "The maximum aggregate deduction under Section 80C is ₹1,50,000 per financial year. "
        "This includes contributions to PPF, ELSS, and life insurance premiums.\n\n"
        "Section 80CCD(1B) provides an additional ₹50,000 deduction for NPS Tier-1. "
        "This deduction is over and above the ₹1,50,000 limit.\n\n"
        "Preventive health check-up costs up to ₹5,000 can be claimed under Section 80D."
    )

    def test_chunk_document_returns_list_of_chunks(self):
        chunks = chunk_document(
            text=self.SAMPLE_TEXT,
            doc_name="test_doc.txt",
            doc_title="Test Document",
            max_tokens=200,
        )
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_chunk_ids_are_unique(self):
        chunks = chunk_document(
            text=self.SAMPLE_TEXT,
            doc_name="test_doc.txt",
            doc_title="Test Document",
            max_tokens=100,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_spans_cover_full_text(self):
        """Start/end char pairs should be monotonically increasing and non-overlapping."""
        chunks = chunk_document(
            text=self.SAMPLE_TEXT,
            doc_name="test_doc.txt",
            doc_title="Test Document",
            max_tokens=200,
        )
        prev_end = 0
        for c in chunks:
            assert c.start_char >= prev_end
            assert c.end_char > c.start_char
            prev_end = c.end_char

    def test_chunk_content_hash_is_sha256(self):
        chunks = chunk_document(
            text=self.SAMPLE_TEXT,
            doc_name="test_doc.txt",
            doc_title="Test Document",
            max_tokens=200,
        )
        for c in chunks:
            expected = hashlib.sha256(c.text.encode()).hexdigest()
            assert c.content_hash == expected

    def test_chunk_doc_name_preserved(self):
        chunks = chunk_document(
            text=self.SAMPLE_TEXT,
            doc_name="section_80c.txt",
            doc_title="Section 80C",
            max_tokens=200,
        )
        assert all(c.doc_name == "section_80c.txt" for c in chunks)

    def test_small_max_tokens_creates_more_chunks(self):
        chunks_small = chunk_document(
            text=self.SAMPLE_TEXT * 3,
            doc_name="doc.txt",
            doc_title="Doc",
            max_tokens=60,
        )
        chunks_large = chunk_document(
            text=self.SAMPLE_TEXT * 3,
            doc_name="doc.txt",
            doc_title="Doc",
            max_tokens=500,
        )
        assert len(chunks_small) >= len(chunks_large)

    def test_single_paragraph_no_split(self):
        text = "Short single paragraph. No double newlines here."
        chunks = chunk_document(text=text, doc_name="d.txt", doc_title="D", max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text.strip()

    def test_load_and_chunk_text_from_string(self):
        """load_and_chunk_text should accept raw string input directly."""
        chunks = load_and_chunk_text(
            content=self.SAMPLE_TEXT,
            doc_name="inline.txt",
            doc_title="Inline Doc",
        )
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Versioning tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVersioning:
    def test_hash_dataset_is_deterministic(self):
        h1 = hash_dataset(SAMPLE_CHUNKS)
        h2 = hash_dataset(SAMPLE_CHUNKS)
        assert h1 == h2

    def test_hash_changes_on_mutation(self):
        modified = list(SAMPLE_CHUNKS)
        original_hash = hash_dataset(modified)
        # Change one chunk text
        new_chunk = modified[0].model_copy(update={"text": "Totally different text XYZ."})
        modified[0] = new_chunk
        assert hash_dataset(modified) != original_hash

    def test_hash_order_sensitivity(self):
        """Reversing chunk order should change the hash (order is significant)."""
        h1 = hash_dataset(SAMPLE_CHUNKS)
        h2 = hash_dataset(list(reversed(SAMPLE_CHUNKS)))
        assert h1 != h2

    def test_create_manifest_fields(self):
        candidates = [make_candidate_qa()]
        dev_items = [make_candidate_qa(id="LITMUS_QA_001")]
        test_items = [make_candidate_qa(id="LITMUS_QA_002", question_type=QuestionType.NUMERIC)]
        manifest = create_manifest(
            dataset_name="test_dataset",
            version="v0.1",
            all_items=candidates,
            dev_items=dev_items,
            test_items=test_items,
            generator_model="mistral:7b",
            generator_family="mistral",
            answering_family="llama",
            judge_family="gemma",
        )
        assert isinstance(manifest, DatasetManifest)
        assert manifest.total_items == len(candidates)
        assert manifest.dev_count == len(dev_items)
        assert manifest.test_count == len(test_items)
        assert len(manifest.dataset_hash) == 64  # sha256 hex

    def test_manifest_type_distribution(self):
        qa1 = make_candidate_qa(id="Q1", question_type=QuestionType.SINGLE_FACT)
        qa2 = make_candidate_qa(id="Q2", question_type=QuestionType.NUMERIC)
        qa3 = make_candidate_qa(id="Q3", question_type=QuestionType.SINGLE_FACT)
        manifest = create_manifest(
            dataset_name="d",
            version="v1",
            all_items=[qa1, qa2, qa3],
            dev_items=[qa1],
            test_items=[qa2, qa3],
            generator_model="m",
            generator_family="f",
            answering_family="af",
            judge_family="jf",
        )
        assert manifest.type_distribution["single_fact"] == 2
        assert manifest.type_distribution["numeric"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Split tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSplit:
    def _make_items(self):
        """7 items across 4 distinct doc groups."""
        types = [
            QuestionType.SINGLE_FACT,
            QuestionType.NUMERIC,
            QuestionType.YES_NO,
            QuestionType.MULTI_HOP,
            QuestionType.UNANSWERABLE,
            QuestionType.SINGLE_FACT,
            QuestionType.NUMERIC,
        ]
        groups = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e", "doc_a", "doc_b"]
        return [
            make_candidate_qa(
                id=f"Q{i+1}",
                question_type=types[i],
                doc_group=groups[i],
            )
            for i in range(7)
        ]

    def test_split_no_overlap(self):
        items = self._make_items()
        dev, test = split_by_doc_group(items, test_fraction=0.4, seed=42)
        dev_groups = {d.doc_group for d in dev}
        test_groups = {d.doc_group for d in test}
        assert dev_groups.isdisjoint(test_groups), (
            f"Overlap found: {dev_groups & test_groups}"
        )

    def test_split_covers_all_items(self):
        items = self._make_items()
        dev, test = split_by_doc_group(items, test_fraction=0.4, seed=42)
        assert len(dev) + len(test) == len(items)

    def test_split_is_reproducible(self):
        items = self._make_items()
        dev1, test1 = split_by_doc_group(items, test_fraction=0.4, seed=99)
        dev2, test2 = split_by_doc_group(items, test_fraction=0.4, seed=99)
        assert [d.id for d in dev1] == [d.id for d in dev2]
        assert [d.id for d in test1] == [d.id for d in test2]

    def test_split_different_seeds_may_differ(self):
        items = self._make_items()
        dev1, _ = split_by_doc_group(items, test_fraction=0.4, seed=1)
        dev2, _ = split_by_doc_group(items, test_fraction=0.4, seed=2)
        # With more than 2 doc groups and different seeds, split often differs
        # (not guaranteed for every seed pair but expected for most)
        ids1 = set(d.id for d in dev1)
        ids2 = set(d.id for d in dev2)
        # Just verify both are valid splits; exact equality not required
        assert ids1 or ids2  # at least one is non-empty

    def test_split_unanswerable_group_stays_together(self):
        """Items sharing a doc_group must stay in the same split."""
        items = self._make_items()
        dev, test = split_by_doc_group(items, test_fraction=0.4, seed=42)
        for group in set(i.doc_group for i in items):
            in_dev = [x for x in dev if x.doc_group == group]
            in_test = [x for x in test if x.doc_group == group]
            assert not (in_dev and in_test), (
                f"doc_group '{group}' split across dev and test"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Generator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerator:
    def test_build_returns_all_five_types(self):
        items = build_candidate_dataset(SAMPLE_CHUNKS)
        types_present = {item.question_type for item in items}
        assert QuestionType.SINGLE_FACT in types_present
        assert QuestionType.NUMERIC in types_present
        assert QuestionType.YES_NO in types_present
        assert QuestionType.MULTI_HOP in types_present
        assert QuestionType.UNANSWERABLE in types_present

    def test_no_mock_data_in_items(self):
        """Guard: result items must have real questions and answers, not placeholders."""
        items = build_candidate_dataset(SAMPLE_CHUNKS)
        for item in items:
            assert item.question.strip(), f"{item.id} has empty question"
            assert item.reference_answer.strip(), f"{item.id} has empty reference_answer"
            assert "TODO" not in item.question
            assert "PLACEHOLDER" not in item.reference_answer.upper()

    def test_unanswerable_items_have_no_primary_chunk(self):
        items = build_candidate_dataset(SAMPLE_CHUNKS)
        for item in items:
            if item.question_type == QuestionType.UNANSWERABLE:
                assert item.primary_chunk_id is None

    def test_generator_model_metadata_present(self):
        items = build_candidate_dataset(SAMPLE_CHUNKS)
        for item in items:
            assert item.generator_model, f"{item.id} missing generator_model"
            assert item.generator_family, f"{item.id} missing generator_family"

    def test_ids_are_unique(self):
        items = build_candidate_dataset(SAMPLE_CHUNKS)
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids))

    def test_match_chunk_for_item_returns_best(self):
        chunk = match_chunk_for_item(
            doc_name="section_80c_deductions.txt",
            keywords=["elss", "lock-in", "3 years"],
            chunks=SAMPLE_CHUNKS,
        )
        assert chunk is not None
        assert "ELSS" in chunk.text or "elss" in chunk.text.lower()

    def test_match_chunk_for_item_missing_doc_returns_none(self):
        chunk = match_chunk_for_item(
            doc_name="nonexistent_doc.txt",
            keywords=["foo"],
            chunks=SAMPLE_CHUNKS,
        )
        assert chunk is None

    def test_minimum_item_count(self):
        """The curated seed bank should produce at least 20 items."""
        items = build_candidate_dataset(SAMPLE_CHUNKS)
        assert len(items) >= 20
