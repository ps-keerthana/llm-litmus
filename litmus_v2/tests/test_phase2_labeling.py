"""
Tests for litmus_v2 Phase 2: Human labeling tooling.
Verifies CSV export format, blank-label enforcement, load/validate logic,
and the hard rule that no label is ever auto-filled.
"""

import csv
import pytest
from pathlib import Path

from litmus_v2.data.schema import QuestionType
from litmus_v2.labeling.labeling import (
    export_labeling_csv,
    load_human_labels,
    validate_labels_complete,
    CSV_COLUMNS,
)
from litmus_v2.tests.fixtures.sample_chunks import SAMPLE_CHUNKS, make_candidate_qa


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_candidates():
    return [
        make_candidate_qa(id="Q001", question_type=QuestionType.SINGLE_FACT),
        make_candidate_qa(id="Q002", question_type=QuestionType.NUMERIC,
                          question="What is the max 80C limit?",
                          reference_answer="₹1,50,000"),
        make_candidate_qa(id="Q003", question_type=QuestionType.UNANSWERABLE,
                          question="Crypto tax?",
                          reference_answer="Does not contain.",
                          primary_chunk_id=None,
                          supporting_chunk_ids=[]),
    ]


@pytest.fixture
def chunk_lookup():
    return {c.chunk_id: c for c in SAMPLE_CHUNKS}


@pytest.fixture
def blank_csv(tmp_path, sample_candidates, chunk_lookup):
    """Exports a blank labeling CSV to a temp path and returns the path."""
    out = str(tmp_path / "labeling_template.csv")
    export_labeling_csv(sample_candidates, chunk_lookup, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Export tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExportCSV:
    def test_creates_file(self, blank_csv):
        assert Path(blank_csv).exists()

    def test_correct_row_count(self, blank_csv, sample_candidates):
        with open(blank_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(sample_candidates)

    def test_has_all_required_columns(self, blank_csv):
        with open(blank_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        for col in CSV_COLUMNS:
            assert col in header, f"Missing column: {col}"

    def test_human_label_columns_are_blank(self, blank_csv):
        """HARD RULE: no auto-filled labels — all human columns must be empty."""
        human_cols = [
            "reference_correct",
            "supported_by_chunk",
            "edited_reference",
            "notes",
            "labeled_by",
            "labeled_at",
        ]
        with open(blank_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for col in human_cols:
                assert row[col] == "", (
                    f"HARD RULE VIOLATION: Column '{col}' was auto-filled "
                    f"with '{row[col]}' for id={row['id']}. "
                    "Human labels must NEVER be auto-filled."
                )

    def test_ids_match_candidates(self, blank_csv, sample_candidates):
        with open(blank_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        exported_ids = [r["id"] for r in rows]
        expected_ids = [qa.id for qa in sample_candidates]
        assert exported_ids == expected_ids

    def test_source_chunk_text_filled_for_answerable(self, blank_csv, chunk_lookup):
        """Answerable items should have source_chunk_text populated from chunk_lookup."""
        with open(blank_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["question_type"] != "unanswerable":
                assert row["source_chunk_text"].strip(), (
                    f"id={row['id']} ({row['question_type']}) should have source_chunk_text"
                )

    def test_unanswerable_source_chunk_is_empty(self, blank_csv):
        """Unanswerable items have no chunk — source_chunk_text should be blank."""
        with open(blank_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["question_type"] == "unanswerable":
                assert row["source_chunk_text"] == "", (
                    f"Unanswerable id={row['id']} should have no source chunk"
                )

    def test_overwrite_false_raises_if_exists(self, blank_csv, sample_candidates, chunk_lookup):
        with pytest.raises(FileExistsError):
            export_labeling_csv(
                sample_candidates, chunk_lookup, blank_csv, overwrite=False
            )

    def test_overwrite_true_succeeds(self, blank_csv, sample_candidates, chunk_lookup):
        # Should not raise
        export_labeling_csv(
            sample_candidates, chunk_lookup, blank_csv, overwrite=True
        )
        assert Path(blank_csv).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Load & validate tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadHumanLabels:
    def _write_labeled_csv(self, path: str, rows: list):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_load_complete_labels(self, tmp_path):
        path = str(tmp_path / "labeled.csv")
        self._write_labeled_csv(path, [
            {
                "id": "Q001",
                "question_type": "single_fact",
                "question": "What is the ELSS lock-in?",
                "reference_answer": "3 years",
                "primary_chunk_id": "chunk_001",
                "source_chunk_text": "Some text",
                "reference_correct": "True",
                "supported_by_chunk": "True",
                "edited_reference": "",
                "notes": "",
                "labeled_by": "keerthana",
                "labeled_at": "2026-10-16T00:00:00+00:00",
            }
        ])
        labels = load_human_labels(path)
        assert len(labels) == 1
        assert labels[0].reference_correct is True
        assert labels[0].supported_by_chunk is True
        assert labels[0].labeled_by == "keerthana"

    def test_load_false_labels(self, tmp_path):
        path = str(tmp_path / "labeled.csv")
        self._write_labeled_csv(path, [
            {
                "id": "Q002",
                "question_type": "numeric",
                "question": "Max 80C limit?",
                "reference_answer": "₹1,50,000",
                "primary_chunk_id": "",
                "source_chunk_text": "",
                "reference_correct": "False",
                "supported_by_chunk": "False",
                "edited_reference": "Corrected: ₹1,50,000",
                "notes": "Reference was wrong",
                "labeled_by": "reviewer2",
                "labeled_at": "2026-10-16T01:00:00+00:00",
            }
        ])
        labels = load_human_labels(path)
        assert labels[0].reference_correct is False
        assert labels[0].supported_by_chunk is False
        assert labels[0].edited_reference == "Corrected: ₹1,50,000"

    def test_load_skips_blank_rows_and_warns(self, tmp_path, capsys):
        """Rows without labels should be skipped (not silently accepted)."""
        path = str(tmp_path / "partial.csv")
        self._write_labeled_csv(path, [
            {
                "id": "Q001",
                "question_type": "single_fact",
                "question": "Q?",
                "reference_answer": "A.",
                "primary_chunk_id": "",
                "source_chunk_text": "",
                "reference_correct": "",    # ← blank
                "supported_by_chunk": "",   # ← blank
                "edited_reference": "",
                "notes": "",
                "labeled_by": "",
                "labeled_at": "",
            }
        ])
        labels = load_human_labels(path)
        assert len(labels) == 0
        captured = capsys.readouterr()
        assert "unlabeled" in captured.out.lower()

    def test_load_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_human_labels(str(tmp_path / "nonexistent.csv"))


class TestValidateLabelsComplete:
    def _make_label(self, **kwargs):
        from litmus_v2.data.schema import HumanLabel
        defaults = {
            "id": "Q001",
            "question": "Q?",
            "reference_answer": "A.",
            "question_type": "single_fact",
            "reference_correct": True,
            "supported_by_chunk": True,
            "labeled_by": "reviewer",
        }
        defaults.update(kwargs)
        return HumanLabel(**defaults)

    def test_complete_labels_pass(self):
        labels = [self._make_label(id="Q001"), self._make_label(id="Q002")]
        assert validate_labels_complete(labels) is True

    def test_missing_reference_correct_raises(self):
        labels = [self._make_label(reference_correct=None)]
        with pytest.raises(ValueError, match="reference_correct"):
            validate_labels_complete(labels)

    def test_missing_supported_by_chunk_raises(self):
        labels = [self._make_label(supported_by_chunk=None)]
        with pytest.raises(ValueError, match="supported_by_chunk"):
            validate_labels_complete(labels)

    def test_missing_labeled_by_raises(self):
        labels = [self._make_label(labeled_by="")]
        with pytest.raises(ValueError, match="labeled_by"):
            validate_labels_complete(labels)

    def test_multiple_incomplete_reported(self):
        labels = [
            self._make_label(id="Q001", reference_correct=None),
            self._make_label(id="Q002", supported_by_chunk=None),
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_labels_complete(labels)
        msg = str(exc_info.value)
        assert "Q001" in msg
        assert "Q002" in msg
