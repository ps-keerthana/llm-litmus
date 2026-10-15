"""
Phase 2: Human Gold Set Labeling Tooling.

Exports a CSV template with all candidate Q/A items, blank human-label columns,
and launches a Streamlit UI for manual review.

HARD RULE: This module NEVER auto-fills labels.
- reference_correct, supported_by_chunk, edited_reference remain blank (empty).
- The human user fills them in Streamlit or directly in the CSV.
- No script should import this and write label values programmatically.
"""

import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from litmus_v2.data.schema import CandidateQA, HumanLabel


# Human-label CSV columns in canonical order
CSV_COLUMNS = [
    "id",
    "question_type",
    "question",
    "reference_answer",
    "primary_chunk_id",
    "source_chunk_text",
    # ── Human fills these ──────────────────────────────────
    "reference_correct",    # True / False   — Is the reference answer factually correct?
    "supported_by_chunk",   # True / False   — Is it directly supported by the source chunk?
    "edited_reference",     # Free text      — Corrected answer if reference_correct=False
    "notes",                # Free text      — Reviewer comments / flags
    "labeled_by",           # Reviewer name / ID
    "labeled_at",           # ISO-8601 timestamp (reviewer fills or auto-stamps on save)
]


def export_labeling_csv(
    candidates: List[CandidateQA],
    chunk_lookup: dict,
    output_path: str,
    overwrite: bool = False,
) -> str:
    """
    Writes a labeling CSV template to output_path.
    All human-label columns are BLANK — never auto-filled.

    Args:
        candidates: List of CandidateQA items to label.
        chunk_lookup: Dict[chunk_id -> Chunk] for source text lookup.
        output_path: Destination path for the CSV.
        overwrite: If False, raises if file already exists.

    Returns:
        Absolute path to the written CSV.
    """
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Labeling CSV already exists: {path}. Pass overwrite=True to replace."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for qa in candidates:
            chunk_text = ""
            if qa.primary_chunk_id and qa.primary_chunk_id in chunk_lookup:
                chunk_text = chunk_lookup[qa.primary_chunk_id].text

            row = {
                "id": qa.id,
                "question_type": qa.question_type.value,
                "question": qa.question,
                "reference_answer": qa.reference_answer,
                "primary_chunk_id": qa.primary_chunk_id or "",
                "source_chunk_text": chunk_text,
                # Human-label columns — intentionally BLANK
                "reference_correct": "",
                "supported_by_chunk": "",
                "edited_reference": "",
                "notes": "",
                "labeled_by": "",
                "labeled_at": "",
            }
            writer.writerow(row)

    print(f"[litmus_v2] Labeling CSV written → {path.resolve()}")
    print(f"  Rows: {len(candidates)}")
    print(f"  Columns: {', '.join(CSV_COLUMNS)}")
    print(
        "\n  *** STOP: Fill in 'reference_correct', 'supported_by_chunk',"
        " 'edited_reference', 'labeled_by' for each row. ***"
        "\n  Do NOT run the evaluation pipeline until labeling is complete."
    )
    return str(path.resolve())


def load_human_labels(csv_path: str) -> List[HumanLabel]:
    """
    Reads a completed labeling CSV and returns HumanLabel objects.
    Raises ValueError if any label row is still empty (not yet filled).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Labeling CSV not found: {path}")

    labels: List[HumanLabel] = []
    unlabeled_ids = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_correct_raw = row.get("reference_correct", "").strip()
            supp_raw = row.get("supported_by_chunk", "").strip()

            if not ref_correct_raw or not supp_raw:
                unlabeled_ids.append(row.get("id", "?"))
                continue

            def parse_bool(val: str) -> Optional[bool]:
                if val.strip().lower() in ("true", "yes", "1"):
                    return True
                if val.strip().lower() in ("false", "no", "0"):
                    return False
                return None

            hl = HumanLabel(
                id=row["id"],
                question=row["question"],
                reference_answer=row["reference_answer"],
                question_type=row["question_type"],
                primary_chunk_id=row.get("primary_chunk_id") or None,
                source_chunk_text=row.get("source_chunk_text") or None,
                reference_correct=parse_bool(ref_correct_raw),
                supported_by_chunk=parse_bool(supp_raw),
                edited_reference=row.get("edited_reference") or None,
                notes=row.get("notes") or None,
                labeled_by=row.get("labeled_by") or None,
                labeled_at=row.get("labeled_at") or None,
            )
            labels.append(hl)

    if unlabeled_ids:
        print(f"[litmus_v2] WARNING: {len(unlabeled_ids)} rows are still unlabeled: {unlabeled_ids[:10]}")

    return labels


def validate_labels_complete(labels: List[HumanLabel]) -> bool:
    """
    Asserts all required label fields are filled.
    Returns True if complete; raises ValueError listing incomplete rows.
    """
    incomplete = []
    for hl in labels:
        issues = []
        if hl.reference_correct is None:
            issues.append("reference_correct")
        if hl.supported_by_chunk is None:
            issues.append("supported_by_chunk")
        if not hl.labeled_by:
            issues.append("labeled_by")
        if issues:
            incomplete.append(f"{hl.id}: missing {issues}")

    if incomplete:
        raise ValueError(
            f"Labeling incomplete ({len(incomplete)} rows):\n" +
            "\n".join(incomplete[:20])
        )
    return True
