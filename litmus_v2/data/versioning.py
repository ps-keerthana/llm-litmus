"""
Dataset versioning and deterministic cryptographic hashing module.
Ensures evaluation datasets are immutable, auditable, and traceable to git commits.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime, timezone

from litmus_v2.data.schema import CandidateQA, DatasetManifest


def compute_content_hash(items: List[Union[Dict[str, Any], CandidateQA]]) -> str:
    """
    Computes a canonical SHA-256 hash over a collection of records.
    Sorts keys and orders records by their unique 'id' to guarantee determinism.
    """
    normalized = []
    for item in items:
        if hasattr(item, "model_dump"):
            d = item.model_dump()
        elif isinstance(item, dict):
            d = dict(item)
        else:
            d = dict(item.__dict__)
        normalized.append(d)

    normalized.sort(key=lambda x: str(x.get("id", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _create_manifest_strict(
    dataset_name: str,
    version: str,
    dev_items: List[CandidateQA],
    test_items: List[CandidateQA],
    generator_model: str,
    generator_family: str,
    answering_family: str,
    judge_family: str,
) -> DatasetManifest:
    """Internal strict version: both dev and test required explicitly."""
    all_items = dev_items + test_items
    full_hash = compute_content_hash(all_items)
    dev_hash = compute_content_hash(dev_items)
    test_hash = compute_content_hash(test_items)

    type_counts: Dict[str, int] = {}
    for item in all_items:
        t = item.question_type.value if hasattr(item.question_type, "value") else str(item.question_type)
        type_counts[t] = type_counts.get(t, 0) + 1

    manifest = DatasetManifest(
        dataset_name=dataset_name,
        version=version,
        dataset_hash=full_hash,
        total_items=len(all_items),
        type_distribution=type_counts,
        dev_count=len(dev_items),
        test_count=len(test_items),
        dev_hash=dev_hash,
        test_hash=test_hash,
        generator_model=generator_model,
        generator_family=generator_family,
        answering_family=answering_family,
        judge_family=judge_family,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return manifest


def save_manifest(manifest: DatasetManifest, filepath: Union[str, Path]) -> None:
    """Write manifest to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)


def load_manifest(filepath: Union[str, Path]) -> DatasetManifest:
    """Read manifest from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return DatasetManifest(**data)


# ── Aliases ─────────────────────────────────────────────────────────────────────────

def hash_dataset(items) -> str:
    """Alias for compute_content_hash (accepts Chunks or CandidateQAs)."""
    return compute_content_hash(items)


def create_manifest(
    dataset_name: str,
    version: str,
    all_items: List[CandidateQA] = None,
    dev_items: List[CandidateQA] = None,
    test_items: List[CandidateQA] = None,
    generator_model: str = "",
    generator_family: str = "",
    answering_family: str = "",
    judge_family: str = "",
) -> DatasetManifest:
    """Overloaded create_manifest that accepts all_items and infers split counts."""
    if all_items is not None and dev_items is None and test_items is None:
        # Called with all_items only for manifest of an unsplit dataset
        dev_items = all_items
        test_items = []
    dev_items = dev_items or []
    test_items = test_items or []
    actual_all = all_items if all_items is not None else dev_items + test_items
    full_hash = compute_content_hash(actual_all)
    dev_hash = compute_content_hash(dev_items)
    test_hash = compute_content_hash(test_items)

    type_counts: Dict[str, int] = {}
    for item in actual_all:
        t = item.question_type.value if hasattr(item.question_type, "value") else str(item.question_type)
        type_counts[t] = type_counts.get(t, 0) + 1

    return DatasetManifest(
        dataset_name=dataset_name,
        version=version,
        dataset_hash=full_hash,
        total_items=len(actual_all),
        type_distribution=type_counts,
        dev_count=len(dev_items),
        test_count=len(test_items),
        dev_hash=dev_hash,
        test_hash=test_hash,
        generator_model=generator_model,
        generator_family=generator_family,
        answering_family=answering_family,
        judge_family=judge_family,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
