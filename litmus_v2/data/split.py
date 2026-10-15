"""
Leakage-free dataset splitting module.
Partitions candidate Q/A items by document/chunk group into distinct Dev and Test sets.
Guarantees zero overlap in source documents between training/development and test evaluation.
"""

import random
from typing import List, Tuple, Set, Dict
from litmus_v2.data.schema import CandidateQA


def split_dataset_by_group(
    items: List[CandidateQA],
    test_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[CandidateQA], List[CandidateQA]]:
    """
    Performs a leak-free group split by `doc_group`.
    Every document belongs either entirely to the dev set or entirely to the test set.

    Returns:
        (dev_items, test_items)
    """
    if not items:
        return [], []

    # Map groups to items
    group_to_items: Dict[str, List[CandidateQA]] = {}
    for item in items:
        group_to_items.setdefault(item.doc_group, []).append(item)

    groups = sorted(list(group_to_items.keys()))
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_items = len(items)
    target_test_count = int(total_items * test_ratio)

    test_groups: Set[str] = set()
    current_test_count = 0

    for g in groups:
        g_len = len(group_to_items[g])
        if current_test_count + g_len <= target_test_count or not test_groups:
            test_groups.add(g)
            current_test_count += g_len
        else:
            break

    dev_groups = set(groups) - test_groups

    dev_items = [item for g in sorted(dev_groups) for item in group_to_items[g]]
    test_items = [item for g in sorted(test_groups) for item in group_to_items[g]]

    # Assert 0 leakage
    assert dev_groups.isdisjoint(test_groups), "Leakage detected: shared document group between dev and test!"

    return dev_items, test_items


def verify_no_leakage(dev_items: List[CandidateQA], test_items: List[CandidateQA]) -> bool:
    """Verifies that dev and test items do not share any document groups or chunk IDs."""
    dev_groups = {it.doc_group for it in dev_items}
    test_groups = {it.doc_group for it in test_items}

    dev_chunks = {it.primary_chunk_id for it in dev_items if it.primary_chunk_id}
    test_chunks = {it.primary_chunk_id for it in test_items if it.primary_chunk_id}

    for it in dev_items:
        dev_chunks.update(it.supporting_chunk_ids)
    for it in test_items:
        test_chunks.update(it.supporting_chunk_ids)

    group_overlap = dev_groups.intersection(test_groups)
    chunk_overlap = dev_chunks.intersection(test_chunks)

    if group_overlap:
        raise ValueError(f"Document group leakage detected: {group_overlap}")
    if chunk_overlap:
        raise ValueError(f"Chunk ID leakage detected: {chunk_overlap}")

    return True


# ── Public alias ─────────────────────────────────────────────────────────────

def split_by_doc_group(
    items: List[CandidateQA],
    test_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[List[CandidateQA], List[CandidateQA]]:
    """Alias for split_dataset_by_group with test_fraction parameter name."""
    return split_dataset_by_group(items, test_ratio=test_fraction, seed=seed)
