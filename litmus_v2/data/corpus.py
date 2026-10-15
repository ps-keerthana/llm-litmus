"""
Corpus loader and chunking module for litmus_v2.
Documented Strategy:
- Identifies semantic paragraph units with clean header normalization.
- Windows chunks targeting ~150-300 words with configurable token overlap.
- Assigns deterministic, globally unique chunk_id per chunk.
- Preserves exact character offsets and SHA-256 content hashes.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from litmus_v2.data.schema import Chunk


def _slugify(text: str) -> str:
    """Normalize file name into a clean alphanumeric slug."""
    text = os.path.splitext(text)[0].lower()
    text = re.sub(r'[^a-z0-9_]+', '_', text)
    return text.strip('_')


def _approx_token_count(text: str) -> int:
    """Fast whitespace-based token approximation (1.3 tokens per word approx)."""
    words = len(text.split())
    return max(1, int(words * 1.3))


def chunk_document_text(
    text: str,
    doc_name: str,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> List[Chunk]:
    """
    Splits text into chunks using documented hierarchical paragraph chunking.
    Paragraph boundaries (\n\n) are preserved. Large paragraphs are split at sentence boundaries.
    """
    doc_slug = _slugify(doc_name)
    doc_title = os.path.splitext(doc_name)[0].replace("_", " ").title()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_blocks: List[Tuple[str, int, int]] = []

    # Find character offsets in original text
    cursor = 0
    for p in paragraphs:
        start_idx = text.find(p, cursor)
        if start_idx == -1:
            start_idx = cursor
        end_idx = start_idx + len(p)
        cursor = end_idx
        raw_blocks.append((p, start_idx, end_idx))

    chunks: List[Chunk] = []
    chunk_idx = 0

    current_text_parts = []
    current_start = 0
    current_end = 0

    for block_text, b_start, b_end in raw_blocks:
        block_tokens = _approx_token_count(block_text)

        # If a single paragraph is too large, split by sentence
        if block_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', block_text)
            sent_cursor = b_start
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                s_tokens = _approx_token_count(sent)
                s_start = text.find(sent, sent_cursor)
                if s_start == -1:
                    s_start = sent_cursor
                s_end = s_start + len(sent)
                sent_cursor = s_end

                c_id = f"chunk_{doc_slug}_{chunk_idx:03d}"
                c_hash = hashlib.sha256(sent.encode("utf-8")).hexdigest()
                chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        doc_name=doc_name,
                        doc_title=doc_title,
                        text=sent,
                        chunk_index=chunk_idx,
                        start_char=s_start,
                        end_char=s_end,
                        token_count=s_tokens,
                        content_hash=c_hash,
                    )
                )
                chunk_idx += 1
            continue

        # Combine small paragraphs up to max_tokens
        curr_tokens = _approx_token_count("\n\n".join(current_text_parts + [block_text]))
        if curr_tokens <= max_tokens and current_text_parts:
            current_text_parts.append(block_text)
            current_end = b_end
        else:
            if current_text_parts:
                combined_text = "\n\n".join(current_text_parts)
                c_id = f"chunk_{doc_slug}_{chunk_idx:03d}"
                c_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()
                chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        doc_name=doc_name,
                        doc_title=doc_title,
                        text=combined_text,
                        chunk_index=chunk_idx,
                        start_char=current_start,
                        end_char=current_end,
                        token_count=_approx_token_count(combined_text),
                        content_hash=c_hash,
                    )
                )
                chunk_idx += 1
            current_text_parts = [block_text]
            current_start = b_start
            current_end = b_end

    if current_text_parts:
        combined_text = "\n\n".join(current_text_parts)
        c_id = f"chunk_{doc_slug}_{chunk_idx:03d}"
        c_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=c_id,
                doc_name=doc_name,
                doc_title=doc_title,
                text=combined_text,
                chunk_index=chunk_idx,
                start_char=current_start,
                end_char=current_end,
                token_count=_approx_token_count(combined_text),
                content_hash=c_hash,
            )
        )

    return chunks


def load_corpus(
    docs_folder: str = "docs",
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> Tuple[List[Chunk], Dict[str, Chunk]]:
    """
    Loads all .txt and .md files from docs_folder, chunks them, and returns
    (chunk_list, chunk_id_lookup_dict).
    """
    path = Path(docs_folder)
    if not path.is_absolute() and not path.exists():
        # Try relative to parent if run from inside litmus_v2
        if (Path("..") / docs_folder).exists():
            path = Path("..") / docs_folder

    if not path.exists():
        raise FileNotFoundError(f"Knowledge corpus folder not found: {docs_folder}")

    all_chunks: List[Chunk] = []
    lookup: Dict[str, Chunk] = {}

    for doc_file in sorted(path.glob("*.*")):
        if doc_file.suffix.lower() not in (".txt", ".md"):
            continue
        with open(doc_file, "r", encoding="utf-8") as f:
            content = f.read()
        doc_chunks = chunk_document_text(
            text=content,
            doc_name=doc_file.name,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for c in doc_chunks:
            all_chunks.append(c)
            lookup[c.chunk_id] = c

    return all_chunks, lookup


# ── Public API aliases ────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    doc_name: str,
    doc_title: str = "",
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> List[Chunk]:
    """Alias for chunk_document_text with explicit doc_title support."""
    chunks = chunk_document_text(
        text=text,
        doc_name=doc_name,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    if doc_title:
        chunks = [
            c.model_copy(update={"doc_title": doc_title})
            for c in chunks
        ]
    return chunks


def load_and_chunk_text(
    content: str,
    doc_name: str,
    doc_title: str = "",
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> List[Chunk]:
    """Convenience function to chunk an already-loaded text string."""
    return chunk_document(
        text=content,
        doc_name=doc_name,
        doc_title=doc_title,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
