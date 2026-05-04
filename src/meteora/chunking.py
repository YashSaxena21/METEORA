from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from .types import Chunk

_WORD_RE = re.compile(r"\S+")


def chunk_text(text: str, chunk_size: int = 256, overlap: Optional[int] = None) -> List[Chunk]:
    """Split text into word chunks while preserving character offsets."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return []

    if overlap is None:
        overlap = max(0, chunk_size // 10)
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    for start_word in range(0, len(matches), step):
        end_word = min(start_word + chunk_size, len(matches))
        start_pos = matches[start_word].start()
        end_pos = matches[end_word - 1].end()
        chunk = text[start_pos:end_pos]
        chunks.append(
            Chunk(text=chunk, index=len(chunks), start_pos=start_pos, end_pos=end_pos)
        )
        if end_word >= len(matches):
            break
    return chunks


def find_spanning_chunks(
    span_start: Optional[int], span_end: Optional[int], chunks: Sequence[Chunk]
) -> List[int]:
    """Return indices of chunks that overlap a source-character span."""

    if span_start is None or span_end is None:
        return []
    if span_start > span_end:
        raise ValueError("span_start must be <= span_end.")

    overlapping = [
        chunk.index
        for chunk in chunks
        if chunk.start_pos is not None
        and chunk.end_pos is not None
        and chunk.start_pos <= span_end
        and chunk.end_pos >= span_start
    ]
    if not overlapping:
        return []
    return list(range(min(overlapping), max(overlapping) + 1))


def chunk_texts(texts: Iterable[str], chunk_size: int = 256, overlap: Optional[int] = None) -> List[Chunk]:
    """Chunk multiple documents and retain each document id in chunk metadata."""

    all_chunks: List[Chunk] = []
    for doc_id, text in enumerate(texts):
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            all_chunks.append(
                Chunk(
                    text=chunk.text,
                    index=len(all_chunks),
                    start_pos=chunk.start_pos,
                    end_pos=chunk.end_pos,
                    metadata={"document_index": doc_id, "local_chunk_index": chunk.index},
                )
            )
    return all_chunks
