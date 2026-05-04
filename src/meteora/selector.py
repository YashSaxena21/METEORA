from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Union

import numpy as np

from .embeddings import EncoderLike, cosine_similarity_matrix, encode_texts
from .rationales import coerce_rationales
from .types import (
    Chunk,
    ChunkLike,
    ChunkScore,
    Rationale,
    RationaleLike,
    RationaleMatch,
    SelectionDetails,
    SelectionResult,
    normalize_chunks,
)


@dataclass(frozen=True)
class MeteoraConfig:
    """Selection settings for METEORA's rank-free stages."""

    expansion_window: int = 1
    elbow_z_threshold: float = 1.0
    enable_pairing: bool = True
    enable_pooling: bool = True
    enable_expansion: bool = True


class MeteoraSelector:
    """Rationale-driven, rank-free evidence selector.

    The selector implements the three-stage ECSE routine used by METEORA:

    1. Pair each rationale to its best matching chunk, allowing convergence.
    2. Pool rationale embeddings and select chunks up to a statistical elbow.
    3. Expand by neighboring context windows.
    """

    def __init__(self, encoder: EncoderLike, config: Optional[MeteoraConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either a MeteoraConfig or keyword config values, not both.")
        self.encoder = encoder
        self.config = config or MeteoraConfig(**kwargs)
        if self.config.expansion_window < 0:
            raise ValueError("expansion_window must be non-negative.")

    def select(
        self,
        chunks: Iterable[ChunkLike],
        rationales: Union[Iterable[RationaleLike], str],
    ) -> SelectionResult:
        normalized_chunks = tuple(normalize_chunks(chunks))
        normalized_rationales = tuple(coerce_rationales(rationales))

        if not normalized_chunks:
            return _empty_result(normalized_rationales)
        if not normalized_rationales:
            return _result_from_indices(normalized_chunks, normalized_rationales, set(), [], [], None, {})

        chunk_embeddings = encode_texts(self.encoder, [chunk.text for chunk in normalized_chunks])
        rationale_embeddings = encode_texts(
            self.encoder, [rationale.text for rationale in normalized_rationales]
        )

        pairing_positions: Set[int] = set()
        matches: List[RationaleMatch] = []
        contributions = {}
        if self.config.enable_pairing:
            pairing_positions, matches, contributions = self._pair_rationales(
                normalized_chunks, normalized_rationales, chunk_embeddings, rationale_embeddings
            )

        pooled_positions: Set[int] = set()
        pooled_scores: List[ChunkScore] = []
        elbow_index: Optional[int] = None
        if self.config.enable_pooling:
            pooled_positions, pooled_scores, elbow_index = self._pool_rationales(
                normalized_chunks, chunk_embeddings, rationale_embeddings
            )

        combined_positions = pairing_positions.union(pooled_positions)
        expansion_positions = (
            self._expand(combined_positions, len(normalized_chunks))
            if self.config.enable_expansion
            else set()
        )
        selected_positions = combined_positions.union(expansion_positions)

        return _result_from_positions(
            chunks=normalized_chunks,
            rationales=normalized_rationales,
            selected_positions=selected_positions,
            pairing_positions=pairing_positions,
            pooled_positions=pooled_positions,
            expansion_positions=expansion_positions,
            matches=matches,
            pooled_scores=pooled_scores,
            elbow_index=elbow_index,
            contributions=contributions,
        )

    def _pair_rationales(
        self,
        chunks: Sequence[Chunk],
        rationales: Sequence[Rationale],
        chunk_embeddings: np.ndarray,
        rationale_embeddings: np.ndarray,
    ):
        scores = cosine_similarity_matrix(rationale_embeddings, chunk_embeddings)
        selected_positions: Set[int] = set()
        matches: List[RationaleMatch] = []
        contributions = {}

        for rationale_pos, rationale in enumerate(rationales):
            best_position = int(np.argmax(scores[rationale_pos]))
            selected_positions.add(best_position)
            chunk_index = chunks[best_position].index
            contributions.setdefault(chunk_index, []).append(rationale.index)
            matches.append(
                RationaleMatch(
                    rationale_index=rationale.index,
                    chunk_index=chunk_index,
                    score=float(scores[rationale_pos, best_position]),
                )
            )

        return selected_positions, matches, contributions

    def _pool_rationales(
        self,
        chunks: Sequence[Chunk],
        chunk_embeddings: np.ndarray,
        rationale_embeddings: np.ndarray,
    ):
        pooled_embedding = rationale_embeddings.mean(axis=0, keepdims=True)
        scores = cosine_similarity_matrix(pooled_embedding, chunk_embeddings)[0]
        ordered_positions = list(np.argsort(scores)[::-1])
        ordered_scores = [float(scores[position]) for position in ordered_positions]
        elbow_index = statistical_elbow(ordered_scores, z_threshold=self.config.elbow_z_threshold)
        if elbow_index is None:
            return set(), [], None

        selected_positions = set(ordered_positions[: elbow_index + 1])
        pooled_scores = [
            ChunkScore(chunk_index=chunks[position].index, score=float(scores[position]))
            for position in ordered_positions
        ]
        return selected_positions, pooled_scores, elbow_index

    def _expand(self, positions: Iterable[int], total_chunks: int) -> Set[int]:
        expanded: Set[int] = set()
        window = self.config.expansion_window
        for position in positions:
            start = max(0, position - window)
            stop = min(total_chunks - 1, position + window)
            for neighbor in range(start, stop + 1):
                if neighbor != position:
                    expanded.add(neighbor)
        return expanded


def statistical_elbow(scores: Sequence[float], z_threshold: float = 1.0) -> Optional[int]:
    """Return the inclusive elbow index for a descending score list."""

    if not scores:
        return None
    if len(scores) <= 2:
        return len(scores) - 1

    score_array = np.asarray(scores, dtype=np.float64)
    diffs = score_array[:-1] - score_array[1:]
    if diffs.size == 0:
        return 0

    std = float(diffs.std())
    if std > 0.0:
        z_scores = (diffs - float(diffs.mean())) / std
        significant = np.flatnonzero(z_scores > z_threshold)
        if significant.size:
            return int(significant[0])

    second_diffs = np.diff(diffs)
    if second_diffs.size:
        return int(np.argmax(second_diffs))
    return 0


def _empty_result(rationales: Sequence[Rationale]) -> SelectionResult:
    details = SelectionDetails(
        pairing_indices=[],
        pooled_indices=[],
        expansion_indices=[],
        elbow_index=None,
        rationale_matches=[],
        pooled_scores=[],
        rationale_contributions={},
    )
    return SelectionResult(selected_indices=[], selected_chunks=[], rationales=rationales, details=details)


def _result_from_indices(
    chunks: Sequence[Chunk],
    rationales: Sequence[Rationale],
    selected_indices: Set[int],
    matches: List[RationaleMatch],
    pooled_scores: List[ChunkScore],
    elbow_index: Optional[int],
    contributions,
) -> SelectionResult:
    selected_chunks = [chunk for chunk in chunks if chunk.index in selected_indices]
    details = SelectionDetails(
        pairing_indices=[],
        pooled_indices=[],
        expansion_indices=[],
        elbow_index=elbow_index,
        rationale_matches=matches,
        pooled_scores=pooled_scores,
        rationale_contributions=contributions,
    )
    return SelectionResult(
        selected_indices=sorted(selected_indices),
        selected_chunks=selected_chunks,
        rationales=rationales,
        details=details,
    )


def _result_from_positions(
    chunks: Sequence[Chunk],
    rationales: Sequence[Rationale],
    selected_positions: Set[int],
    pairing_positions: Set[int],
    pooled_positions: Set[int],
    expansion_positions: Set[int],
    matches: List[RationaleMatch],
    pooled_scores: List[ChunkScore],
    elbow_index: Optional[int],
    contributions,
) -> SelectionResult:
    selected_indices = sorted(chunks[position].index for position in selected_positions)
    selected_chunks = [chunk for position, chunk in enumerate(chunks) if position in selected_positions]
    details = SelectionDetails(
        pairing_indices=sorted(chunks[position].index for position in pairing_positions),
        pooled_indices=sorted(chunks[position].index for position in pooled_positions),
        expansion_indices=sorted(chunks[position].index for position in expansion_positions),
        elbow_index=elbow_index,
        rationale_matches=matches,
        pooled_scores=pooled_scores,
        rationale_contributions=contributions,
    )
    return SelectionResult(
        selected_indices=selected_indices,
        selected_chunks=selected_chunks,
        rationales=rationales,
        details=details,
    )
