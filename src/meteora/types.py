from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class Chunk:
    """A document chunk plus optional source offsets."""

    text: str
    index: int
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"index": self.index, "text": self.text}
        if self.start_pos is not None:
            payload["start_pos"] = self.start_pos
        if self.end_pos is not None:
            payload["end_pos"] = self.end_pos
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class Rationale:
    """A generated search rationale used to select evidence."""

    text: str
    index: int
    label: Optional[str] = None
    flag_instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"index": self.index, "text": self.text}
        if self.label:
            payload["label"] = self.label
        if self.flag_instructions:
            payload["flag_instructions"] = self.flag_instructions
        return payload


@dataclass(frozen=True)
class RationaleMatch:
    rationale_index: int
    chunk_index: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rationale_index": self.rationale_index,
            "chunk_index": self.chunk_index,
            "score": self.score,
        }


@dataclass(frozen=True)
class ChunkScore:
    chunk_index: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_index": self.chunk_index, "score": self.score}


@dataclass(frozen=True)
class SelectionDetails:
    pairing_indices: Sequence[int]
    pooled_indices: Sequence[int]
    expansion_indices: Sequence[int]
    elbow_index: Optional[int]
    rationale_matches: Sequence[RationaleMatch]
    pooled_scores: Sequence[ChunkScore]
    rationale_contributions: Mapping[int, Sequence[int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairing_indices": list(self.pairing_indices),
            "pooled_indices": list(self.pooled_indices),
            "expansion_indices": list(self.expansion_indices),
            "pairing_and_pooling": sorted(set(self.pairing_indices).union(self.pooled_indices)),
            "all_stages": sorted(
                set(self.pairing_indices).union(self.pooled_indices).union(self.expansion_indices)
            ),
            "elbow_index": self.elbow_index,
            "rationale_matches": [match.to_dict() for match in self.rationale_matches],
            "pooled_scores": [score.to_dict() for score in self.pooled_scores],
            "rationale_contributions": {
                str(chunk_index): list(rationale_indices)
                for chunk_index, rationale_indices in self.rationale_contributions.items()
            },
        }


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: Sequence[int]
    selected_chunks: Sequence[Chunk]
    rationales: Sequence[Rationale]
    details: SelectionDetails

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_indices": list(self.selected_indices),
            "selected_chunks": [chunk.to_dict() for chunk in self.selected_chunks],
            "rationales": [rationale.to_dict() for rationale in self.rationales],
            "details": self.details.to_dict(),
        }


ChunkLike = Union[str, Mapping[str, Any], Chunk]
RationaleLike = Union[str, Sequence[Any], Mapping[str, Any], Rationale]


def normalize_chunks(chunks: Iterable[ChunkLike]) -> Sequence[Chunk]:
    normalized = []
    for position, chunk in enumerate(chunks):
        if isinstance(chunk, Chunk):
            normalized.append(chunk)
            continue
        if isinstance(chunk, str):
            normalized.append(Chunk(text=chunk, index=position))
            continue
        if isinstance(chunk, Mapping):
            if "text" not in chunk:
                raise ValueError(f"Chunk at position {position} is missing a 'text' field.")
            metadata = {
                key: value
                for key, value in chunk.items()
                if key not in {"text", "index", "start_pos", "end_pos"}
            }
            normalized.append(
                Chunk(
                    text=str(chunk["text"]),
                    index=int(chunk.get("index", position)),
                    start_pos=_optional_int(chunk.get("start_pos")),
                    end_pos=_optional_int(chunk.get("end_pos")),
                    metadata=metadata,
                )
            )
            continue
        raise TypeError(f"Unsupported chunk type at position {position}: {type(chunk)!r}")
    return tuple(normalized)


def normalize_rationales(rationales: Iterable[RationaleLike]) -> Sequence[Rationale]:
    normalized = []
    for position, rationale in enumerate(rationales, start=1):
        if isinstance(rationale, Rationale):
            normalized.append(rationale)
            continue
        if isinstance(rationale, str):
            normalized.append(Rationale(text=rationale, index=position))
            continue
        if isinstance(rationale, Mapping):
            text = rationale.get("text") or rationale.get("rationale")
            if not text:
                raise ValueError(f"Rationale at position {position} is missing text.")
            normalized.append(
                Rationale(
                    text=str(text),
                    index=int(rationale.get("index", position)),
                    label=_optional_str(rationale.get("label")),
                    flag_instructions=_optional_str(rationale.get("flag_instructions")),
                )
            )
            continue
        if isinstance(rationale, Sequence) and len(rationale) >= 2:
            normalized.append(Rationale(text=str(rationale[1]), index=int(rationale[0])))
            continue
        raise TypeError(f"Unsupported rationale type at position {position}: {type(rationale)!r}")
    return tuple(normalized)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
