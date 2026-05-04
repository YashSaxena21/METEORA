from __future__ import annotations

import json
import re
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, Union

from .prompts import build_verifier_prompt
from .rationales import coerce_rationales
from .types import Chunk, ChunkLike, Rationale, RationaleLike, SelectionResult, normalize_chunks


class VerifierModel(Protocol):
    def __call__(self, prompt: str) -> Any:
        ...


VerifierModelLike = Union[VerifierModel, Callable[[str], Any]]


@dataclass(frozen=True)
class VerificationResult:
    """Verifier decision for one selected evidence chunk."""

    chunk: Chunk
    relevant: bool
    flagged: bool
    reason: str = ""
    flag_types: Sequence[str] = field(default_factory=tuple)
    confidence: Optional[float] = None
    raw_response: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.relevant and not self.flagged

    def to_dict(self):
        payload = {
            "chunk": self.chunk.to_dict(),
            "relevant": self.relevant,
            "flagged": self.flagged,
            "accepted": self.accepted,
            "reason": self.reason,
            "flag_types": list(self.flag_types),
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.metadata:
            payload["metadata"] = _json_safe(self.metadata)
        if self.raw_response is not None:
            payload["raw_response"] = _json_safe(self.raw_response)
        return payload


@dataclass(frozen=True)
class VerifiedSelection:
    """A selected evidence set after verifier filtering."""

    query: str
    rationales: Sequence[Rationale]
    results: Sequence[VerificationResult]

    @property
    def accepted_chunks(self) -> Sequence[Chunk]:
        return tuple(result.chunk for result in self.results if result.accepted)

    @property
    def flagged_chunks(self) -> Sequence[Chunk]:
        return tuple(result.chunk for result in self.results if result.flagged)

    @property
    def rejected_chunks(self) -> Sequence[Chunk]:
        return tuple(result.chunk for result in self.results if not result.accepted)

    @property
    def accepted_indices(self) -> Sequence[int]:
        return tuple(chunk.index for chunk in self.accepted_chunks)

    @property
    def flagged_indices(self) -> Sequence[int]:
        return tuple(chunk.index for chunk in self.flagged_chunks)

    @property
    def rejected_indices(self) -> Sequence[int]:
        return tuple(chunk.index for chunk in self.rejected_chunks)

    def to_dict(self):
        return {
            "query": self.query,
            "accepted_indices": list(self.accepted_indices),
            "flagged_indices": list(self.flagged_indices),
            "rejected_indices": list(self.rejected_indices),
            "accepted_chunks": [chunk.to_dict() for chunk in self.accepted_chunks],
            "flagged_chunks": [chunk.to_dict() for chunk in self.flagged_chunks],
            "rejected_chunks": [chunk.to_dict() for chunk in self.rejected_chunks],
            "rationales": [rationale.to_dict() for rationale in self.rationales],
            "results": [result.to_dict() for result in self.results],
        }


class MeteoraVerifier:
    """Runnable verifier stage for filtering selected METEORA evidence."""

    def __init__(self, model: VerifierModelLike, fail_closed: bool = True) -> None:
        self.model = model
        self.fail_closed = fail_closed

    def verify_chunk(
        self,
        query: str,
        chunk: ChunkLike,
        rationales: Union[Iterable[RationaleLike], str],
        *,
        accepted_chunks: Optional[Sequence[ChunkLike]] = None,
    ) -> VerificationResult:
        normalized_chunk = normalize_chunks([chunk])[0]
        normalized_rationales = tuple(coerce_rationales(rationales))
        normalized_accepted = tuple(normalize_chunks(accepted_chunks or []))
        prompt = build_verifier_prompt(
            query=query,
            chunk=normalized_chunk,
            rationales=normalized_rationales,
            accepted_chunks=normalized_accepted,
        )
        raw_response = _call_model(self.model, prompt)
        return parse_verifier_response(
            raw_response,
            chunk=normalized_chunk,
            fail_closed=self.fail_closed,
        )

    def verify_chunks(
        self,
        query: str,
        chunks: Iterable[ChunkLike],
        rationales: Union[Iterable[RationaleLike], str],
    ) -> VerifiedSelection:
        normalized_chunks = tuple(normalize_chunks(chunks))
        normalized_rationales = tuple(coerce_rationales(rationales))
        accepted_chunks = []
        results = []

        for chunk in normalized_chunks:
            result = self.verify_chunk(
                query=query,
                chunk=chunk,
                rationales=normalized_rationales,
                accepted_chunks=accepted_chunks,
            )
            results.append(result)
            if result.accepted:
                accepted_chunks.append(result.chunk)

        return VerifiedSelection(query=query, rationales=normalized_rationales, results=tuple(results))

    def verify_selection(self, query: str, selection: SelectionResult) -> VerifiedSelection:
        return self.verify_chunks(
            query=query,
            chunks=selection.selected_chunks,
            rationales=selection.rationales,
        )


def parse_verifier_response(
    response: Any,
    *,
    chunk: Chunk,
    fail_closed: bool = True,
) -> VerificationResult:
    """Parse a verifier model response into a structured decision."""

    try:
        payload = _coerce_response_payload(response)
    except ValueError as exc:
        flagged = bool(fail_closed)
        return VerificationResult(
            chunk=chunk,
            relevant=not flagged,
            flagged=flagged,
            reason=f"Could not parse verifier response: {exc}",
            flag_types=("PARSE_ERROR",) if flagged else (),
            raw_response=response,
        )

    flagged = _to_bool(payload.get("flagged"), default=False)
    relevant = _to_bool(payload.get("relevant"), default=not flagged)
    reason = str(payload.get("reason") or payload.get("explanation") or "").strip()
    confidence = _optional_float(payload.get("confidence"))
    flag_types = _normalize_flag_types(payload.get("flag_types") or payload.get("flags"))
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"relevant", "flagged", "reason", "explanation", "confidence", "flag_types", "flags"}
    }

    if flagged and not flag_types:
        flag_types = ("FLAGGED",)

    return VerificationResult(
        chunk=chunk,
        relevant=relevant,
        flagged=flagged,
        reason=reason,
        flag_types=flag_types,
        confidence=confidence,
        raw_response=response,
        metadata=metadata,
    )


def _call_model(model: VerifierModelLike, prompt: str) -> Any:
    if callable(model):
        return model(prompt)
    for method_name in ("generate", "complete", "invoke"):
        method = getattr(model, method_name, None)
        if method is not None:
            return method(prompt)
    raise TypeError("Verifier model must be callable or expose generate, complete, or invoke.")


def _coerce_response_payload(response: Any) -> Mapping[str, Any]:
    if isinstance(response, MappingABC):
        return response

    content = _extract_text_response(response)
    if content is None:
        raise ValueError(f"unsupported response type {type(response)!r}")

    json_text = _extract_json_object(content)
    if json_text:
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
        if isinstance(payload, MappingABC):
            return payload
        raise ValueError("verifier JSON response must be an object")

    fallback = _parse_boolean_fallback(content)
    if fallback:
        return fallback
    raise ValueError("no JSON object or boolean verifier fields found")


def _extract_text_response(response: Any) -> Optional[str]:
    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace")
    for attr in ("content", "text", "output_text"):
        value = getattr(response, attr, None)
        if value is not None:
            return str(value)
    return None


def _extract_json_object(text: str) -> Optional[str]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def _parse_boolean_fallback(text: str) -> Optional[Mapping[str, Any]]:
    flagged = _match_bool_field(text, "flagged")
    relevant = _match_bool_field(text, "relevant")
    if flagged is None and relevant is None:
        return None
    reason_match = re.search(r"reason\s*[:=-]\s*(.+)", text, re.IGNORECASE)
    return {
        "flagged": flagged if flagged is not None else False,
        "relevant": relevant if relevant is not None else not bool(flagged),
        "reason": reason_match.group(1).strip() if reason_match else "",
    }


def _match_bool_field(text: str, field: str) -> Optional[bool]:
    match = re.search(rf"\b{field}\b\s*[:=-]\s*(true|false|yes|no)", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() in {"true", "yes"}


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "accepted"}:
            return True
        if normalized in {"false", "no", "0", "rejected"}:
            return False
    return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_flag_types(value: Any) -> Sequence[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, IterableABC):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, IterableABC):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)
