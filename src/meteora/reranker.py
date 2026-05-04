from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, Union

from .embeddings import EncoderLike
from .rationales import coerce_rationales
from .selector import MeteoraConfig, MeteoraSelector
from .types import Chunk, ChunkLike, Rationale, RationaleLike, SelectionResult
from .verifier import MeteoraVerifier, VerificationResult, VerifiedSelection, VerifierModelLike


class RationaleGenerator(Protocol):
    def __call__(self, query: str, documents: Sequence[str]) -> Any:
        ...


RationaleGeneratorLike = Union[RationaleGenerator, Callable[..., Any]]


@dataclass(frozen=True)
class RerankResult:
    """Reranker-compatible result for one METEORA-selected document."""

    index: int
    text: str
    score: float
    rank: int
    document: Any
    chunk: Chunk
    verification: Optional[VerificationResult] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.verification.accepted if self.verification is not None else True

    def to_dict(self):
        payload = {
            "index": self.index,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "accepted": self.accepted,
            "chunk": self.chunk.to_dict(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.verification is not None:
            payload["verification"] = self.verification.to_dict()
        return payload


@dataclass(frozen=True)
class MeteoraRerankTrace:
    """Diagnostics from one reranker-style METEORA call."""

    query: str
    rationales: Sequence[Rationale]
    selection: SelectionResult
    verification: Optional[VerifiedSelection]
    results: Sequence[RerankResult]

    @property
    def documents(self) -> Sequence[Any]:
        return tuple(result.document for result in self.results)

    @property
    def indices(self) -> Sequence[int]:
        return tuple(result.index for result in self.results)

    def to_dict(self):
        return {
            "query": self.query,
            "indices": list(self.indices),
            "rationales": [rationale.to_dict() for rationale in self.rationales],
            "selection": self.selection.to_dict(),
            "verification": self.verification.to_dict() if self.verification else None,
            "results": [result.to_dict() for result in self.results],
        }


class MeteoraReranker:
    """Drop-in facade for replacing a conventional reranker in RAG pipelines.

    `rerank(query, documents)` returns METEORA-selected documents as scored result
    objects. No fixed top-k is applied unless `top_k` is explicitly passed for
    compatibility with an existing pipeline.
    """

    def __init__(
        self,
        encoder: EncoderLike,
        *,
        rationale_generator: Optional[RationaleGeneratorLike] = None,
        verifier: Optional[Union[MeteoraVerifier, VerifierModelLike]] = None,
        selector: Optional[MeteoraSelector] = None,
        config: Optional[MeteoraConfig] = None,
        fallback_to_query_rationale: bool = True,
        default_order: str = "score",
    ) -> None:
        reranker_config = config or MeteoraConfig(expansion_window=0)
        self.selector = selector or MeteoraSelector(encoder, config=reranker_config)
        self.rationale_generator = rationale_generator
        self.verifier = _coerce_verifier(verifier)
        self.fallback_to_query_rationale = fallback_to_query_rationale
        self.default_order = default_order

    def rerank(
        self,
        query: str,
        documents: Iterable[Any],
        *,
        rationales: Optional[Union[Iterable[RationaleLike], str]] = None,
        top_k: Optional[int] = None,
        verify: Optional[bool] = None,
        order: Optional[str] = None,
    ) -> Sequence[RerankResult]:
        """Return METEORA-selected documents in a reranker-style result list."""

        return self.trace(
            query=query,
            documents=documents,
            rationales=rationales,
            top_k=top_k,
            verify=verify,
            order=order,
        ).results

    def filter(
        self,
        query: str,
        documents: Iterable[Any],
        *,
        rationales: Optional[Union[Iterable[RationaleLike], str]] = None,
        top_k: Optional[int] = None,
        verify: Optional[bool] = None,
        order: Optional[str] = None,
    ) -> Sequence[Any]:
        """Return only the selected original documents for pipeline drop-in use."""

        return tuple(
            result.document
            for result in self.rerank(
                query=query,
                documents=documents,
                rationales=rationales,
                top_k=top_k,
                verify=verify,
                order=order,
            )
        )

    def trace(
        self,
        query: str,
        documents: Iterable[Any],
        *,
        rationales: Optional[Union[Iterable[RationaleLike], str]] = None,
        top_k: Optional[int] = None,
        verify: Optional[bool] = None,
        order: Optional[str] = None,
    ) -> MeteoraRerankTrace:
        """Return selected documents plus selection and verification diagnostics."""

        prepared = tuple(_prepare_documents(documents))
        chunks = tuple(item[0] for item in prepared)
        original_documents = {id(chunk): original for chunk, original in prepared}

        resolved_rationales = self._resolve_rationales(query, chunks, rationales)
        selection = self.selector.select(chunks, resolved_rationales)

        should_verify = self.verifier is not None if verify is None else verify
        verification = None
        verification_by_chunk_id = {}
        accepted_chunk_ids = {id(chunk) for chunk in selection.selected_chunks}
        if should_verify:
            if self.verifier is None:
                raise ValueError("verify=True requires a verifier.")
            verification = self.verifier.verify_selection(query, selection)
            verification_by_chunk_id = {
                id(result.chunk): result for result in verification.results
            }
            accepted_chunk_ids = {
                id(result.chunk) for result in verification.results if result.accepted
            }

        score_by_index = _score_selected_chunks(selection)
        items = [
            RerankResult(
                index=chunk.index,
                text=chunk.text,
                score=score_by_index.get(chunk.index, 0.0),
                rank=0,
                document=original_documents[id(chunk)],
                chunk=chunk,
                verification=verification_by_chunk_id.get(id(chunk)),
                metadata=chunk.metadata,
            )
            for chunk in selection.selected_chunks
            if id(chunk) in accepted_chunk_ids
        ]
        ordered = _order_results(items, order or self.default_order)
        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k must be non-negative.")
            ordered = ordered[:top_k]
        ranked = tuple(
            RerankResult(
                index=item.index,
                text=item.text,
                score=item.score,
                rank=rank,
                document=item.document,
                chunk=item.chunk,
                verification=item.verification,
                metadata=item.metadata,
            )
            for rank, item in enumerate(ordered, start=1)
        )

        return MeteoraRerankTrace(
            query=query,
            rationales=resolved_rationales,
            selection=selection,
            verification=verification,
            results=ranked,
        )

    def _resolve_rationales(
        self,
        query: str,
        chunks: Sequence[Chunk],
        rationales: Optional[Union[Iterable[RationaleLike], str]],
    ) -> Sequence[Rationale]:
        if rationales is not None:
            return tuple(coerce_rationales(rationales))

        if self.rationale_generator is not None:
            generated = _call_rationale_generator(
                self.rationale_generator,
                query=query,
                documents=[chunk.text for chunk in chunks],
            )
            return tuple(coerce_rationales(generated))

        if self.fallback_to_query_rationale:
            return tuple(coerce_rationales([query]))

        raise ValueError(
            "No rationales supplied. Pass rationales, configure a rationale_generator, "
            "or enable fallback_to_query_rationale."
        )


def _prepare_documents(documents: Iterable[Any]) -> Sequence[tuple[Chunk, Any]]:
    prepared = []
    for index, document in enumerate(documents):
        chunk = _document_to_chunk(document, index)
        prepared.append((chunk, document))
    return tuple(prepared)


def _document_to_chunk(document: Any, index: int) -> Chunk:
    if isinstance(document, Chunk):
        return document
    if isinstance(document, str):
        return Chunk(text=document, index=index)
    if isinstance(document, Mapping):
        text = document.get("text") or document.get("page_content") or document.get("content")
        if text is None:
            raise ValueError(
                "Document mappings must include one of: 'text', 'page_content', or 'content'."
            )
        metadata = {
            key: value
            for key, value in document.items()
            if key not in {"text", "page_content", "content", "index", "start_pos", "end_pos"}
        }
        return Chunk(
            text=str(text),
            index=int(document.get("index", index)),
            start_pos=_optional_int(document.get("start_pos")),
            end_pos=_optional_int(document.get("end_pos")),
            metadata=metadata,
        )

    text = getattr(document, "page_content", None)
    if text is None:
        text = getattr(document, "text", None)
    if text is None:
        raise TypeError(
            "Documents must be strings, mappings, Chunk objects, or objects with "
            "`page_content` or `text` attributes."
        )
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    return Chunk(text=str(text), index=index, metadata=metadata)


def _score_selected_chunks(selection: SelectionResult) -> Mapping[int, float]:
    scores = {score.chunk_index: score.score for score in selection.details.pooled_scores}
    for match in selection.details.rationale_matches:
        scores[match.chunk_index] = max(scores.get(match.chunk_index, 0.0), match.score)
    return scores


def _order_results(items: Sequence[RerankResult], order: str) -> Sequence[RerankResult]:
    if order == "score":
        return tuple(sorted(items, key=lambda item: (-item.score, item.index)))
    if order in {"document", "input", "context"}:
        return tuple(sorted(items, key=lambda item: item.index))
    raise ValueError("order must be one of: 'score', 'document', 'input', or 'context'.")


def _call_rationale_generator(generator: RationaleGeneratorLike, query: str, documents: Sequence[str]) -> Any:
    if callable(generator):
        try:
            return generator(query, documents)
        except TypeError:
            return generator(query)
    for method_name in ("generate", "complete", "invoke"):
        method = getattr(generator, method_name, None)
        if method is not None:
            try:
                return method(query=query, documents=documents)
            except TypeError:
                return method(query)
    raise TypeError("Rationale generator must be callable or expose generate, complete, or invoke.")


def _coerce_verifier(
    verifier: Optional[Union[MeteoraVerifier, VerifierModelLike]]
) -> Optional[MeteoraVerifier]:
    if verifier is None:
        return None
    if isinstance(verifier, MeteoraVerifier):
        return verifier
    return MeteoraVerifier(verifier)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
