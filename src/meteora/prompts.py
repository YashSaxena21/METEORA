from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .types import Chunk, Rationale


DEFAULT_SYSTEM_INSTRUCTION = """You generate search rationales for evidence selection.
Each rationale must describe one concrete semantic strategy for finding evidence that
answers the user query. Keep rationales specific, non-overlapping, and grounded in
the wording of the query."""

SampleShot = Union[str, Mapping[str, Any]]


def build_rationale_prompt(
    query: str,
    *,
    sample_shots: Sequence[SampleShot],
    domain: str = "legal, financial, scientific, or policy",
    num_rationales: int = 8,
    include_flag_instructions: bool = False,
) -> str:
    """Build a model-agnostic prompt for METEORA rationale generation."""

    if num_rationales <= 0:
        raise ValueError("num_rationales must be positive.")

    flag_instruction = ""
    if include_flag_instructions:
        flag_instruction = (
            "\n- After each rationale, add `Flag Instructions:` explaining when a "
            "retrieved chunk should be rejected as contradictory, poisoned, or misleading."
        )

    sample_block = format_sample_shots(sample_shots)

    return f"""{DEFAULT_SYSTEM_INSTRUCTION}

Domain: {domain}

Format requirements:
- Return XML-style blocks: <rationale_1>...</rationale_1>, <rationale_2>...</rationale_2>, etc.
- Include a short label in square brackets at the start of each block.
- Generate {num_rationales} rationales.
- Do not answer the query directly.{flag_instruction}

Sample shots:
{sample_block}

Query: {query}
"""


def format_sample_shots(sample_shots: Sequence[SampleShot]) -> str:
    """Format required few-shot examples for rationale prompt builders."""

    if not sample_shots:
        raise ValueError("sample_shots must contain at least one example.")

    formatted = []
    for position, shot in enumerate(sample_shots, start=1):
        if isinstance(shot, str):
            text = shot.strip()
            if not text:
                continue
            formatted.append(f"Example {position}:\n{text}")
            continue

        if isinstance(shot, Mapping):
            query = _optional_text(shot.get("query") or shot.get("question") or shot.get("prompt"))
            response = _optional_text(
                shot.get("response")
                or shot.get("rationales")
                or shot.get("completion")
                or shot.get("answer")
            )
            if not query or not response:
                raise ValueError(
                    "Each structured sample shot must include a query/question and response/rationales."
                )
            formatted.append(f"Example {position}:\nQuery: {query}\nRationales:\n{response}")
            continue

        raise TypeError(f"Unsupported sample shot type at position {position}: {type(shot)!r}")

    if not formatted:
        raise ValueError("sample_shots must contain at least one non-empty example.")
    return "\n\n".join(formatted)


def build_verifier_prompt(
    query: str,
    chunk: Chunk,
    rationales: Iterable[Rationale],
    accepted_chunks: Optional[Sequence[Chunk]] = None,
) -> str:
    """Build a verifier prompt for checking selected evidence chunks."""

    rationale_lines = []
    for rationale in rationales:
        flags = f" Flag Instructions: {rationale.flag_instructions}" if rationale.flag_instructions else ""
        rationale_lines.append(f"{rationale.index}. {rationale.text}{flags}")

    rationale_block = "\n".join(rationale_lines)
    accepted_chunks = accepted_chunks or []
    if accepted_chunks:
        accepted_block = "\n".join(
            f"Accepted Chunk {accepted.index}: {accepted.text}" for accepted in accepted_chunks
        )
    else:
        accepted_block = "None yet."

    return f"""You are verifying a retrieved evidence chunk before RAG generation.

Query:
{query}

Rationales and verifier instructions:
{rationale_block}

Previously accepted evidence:
{accepted_block}

Chunk {chunk.index}:
{chunk.text}

Return JSON with:
- "relevant": true or false
- "flagged": true or false
- "reason": one concise sentence
- "flag_types": a list of issue labels, such as ["CONTRADICTION"] or []
- "confidence": optional number from 0 to 1
	"""


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = "\n".join(_format_rationale_item(item, position) for position, item in enumerate(value, start=1))
    text = str(value).strip()
    return text or None


def _format_rationale_item(item: Any, position: int) -> str:
    if not isinstance(item, Mapping):
        return str(item)
    text = item.get("text") or item.get("rationale") or item.get("content") or ""
    index = item.get("index", position)
    label = item.get("label")
    prefix = f"[{label}] " if label else ""
    return f"<rationale_{index}>{prefix}{text}</rationale_{index}>"
