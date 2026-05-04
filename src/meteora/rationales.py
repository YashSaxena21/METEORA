from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .types import Rationale, RationaleLike, normalize_rationales

_XML_RE = re.compile(
    r"<rationale_(?P<num>\d+)>\s*(?P<body>.*?)\s*</rationale_(?P=num)>",
    re.DOTALL | re.IGNORECASE,
)
_LOOSE_XML_RE = re.compile(
    r"<rationale_(?P<num>\d+)>\s*(?P<body>.*?)(?=<rationale_\d+>|$)",
    re.DOTALL | re.IGNORECASE,
)
_NUMBERED_RE = re.compile(
    r"(?:^|\n)\s*(?P<num>\d+)[.)]\s*(?P<body>.*?)(?=\n\s*\d+[.)]\s+|\Z)",
    re.DOTALL,
)
_RATIONALE_RE = re.compile(
    r"(?:^|\n)\s*Rationale\s+(?P<num>\d+)\s*:\s*(?P<body>.*?)(?=\n\s*Rationale\s+\d+\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_LABEL_RE = re.compile(r"^\s*\[(?P<label>[^\]]+)\]\s*(?P<body>.*)$", re.DOTALL)
_FLAG_RE = re.compile(r"\bFlag Instructions:\s*(?P<flags>.*)$", re.DOTALL | re.IGNORECASE)


def parse_rationales(response: str, *, split_plain_text: bool = False) -> List[Rationale]:
    """Extract rationales from common LLM output formats."""

    block = _last_query_block(response)
    matches = _find_matches(block, split_plain_text=split_plain_text)
    rationales: List[Rationale] = []
    seen = set()

    for fallback_index, (number, raw_body) in enumerate(matches, start=1):
        body = _strip_markup(raw_body)
        label, body = _split_label(body)
        body, flags = _split_flags(body)
        body = _strip_markup(body)
        if not body:
            continue
        index = number or fallback_index
        key = (index, body)
        if key in seen:
            continue
        seen.add(key)
        rationales.append(
            Rationale(text=body, index=index, label=label, flag_instructions=flags)
        )

    rationales.sort(key=lambda rationale: rationale.index)
    return rationales


def parse_rationale_texts(response: str) -> List[str]:
    return [rationale.text for rationale in parse_rationales(response)]


def extract_flag_instructions(response: str) -> List[Tuple[int, str]]:
    return [
        (rationale.index, rationale.flag_instructions)
        for rationale in parse_rationales(response)
        if rationale.flag_instructions
    ]


def coerce_rationales(rationales: Union[Iterable[RationaleLike], str]) -> Sequence[Rationale]:
    if isinstance(rationales, str):
        parsed = parse_rationales(rationales, split_plain_text=True)
        if parsed:
            return tuple(parsed)
        lines = [line.strip() for line in rationales.splitlines() if line.strip()]
        return tuple(normalize_rationales(lines))
    return normalize_rationales(rationales)


def _last_query_block(text: str) -> str:
    parts = re.split(r"\n?\s*Query\s*:\s*", text)
    return parts[-1] if len(parts) > 1 else text


def _find_matches(block: str, *, split_plain_text: bool = False) -> List[Tuple[Optional[int], str]]:
    for pattern in (_XML_RE, _LOOSE_XML_RE, _RATIONALE_RE, _NUMBERED_RE):
        matches = [
            (int(match.group("num")), match.group("body").strip())
            for match in pattern.finditer(block)
        ]
        if matches:
            return matches
    stripped = block.strip()
    if split_plain_text:
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1:
            return [(None, line) for line in lines]
    return [(1, stripped)] if stripped else []


def _split_label(body: str) -> Tuple[Optional[str], str]:
    match = _LABEL_RE.match(body)
    if not match:
        return None, body.strip()
    return match.group("label").strip(), match.group("body").strip()


def _split_flags(body: str) -> Tuple[str, Optional[str]]:
    match = _FLAG_RE.search(body)
    if not match:
        return body.strip(), None
    rationale_text = body[: match.start()].strip()
    flags = _strip_markup(match.group("flags")).strip()
    return rationale_text, flags or None


def _strip_markup(text: str) -> str:
    text = re.sub(r"</?rationale_\d+>", "", text, flags=re.IGNORECASE)
    return text.strip()
