"""PII detection helpers that run entirely offline."""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){7,}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def extract_pii_entities(text: str) -> List[Dict[str, object]]:
    """Extract simple PII entities from text using regex heuristics."""

    entities: List[Dict[str, object]] = []
    seen: Set[Tuple[str, str]] = set()

    def _add(span: str, label: str, conf: float) -> None:
        key = (span.lower(), label)
        if span and key not in seen:
            seen.add(key)
            entities.append({"type": label, "span": span, "conf": conf})

    for match in _EMAIL_RE.finditer(text):
        _add(match.group(0), "EMAIL", 0.85)

    for match in _PHONE_RE.finditer(text):
        span = match.group(0)
        digits = sum(ch.isdigit() for ch in span)
        if digits >= 7:
            _add(span, "PHONE", 0.75)

    for match in _SSN_RE.finditer(text):
        _add(match.group(0), "SSN", 0.9)

    for match in _NAME_RE.finditer(text):
        candidate = match.group(1)
        if len(candidate.split()) >= 2:
            _add(candidate, "PERSON", 0.55)

    return entities
