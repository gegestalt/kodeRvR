"""Exact and near-exact public-code reuse fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable


def normalized_tokens(code: str) -> tuple[str, ...]:
    code = re.sub(r"#.*?$|//.*?$|/\*.*?\*/", " ", code, flags=re.MULTILINE | re.DOTALL)
    return tuple(re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", code.lower()))


def token_shingles(code: str, width: int = 7) -> set[str]:
    if width < 2:
        raise ValueError("shingle width must be at least two")
    tokens = normalized_tokens(code)
    return {
        hashlib.sha256("\x1f".join(tokens[i:i + width]).encode()).hexdigest()[:20]
        for i in range(max(0, len(tokens) - width + 1))
    }


@dataclass
class PublicReuseIndex:
    """Local corpus index; provenance URLs/licenses stay outside model features."""

    width: int = 7

    def __post_init__(self) -> None:
        self._shingles: set[str] = set()
        self._documents = 0

    def add(self, code_documents: Iterable[str]) -> None:
        for code in code_documents:
            self._shingles.update(token_shingles(code, self.width))
            self._documents += 1

    def overlap_fraction(self, code: str) -> float:
        query = token_shingles(code, self.width)
        return len(query & self._shingles) / len(query) if query else 0.0

    @property
    def documents(self) -> int:
        return self._documents
