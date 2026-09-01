"""Deterministic keyword retrieval for repository chunks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .loader import RepositoryChunk

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_TOP_K = 5


def normalize_words(text: str) -> set[str]:
    """Return normalized, unique words from ``text``.

    Punctuation is treated as a separator and matching is case-insensitive.
    Keeping this as a small helper makes the Day 9 baseline easy to test and
    replace with semantic or hybrid retrieval later.
    """

    return {match.casefold() for match in _WORD_RE.findall(text)}


def keyword_score(question_words: set[str], chunk_text: str) -> int:
    """Count distinct question terms present in a chunk."""

    return len(question_words & normalize_words(chunk_text))


def retrieve(
    question: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[Mapping[str, Any]]:
    """Return the highest-scoring relevant chunks for ``question``.

    Scores are the number of distinct normalized question terms found in each
    chunk.  Results are ordered by descending score, with original input order
    breaking ties.  Chunks with a zero score are excluded.
    """

    if top_k <= 0:
        return []

    question_words = normalize_words(question)
    if not question_words:
        return []

    scored: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        chunk_text = chunk.get("text")
        if not isinstance(chunk_text, str):
            continue
        score = keyword_score(question_words, chunk_text)
        if score:
            scored.append((score, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored[:top_k]]


def retrieve_chunks(
    question: str,
    chunks: Sequence[RepositoryChunk],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[RepositoryChunk]:
    """Typed convenience wrapper around :func:`retrieve`."""

    return list(retrieve(question, chunks, top_k=top_k))  # type: ignore[return-value]


__all__ = ["DEFAULT_TOP_K", "keyword_score", "normalize_words", "retrieve", "retrieve_chunks"]
