"""Deterministic BM25, vector, and hybrid retrieval for repository chunks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from rank_bm25 import BM25Okapi

from .embeddings import (
    EmbeddingFunction,
    InMemoryVectorIndex,
    SentenceTransformerAdapter,
)
from .loader import RepositoryChunk

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
DEFAULT_HYBRID_TOP_K = 4
DEFAULT_RRF_K = 60
DEFAULT_MIN_VECTOR_SCORE = 0.25

_DEFAULT_EMBEDDER: SentenceTransformerAdapter | None = None


def _get_default_embedder() -> SentenceTransformerAdapter:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = SentenceTransformerAdapter()
    return _DEFAULT_EMBEDDER


def chunk_identifier(chunk: Mapping[str, Any]) -> str:
    """Return a stable chunk identifier in ``file_path:start_line-end_line`` format."""

    file_path = chunk.get("file_path", "")
    start_line = chunk.get("start_line", 0)
    end_line = chunk.get("end_line", 0)
    return f"{file_path}:{start_line}-{end_line}"


def normalize_words(text: str) -> set[str]:
    """Return normalized, unique words from ``text``.

    Punctuation is treated as a separator and matching is case-insensitive.
    """

    return {match.casefold() for match in _WORD_RE.findall(text)}


def tokenize_words(text: str) -> list[str]:
    """Return normalized word tokens from ``text``."""

    return [match.casefold() for match in _WORD_RE.findall(text)]


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


def bm25_search(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_CANDIDATE_K,
) -> list[RepositoryChunk]:
    """Return the highest-scoring chunks using BM25Okapi.

    Tokens are normalized using :func:`tokenize_words`. Chunks with a score of
    zero (no matching query terms) are excluded. Ties retain input order.
    """

    if top_k <= 0:
        return []

    tokenized_query = tokenize_words(query)
    if not tokenized_query or not chunks:
        return []

    tokenized_corpus = [
        tokenize_words(str(chunk.get("text", ""))) for chunk in chunks
    ]

    if not any(tokenized_corpus):
        return []

    bm25 = BM25Okapi(tokenized_corpus)

    # Smooth zero or negative IDF values so small corpora (e.g. 1-2 chunks in tests)
    # still produce positive scores for matching terms.
    corpus_size = len(tokenized_corpus)
    for term in tokenized_query:
        if term in bm25.idf and bm25.idf[term] <= 0:
            doc_freq = sum(1 for doc in tokenized_corpus if term in doc)
            bm25.idf[term] = math.log(1.0 + (corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))

    scores = bm25.get_scores(tokenized_query)
    scored: list[tuple[float, int, RepositoryChunk]] = []
    for index, (score, chunk) in enumerate(zip(scores, chunks)):
        if score > 0:
            scored.append((float(score), index, cast(RepositoryChunk, chunk)))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored[:top_k]]


def vector_search(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_CANDIDATE_K,
    embedding_fn: EmbeddingFunction | None = None,
    min_score: float | None = None,
    vector_index: InMemoryVectorIndex[RepositoryChunk] | None = None,
) -> list[RepositoryChunk]:
    """Return the nearest chunks ranked by cosine similarity."""

    if top_k <= 0 or not chunks or not query.strip():
        return []

    typed_chunks = [cast(RepositoryChunk, chunk) for chunk in chunks]

    if vector_index is not None:
        index = vector_index
    else:
        fn = embedding_fn if embedding_fn is not None else _get_default_embedder()
        index = InMemoryVectorIndex(
            typed_chunks,
            fn,
            text_getter=lambda chunk: str(chunk.get("text", "")),
        )

    matches = index.search(query, top_k=top_k, min_score=min_score)
    return [match.document for match in matches]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]] | None = None,
    *extra_lists: Sequence[Mapping[str, Any]],
    k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_HYBRID_TOP_K,
) -> list[RepositoryChunk]:
    """Fuse ranked lists using Reciprocal Rank Fusion (RRF).

    Chunks are identified and deduplicated using :func:`chunk_identifier`.
    Scores are never averaged; ordinal ranks are fused using:
        score(d) = sum_{list} 1 / (k + rank(d))
    """

    if top_k <= 0:
        return []

    all_lists: list[Sequence[Mapping[str, Any]]]
    if ranked_lists is None:
        all_lists = list(extra_lists)
    elif extra_lists:
        all_lists = [ranked_lists, *extra_lists]  # type: ignore[list-item]
    elif ranked_lists and isinstance(ranked_lists[0], (list, tuple)):
        all_lists = list(ranked_lists)  # type: ignore[arg-type]
    elif ranked_lists:
        all_lists = [ranked_lists]  # type: ignore[list-item]
    else:
        all_lists = []

    rrf_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RepositoryChunk] = {}
    first_seen_order: dict[str, int] = {}
    seen_counter = 0

    for ranked_list in all_lists:
        seen_in_this_list: set[str] = set()
        for rank, chunk in enumerate(ranked_list, start=1):
            identifier = chunk_identifier(chunk)
            if identifier in seen_in_this_list:
                continue
            seen_in_this_list.add(identifier)

            if identifier not in chunks_by_id:
                chunks_by_id[identifier] = cast(RepositoryChunk, chunk)
                first_seen_order[identifier] = seen_counter
                seen_counter += 1
                rrf_scores[identifier] = 0.0

            rrf_scores[identifier] += 1.0 / (k + rank)

    sorted_identifiers = sorted(
        rrf_scores.keys(),
        key=lambda item_id: (-rrf_scores[item_id], first_seen_order[item_id]),
    )
    return [chunks_by_id[item_id] for item_id in sorted_identifiers[:top_k]]


def hybrid_search(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedding_fn: EmbeddingFunction | None = None,
    vector_index: InMemoryVectorIndex[RepositoryChunk] | None = None,
    min_vector_score: float | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RepositoryChunk]:
    """Retrieve chunks combining BM25 and dense vector search via RRF.

    BM25 and vector search each retrieve up to ``candidate_k`` candidates.
    Their ranks are fused via Reciprocal Rank Fusion, returning the top ``top_k``
    deduplicated chunks.
    """

    if top_k <= 0 or not chunks or not query.strip():
        return []

    bm25_candidates = bm25_search(query, chunks, top_k=candidate_k)
    vector_candidates = vector_search(
        query,
        chunks,
        top_k=candidate_k,
        embedding_fn=embedding_fn,
        min_score=min_vector_score,
        vector_index=vector_index,
    )

    return reciprocal_rank_fusion(
        bm25_candidates,
        vector_candidates,
        k=rrf_k,
        top_k=top_k,
    )


__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_HYBRID_TOP_K",
    "DEFAULT_MIN_VECTOR_SCORE",
    "DEFAULT_RRF_K",
    "DEFAULT_TOP_K",
    "bm25_search",
    "chunk_identifier",
    "hybrid_search",
    "keyword_score",
    "normalize_words",
    "reciprocal_rank_fusion",
    "retrieve",
    "retrieve_chunks",
    "tokenize_words",
    "vector_search",
]

