"""Small local embedding utilities for in-memory vector retrieval.

The production adapter uses Sentence Transformers lazily.  Retrieval itself
stores only a plain Python embedding matrix, which keeps this small repository
free of a vector-database dependency and makes the embedding function easy to
replace with a deterministic fake in tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .loader import RepositoryChunk

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4

Embedding = tuple[float, ...]
EmbeddingFunction = Callable[[Sequence[str]], Sequence[Sequence[float]]]
T = TypeVar("T")


def _as_list(value: Any) -> Any:
    """Convert array-like model output to ordinary Python lists."""

    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _coerce_vector(vector: Sequence[float]) -> Embedding:
    """Validate and normalize one embedding vector."""

    values = _as_list(vector)
    if isinstance(values, (str, bytes)):
        raise TypeError("An embedding must be a sequence of numbers.")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError("An embedding must be a sequence of numbers.") from exc
    if not result:
        raise ValueError("An embedding must contain at least one value.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Embedding values must be finite numbers.")
    return result


def _coerce_vectors(raw_vectors: Any, expected_count: int) -> list[Embedding]:
    """Validate a batch of model output, including a one-item 1-D result."""

    values = _as_list(raw_vectors)
    if expected_count == 0:
        return []
    if isinstance(values, (str, bytes)):
        raise TypeError("Embedding output must be a sequence of vectors.")
    try:
        values = list(values)
    except TypeError as exc:
        raise TypeError("Embedding output must be a sequence of vectors.") from exc

    # Some model-like callables return [0.1, 0.2] for a one-item batch.
    if expected_count == 1 and values and all(
        isinstance(value, (int, float)) for value in values
    ):
        values = [values]
    if len(values) != expected_count:
        raise ValueError(
            f"Embedding function returned {len(values)} vectors for "
            f"{expected_count} texts."
        )
    vectors = [_coerce_vector(vector) for vector in values]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("All embeddings in a batch must have the same dimension.")
    return vectors


def create_document_embeddings(
    documents: Sequence[str],
    embedding_fn: EmbeddingFunction,
) -> list[Embedding]:
    """Create one embedding per document using an injectable function."""

    texts = list(documents)
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Documents must be strings.")
    return _coerce_vectors(embedding_fn(texts), len(texts))


def create_query_embedding(query: str, embedding_fn: EmbeddingFunction) -> Embedding:
    """Create one embedding for a query using the same model as documents."""

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")
    return create_document_embeddings([query], embedding_fn)[0]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, treating a zero vector as having score 0."""

    left_vector = _coerce_vector(left)
    right_vector = _coerce_vector(right)
    if len(left_vector) != len(right_vector):
        raise ValueError("Embedding vectors must have the same dimension.")

    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left_vector, right_vector)) / (
        left_norm * right_norm
    )


def cosine_similarity_search(
    query_embedding: Sequence[float],
    document_embeddings: Sequence[Sequence[float]],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
) -> list[tuple[int, float]]:
    """Rank document-vector indexes by cosine similarity.

    Ties retain document order.  ``min_score`` is optional because dense
    retrieval normally returns the nearest neighbours even for weak matches;
    callers that need an evidence threshold can set it explicitly.
    """

    if top_k <= 0:
        return []
    query_vector = _coerce_vector(query_embedding)
    scored: list[tuple[float, int]] = []
    for index, document_embedding in enumerate(document_embeddings):
        score = cosine_similarity(query_vector, document_embedding)
        if min_score is None or score >= min_score:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(index, score) for score, index in scored[:top_k]]


@dataclass(frozen=True)
class VectorMatch(Generic[T]):
    """A document and its cosine-similarity score."""

    document: T
    score: float


class InMemoryVectorIndex(Generic[T]):
    """A small document collection backed by an in-memory embedding matrix."""

    def __init__(
        self,
        documents: Sequence[T],
        embedding_fn: EmbeddingFunction,
        *,
        text_getter: Callable[[T], str] | None = None,
    ) -> None:
        self.documents = tuple(documents)
        getter = text_getter or self._default_text_getter
        self.embeddings = tuple(
            create_document_embeddings([getter(document) for document in self.documents], embedding_fn)
        )
        self._embedding_fn = embedding_fn

    @staticmethod
    def _default_text_getter(document: T) -> str:
        if not isinstance(document, str):
            raise TypeError("Use text_getter when indexing non-string documents.")
        return document

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float | None = None,
    ) -> list[VectorMatch[T]]:
        """Embed ``query`` and return the nearest indexed documents."""

        query_embedding = create_query_embedding(query, self._embedding_fn)
        ranked = cosine_similarity_search(
            query_embedding,
            self.embeddings,
            top_k=top_k,
            min_score=min_score,
        )
        return [VectorMatch(self.documents[index], score) for index, score in ranked]


class SentenceTransformerAdapter:
    """Lazy local adapter around ``sentence_transformers.SentenceTransformer``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        model: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        if model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": device} if device is not None else {}
            model = SentenceTransformer(model_name, **kwargs)
        self._model = model

    def __call__(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed a batch, making the adapter injectable as an embedding function."""

        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("Texts must be strings.")
        if not values:
            return []
        encoded = self._model.encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _coerce_vectors(encoded, len(values))

    def embed_documents(self, documents: Sequence[str]) -> list[Embedding]:
        """Create normalized embeddings for documents."""

        return self(documents)

    def embed_query(self, query: str) -> Embedding:
        """Create a normalized embedding for one query."""

        return self([query])[0]


def vector_search_chunks(
    query: str,
    chunks: Sequence[RepositoryChunk],
    embedding_fn: EmbeddingFunction,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
) -> list[VectorMatch[RepositoryChunk]]:
    """Search line-aware repository chunks with an injected embedder."""

    index = InMemoryVectorIndex(chunks, embedding_fn, text_getter=lambda chunk: chunk["text"])
    return index.search(query, top_k=top_k, min_score=min_score)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TOP_K",
    "Embedding",
    "EmbeddingFunction",
    "InMemoryVectorIndex",
    "SentenceTransformerAdapter",
    "VectorMatch",
    "cosine_similarity",
    "cosine_similarity_search",
    "create_document_embeddings",
    "create_query_embedding",
    "vector_search_chunks",
]
