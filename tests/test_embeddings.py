"""Deterministic tests for in-memory embedding retrieval."""

from __future__ import annotations

from repo_assistant.embeddings import (
    InMemoryVectorIndex,
    SentenceTransformerAdapter,
    cosine_similarity,
    cosine_similarity_search,
    create_document_embeddings,
    create_query_embedding,
)


def test_fake_embedding_function_supports_document_query_and_search() -> None:
    vectors = {
        "alpha document": [1.0, 0.0],
        "beta document": [0.0, 1.0],
        "find alpha": [1.0, 0.0],
    }

    def fake_embedding(texts: list[str]) -> list[list[float]]:
        return [vectors[text] for text in texts]

    documents = ["alpha document", "beta document"]
    assert create_document_embeddings(documents, fake_embedding) == [
        (1.0, 0.0),
        (0.0, 1.0),
    ]
    assert create_query_embedding("find alpha", fake_embedding) == (1.0, 0.0)

    index = InMemoryVectorIndex(documents, fake_embedding)
    matches = index.search("find alpha", top_k=2)

    assert [match.document for match in matches] == documents
    assert matches[0].score == 1.0
    assert matches[1].score == 0.0


def test_cosine_search_is_descending_and_stable_for_ties() -> None:
    assert cosine_similarity([3.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity_search(
        [1.0, 0.0],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        top_k=3,
    ) == [(1, 1.0), (0, 0.0), (2, 0.0)]


def test_sentence_transformer_adapter_accepts_a_fake_model() -> None:
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["show_progress_bar"] is False
            return [[float(len(text)), 1.0] for text in texts]

    adapter = SentenceTransformerAdapter(model=FakeModel())

    assert adapter.embed_query("abcd") == (4.0, 1.0)
    assert adapter.embed_documents(["a", "bb"]) == [(1.0, 1.0), (2.0, 1.0)]
