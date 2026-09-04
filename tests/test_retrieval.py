"""Deterministic tests for repository retrieval."""

from __future__ import annotations

from repo_assistant.retrieval import (
    bm25_search,
    chunk_identifier,
    hybrid_search,
    normalize_words,
    reciprocal_rank_fusion,
    retrieve,
    vector_search,
)

FIXTURE_CHUNKS = [
    {
        "file_path": "backend/approvals.py",
        "start_line": 4,
        "end_line": 8,
        "text": "Sensitive tools require approval before execution.",
    },
    {
        "file_path": "README.md",
        "start_line": 1,
        "end_line": 2,
        "text": "This project contains installation instructions.",
    },
]


def test_retrieval_returns_expected_fixture_file() -> None:
    matches = retrieve("Where is approval required for sensitive tools?", FIXTURE_CHUNKS)

    assert matches == [FIXTURE_CHUNKS[0]]


def test_retrieval_is_case_insensitive_and_excludes_zero_scores() -> None:
    assert normalize_words("Approval, APPROVAL!") == {"approval"}
    assert retrieve("database", FIXTURE_CHUNKS) == []


def test_chunk_identifier_formats_consistently() -> None:
    assert chunk_identifier(FIXTURE_CHUNKS[0]) == "backend/approvals.py:4-8"
    assert chunk_identifier(FIXTURE_CHUNKS[1]) == "README.md:1-2"


def test_bm25_search_returns_matches_and_excludes_zero_scores() -> None:
    results = bm25_search("approval required", FIXTURE_CHUNKS)
    assert len(results) == 1
    assert results[0]["file_path"] == "backend/approvals.py"

    assert bm25_search("unrelated database query", FIXTURE_CHUNKS) == []
    assert bm25_search("", FIXTURE_CHUNKS) == []
    assert bm25_search("approval", []) == []


def test_vector_search_with_fake_embedder() -> None:
    vectors = {
        "Sensitive tools require approval before execution.": [1.0, 0.0],
        "This project contains installation instructions.": [0.0, 1.0],
        "need approval": [1.0, 0.0],
        "install": [0.0, 1.0],
    }

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        return [vectors[t] for t in texts]

    results = vector_search("need approval", FIXTURE_CHUNKS, embedding_fn=fake_embedder, top_k=1)
    assert len(results) == 1
    assert results[0]["file_path"] == "backend/approvals.py"

    # min_score filtering
    assert vector_search(
        "need approval", FIXTURE_CHUNKS, embedding_fn=fake_embedder, min_score=0.99, top_k=2
    ) == [FIXTURE_CHUNKS[0]]


def test_reciprocal_rank_fusion_fuses_and_deduplicates() -> None:
    chunk_a = {"file_path": "a.py", "start_line": 1, "end_line": 5, "text": "chunk a"}
    chunk_b = {"file_path": "b.py", "start_line": 1, "end_line": 5, "text": "chunk b"}
    chunk_c = {"file_path": "c.py", "start_line": 1, "end_line": 5, "text": "chunk c"}

    # List 1: [a (rank 1), b (rank 2)]
    # List 2: [b (rank 1), c (rank 2)]
    # For k=60:
    # a: 1/61 + 0 = ~0.01639
    # b: 1/62 + 1/61 = ~0.03252
    # c: 0 + 1/62 = ~0.01613
    # Ranking should be: b, a, c
    fused = reciprocal_rank_fusion([chunk_a, chunk_b], [chunk_b, chunk_c], k=60, top_k=3)
    assert fused == [chunk_b, chunk_a, chunk_c]

    # Supports passing a list of lists as well
    fused_list = reciprocal_rank_fusion([[chunk_a, chunk_b], [chunk_b, chunk_c]], k=60, top_k=2)
    assert fused_list == [chunk_b, chunk_a]


def test_hybrid_search_defaults_to_four_chunks() -> None:
    chunks = [
        {
            "file_path": f"file_{i}.py",
            "start_line": 1,
            "end_line": 10,
            "text": f"function_{i} implementation keyword",
        }
        for i in range(10)
    ]
    vectors = {c["text"]: [float(i), 1.0] for i, c in enumerate(chunks)}
    vectors["search keyword"] = [5.0, 1.0]

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        return [vectors.get(t, [0.0, 1.0]) for t in texts]

    results = hybrid_search("search keyword", chunks, embedding_fn=fake_embedder)
    assert len(results) == 4
    identifiers = [chunk_identifier(c) for c in results]
    assert len(set(identifiers)) == 4

