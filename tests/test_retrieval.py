"""Deterministic tests for repository retrieval."""

from __future__ import annotations

from repo_assistant.retrieval import normalize_words, retrieve

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
