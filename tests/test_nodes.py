"""Provider-free tests for repository-assistant graph nodes."""

from __future__ import annotations

from pathlib import Path

from repo_assistant.nodes import (
    generate_answer,
    handle_failure,
    load_repository,
    retrieve_context,
    route_question,
    validate_repository,
)


def _state(**updates: object) -> dict[str, object]:
    state: dict[str, object] = {
        "repo_path": "",
        "question": "",
        "route": "repo_question",
        "chunks": [],
        "answer": "",
        "citations": [],
        "status": "running",
        "errors": [],
    }
    state.update(updates)
    return state


def test_validate_and_load_repository(tmp_path: Path) -> None:
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    validated = validate_repository(_state(repo_path=str(tmp_path)))
    assert validated["status"] == "running"

    loaded = load_repository(_state(repo_path=str(tmp_path)))
    chunks = loaded["chunks"]
    assert loaded["status"] == "running"
    assert chunks[0]["file_path"] == "src/main.py"  # type: ignore[index]


def test_invalid_repository_returns_error(tmp_path: Path) -> None:
    missing = validate_repository(_state(repo_path=str(tmp_path / "missing")))
    assert missing["status"] == "error"
    assert "does not exist" in missing["errors"][0]  # type: ignore[index]


def test_route_question() -> None:
    assert route_question(_state(question="How does the repository load files?"))["route"] == (
        "repo_question"
    )
    assert route_question(_state(question="Give me a summary of the repository."))["route"] == (
        "repo_summary"
    )
    assert route_question(_state(question="What is the weather today?"))["route"] == "unsupported"


def test_retrieve_and_generate_use_injected_fake_model() -> None:
    chunks = [
        {"file_path": "src/main.py", "start_line": 1, "end_line": 2, "text": "def main(): return 1"},
        {"file_path": "README.md", "start_line": 1, "end_line": 1, "text": "unrelated documentation"},
    ]
    retrieved = retrieve_context(
        _state(question="How does main work?", route="repo_question", chunks=chunks)
    )
    selected = retrieved["chunks"]
    assert selected == [chunks[0]]

    received: dict[str, object] = {}

    def fake_model(question: str, model_chunks: object) -> str:
        received["question"] = question
        received["chunks"] = model_chunks
        return "main returns a value [src/main.py:1-2]"

    generated = generate_answer(
        _state(question="How does main work?", chunks=selected), model_fn=fake_model
    )
    assert received == {"question": "How does main work?", "chunks": selected}
    assert generated["status"] == "completed"
    assert generated["citations"] == ["src/main.py:1-2"]


def test_generate_answer_returns_insufficient_data_without_calling_model() -> None:
    called = False

    def fake_model(question: str, chunks: object) -> str:
        nonlocal called
        called = True
        return "should not be called"

    result = generate_answer(
        _state(question="Explain the repository", chunks=[]), model_fn=fake_model
    )
    assert result["status"] == "insufficient_data"
    assert result["answer"] == "INSUFFICIENT_DATA"
    assert called is False


def test_generate_answer_requires_citations() -> None:
    result = generate_answer(
        _state(
            question="How does main work?",
            chunks=[
                {
                    "file_path": "src/main.py",
                    "start_line": 1,
                    "end_line": 2,
                    "text": "def main(): return 1",
                }
            ],
        ),
        model_fn=lambda question, chunks: "It returns one.",
    )

    assert result["status"] == "error"
    assert "must include a citation" in result["errors"][0]  # type: ignore[index]


def test_handle_failure_for_unsupported_request() -> None:
    result = handle_failure(_state(route="unsupported"))
    assert result["status"] == "error"
    assert "only answer questions about the repository" in result["answer"]


def test_citation_path_must_exist_under_selected_repository(tmp_path: Path) -> None:
    chunks = [
        {"file_path": "src/missing.py", "start_line": 1, "end_line": 5, "text": "def foo(): pass"}
    ]
    result = generate_answer(
        _state(
            repo_path=str(tmp_path),
            question="Where is missing?",
            chunks=chunks,
        ),
        model_fn=lambda q, c: "Missing function [src/missing.py:1-5]",
    )

    assert result["status"] == "error"
    assert "does not exist under the selected repository" in result["errors"][0]


def test_citation_line_range_must_be_valid(tmp_path: Path) -> None:
    chunks = [
        {"file_path": "src/main.py", "start_line": 1, "end_line": 5, "text": "def main(): pass"}
    ]

    # Zero start line
    zero_start = generate_answer(
        _state(question="How does main work?", chunks=chunks),
        model_fn=lambda q, c: "Main function [src/main.py:0-5]",
    )
    assert zero_start["status"] == "error"
    assert "Invalid citation line range" in zero_start["errors"][0]

    # Inverted line range (start > end)
    inverted = generate_answer(
        _state(question="How does main work?", chunks=chunks),
        model_fn=lambda q, c: "Main function [src/main.py:5-2]",
    )
    assert inverted["status"] == "error"
    assert "Invalid citation line range" in inverted["errors"][0]

    # Line range exceeding actual file line count
    file_path = tmp_path / "src" / "main.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
    chunks = [
        {"file_path": "src/main.py", "start_line": 1, "end_line": 100, "text": "line 1\nline 2\nline 3\n"}
    ]
    exceeding = generate_answer(
        _state(repo_path=str(tmp_path), question="How does main work?", chunks=chunks),
        model_fn=lambda q, c: "Main function [src/main.py:1-50]",
    )
    assert exceeding["status"] == "error"
    assert "exceeds file length" in exceeding["errors"][0]


def test_citation_must_correspond_to_retrieved_chunks(tmp_path: Path) -> None:
    file_a = tmp_path / "src" / "a.py"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("def a(): return 'a'\n", encoding="utf-8")
    file_b = tmp_path / "src" / "b.py"
    file_b.write_text("def b(): return 'b'\n", encoding="utf-8")

    retrieved_chunks = [
        {"file_path": "src/a.py", "start_line": 1, "end_line": 1, "text": "def a(): return 'a'"}
    ]

    # File exists in repo but was not in retrieved chunks
    unretrieved = generate_answer(
        _state(repo_path=str(tmp_path), question="Where is b?", chunks=retrieved_chunks),
        model_fn=lambda q, c: "Here is b [src/b.py:1-1]",
    )
    assert unretrieved["status"] == "error"
    assert "does not correspond to any retrieved chunk" in unretrieved["errors"][0]

    # Line range outside the retrieved chunk
    out_of_range = generate_answer(
        _state(repo_path=str(tmp_path), question="Where is a?", chunks=retrieved_chunks),
        model_fn=lambda q, c: "Here is a [src/a.py:10-20]",
    )
    assert out_of_range["status"] == "error"
    assert "does not correspond to any retrieved chunk" in out_of_range["errors"][0]


def test_fabricated_citation_causes_error(tmp_path: Path) -> None:
    chunks = [
        {"file_path": "src/main.py", "start_line": 1, "end_line": 2, "text": "def main(): return 1"}
    ]
    fabricated = generate_answer(
        _state(repo_path=str(tmp_path), question="How does auth work?", chunks=chunks),
        model_fn=lambda q, c: "Auth is handled in [src/security/auth.py:42-88]",
    )
    assert fabricated["status"] == "error"
    assert (
        "does not correspond to any retrieved chunk" in fabricated["errors"][0]
        or "does not exist under the selected repository" in fabricated["errors"][0]
    )


def test_question_length_boundary_is_enforced_in_nodes() -> None:
    blank = route_question(_state(question=" "))
    oversized = route_question(_state(question="x" * 501))

    assert blank["status"] == "error"
    assert oversized["status"] == "error"
    assert "Question" in oversized["errors"][0]
