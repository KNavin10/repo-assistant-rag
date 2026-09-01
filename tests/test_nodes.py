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
