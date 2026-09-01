"""Provider-free end-to-end tests for the LangGraph workflow."""

from __future__ import annotations

from pathlib import Path

from repo_assistant.graph import build_graph


def _initial_state(repo_path: str, question: str) -> dict[str, object]:
    return {
        "repo_path": repo_path,
        "question": question,
        "route": "repo_question",
        "chunks": [],
        "answer": "",
        "citations": [],
        "status": "running",
        "errors": [],
    }


def test_graph_runs_supported_question_with_thread_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    calls: list[tuple[str, object]] = []

    def fake_model(question: str, chunks: object) -> str:
        calls.append((question, chunks))
        return "main returns ok [src/main.py:1-2]"

    graph = build_graph(model_fn=fake_model)
    config = {"configurable": {"thread_id": "day9-demo"}}
    result = graph.invoke(
        _initial_state(str(tmp_path), "How does the main function work?"), config
    )

    assert result["route"] == "repo_question"
    assert result["status"] == "completed"
    assert result["answer"] == "main returns ok [src/main.py:1-2]"
    assert result["citations"] == ["src/main.py:1-2"]
    assert len(calls) == 1
    assert calls[0][0] == "How does the main function work?"

    checkpointed = graph.get_state(config)
    assert checkpointed.values["status"] == "completed"
    assert checkpointed.config["configurable"]["thread_id"] == "day9-demo"


def test_graph_handles_invalid_repository(tmp_path: Path) -> None:
    graph = build_graph()
    result = graph.invoke(
        _initial_state(str(tmp_path / "missing"), "Explain the repository code."),
        {"configurable": {"thread_id": "day9-invalid"}},
    )

    assert result["status"] == "error"
    assert "does not exist" in result["answer"]


def test_graph_handles_unsupported_request(tmp_path: Path) -> None:
    called = False

    def model_must_not_run(question: str, chunks: object) -> str:
        nonlocal called
        called = True
        raise AssertionError("unsupported questions must not call the model")

    graph = build_graph(model_fn=model_must_not_run)
    result = graph.invoke(
        _initial_state(str(tmp_path), "What is the weather today?"),
        {"configurable": {"thread_id": "day9-unsupported"}},
    )

    assert result["status"] == "error"
    assert "only answer questions about the repository" in result["answer"]
    assert called is False


def test_graph_handles_missing_evidence(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("installation instructions", encoding="utf-8")

    called = False

    def model_must_not_run(question: str, chunks: object) -> str:
        nonlocal called
        called = True
        raise AssertionError("missing evidence must not call the model")

    graph = build_graph(model_fn=model_must_not_run)
    result = graph.invoke(
        _initial_state(str(tmp_path), "What does the deployment function do?"),
        {"configurable": {"thread_id": "day9-no-evidence"}},
    )

    assert result["status"] == "insufficient_data"
    assert result["answer"] == "INSUFFICIENT_DATA"
    assert called is False
