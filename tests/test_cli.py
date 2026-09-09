"""Provider-free tests for the repository-assistant CLI."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from repo_assistant import cli
from repo_assistant.cli import main


def test_cli_prints_answer_and_sources(tmp_path: Path, capsys) -> None:
    source = tmp_path / "backend" / "approvals.py"
    source.parent.mkdir()
    source.write_text(
        "def enforce_approval():\n    approval is required before sensitive tools execute\n",
        encoding="utf-8",
    )

    def fake_model(question: str, chunks: object) -> str:
        return "Sensitive tools are paused before execution. [backend/approvals.py:1-2]"

    exit_code = main(
        [
            "--repo",
            str(tmp_path),
            "--question",
            "Where is approval for sensitive tools enforced?",
            "--thread-id",
            "day9-demo",
        ],
        model_fn=fake_model,
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "Answer:\n"
        "Sensitive tools are paused before execution. [backend/approvals.py:1-2]\n"
        "\nSources:\n"
        "backend/approvals.py:1-2\n"
    )


def test_cli_prints_insufficient_data(tmp_path: Path, capsys) -> None:
    (tmp_path / "README.md").write_text("installation instructions", encoding="utf-8")

    def model_must_not_run(question: str, chunks: object) -> str:
        raise AssertionError("the model must not run without evidence")

    exit_code = main(
        [
            "--repo",
            str(tmp_path),
            "--question",
            "Where is approval for sensitive tools enforced?",
            "--thread-id",
            "day9-no-evidence",
        ],
        model_fn=model_must_not_run,
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "INSUFFICIENT_DATA\n"


def test_run_configures_langsmith_metadata(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeGraph:
        def invoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            captured["state"] = state
            captured["config"] = config
            return {"status": "insufficient_data"}

    monkeypatch.setattr(cli, "build_graph", lambda model_fn: FakeGraph())

    fake_model = lambda question, chunks: "unused"
    result = cli.run(
        "tests/fixtures/eval_repo",
        "Where is validation?",
        "day11-demo-001",
        model_fn=fake_model,
    )

    assert result == {"status": "insufficient_data"}
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["run_name"] == "repo-assistant-question"
    assert config["tags"] == ["cli"]
    assert config["configurable"] == {"thread_id": "day11-demo-001"}
    assert config["metadata"]["thread_id"] == "day11-demo-001"
    UUID(config["metadata"]["request_id"])


def test_default_groq_path_requires_explicit_consent(tmp_path: Path, monkeypatch) -> None:
    graph_built = False

    def fake_build_graph(model_fn):
        nonlocal graph_built
        graph_built = True
        raise AssertionError("graph must not be built before consent")

    monkeypatch.setattr(cli, "build_graph", fake_build_graph)

    with pytest.raises(PermissionError, match="allow-external-model"):
        cli.run(str(tmp_path), "Explain this repository", "no-consent")
    assert graph_built is False


def test_external_model_flag_allows_default_model_path(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeGraph:
        def invoke(self, state, config):
            return {"status": "insufficient_data"}

    def fake_build_graph(model_fn):
        captured["model_fn"] = model_fn
        return FakeGraph()

    monkeypatch.setattr(cli, "build_graph", fake_build_graph)

    cli.run(
        str(tmp_path),
        "Explain this repository",
        "with-consent",
        allow_external_model=True,
    )
    assert captured["model_fn"] is cli.groq_model


def test_fake_model_is_exempt_from_external_consent(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("repository overview\n", encoding="utf-8")

    result = cli.run(
        str(tmp_path),
        "Summarize this repository",
        "fake-model",
        model_fn=lambda question, chunks: "Overview [README.md:1-1]",
    )

    assert result["status"] == "completed"


def test_non_fixture_langsmith_tracing_requires_separate_consent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    fake_model = lambda question, chunks: "unused"

    with pytest.raises(PermissionError, match="allow-non-fixture-tracing"):
        cli.run(str(tmp_path), "Explain this repository", "trace", model_fn=fake_model)


def test_fixture_tracing_and_explicit_non_fixture_tracing_are_allowed(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeGraph:
        def invoke(self, state, config):
            return {"status": "insufficient_data"}

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(cli, "build_graph", lambda model_fn: FakeGraph())
    fake_model = lambda question, chunks: "unused"

    fixture_result = cli.run(
        "tests/fixtures/eval_repo", "Explain this repository", "fixture", model_fn=fake_model
    )
    opted_in_result = cli.run(
        str(tmp_path),
        "Explain this repository",
        "opted-in",
        model_fn=fake_model,
        allow_non_fixture_tracing=True,
    )

    assert fixture_result["status"] == "insufficient_data"
    assert opted_in_result["status"] == "insufficient_data"


@pytest.mark.parametrize("question", ["", " ", "x" * 501])
def test_cli_rejects_question_outside_bounds(tmp_path: Path, question: str) -> None:
    with pytest.raises(ValueError, match="Question"):
        cli.run(str(tmp_path), question, "question-bound", model_fn=lambda q, c: "unused")
