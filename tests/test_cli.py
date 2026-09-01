"""Provider-free tests for the repository-assistant CLI."""

from __future__ import annotations

from pathlib import Path

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
