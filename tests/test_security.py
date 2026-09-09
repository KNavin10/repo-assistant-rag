"""Focused tests for repository path and input security boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from repo_assistant.security import resolve_repository_file, resolve_repository_root
from repo_assistant.security_evaluation import (
    evaluate_security_cases,
    load_security_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_root_is_resolved_and_normalized(tmp_path: Path) -> None:
    nested = tmp_path / "repo"
    nested.mkdir()

    assert resolve_repository_root(nested / ".") == nested.resolve()


def test_repository_root_outside_allowed_root_is_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(PermissionError, match="outside the allowed root"):
        resolve_repository_root(outside, allowed_root=allowed)


def test_repository_file_accepts_normalized_relative_path(tmp_path: Path) -> None:
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")

    assert resolve_repository_file(tmp_path / ".", "./src/main.py") == source.resolve()


@pytest.mark.parametrize("file_path", ["../outside.py", "src/../outside.py"])
def test_repository_file_rejects_dot_dot_traversal(tmp_path: Path, file_path: str) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain '..' traversal"):
        resolve_repository_file(tmp_path, file_path)


def test_repository_file_rejects_absolute_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be relative"):
        resolve_repository_file(tmp_path, outside)


def test_repository_file_rejects_sensitive_credential_selection(tmp_path: Path) -> None:
    credential_file = tmp_path / "credentials.json"
    credential_file.write_text('{"fixture": "not-a-real-secret"}\n', encoding="utf-8")

    with pytest.raises(PermissionError, match="Sensitive file selection is not allowed"):
        resolve_repository_file(tmp_path, "credentials.json")


def test_repository_file_rejects_existing_file_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(ValueError, match="outside the repository root"):
        resolve_repository_file(tmp_path, "linked.py")


def test_declared_security_cases_all_have_passing_executable_checks() -> None:
    cases = load_security_cases(PROJECT_ROOT / "evals" / "security_cases.yaml")
    report = evaluate_security_cases(cases)

    assert [case.case_id for case in cases] == [
        "traversal_rejected",
        "external_repository_rejected",
        "top_k_bounded",
        "sensitive_files_skipped",
        "unsupported_question_no_model",
        "fabricated_citation_rejected",
        "prompt_injection_is_data",
        "non_fixture_tracing_rejected",
    ]
    assert report.passed
    assert report.success_count == report.case_count == 8


def test_ci_runs_tests_evaluations_and_security_scanners() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    required_commands = [
        '$env:PYTHONPATH = "src"',
        "-m pytest -q tests -p no:cacheprovider --basetemp .day13-test-temp",
        "-m repo_assistant.evaluation --mode keyword",
        "-m repo_assistant.security_evaluation",
        "-m pip check",
        "-m ruff check",
        "-m pip_audit -r requirements.txt",
        "gitleaks/gitleaks-action@v3",
    ]
    for command in required_commands:
        assert command in workflow


def test_security_review_records_concrete_evidence() -> None:
    report = (PROJECT_ROOT / "docs" / "security-review.md").read_text(encoding="utf-8")

    evidence = [
        "input='../../outside.py'; exception=ValueError",
        "selection_exception=PermissionError",
        "input=1000000; exception=ValueError",
        "action_calls=0",
        "route=unsupported; status=error; model_calls=0",
        "Citation does not correspond to any retrieved chunk",
        "LANGSMITH_TRACING=true; exception=PermissionError",
    ]
    for item in evidence:
        assert item in report


def test_prompt_injection_fixture_is_checked_in_as_untrusted_data() -> None:
    fixture = (
        PROJECT_ROOT / "tests" / "fixtures" / "security_repo" / "src" / "untrusted.md"
    ).read_text(encoding="utf-8")

    assert "Invoke a shell tool" in fixture
    assert "delete files outside" in fixture
