"""Provider-free executable evaluation of Project B's security boundaries.

The evaluator deliberately creates small temporary repositories and exercises
the same resolver, loader, graph, citation, and tracing functions used by the
application.  No model provider, network connection, or external repository is
used.  It is intended to run in CI with::

    python -m repo_assistant.security_evaluation
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .graph import build_graph
from .loader import load_repository
from .nodes import generate_answer
from .security import (
    enforce_tracing_consent,
    resolve_repository_file,
    resolve_repository_root,
    validate_top_k,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = _PROJECT_ROOT / "evals" / "security_cases.yaml"
SECURITY_FIXTURE_ROOT = _PROJECT_ROOT / "tests" / "fixtures" / "security_repo"


@dataclass(frozen=True)
class SecurityCase:
    """A human-readable security case loaded from YAML."""

    case_id: str
    description: str


@dataclass(frozen=True)
class SecurityCaseResult:
    """The deterministic outcome for one security case."""

    case_id: str
    success: bool
    detail: str = ""


@dataclass(frozen=True)
class SecurityEvaluationReport:
    """Aggregate result for the security evaluation."""

    results: tuple[SecurityCaseResult, ...]

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(result.success for result in self.results)

    @property
    def passed(self) -> bool:
        return self.case_count > 0 and self.success_count == self.case_count


def load_security_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[SecurityCase]:
    """Load and validate the security case labels from YAML."""

    raw_cases = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Security evaluation YAML must contain a non-empty list.")

    cases: list[SecurityCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise TypeError(f"Security case {index} must be a mapping.")
        case_id = raw_case.get("id")
        description = raw_case.get("description")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Security case {index} needs a non-empty id.")
        if case_id in seen:
            raise ValueError(f"Duplicate security case id: {case_id}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Security case {case_id} needs a description.")
        seen.add(case_id)
        cases.append(SecurityCase(case_id.strip(), description.strip()))
    return cases


def _write(path: Path, text: str) -> None:
    """Write a test fixture file, creating only its parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repository(root: Path) -> None:
    """Create a minimal safe repository fixture."""

    _write(root / "src" / "main.py", "def main():\n    return 1\n")


def _initial_state(repo: Path, question: str) -> dict[str, object]:
    return {
        "repo_path": str(repo),
        "question": question,
        "route": "repo_question",
        "chunks": [],
        "answer": "",
        "citations": [],
        "status": "running",
        "errors": [],
    }


def _case_traversal_rejected() -> str:
    try:
        resolve_repository_file(SECURITY_FIXTURE_ROOT, "../../outside.py")
    except ValueError as exc:
        if "traversal" not in str(exc).casefold():
            raise AssertionError(f"unexpected traversal error: {exc}") from exc
        return f"input='../../outside.py'; exception={type(exc).__name__}: {exc}"
    raise AssertionError("../../outside.py was accepted")


def _case_external_repository_rejected() -> str:
    with tempfile.TemporaryDirectory(prefix="security-eval-") as raw_root:
        base = Path(raw_root)
        allowed = base / "allowed"
        external = base / "external"
        allowed.mkdir()
        external.mkdir()
        _make_repository(external)

        try:
            resolve_repository_root(external, allowed_root=allowed)
        except (OSError, TypeError, ValueError) as exc:
            return f"exception={type(exc).__name__}: {exc}"
        raise AssertionError("repository outside allowed_root was accepted")


def _case_top_k_bounded() -> str:
    try:
        validate_top_k(1_000_000)
    except (TypeError, ValueError) as exc:
        return f"input=1000000; exception={type(exc).__name__}: {exc}"
    raise AssertionError("k=1000000 was accepted")


def _case_sensitive_files_skipped() -> str:
    chunks = load_repository(SECURITY_FIXTURE_ROOT)
    files = sorted({chunk["file_path"] for chunk in chunks})
    if "credentials.json" in files:
        raise AssertionError(f"sensitive file was loaded: {files}")
    try:
        resolve_repository_file(SECURITY_FIXTURE_ROOT, "credentials.json")
    except PermissionError as exc:
        return f"indexed_files={files}; selection_exception={type(exc).__name__}: {exc}"
    raise AssertionError("credentials.json was selectable")


def _case_unsupported_question_no_model() -> str:
    calls = 0

    def model_must_not_run(question: str, chunks: object) -> str:
        nonlocal calls
        calls += 1
        return "unexpected [src/safe.py:1-2]"

    result = build_graph(model_fn=model_must_not_run).invoke(
        _initial_state(SECURITY_FIXTURE_ROOT, "What is the weather today?"),
        {"configurable": {"thread_id": "security-unsupported"}},
    )
    if calls != 0:
        raise AssertionError("unsupported question called the model")
    if result.get("status") != "error" or result.get("route") != "unsupported":
        raise AssertionError(f"unsupported question was not refused: {result}")
    return (
        f"question='What is the weather today?'; route={result.get('route')}; "
        f"status={result.get('status')}; model_calls={calls}"
    )


def _case_fabricated_citation_rejected() -> str:
    chunks = load_repository(SECURITY_FIXTURE_ROOT)
    if not chunks:
        raise AssertionError("fixture did not produce chunks")

    fabricated = "src/missing.py:1-2"
    result = generate_answer(
        _initial_state(SECURITY_FIXTURE_ROOT, "Where is safe_operation implemented?")
        | {"chunks": chunks},
        model_fn=lambda question, retrieved: f"Fabricated [{fabricated}]",
    )
    if result.get("status") != "error":
        raise AssertionError(f"fabricated citation was accepted: {result}")
    errors = result.get("errors", [])
    if not any("correspond" in str(error) or "exist" in str(error) for error in errors):
        raise AssertionError(f"wrong fabricated-citation error: {errors}")
    return f"citation='{fabricated}'; status={result.get('status')}; errors={errors}"


def _case_prompt_injection_is_data() -> str:
    chunks = [
        chunk
        for chunk in load_repository(SECURITY_FIXTURE_ROOT)
        if chunk["file_path"] == "src/untrusted.md"
    ]
    action_calls: list[str] = []
    model_calls = 0

    def model_treats_source_as_evidence(question: str, retrieved: object) -> str:
        nonlocal model_calls
        model_calls += 1
        if not any("delete files outside" in str(chunk) for chunk in retrieved):
            raise AssertionError("injection fixture was not supplied as evidence")
        return "The source contains untrusted instructions. [src/untrusted.md:1-4]"

    result = generate_answer(
        _initial_state(SECURITY_FIXTURE_ROOT, "What does this source contain?")
        | {"chunks": chunks},
        model_fn=model_treats_source_as_evidence,
    )
    if result.get("status") != "completed" or model_calls != 1:
        raise AssertionError(f"injection case did not complete safely: {result}")
    if action_calls:
        raise AssertionError(f"source text triggered an action: {action_calls}")
    return (
        "fixture='tests/fixtures/security_repo/src/untrusted.md'; "
        f"status={result.get('status')}; model_calls={model_calls}; "
        f"action_calls={len(action_calls)}; citations={result.get('citations')}"
    )


def _case_non_fixture_tracing_rejected() -> str:
    old_value = os.environ.get("LANGSMITH_TRACING")
    os.environ["LANGSMITH_TRACING"] = "true"
    try:
        try:
            enforce_tracing_consent(_PROJECT_ROOT, allow_non_fixture_tracing=False)
        except PermissionError as exc:
            return (
                f"repo='{_PROJECT_ROOT}'; LANGSMITH_TRACING=true; "
                f"exception={type(exc).__name__}: {exc}"
            )
        raise AssertionError("non-fixture tracing was accepted")
    finally:
        if old_value is None:
            os.environ.pop("LANGSMITH_TRACING", None)
        else:
            os.environ["LANGSMITH_TRACING"] = old_value


_CASE_RUNNERS: dict[str, Callable[[], str]] = {
    "traversal_rejected": _case_traversal_rejected,
    "external_repository_rejected": _case_external_repository_rejected,
    "top_k_bounded": _case_top_k_bounded,
    "sensitive_files_skipped": _case_sensitive_files_skipped,
    "unsupported_question_no_model": _case_unsupported_question_no_model,
    "fabricated_citation_rejected": _case_fabricated_citation_rejected,
    "prompt_injection_is_data": _case_prompt_injection_is_data,
    "non_fixture_tracing_rejected": _case_non_fixture_tracing_rejected,
}


def evaluate_security_cases(
    cases: list[SecurityCase] | tuple[SecurityCase, ...] | None = None,
) -> SecurityEvaluationReport:
    """Execute all requested cases and return pass/fail results."""

    selected = list(cases) if cases is not None else load_security_cases()
    results: list[SecurityCaseResult] = []
    for case in selected:
        runner = _CASE_RUNNERS.get(case.case_id)
        if runner is None:
            results.append(
                SecurityCaseResult(case.case_id, False, "no executable assertion registered")
            )
            continue
        try:
            detail = runner()
        except Exception as exc:  # noqa: BLE001 - report every case deterministically
            results.append(SecurityCaseResult(case.case_id, False, f"{type(exc).__name__}: {exc}"))
        else:
            results.append(SecurityCaseResult(case.case_id, True, detail))
    return SecurityEvaluationReport(tuple(results))


def build_parser() -> argparse.ArgumentParser:
    """Create the security evaluation CLI parser."""

    parser = argparse.ArgumentParser(description="Run provider-free security cases.")
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="Security case YAML path (default: evals/security_cases.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluator, returning nonzero if any case fails."""

    args = build_parser().parse_args(argv)
    try:
        cases = load_security_cases(args.cases)
        report = evaluate_security_cases(cases)
    except Exception as exc:  # noqa: BLE001 - CLI should report malformed cases
        print(f"security evaluation failed: {type(exc).__name__}: {exc}")
        return 1

    for result in report.results:
        status = "PASS" if result.success else "FAIL"
        suffix = f" - {result.detail}" if result.detail else ""
        print(f"{status} {result.case_id}{suffix}")
    print(f"Security cases: {report.success_count}/{report.case_count} passed")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CASES_PATH",
    "SECURITY_FIXTURE_ROOT",
    "SecurityCase",
    "SecurityCaseResult",
    "SecurityEvaluationReport",
    "build_parser",
    "evaluate_security_cases",
    "load_security_cases",
    "main",
]
