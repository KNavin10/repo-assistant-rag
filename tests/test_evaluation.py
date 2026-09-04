"""Deterministic tests for the retrieval evaluation set."""

from __future__ import annotations

from pathlib import Path

from repo_assistant.evaluation import (
    EvaluationCase,
    evaluate_cases,
    evaluate_hybrid_cases,
    evaluate_vector_cases,
    load_evaluation_cases,
    render_markdown,
    run_evaluation,
)
from repo_assistant.loader import load_repository

ROOT = Path(__file__).parent.parent
CASES_PATH = ROOT / "evals" / "retrieval_cases.yaml"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "eval_repo"


def test_evaluation_yaml_contains_ten_labelled_cases() -> None:
    cases = load_evaluation_cases(CASES_PATH)

    assert len(cases) == 10
    assert cases[0].question == "Where is build_graph implemented?"
    assert cases[0].expected_source == "src/repo_assistant/graph.py"
    assert cases[-1].expected_source is None


def test_keyword_baseline_reports_all_fixture_cases() -> None:
    report = evaluate_cases(load_evaluation_cases(CASES_PATH), load_repository(FIXTURE_PATH))

    assert report.top_k == 4
    assert report.case_count == 10
    assert report.success_count == 10
    assert report.hit_rate == 1.0
    assert report.median_latency_ms >= 0
    assert all(result.success for result in report.results)


def test_markdown_report_contains_metrics_and_per_question_statuses() -> None:
    report = evaluate_cases(load_evaluation_cases(CASES_PATH), load_repository(FIXTURE_PATH))
    markdown = render_markdown(report)

    assert "# day9_keyword_baseline" in markdown
    assert "`hit_rate@4`: 100.0% (10/10)" in markdown
    assert "Median retrieval latency:" in markdown
    assert "Evaluated cases: 10" in markdown
    assert markdown.count("| success |") == 10


def test_run_evaluation_writes_results_file(tmp_path: Path) -> None:
    output_path = tmp_path / "results.md"

    report = run_evaluation(FIXTURE_PATH, CASES_PATH, output_path=output_path)

    assert report.case_count == 10
    assert output_path.read_text(encoding="utf-8").startswith("# day9_keyword_baseline\n")


def test_vector_evaluation_accepts_an_injected_fake_embedding_function() -> None:
    chunks = [
        {"file_path": "a.py", "start_line": 1, "end_line": 1, "text": "alpha"},
        {"file_path": "b.py", "start_line": 1, "end_line": 1, "text": "beta"},
    ]
    vectors = {
        "alpha": [1.0, 0.0],
        "beta": [0.0, 1.0],
        "find alpha": [1.0, 0.0],
        "find beta": [0.0, 1.0],
    }

    def fake_embedding(texts: list[str]) -> list[list[float]]:
        return [vectors[text] for text in texts]

    report = evaluate_vector_cases(
        [
            EvaluationCase("find alpha", "a.py"),
            EvaluationCase("find beta", "b.py"),
        ],
        chunks,
        embedding_fn=fake_embedding,
        top_k=1,
    )

    assert report.success_count == 2
    assert report.hit_rate == 1.0
    assert report.median_latency_ms >= 0


def test_hybrid_evaluation_accepts_an_injected_fake_embedding_function() -> None:
    chunks = [
        {"file_path": "a.py", "start_line": 1, "end_line": 1, "text": "alpha algorithm"},
        {"file_path": "b.py", "start_line": 1, "end_line": 1, "text": "beta algorithm"},
    ]
    vectors = {
        "alpha algorithm": [1.0, 0.0],
        "beta algorithm": [0.0, 1.0],
        "find alpha": [1.0, 0.0],
        "find beta": [0.0, 1.0],
    }

    def fake_embedding(texts: list[str]) -> list[list[float]]:
        return [vectors[text] for text in texts]

    report = evaluate_hybrid_cases(
        [
            EvaluationCase("find alpha", "a.py"),
            EvaluationCase("find beta", "b.py"),
        ],
        chunks,
        embedding_fn=fake_embedding,
        top_k=1,
    )

    assert report.success_count == 2
    assert report.hit_rate == 1.0
    assert report.median_latency_ms >= 0

