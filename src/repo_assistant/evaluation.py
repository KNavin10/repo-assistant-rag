"""Provider-free retrieval evaluation for the Day 9 keyword baseline."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .embeddings import EmbeddingFunction, InMemoryVectorIndex
from .loader import RepositoryChunk, load_repository
from .retrieval import hybrid_search, retrieve_chunks

DEFAULT_EVAL_TOP_K = 4


@dataclass(frozen=True)
class EvaluationCase:
    """One labelled retrieval question."""

    question: str
    expected_source: str | None


@dataclass(frozen=True)
class CaseResult:
    """Measured outcome for one evaluation case."""

    question: str
    expected_source: str | None
    retrieved_sources: tuple[str, ...]
    latency_ms: float
    success: bool


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate and per-question retrieval results."""

    top_k: int
    results: tuple[CaseResult, ...]

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(result.success for result in self.results)

    @property
    def hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.success_count / self.case_count

    @property
    def median_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return statistics.median(result.latency_ms for result in self.results)


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load and validate question/source labels from a YAML file."""

    raw_cases = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise TypeError("Evaluation YAML must contain a list of cases.")

    cases: list[EvaluationCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise TypeError(f"Evaluation case {index} must be a mapping.")
        question = raw_case.get("question")
        expected_source = raw_case.get("expected_source")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Evaluation case {index} needs a non-empty question.")
        if expected_source is not None and (
            not isinstance(expected_source, str) or not expected_source.strip()
        ):
            raise ValueError(
                f"Evaluation case {index} expected_source must be a path or null."
            )
        cases.append(EvaluationCase(question.strip(), expected_source))
    return cases


def _source_paths(chunks: list[RepositoryChunk]) -> tuple[str, ...]:
    """Return unique file paths in retrieval order."""

    paths: list[str] = []
    for chunk in chunks:
        path = chunk.get("file_path")
        if isinstance(path, str) and path not in paths:
            paths.append(path)
    return tuple(paths)


def evaluate_cases(
    cases: list[EvaluationCase] | tuple[EvaluationCase, ...],
    chunks: list[RepositoryChunk] | tuple[RepositoryChunk, ...],
    *,
    top_k: int = DEFAULT_EVAL_TOP_K,
) -> EvaluationReport:
    """Evaluate keyword retrieval and measure only retrieval latency.

    A labelled source is a hit when it appears in the top ``top_k`` files.
    A case with ``expected_source: null`` is successful when retrieval returns
    no chunks, which makes unsupported evidence requests measurable too.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        matches = retrieve_chunks(case.question, chunks, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        sources = _source_paths(matches)
        success = (
            case.expected_source in sources
            if case.expected_source is not None
            else not matches
        )
        results.append(
            CaseResult(
                question=case.question,
                expected_source=case.expected_source,
                retrieved_sources=sources,
                latency_ms=latency_ms,
                success=success,
            )
        )
    return EvaluationReport(top_k=top_k, results=tuple(results))


def evaluate_vector_cases(
    cases: list[EvaluationCase] | tuple[EvaluationCase, ...],
    chunks: list[RepositoryChunk] | tuple[RepositoryChunk, ...],
    *,
    embedding_fn: EmbeddingFunction,
    top_k: int = DEFAULT_EVAL_TOP_K,
    min_score: float | None = None,
) -> EvaluationReport:
    """Evaluate vector-only retrieval using one in-memory embedding matrix.

    The matrix is built once before the cases run.  Per-case latency therefore
    measures query embedding plus cosine search, matching the retrieval-only
    latency convention used by the keyword baseline.  ``min_score`` remains
    optional so this benchmark records nearest-neighbour behaviour without
    silently tuning an evidence threshold.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    index = InMemoryVectorIndex(
        chunks,
        embedding_fn,
        text_getter=lambda chunk: chunk["text"],
    )
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        matches = index.search(case.question, top_k=top_k, min_score=min_score)
        latency_ms = (time.perf_counter() - started) * 1000
        sources = _source_paths([match.document for match in matches])
        success = (
            case.expected_source in sources
            if case.expected_source is not None
            else not matches
        )
        results.append(
            CaseResult(
                question=case.question,
                expected_source=case.expected_source,
                retrieved_sources=sources,
                latency_ms=latency_ms,
                success=success,
            )
        )
    return EvaluationReport(top_k=top_k, results=tuple(results))


def run_evaluation(
    repository_path: str | Path,
    cases_path: str | Path,
    *,
    top_k: int = DEFAULT_EVAL_TOP_K,
    output_path: str | Path | None = None,
    baseline_name: str = "day9_keyword_baseline",
) -> EvaluationReport:
    """Load a repository and case file, evaluate it, and optionally write Markdown."""

    report = evaluate_cases(
        load_evaluation_cases(cases_path),
        load_repository(repository_path),
        top_k=top_k,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report, baseline_name=baseline_name), encoding="utf-8")
    return report


def run_vector_evaluation(
    repository_path: str | Path,
    cases_path: str | Path,
    *,
    embedding_fn: EmbeddingFunction,
    top_k: int = DEFAULT_EVAL_TOP_K,
    min_score: float | None = None,
    output_path: str | Path | None = None,
    baseline_name: str = "vector_only",
) -> EvaluationReport:
    """Load a repository, run vector-only evaluation, and optionally write Markdown."""

    report = evaluate_vector_cases(
        load_evaluation_cases(cases_path),
        load_repository(repository_path),
        embedding_fn=embedding_fn,
        top_k=top_k,
        min_score=min_score,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report, baseline_name=baseline_name), encoding="utf-8")
    return report


def evaluate_hybrid_cases(
    cases: list[EvaluationCase] | tuple[EvaluationCase, ...],
    chunks: list[RepositoryChunk] | tuple[RepositoryChunk, ...],
    *,
    embedding_fn: EmbeddingFunction,
    top_k: int = DEFAULT_EVAL_TOP_K,
    candidate_k: int = 20,
    min_score: float | None = None,
) -> EvaluationReport:
    """Evaluate hybrid BM25 + dense vector retrieval using Reciprocal Rank Fusion.

    The vector index is prebuilt once before evaluating cases. Per-case latency
    measures query tokenization, BM25 scoring, query embedding, cosine search,
    and RRF ranking.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    index = InMemoryVectorIndex(
        chunks,
        embedding_fn,
        text_getter=lambda chunk: chunk["text"],
    )
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        matches = hybrid_search(
            case.question,
            chunks,
            top_k=top_k,
            candidate_k=candidate_k,
            vector_index=index,
            min_vector_score=min_score,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        sources = _source_paths(matches)
        success = (
            case.expected_source in sources
            if case.expected_source is not None
            else not matches
        )
        results.append(
            CaseResult(
                question=case.question,
                expected_source=case.expected_source,
                retrieved_sources=sources,
                latency_ms=latency_ms,
                success=success,
            )
        )
    return EvaluationReport(top_k=top_k, results=tuple(results))


def run_hybrid_evaluation(
    repository_path: str | Path,
    cases_path: str | Path,
    *,
    embedding_fn: EmbeddingFunction,
    top_k: int = DEFAULT_EVAL_TOP_K,
    candidate_k: int = 20,
    min_score: float | None = None,
    output_path: str | Path | None = None,
    baseline_name: str = "hybrid_rrf",
) -> EvaluationReport:
    """Load a repository, run hybrid evaluation, and optionally write Markdown."""

    report = evaluate_hybrid_cases(
        load_evaluation_cases(cases_path),
        load_repository(repository_path),
        embedding_fn=embedding_fn,
        top_k=top_k,
        candidate_k=candidate_k,
        min_score=min_score,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report, baseline_name=baseline_name), encoding="utf-8")
    return report


def _markdown_value(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""

    if value is None:
        return "none"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(
    report: EvaluationReport,
    *,
    baseline_name: str = "day9_keyword_baseline",
) -> str:
    """Render aggregate metrics and every case outcome as Markdown."""

    lines = [
        f"# {baseline_name}",
        "",
        f"- `hit_rate@{report.top_k}`: {report.hit_rate:.1%} ({report.success_count}/{report.case_count})",
        f"- Median retrieval latency: {report.median_latency_ms:.3f} ms",
        f"- Evaluated cases: {report.case_count}",
        "",
        "## Per-question results",
        "",
        "| # | Result | Question | Expected source | Retrieved sources |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, result in enumerate(report.results, start=1):
        retrieved = ", ".join(result.retrieved_sources) or "none"
        lines.append(
            f"| {index} | {'success' if result.success else 'failure'} | "
            f"{_markdown_value(result.question)} | {_markdown_value(result.expected_source)} | "
            f"{_markdown_value(retrieved)} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Create the evaluation command-line parser."""

    parser = argparse.ArgumentParser(description="Evaluate repository retrieval.")
    parser.add_argument(
        "--repo",
        default="tests/fixtures/eval_repo",
        help="Repository fixture to evaluate (default: tests/fixtures/eval_repo).",
    )
    parser.add_argument(
        "--cases",
        default="evals/retrieval_cases.yaml",
        help="YAML retrieval cases (default: evals/retrieval_cases.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional Markdown results path.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_EVAL_TOP_K)
    parser.add_argument(
        "--mode",
        choices=["keyword", "vector", "hybrid"],
        default="keyword",
        help="Retrieval engine to evaluate (default: keyword).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum vector score threshold.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate count per retrieval stream for hybrid search.",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default=None,
        help="Optional custom baseline name for report title.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluator and write its Markdown report."""

    args = build_parser().parse_args(argv)
    if args.mode == "hybrid":
        from .embeddings import SentenceTransformerAdapter

        adapter = SentenceTransformerAdapter()
        baseline_name = args.baseline_name or "hybrid_rrf"
        report = run_hybrid_evaluation(
            args.repo,
            args.cases,
            embedding_fn=adapter,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            min_score=args.min_score,
            output_path=args.output,
            baseline_name=baseline_name,
        )
    elif args.mode == "vector":
        from .embeddings import SentenceTransformerAdapter

        adapter = SentenceTransformerAdapter()
        baseline_name = args.baseline_name or "vector_only"
        report = run_vector_evaluation(
            args.repo,
            args.cases,
            embedding_fn=adapter,
            top_k=args.top_k,
            min_score=args.min_score,
            output_path=args.output,
            baseline_name=baseline_name,
        )
    else:
        baseline_name = args.baseline_name or "day9_keyword_baseline"
        report = run_evaluation(
            args.repo,
            args.cases,
            top_k=args.top_k,
            output_path=args.output,
            baseline_name=baseline_name,
        )
    print(render_markdown(report, baseline_name=baseline_name), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EVAL_TOP_K",
    "CaseResult",
    "EvaluationCase",
    "EvaluationReport",
    "build_parser",
    "evaluate_cases",
    "evaluate_hybrid_cases",
    "evaluate_vector_cases",
    "load_evaluation_cases",
    "main",
    "render_markdown",
    "run_evaluation",
    "run_hybrid_evaluation",
    "run_vector_evaluation",
]
