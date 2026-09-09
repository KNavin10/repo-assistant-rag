"""Command-line interface for the repository assistant."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from uuid import uuid4

from .graph import build_graph
from .model import groq_model
from .nodes import ModelFn
from .security import (
    enforce_tracing_consent,
    resolve_repository_root,
    validate_question,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Ask a question about a repository.")
    parser.add_argument("--repo", required=True, help="Path to the repository directory.")
    parser.add_argument("--question", required=True, help="Question to ask about the repository.")
    parser.add_argument("--thread-id", required=True, help="LangGraph checkpoint thread ID.")
    parser.add_argument(
        "--allow-external-model",
        action="store_true",
        help="Explicitly allow retrieved repository content to be sent to Groq.",
    )
    parser.add_argument(
        "--allow-non-fixture-tracing",
        action="store_true",
        help="Explicitly allow LangSmith tracing for a repository outside tests/fixtures.",
    )
    return parser


def run(
    repo_path: str,
    question: str,
    thread_id: str,
    *,
    model_fn: ModelFn | None = None,
    allow_external_model: bool = False,
    allow_non_fixture_tracing: bool = False,
) -> dict[str, object]:
    """Invoke the compiled graph for one CLI request."""

    root = resolve_repository_root(repo_path)
    normalized_question = validate_question(question)
    enforce_tracing_consent(
        root,
        allow_non_fixture_tracing=allow_non_fixture_tracing,
    )
    if model_fn is None and not allow_external_model:
        raise PermissionError(
            "External model access is disabled. Pass --allow-external-model "
            "to send retrieved repository content to Groq."
        )

    graph = build_graph(model_fn=model_fn if model_fn is not None else groq_model)
    initial_state = {
        "repo_path": str(root),
        "question": normalized_question,
        "route": "repo_question",
        "chunks": [],
        "answer": "",
        "citations": [],
        "status": "running",
        "errors": [],
    }
    request_id = str(uuid4())
    config = {
        "run_name": "repo-assistant-question",
        "tags": ["cli"],
        "metadata": {"request_id": request_id, "thread_id": thread_id},
        "configurable": {"thread_id": thread_id},
    }
    return graph.invoke(initial_state, config)


def format_result(result: dict[str, object]) -> str:
    """Format a graph result for terminal output."""

    if result.get("status") == "insufficient_data":
        return "INSUFFICIENT_DATA"

    if result.get("status") != "completed":
        errors = result.get("errors", [])
        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)) and errors:
            return f"Error: {errors[0]}"
        return "Error: The repository request could not be completed."

    answer = str(result.get("answer", ""))
    citations = result.get("citations", [])
    lines = ["Answer:", answer]
    if isinstance(citations, Sequence) and not isinstance(citations, (str, bytes)):
        lines.extend(["", "Sources:", *(str(citation) for citation in citations)])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, model_fn: ModelFn | None = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        result = run(
            args.repo,
            args.question,
            args.thread_id,
            model_fn=model_fn,
            allow_external_model=args.allow_external_model,
            allow_non_fixture_tracing=args.allow_non_fixture_tracing,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(format_result(result))
    return 0 if result.get("status") in {"completed", "insufficient_data"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "format_result", "main", "run"]
