"""Command-line interface for the repository assistant."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .graph import build_graph
from .model import groq_model
from .nodes import ModelFn


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Ask a question about a repository.")
    parser.add_argument("--repo", required=True, help="Path to the repository directory.")
    parser.add_argument("--question", required=True, help="Question to ask about the repository.")
    parser.add_argument("--thread-id", required=True, help="LangGraph checkpoint thread ID.")
    return parser


def run(
    repo_path: str,
    question: str,
    thread_id: str,
    *,
    model_fn: ModelFn | None = None,
) -> dict[str, object]:
    """Invoke the compiled graph for one CLI request."""

    graph = build_graph(model_fn=model_fn or groq_model)
    initial_state = {
        "repo_path": repo_path,
        "question": question,
        "route": "repo_question",
        "chunks": [],
        "answer": "",
        "citations": [],
        "status": "running",
        "errors": [],
    }
    config = {"configurable": {"thread_id": thread_id}}
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
    result = run(args.repo, args.question, args.thread_id, model_fn=model_fn)
    print(format_result(result))
    return 0 if result.get("status") in {"completed", "insufficient_data"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "format_result", "main", "run"]
