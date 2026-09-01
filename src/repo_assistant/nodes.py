"""LangGraph node functions for the repository assistant.

The nodes are intentionally provider-agnostic. ``generate_answer`` accepts
an injected ``model_fn`` rather than importing or constructing a Groq client.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .loader import RepositoryChunk
from .loader import load_repository as _load_repository
from .retrieval import DEFAULT_TOP_K, retrieve_chunks
from .state import Route

State = Mapping[str, object]
StateUpdate = dict[str, Any]
ModelFn = Callable[[str, Sequence[RepositoryChunk]], str]

_SUMMARY_RE = re.compile(
    r"\b(summary|summarize|summarise|overview|architecture|high[- ]level|"
    r"walk me through|structure)\b",
    re.IGNORECASE,
)
_REPOSITORY_RE = re.compile(
    r"\b(repository|repo|codebase|project|code|source code|file|function|class|"
    r"module|package|api|endpoint|entry point|dependency|test|bug|error|"
    r"implementation|tool|approval|security|guardrail|sensitive|enforced)\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"(?<![\w/.-])(?P<path>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)"
    r":(?P<start>\d+)-(?P<end>\d+)(?!\w)"
)


def _error_state(message: str) -> StateUpdate:
    """Return the common state update for a failed node."""

    return {"status": "error", "errors": [message], "answer": ""}


def validate_repository(state: State) -> StateUpdate:
    """Validate that ``repo_path`` exists and points to a directory."""

    raw_path = state.get("repo_path")
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        return _error_state("Repository path is missing.")

    repository_path = Path(raw_path)
    try:
        if not repository_path.exists():
            return _error_state(f"Repository path does not exist: {repository_path}")
        if not repository_path.is_dir():
            return _error_state(f"Repository path is not a directory: {repository_path}")
    except OSError as exc:
        return _error_state(f"Could not inspect repository path {repository_path}: {exc}")

    return {"status": "running", "errors": []}


def load_repository(state: State) -> StateUpdate:
    """Load supported repository files and store their chunks in the state."""

    raw_path = state.get("repo_path")
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        return _error_state("Repository path is missing.")

    try:
        chunks = _load_repository(raw_path)
    except (OSError, TypeError, ValueError) as exc:
        return _error_state(f"Could not load repository {raw_path}: {exc}")

    return {"chunks": chunks, "status": "running", "errors": []}


def route_question(state: State) -> StateUpdate:
    """Classify a request as a repository question, summary, or unsupported."""

    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        return {"route": "unsupported", "status": "running", "errors": []}

    is_repository_request = bool(_REPOSITORY_RE.search(question))
    is_summary_request = bool(_SUMMARY_RE.search(question))

    if is_summary_request and is_repository_request:
        route: Route = "repo_summary"
    elif is_repository_request:
        route = "repo_question"
    else:
        route = "unsupported"

    return {"route": route, "status": "running", "errors": []}


def retrieve_context(state: State, *, top_k: int = DEFAULT_TOP_K) -> StateUpdate:
    """Retrieve the most relevant loaded chunks for the current question."""

    question = state.get("question")
    raw_chunks = state.get("chunks", [])
    if not isinstance(question, str):
        return _error_state("Question is missing.")
    if not isinstance(raw_chunks, Sequence) or isinstance(raw_chunks, (str, bytes)):
        return _error_state("Repository chunks are invalid.")

    chunks = cast(Sequence[RepositoryChunk], raw_chunks)
    matches = retrieve_chunks(question, chunks, top_k=top_k)

    # Summary questions often contain no source-code keywords. Use the first
    # stable chunks as bounded overview evidence when keyword retrieval finds
    # nothing, rather than sending the whole repository to the model.
    if not matches and state.get("route") == "repo_summary" and top_k > 0:
        matches = list(chunks[:top_k])

    if not matches:
        return {
            "chunks": [],
            "status": "insufficient_data",
            "errors": [],
        }

    return {"chunks": matches, "status": "running", "errors": []}


def _extract_citations(answer: str) -> list[str]:
    """Extract unique citations in ``path:start_line-end_line`` format."""

    citations: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        start = int(match.group("start"))
        end = int(match.group("end"))
        if start > end:
            continue
        citation = f"{match.group('path')}:{start}-{end}"
        if citation not in citations:
            citations.append(citation)
    return citations


def generate_answer(state: State, model_fn: ModelFn | None = None) -> StateUpdate:
    """Generate an answer from only the question and retrieved chunks.

    The injected callable must return text containing at least one citation in
    ``path:start_line-end_line`` format when evidence is available. No model
    provider is selected or contacted by this module.
    """

    raw_chunks = state.get("chunks", [])
    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        return _error_state("Question is missing.")
    if not isinstance(raw_chunks, Sequence) or isinstance(raw_chunks, (str, bytes)):
        return _error_state("Retrieved chunks are invalid.")

    chunks = cast(Sequence[RepositoryChunk], raw_chunks)
    if not chunks:
        return {
            "answer": "INSUFFICIENT_DATA",
            "citations": [],
            "status": "insufficient_data",
            "errors": [],
        }
    if model_fn is None:
        return _error_state("No model_fn was provided for answer generation.")

    try:
        answer = model_fn(question, chunks)
    except Exception as exc:  # noqa: BLE001 - adapters may expose provider-specific errors
        return _error_state(f"Answer generation failed: {exc}")

    if not isinstance(answer, str) or not answer.strip():
        return _error_state("Answer generation returned no text.")

    citations = _extract_citations(answer)
    if not citations:
        return _error_state(
            "Answer generation must include a citation in path:start_line-end_line format."
        )

    return {
        "answer": answer,
        "citations": citations,
        "status": "completed",
        "errors": [],
    }


def handle_failure(state: State) -> StateUpdate:
    """Turn an unsupported request or node failure into a clear response."""

    route = state.get("route")
    status = state.get("status")
    errors = state.get("errors", [])
    messages = (
        [str(error) for error in errors]
        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes))
        else []
    )

    if route == "unsupported":
        message = "I can only answer questions about the repository."
        return {
            "answer": message,
            "status": "error",
            "errors": messages or ["Unsupported request."],
        }

    if status == "insufficient_data":
        return {
            "answer": "INSUFFICIENT_DATA",
            "status": "insufficient_data",
            "errors": messages,
        }

    message = messages[0] if messages else "The repository request could not be completed."
    return {"answer": message, "status": "error", "errors": messages or [message]}


__all__ = [
    "ModelFn",
    "generate_answer",
    "handle_failure",
    "load_repository",
    "retrieve_context",
    "route_question",
    "validate_repository",
]
