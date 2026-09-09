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
from .retrieval import DEFAULT_HYBRID_TOP_K, DEFAULT_MIN_VECTOR_SCORE, hybrid_search
from .security import (
    resolve_repository_file,
    resolve_repository_root,
    validate_question,
)
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

    try:
        repository_path = resolve_repository_root(raw_path)
    except (OSError, TypeError, ValueError) as exc:
        return _error_state(f"Could not inspect repository path {raw_path}: {exc}")

    return {"repo_path": str(repository_path), "status": "running", "errors": []}


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
    try:
        normalized_question = validate_question(question)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        return {"route": "unsupported", **_error_state(str(exc))}

    is_repository_request = bool(_REPOSITORY_RE.search(normalized_question))
    is_summary_request = bool(_SUMMARY_RE.search(normalized_question))

    if is_summary_request and is_repository_request:
        route: Route = "repo_summary"
    elif is_repository_request:
        route = "repo_question"
    else:
        route = "unsupported"

    return {"route": route, "status": "running", "errors": []}


def retrieve_context(
    state: State,
    *,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    min_vector_score: float | None = DEFAULT_MIN_VECTOR_SCORE,
) -> StateUpdate:
    """Retrieve the most relevant loaded chunks for the current question."""

    question = state.get("question")
    raw_chunks = state.get("chunks", [])
    try:
        normalized_question = validate_question(question)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        return _error_state(str(exc))
    if not isinstance(raw_chunks, Sequence) or isinstance(raw_chunks, (str, bytes)):
        return _error_state("Repository chunks are invalid.")

    chunks = cast(Sequence[RepositoryChunk], raw_chunks)
    matches = hybrid_search(
        normalized_question,
        chunks,
        top_k=top_k,
        min_vector_score=min_vector_score,
    )

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
        if start < 1 or start > end:
            continue
        citation = f"{match.group('path')}:{start}-{end}"
        if citation not in citations:
            citations.append(citation)
    return citations


def _validate_and_extract_citations(
    answer: str,
    chunks: Sequence[RepositoryChunk],
    repo_path: str | Path | None = None,
) -> tuple[list[str], str | None]:
    """Validate citations against line rules, retrieved chunks, and repository.

    Returns a tuple of (unique_citations, error_message). If validation fails,
    error_message describes the reason and citations will be empty.
    """

    matches = list(_CITATION_RE.finditer(answer))
    if not matches:
        return [], "Answer generation must include a citation in path:start_line-end_line format."

    citations: list[str] = []
    try:
        repo_root = (
            resolve_repository_root(repo_path)
            if isinstance(repo_path, (str, Path)) and str(repo_path).strip()
            else None
        )
    except (OSError, TypeError, ValueError) as exc:
        return [], f"Could not inspect selected repository for citations: {exc}"

    for match in matches:
        path = match.group("path")
        start = int(match.group("start"))
        end = int(match.group("end"))

        if start < 1:
            return [], f"Invalid citation line range start must be at least 1: {path}:{start}-{end}"
        if start > end:
            return (
                [],
                f"Invalid citation line range: start line {start} exceeds end line {end} ({path}:{start}-{end})",
            )

        # Every citation must correspond to one of the retrieved chunks
        corresponds = any(
            chunk.get("file_path") == path
            and chunk.get("start_line", 0) <= start
            and end <= chunk.get("end_line", 0)
            for chunk in chunks
        )
        if not corresponds:
            return [], f"Citation does not correspond to any retrieved chunk: {path}:{start}-{end}"

        # Every cited path must exist under the selected repository
        if repo_root is not None:
            try:
                cited_file = resolve_repository_file(repo_root, path)
                line_count = len(
                    cited_file.read_text(encoding="utf-8", errors="replace").splitlines()
                )
                if end > line_count:
                    return (
                        [],
                        f"Citation line range {start}-{end} exceeds file length ({line_count} lines): {path}",
                    )
            except FileNotFoundError:
                return [], f"Cited path does not exist under the selected repository: {path}"
            except (OSError, TypeError, ValueError) as exc:
                return [], f"Cited path is invalid under the selected repository: {path}: {exc}"

        citation = f"{path}:{start}-{end}"
        if citation not in citations:
            citations.append(citation)

    return citations, None


def generate_answer(state: State, model_fn: ModelFn | None = None) -> StateUpdate:
    """Generate an answer from only the question and retrieved chunks.

    The injected callable must return text containing at least one citation in
    ``path:start_line-end_line`` format when evidence is available. No model
    provider is selected or contacted by this module.
    """

    raw_chunks = state.get("chunks", [])
    question = state.get("question")
    try:
        normalized_question = validate_question(question)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        return _error_state(str(exc))
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
        answer = model_fn(normalized_question, chunks)
    except Exception as exc:  # noqa: BLE001 - adapters may expose provider-specific errors
        return _error_state(f"Answer generation failed: {exc}")

    if not isinstance(answer, str) or not answer.strip():
        return _error_state("Answer generation returned no text.")

    raw_repo = state.get("repo_path")
    repo_path = raw_repo if isinstance(raw_repo, (str, Path)) else None
    citations, error_message = _validate_and_extract_citations(
        answer,
        chunks,
        repo_path=repo_path,
    )
    if error_message is not None:
        return _error_state(error_message)

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

    if status == "error" and messages:
        return {"answer": messages[0], "status": "error", "errors": messages}

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
