"""Shared input, path, and external-service security boundaries."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath

MIN_QUESTION_LENGTH = 1
MAX_QUESTION_LENGTH = 500
MIN_TOP_K = 1
MAX_TOP_K = 8

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES_ROOT = (_PROJECT_ROOT / "tests" / "fixtures").resolve()
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_PRIVATE_MATERIAL_EXTENSIONS = frozenset(
    {".key", ".pem", ".crt", ".cer", ".p12", ".pfx", ".der", ".csr"}
)
_STRUCTURED_SECRET_EXTENSIONS = frozenset({".json", ".yaml", ".yml"})
_STRUCTURED_SECRET_RE = re.compile(
    r"(?:^|[._-])(?:credential|credentials|secret|secrets|service[-_]?account)(?:[._-]|$)",
    re.IGNORECASE,
)


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether ``path`` is contained by ``root``."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_sensitive_file(path: Path) -> bool:
    """Return whether a file name indicates credentials or private material."""

    name = path.name.casefold()
    if name.startswith(".env"):
        return True
    if path.suffix.casefold() in _PRIVATE_MATERIAL_EXTENSIONS:
        return True
    return (
        path.suffix.casefold() in _STRUCTURED_SECRET_EXTENSIONS
        and _STRUCTURED_SECRET_RE.search(name) is not None
    )


def resolve_repository_root(
    repository_path: str | os.PathLike[str],
    *,
    allowed_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a repository directory and optionally enforce an allowed root."""

    if not isinstance(repository_path, (str, os.PathLike)) or not str(repository_path).strip():
        raise ValueError("Repository path is missing.")

    candidate = Path(repository_path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Repository path does not exist: {candidate}")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")
    if allowed_root is not None:
        boundary = resolve_repository_root(allowed_root)
        if not _is_relative_to(root, boundary):
            raise PermissionError(
                f"Repository path is outside the allowed root {boundary}: {root}"
            )
    return root


def resolve_repository_file(
    repository_root: str | os.PathLike[str],
    file_path: str | os.PathLike[str],
) -> Path:
    """Resolve a relative file path and require it to remain inside the root."""

    root = resolve_repository_root(repository_root)
    if not isinstance(file_path, (str, os.PathLike)) or not str(file_path).strip():
        raise ValueError("File path is missing.")

    candidate = Path(file_path)
    if candidate.is_absolute():
        raise ValueError("File path must be relative to the repository root.")
    if ".." in PurePath(candidate).parts:
        raise ValueError("File path must not contain '..' traversal.")

    unresolved = root / candidate
    if not unresolved.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    resolved = unresolved.resolve(strict=True)
    if not _is_relative_to(resolved, root):
        raise ValueError("File path resolves outside the repository root.")
    if not resolved.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if is_sensitive_file(candidate) or is_sensitive_file(resolved):
        raise PermissionError(f"Sensitive file selection is not allowed: {file_path}")
    return resolved


def validate_question(question: str) -> str:
    """Validate and normalize a bounded user question."""

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Question must be at most {MAX_QUESTION_LENGTH} characters.")
    normalized = question.strip()
    if len(normalized) < MIN_QUESTION_LENGTH:
        raise ValueError("Question must contain at least one non-whitespace character.")
    return normalized


def validate_top_k(k: int) -> int:
    """Require a retrieval result count within the public MCP boundary."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if not MIN_TOP_K <= k <= MAX_TOP_K:
        raise ValueError(f"k must be between {MIN_TOP_K} and {MAX_TOP_K}.")
    return k


def langsmith_tracing_enabled() -> bool:
    """Return whether LangSmith/LangChain tracing is enabled by environment."""

    return any(
        os.getenv(name, "").strip().casefold() in _TRUTHY_VALUES
        for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )


def is_fixture_repository(repository_path: str | os.PathLike[str]) -> bool:
    """Return whether a repository is inside this project's test fixtures."""

    root = resolve_repository_root(repository_path)
    return _is_relative_to(root, _FIXTURES_ROOT)


def enforce_tracing_consent(
    repository_path: str | os.PathLike[str],
    *,
    allow_non_fixture_tracing: bool,
) -> None:
    """Refuse tracing repository data unless fixture-safe or explicitly allowed."""

    if (
        langsmith_tracing_enabled()
        and not allow_non_fixture_tracing
        and not is_fixture_repository(repository_path)
    ):
        raise PermissionError(
            "LangSmith tracing is enabled for a non-fixture repository. "
            "Disable tracing or pass --allow-non-fixture-tracing explicitly."
        )


__all__ = [
    "MAX_QUESTION_LENGTH",
    "MAX_TOP_K",
    "MIN_QUESTION_LENGTH",
    "MIN_TOP_K",
    "enforce_tracing_consent",
    "is_fixture_repository",
    "is_sensitive_file",
    "langsmith_tracing_enabled",
    "resolve_repository_file",
    "resolve_repository_root",
    "validate_question",
    "validate_top_k",
]
