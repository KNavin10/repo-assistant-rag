"""Load supported repository files into small, line-aware text chunks.

The loader deliberately stops at plain text.  Embeddings and retrieval belong
in later stages of the application.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

SUPPORTED_EXTENSIONS = frozenset(
    {".py", ".ts", ".tsx", ".js", ".html", ".css", ".md", ".json", ".yaml", ".yml"}
)
IGNORED_DIRECTORIES = frozenset(
    {".git", ".venv", "node_modules", "dist", "build", ".next", "__pycache__"}
)

DEFAULT_MAX_FILE_SIZE = 1_048_576  # 1 MiB
DEFAULT_CHUNK_SIZE = 1_200  # characters


class RepositoryChunk(TypedDict):
    """A searchable piece of a repository file.

    ``file_path`` is relative to the repository root and uses POSIX-style
    separators so the result is stable across operating systems.
    ``start_line`` and ``end_line`` are one-based and inclusive.
    """

    file_path: str
    start_line: int
    end_line: int
    text: str


def iter_supported_files(repository_path: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield supported files below ``repository_path`` in stable order.

    Ignored directory names are pruned before walking, so files below them
    are never opened.
    """

    root = Path(repository_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    for current_dir, directory_names, file_names in os.walk(root, topdown=True):
        directory_names[:] = sorted(
            name for name in directory_names if name.lower() not in IGNORED_DIRECTORIES
        )
        for file_name in sorted(file_names):
            path = Path(current_dir) / file_name
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def _read_text(path: Path, max_file_size: int) -> str | None:
    """Read a small UTF-8 text file, returning ``None`` for skipped files."""

    try:
        if path.stat().st_size > max_file_size:
            return None

        raw = path.read_bytes()
    except OSError:
        return None

    # NUL bytes are a reliable and inexpensive binary-file signal for this
    # loader.  Strict UTF-8 decoding catches other non-text content.
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _split_into_chunks(text: str, chunk_size: int) -> Iterator[tuple[int, int, str]]:
    """Split text into chunks of at most ``chunk_size`` characters.

    Chunks are built on line boundaries whenever possible.  A single long
    line is split into smaller pieces while retaining that line's number.
    """

    lines = text.splitlines()
    if not lines:
        return

    chunk_lines: list[str] = []
    chunk_start = 1
    chunk_length = 0

    def flush() -> tuple[int, int, str] | None:
        nonlocal chunk_lines, chunk_start, chunk_length
        if not chunk_lines:
            return None
        result = (chunk_start, chunk_start + len(chunk_lines) - 1, "\n".join(chunk_lines))
        chunk_lines = []
        chunk_length = 0
        return result

    for line_number, line in enumerate(lines, start=1):
        if len(line) > chunk_size:
            pending = flush()
            if pending is not None:
                yield pending
            for offset in range(0, len(line), chunk_size):
                yield line_number, line_number, line[offset : offset + chunk_size]
            chunk_start = line_number + 1
            continue

        added_length = len(line) if not chunk_lines else len(line) + 1
        if chunk_lines and chunk_length + added_length > chunk_size:
            pending = flush()
            if pending is not None:
                yield pending
            chunk_start = line_number
            added_length = len(line)

        if not chunk_lines:
            chunk_start = line_number
        chunk_lines.append(line)
        chunk_length += added_length

    pending = flush()
    if pending is not None:
        yield pending


def load_repository(
    repository_path: str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> list[RepositoryChunk]:
    """Load a repository directory, such as ``policies/my-repository``.

    Unsupported extensions, ignored directories, binary files, unreadable
    files, and files larger than ``max_file_size`` bytes are skipped.  The
    returned dictionaries contain only file location and text metadata; no
    embeddings are created here.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if max_file_size < 0:
        raise ValueError("max_file_size must not be negative")

    root = Path(repository_path)
    chunks: list[RepositoryChunk] = []
    for path in iter_supported_files(root):
        text = _read_text(path, max_file_size)
        if text is None:
            continue

        relative_path = path.relative_to(root).as_posix()
        for start_line, end_line, chunk_text in _split_into_chunks(text, chunk_size):
            chunks.append(
                {
                    "file_path": relative_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": chunk_text,
                }
            )
    return chunks


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_FILE_SIZE",
    "IGNORED_DIRECTORIES",
    "SUPPORTED_EXTENSIONS",
    "RepositoryChunk",
    "iter_supported_files",
    "load_repository",
]
