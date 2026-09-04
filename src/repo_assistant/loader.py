"""Load supported repository files into small, line-aware text chunks.

The loader deliberately stops at plain text.  Embeddings and retrieval belong
in later stages of the application.
"""

from __future__ import annotations

import ast
import os
import re
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


def _node_start_line(node: ast.AST) -> int:
    """Return 1-based start line of an AST node, including decorators."""

    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.decorator_list
    ):
        return min([node.lineno] + [d.lineno for d in node.decorator_list])
    return getattr(node, "lineno", 1)


def _node_end_line(node: ast.AST) -> int:
    """Return 1-based end line of an AST node."""

    return getattr(node, "end_lineno", _node_start_line(node))


def _extract_file_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Collect top-level import statements from a parsed module."""

    return [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]


def _get_relevant_imports(
    imports: list[ast.Import | ast.ImportFrom],
    code: str,
) -> list[str]:
    """Return formatted import statements for symbols referenced in ``code``."""

    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", code))
    relevant: list[str] = []

    for node in imports:
        if isinstance(node, ast.Import):
            matching = [
                alias
                for alias in node.names
                if (alias.asname or alias.name.split(".")[0]) in identifiers
                or alias.name in identifiers
            ]
            if matching:
                stmt = ast.unparse(ast.Import(names=matching))
                if stmt not in code:
                    relevant.append(stmt)
        elif isinstance(node, ast.ImportFrom):
            matching = [
                alias
                for alias in node.names
                if (alias.asname or alias.name) in identifiers
                or alias.name == "*"
            ]
            if matching:
                stmt = ast.unparse(
                    ast.ImportFrom(
                        module=node.module,
                        names=matching,
                        level=node.level,
                    )
                )
                if stmt not in code:
                    relevant.append(stmt)

    return relevant


def _format_chunk_text(file_path: str, code: str, relevant_imports: list[str]) -> str:
    """Prepend a short header with file path and relevant imports to code."""

    header_lines = [f"# File: {file_path}"]
    if relevant_imports:
        header_lines.extend(relevant_imports)
    return "\n".join(header_lines) + "\n\n" + code


def _split_python_file(
    file_path: str,
    text: str,
    chunk_size: int,
) -> list[RepositoryChunk]:
    """Split Python source text into code-aware chunks using AST boundaries."""

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        # Fallback to fixed-size line splitter for syntax-invalid Python
        return [
            {
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "text": chunk_text,
            }
            for start, end, chunk_text in _split_into_chunks(text, chunk_size)
        ]

    source_lines = text.splitlines()
    if not source_lines:
        return []
    if not tree.body:
        if text.strip():
            return [
                {
                    "file_path": file_path,
                    "start_line": start,
                    "end_line": end,
                    "text": chunk_text,
                }
                for start, end, chunk_text in _split_into_chunks(text, chunk_size)
            ]
        return []

    file_imports = _extract_file_imports(tree)
    chunks: list[RepositoryChunk] = []
    current_module_nodes: list[ast.AST] = []

    def flush_module_nodes() -> None:
        nonlocal current_module_nodes
        if not current_module_nodes:
            return
        start = _node_start_line(current_module_nodes[0])
        end = _node_end_line(current_module_nodes[-1])
        code = "\n".join(source_lines[start - 1 : end])
        current_module_nodes = []
        if not code.strip():
            return

        if len(code) <= chunk_size:
            rel = _get_relevant_imports(file_imports, code)
            chunks.append(
                {
                    "file_path": file_path,
                    "start_line": start,
                    "end_line": end,
                    "text": _format_chunk_text(file_path, code, rel),
                }
            )
        else:
            for s, e, piece in _split_into_chunks(code, chunk_size):
                rel = _get_relevant_imports(file_imports, piece)
                chunks.append(
                    {
                        "file_path": file_path,
                        "start_line": start + s - 1,
                        "end_line": start + e - 1,
                        "text": _format_chunk_text(file_path, piece, rel),
                    }
                )

    def process_class(node: ast.ClassDef) -> None:
        start = _node_start_line(node)
        end = _node_end_line(node)
        class_code = "\n".join(source_lines[start - 1 : end])

        # A class is returned as one meaningful unit where size permits
        if len(class_code) <= chunk_size:
            rel = _get_relevant_imports(file_imports, class_code)
            chunks.append(
                {
                    "file_path": file_path,
                    "start_line": start,
                    "end_line": end,
                    "text": _format_chunk_text(file_path, class_code, rel),
                }
            )
            return

        # Where size does not permit, split on method/child boundaries
        child_nodes = [
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not child_nodes:
            for s, e, piece in _split_into_chunks(class_code, chunk_size):
                rel = _get_relevant_imports(file_imports, piece)
                chunks.append(
                    {
                        "file_path": file_path,
                        "start_line": start + s - 1,
                        "end_line": start + e - 1,
                        "text": _format_chunk_text(file_path, piece, rel),
                    }
                )
            return

        first_child_start = _node_start_line(child_nodes[0])
        if first_child_start > start:
            preamble = "\n".join(source_lines[start - 1 : first_child_start - 1])
            if preamble.strip():
                if len(preamble) <= chunk_size:
                    rel = _get_relevant_imports(file_imports, preamble)
                    chunks.append(
                        {
                            "file_path": file_path,
                            "start_line": start,
                            "end_line": first_child_start - 1,
                            "text": _format_chunk_text(file_path, preamble, rel),
                        }
                    )
                else:
                    for s, e, piece in _split_into_chunks(preamble, chunk_size):
                        rel = _get_relevant_imports(file_imports, piece)
                        chunks.append(
                            {
                                "file_path": file_path,
                                "start_line": start + s - 1,
                                "end_line": start + e - 1,
                                "text": _format_chunk_text(file_path, piece, rel),
                            }
                        )

        for child in child_nodes:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c_start = _node_start_line(child)
                c_end = _node_end_line(child)
                c_code = "\n".join(source_lines[c_start - 1 : c_end])
                rel = _get_relevant_imports(file_imports, c_code)
                # A function is never cut in half
                chunks.append(
                    {
                        "file_path": file_path,
                        "start_line": c_start,
                        "end_line": c_end,
                        "text": _format_chunk_text(file_path, c_code, rel),
                    }
                )
            elif isinstance(child, ast.ClassDef):
                process_class(child)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_module_nodes()
            start = _node_start_line(node)
            end = _node_end_line(node)
            func_code = "\n".join(source_lines[start - 1 : end])
            rel = _get_relevant_imports(file_imports, func_code)
            # A function is never cut in half
            chunks.append(
                {
                    "file_path": file_path,
                    "start_line": start,
                    "end_line": end,
                    "text": _format_chunk_text(file_path, func_code, rel),
                }
            )
        elif isinstance(node, ast.ClassDef):
            flush_module_nodes()
            process_class(node)
        else:
            current_module_nodes.append(node)

    flush_module_nodes()
    return chunks


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
        if path.suffix.lower() == ".py":
            chunks.extend(_split_python_file(relative_path, text, chunk_size))
        else:
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

