"""Deterministic tests for repository loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from repo_assistant.loader import iter_supported_files, load_repository


def test_loader_finds_supported_files(tmp_path: Path) -> None:
    supported = {
        "src/main.py",
        "web/app.ts",
        "web/component.TSX",
        "web/client.js",
        "web/index.html",
        "web/styles.css",
        "docs/README.md",
        "config/settings.json",
        "config/service.yaml",
        "config/service.yml",
    }
    for relative_path in supported:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative_path}\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("unsupported", encoding="utf-8")

    chunks = load_repository(tmp_path)

    assert {chunk["file_path"] for chunk in chunks} == supported


def test_ignored_directories_are_not_loaded(tmp_path: Path) -> None:
    visible = tmp_path / "src" / "visible.py"
    visible.parent.mkdir()
    visible.write_text("visible = True\n", encoding="utf-8")
    for directory_name in (".GIT", ".venv", "node_modules", "dist", "build", ".next", "__pycache__"):
        hidden = tmp_path / directory_name / "hidden.py"
        hidden.parent.mkdir()
        hidden.write_text("hidden = True\n", encoding="utf-8")

    paths = [path.relative_to(tmp_path).as_posix() for path in iter_supported_files(tmp_path)]
    chunks = load_repository(tmp_path)

    assert paths == ["src/visible.py"]
    assert {chunk["file_path"] for chunk in chunks} == {"src/visible.py"}


def test_binary_and_oversized_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"not text\x00still not text")
    (tmp_path / "large.md").write_text("x" * 32, encoding="utf-8")
    (tmp_path / "kept.py").write_text("answer = 42\n", encoding="utf-8")

    chunks = load_repository(tmp_path, max_file_size=16)

    assert {chunk["file_path"] for chunk in chunks} == {"kept.py"}
    assert chunks[0]["text"] == "answer = 42"


def test_invalid_repository_path_raises() -> None:
    with pytest.raises(NotADirectoryError):
        load_repository("path-that-does-not-exist")
