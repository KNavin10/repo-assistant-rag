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
    assert "answer = 42" in chunks[0]["text"]
    assert chunks[0]["text"].startswith("# File: kept.py")
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 1
    assert set(chunks[0].keys()) == {"file_path", "start_line", "end_line", "text"}


def test_sensitive_files_are_excluded(tmp_path: Path) -> None:
    visible = tmp_path / "config" / "settings.json"
    visible.parent.mkdir(parents=True)
    visible.write_text('{"environment": "test"}\n', encoding="utf-8")

    sensitive_files = {
        ".env": "GROQ_API_KEY=should-not-load\n",
        ".env.local": "TOKEN=should-not-load\n",
        ".envrc": "TOKEN=should-not-load\n",
        "credentials.json": '{"password": "should-not-load"}\n',
        "service-secret.yaml": "api_key: should-not-load\n",
        "service-account.yml": "private_key: should-not-load\n",
        "server.key": "private key material\n",
        "server.pem": "certificate material\n",
        "server.crt": "certificate material\n",
    }
    for relative_path, contents in sensitive_files.items():
        (tmp_path / relative_path).write_text(contents, encoding="utf-8")

    chunks = load_repository(tmp_path)

    assert {chunk["file_path"] for chunk in chunks} == {"config/settings.json"}


def test_root_gitignore_patterns_are_applied(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "ignored.json\nprivate/*.yaml\nignored-directory/\n",
        encoding="utf-8",
    )
    (tmp_path / "visible.py").write_text("visible = True\n", encoding="utf-8")
    (tmp_path / "ignored.json").write_text('{"hidden": true}\n', encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    (private / "settings.yaml").write_text("hidden: true\n", encoding="utf-8")
    ignored_directory = tmp_path / "ignored-directory"
    ignored_directory.mkdir()
    (ignored_directory / "hidden.py").write_text("hidden = True\n", encoding="utf-8")

    paths = [path.relative_to(tmp_path).as_posix() for path in iter_supported_files(tmp_path)]

    assert paths == ["visible.py"]


def test_scan_file_limit_is_an_explicit_error(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text(f"value = '{name}'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="maximum of 2 files"):
        load_repository(tmp_path, max_files=2)


def test_scan_total_byte_limit_is_an_explicit_error(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("aaaa\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("bbbb\n", encoding="utf-8")

    with pytest.raises(ValueError, match="maximum of 7 total bytes"):
        load_repository(tmp_path, max_total_bytes=7)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("max_files", 0, "max_files must be a positive integer"),
        ("max_total_bytes", 0, "max_total_bytes must be a positive integer"),
    ],
)
def test_scan_limits_require_positive_integers(
    tmp_path: Path, argument: str, value: int, message: str
) -> None:
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_repository(tmp_path, **{argument: value})


def test_invalid_repository_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_repository("path-that-does-not-exist")


def test_function_is_never_cut_in_half(tmp_path: Path) -> None:
    lines = ["def huge_function():"]
    for i in range(100):
        lines.append(f"    variable_{i} = {i} * 42  # filler line to exceed size")
    func_text = "\n".join(lines) + "\n"

    (tmp_path / "huge.py").write_text(func_text, encoding="utf-8")

    chunks = load_repository(tmp_path, chunk_size=300)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["file_path"] == "huge.py"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == len(lines)
    assert "def huge_function():" in chunk["text"]
    assert "variable_99" in chunk["text"]


def test_async_function_is_never_cut_in_half(tmp_path: Path) -> None:
    lines = ["async def async_worker():"]
    for i in range(80):
        lines.append(f"    await step_{i}()  # filler statement")
    func_text = "\n".join(lines) + "\n"

    (tmp_path / "async_mod.py").write_text(func_text, encoding="utf-8")

    chunks = load_repository(tmp_path, chunk_size=300)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["file_path"] == "async_mod.py"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == len(lines)
    assert "async def async_worker():" in chunk["text"]


def test_class_returned_as_one_meaningful_unit_where_size_permits(tmp_path: Path) -> None:
    small_class = (
        "class SmallService:\n"
        '    """A small service."""\n'
        "    timeout = 30\n"
        "\n"
        "    def run(self):\n"
        "        return True\n"
    )
    (tmp_path / "service.py").write_text(small_class, encoding="utf-8")

    chunks = load_repository(tmp_path, chunk_size=1200)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["file_path"] == "service.py"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 6
    assert "class SmallService:" in chunk["text"]
    assert "def run(self):" in chunk["text"]


def test_class_exceeding_size_is_split_on_method_boundaries(tmp_path: Path) -> None:
    class_code = (
        "class LargeService:\n"
        '    """Preamble description."""\n'
        "    version = 1\n"
        "\n"
        "    def method_alpha(self):\n"
        "        return 'alpha'\n"
        "\n"
        "    def method_beta(self):\n"
        "        return 'beta'\n"
    )
    (tmp_path / "large_service.py").write_text(class_code, encoding="utf-8")

    chunks = load_repository(tmp_path, chunk_size=70)
    assert len(chunks) >= 2
    method_alpha_chunk = next(c for c in chunks if "method_alpha" in c["text"])
    method_beta_chunk = next(c for c in chunks if "method_beta" in c["text"])
    assert method_alpha_chunk["start_line"] == 5
    assert method_alpha_chunk["end_line"] == 6
    assert method_beta_chunk["start_line"] == 8
    assert method_beta_chunk["end_line"] == 9


def test_syntax_invalid_python_uses_fallback(tmp_path: Path) -> None:
    invalid_code = (
        "def broken_syntax(:\n"
        "    x = [1, 2,\n"
        "    return x\n"
    )
    (tmp_path / "invalid.py").write_text(invalid_code, encoding="utf-8")

    chunks = load_repository(tmp_path, chunk_size=40)
    assert len(chunks) > 0
    # Fallback line splitter does not prepend the "# File:" header
    assert not any(chunk["text"].startswith("# File:") for chunk in chunks)
    assert any("broken_syntax" in chunk["text"] for chunk in chunks)


def test_chunk_header_includes_path_and_relevant_imports(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "import sys\n"
        "from math import sqrt, pi\n"
        "\n"
        "def calculate_hypotenuse(a, b):\n"
        "    return sqrt(a * a + b * b)\n"
    )
    (tmp_path / "calc.py").write_text(code, encoding="utf-8")

    chunks = load_repository(tmp_path)
    func_chunk = next(c for c in chunks if "calculate_hypotenuse" in c["text"])

    assert "# File: calc.py" in func_chunk["text"]
    assert "from math import sqrt" in func_chunk["text"]
    assert "import os" not in func_chunk["text"]
    assert "import sys" not in func_chunk["text"]
    assert "from math import pi" not in func_chunk["text"]
    assert func_chunk["start_line"] == 5
    assert func_chunk["end_line"] == 6
