"""In-memory tests for the repository assistant MCP server."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession

from repo_assistant.mcp_server import create_mcp_server, mcp, server


def test_mcp_server_exposes_exactly_two_tools() -> None:
    async def _test() -> None:
        async with InMemoryTransport(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tool_names = {t.name for t in tools_result.tools}

                assert tool_names == {"search_code", "explain_file"}

                search_tool = next(t for t in tools_result.tools if t.name == "search_code")
                assert "repo_path" in search_tool.input_schema.get("properties", {})
                assert "question" in search_tool.input_schema.get("properties", {})
                assert "k" in search_tool.input_schema.get("properties", {})

                explain_tool = next(t for t in tools_result.tools if t.name == "explain_file")
                assert "repo_path" in explain_tool.input_schema.get("properties", {})
                assert "file_path" in explain_tool.input_schema.get("properties", {})

    anyio.run(_test)


def test_search_code_returns_ranked_chunks_and_citations(tmp_path: Path) -> None:
    src_file = tmp_path / "src" / "worker.py"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text("def run_job():\n    return 'done'\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("# Project Docs\nWorker service\n", encoding="utf-8")

    fake_embedding = lambda texts: [[1.0, 0.0] for _ in texts]
    test_server = create_mcp_server(embedding_fn=fake_embedding)

    async def _test() -> None:
        async with InMemoryTransport(test_server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                res = await session.call_tool(
                    "search_code",
                    arguments={"repo_path": str(tmp_path), "question": "run_job", "k": 2},
                )
                assert not res.is_error
                data = json.loads(res.content[0].text)  # type: ignore[union-attr]

                assert "citations" in data
                assert "chunks" in data
                assert len(data["chunks"]) <= 2
                assert any("src/worker.py:1-2" in c for c in data["citations"])
                assert data["chunks"][0]["citation"] == "src/worker.py:1-2"
                assert "def run_job():" in data["chunks"][0]["text"]

    anyio.run(_test)


def test_search_code_with_invalid_repo_returns_error() -> None:
    async def _test() -> None:
        async with InMemoryTransport(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                res = await session.call_tool(
                    "search_code",
                    arguments={"repo_path": "path-that-does-not-exist", "question": "test"},
                )
                assert res.is_error

    anyio.run(_test)


def test_explain_file_returns_explanation_and_citations(tmp_path: Path) -> None:
    auth_file = tmp_path / "src" / "auth.py"
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_file.write_text(
        "def authenticate(user, token):\n    return token == 'valid'\n",
        encoding="utf-8",
    )

    def fake_model(prompt: str, chunks: object) -> str:
        return "Authenticates users via token check [src/auth.py:1-2]."

    test_server = create_mcp_server(model_fn=fake_model)

    async def _test() -> None:
        async with InMemoryTransport(test_server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                res = await session.call_tool(
                    "explain_file",
                    arguments={"repo_path": str(tmp_path), "file_path": "src/auth.py"},
                )
                assert not res.is_error
                data = json.loads(res.content[0].text)  # type: ignore[union-attr]

                assert data["file_path"] == "src/auth.py"
                assert "Authenticates users via token check" in data["explanation"]
                assert data["citations"] == ["src/auth.py:1-2"]

    anyio.run(_test)


def test_explain_file_nonexistent_or_unsupported(tmp_path: Path) -> None:
    async def _test() -> None:
        async with InMemoryTransport(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Nonexistent file
                res_missing = await session.call_tool(
                    "explain_file",
                    arguments={"repo_path": str(tmp_path), "file_path": "src/missing.py"},
                )
                assert res_missing.is_error

                # Unsupported extension
                unsupported_file = tmp_path / "data.bin"
                unsupported_file.write_bytes(b"\x00\x01\x02")
                res_unsupported = await session.call_tool(
                    "explain_file",
                    arguments={"repo_path": str(tmp_path), "file_path": "data.bin"},
                )
                assert res_unsupported.is_error

    anyio.run(_test)


def test_server_and_mcp_aliases() -> None:
    assert server is mcp
