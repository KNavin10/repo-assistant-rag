"""Model Context Protocol (MCP) server for the repository assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

try:
    from .embeddings import EmbeddingFunction, SentenceTransformerAdapter
    from .loader import SUPPORTED_EXTENSIONS, load_repository
    from .nodes import ModelFn, _extract_citations
    from .retrieval import hybrid_search
except (ImportError, ValueError):
    from repo_assistant.embeddings import EmbeddingFunction, SentenceTransformerAdapter
    from repo_assistant.loader import SUPPORTED_EXTENSIONS, load_repository
    from repo_assistant.nodes import ModelFn, _extract_citations
    from repo_assistant.retrieval import hybrid_search


def create_mcp_server(
    *,
    model_fn: ModelFn | None = None,
    embedding_fn: EmbeddingFunction | None = None,
    name: str = "repo-assistant",
) -> MCPServer:
    """Create and configure the repository assistant MCP server."""

    server = MCPServer(name=name)

    @server.tool()
    def search_code(repo_path: str, question: str, k: int = 4) -> dict[str, Any]:
        """Search repository code using hybrid retrieval and return ranked chunks with citations.

        Args:
            repo_path: Absolute or relative path to the repository directory.
            question: Search query or technical question.
            k: Maximum number of ranked chunks to return (default: 4).
        """
        root = Path(repo_path)
        if not root.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

        chunks = load_repository(root)
        if not chunks:
            return {"citations": [], "chunks": []}

        embedder = (
            embedding_fn
            if embedding_fn is not None
            else SentenceTransformerAdapter()
        )
        matches = hybrid_search(
            question,
            chunks,
            top_k=k,
            embedding_fn=embedder,
        )

        citations = [
            f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
            for chunk in matches
        ]
        return {
            "citations": citations,
            "chunks": [
                {
                    "file_path": chunk["file_path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "citation": f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}",
                    "text": chunk["text"],
                }
                for chunk in matches
            ],
        }

    @server.tool()
    def explain_file(repo_path: str, file_path: str) -> dict[str, Any]:
        """Explain one supported repository file and return explanation with citations.

        Args:
            repo_path: Absolute or relative path to the repository directory.
            file_path: Relative path to the file within the repository.
        """
        root = Path(repo_path)
        if not root.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

        target_path = (root / file_path).resolve()
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if target_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{target_path.suffix}'. Supported extensions: "
                f"{sorted(SUPPORTED_EXTENSIONS)}"
            )

        all_chunks = load_repository(root)
        normalized_target = Path(file_path).as_posix()
        file_chunks = [c for c in all_chunks if c["file_path"] == normalized_target]
        if not file_chunks:
            raise ValueError(f"No indexable content found in {file_path}")

        prompt = f"Explain the implementation, purpose, and key components of {normalized_target}."
        if model_fn is not None:
            explanation = model_fn(prompt, file_chunks)
        else:
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if api_key:
                from .model import groq_model

                explanation = groq_model(prompt, file_chunks)
            else:
                chunk_summaries = ", ".join(
                    f"{c['start_line']}-{c['end_line']}" for c in file_chunks
                )
                explanation = (
                    f"File '{normalized_target}' consists of {len(file_chunks)} chunk(s) "
                    f"spanning lines {chunk_summaries}. Detailed implementation is cited below "
                    f"[{normalized_target}:{file_chunks[0]['start_line']}-{file_chunks[-1]['end_line']}]."
                )

        extracted = _extract_citations(explanation)
        chunk_citations = [
            f"{c['file_path']}:{c['start_line']}-{c['end_line']}" for c in file_chunks
        ]
        citations = extracted if extracted else chunk_citations

        return {
            "file_path": normalized_target,
            "explanation": explanation,
            "citations": citations,
        }

    return server


server = create_mcp_server()
mcp = server
app = server


def main() -> None:
    """Run the MCP server on stdio transport."""
    server.run("stdio")


if __name__ == "__main__":
    main()


__all__ = [
    "create_mcp_server",
    "main",
    "mcp",
    "server",
]
