"""Typed state shared by the repository-assistant graph."""

from __future__ import annotations

from typing import Literal, TypedDict

from .loader import RepositoryChunk

Route = Literal["repo_question", "repo_summary", "unsupported"]
Status = Literal["running", "completed", "insufficient_data", "error"]


class AgentState(TypedDict):
    """State carried between repository-assistant graph nodes."""

    repo_path: str
    question: str
    route: Route
    chunks: list[RepositoryChunk]
    answer: str
    citations: list[str]
    status: Status
    errors: list[str]


# ``GraphState`` is a descriptive alias for callers that prefer graph
# terminology while keeping one canonical state definition.
GraphState = AgentState


__all__ = ["AgentState", "GraphState", "Route", "Status"]
