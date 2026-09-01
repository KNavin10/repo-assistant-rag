"""LangGraph workflow for the repository assistant."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    ModelFn,
    generate_answer,
    handle_failure,
    load_repository,
    retrieve_context,
    route_question,
    validate_repository,
)
from .state import AgentState


def _after_validation(state: dict[str, Any]) -> str:
    """Choose the success or failure edge after repository validation."""

    return "valid" if state.get("status") == "running" else "invalid"


def _after_routing(state: dict[str, Any]) -> str:
    """Choose the supported or unsupported-question edge."""

    return (
        "supported"
        if state.get("route") in {"repo_question", "repo_summary"}
        else "unsupported"
    )


def _after_retrieval(state: dict[str, Any]) -> str:
    """Choose the evidence or no-evidence edge after retrieval."""

    chunks = state.get("chunks")
    has_evidence = (
        state.get("status") == "running"
        and isinstance(chunks, list)
        and bool(chunks)
    )
    return "evidence" if has_evidence else "no_evidence"


def build_graph(model_fn: ModelFn | None = None):
    """Build and compile the repository-assistant workflow.

    ``model_fn`` is injected into the answer node. It receives only the user
    question and the retrieved repository chunks, allowing tests to use a fake
    model without contacting Groq or any other provider.
    """

    builder = StateGraph(AgentState)
    builder.add_node("validate_repository", validate_repository)
    builder.add_node("load_repository", load_repository)
    builder.add_node("route_question", route_question)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_answer", partial(generate_answer, model_fn=model_fn))
    builder.add_node("handle_failure", handle_failure)

    builder.add_edge(START, "validate_repository")
    builder.add_conditional_edges(
        "validate_repository",
        _after_validation,
        {"valid": "load_repository", "invalid": "handle_failure"},
    )
    builder.add_edge("load_repository", "route_question")
    builder.add_conditional_edges(
        "route_question",
        _after_routing,
        {"supported": "retrieve_context", "unsupported": "handle_failure"},
    )
    builder.add_conditional_edges(
        "retrieve_context",
        _after_retrieval,
        {"evidence": "generate_answer", "no_evidence": "handle_failure"},
    )
    builder.add_edge("generate_answer", END)
    builder.add_edge("handle_failure", END)

    return builder.compile(checkpointer=InMemorySaver())


# Descriptive aliases keep the graph factory convenient for callers using
# either ``create_graph`` or ``compile_graph`` terminology.
create_graph = build_graph
compile_graph = build_graph


__all__ = ["build_graph", "compile_graph", "create_graph"]
