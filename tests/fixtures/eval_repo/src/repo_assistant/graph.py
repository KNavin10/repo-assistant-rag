"""LangGraph workflow for the repository assistant."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


def build_graph():
    """Build and compile the graph with in-memory checkpoints."""

    builder = StateGraph(dict)
    builder.add_edge(START, END)
    return builder.compile(checkpointer=InMemorySaver())
