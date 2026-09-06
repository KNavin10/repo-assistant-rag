"""Repository-assistant workflow nodes."""


INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def route_question(question):
    """Classify repository questions before retrieval."""

    return "repo_question" if "repository" in question.casefold() else "unsupported"


def retrieve_context(question, chunks):
    """Select evidence for a routed repository question."""

    return chunks
