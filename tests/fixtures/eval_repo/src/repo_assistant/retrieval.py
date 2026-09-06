"""Deterministic keyword retrieval."""


def normalize_words(text):
    """Return normalized, unique words from text."""

    return set(text.casefold().split())


def keyword_score(question_words, chunk_text):
    """Count distinct question terms present in a chunk."""

    return len(question_words & normalize_words(chunk_text))


def retrieve(question, chunks, top_k=4):
    """Rank chunks by overlap; tied scores preserve input order."""

    return chunks[:top_k]
