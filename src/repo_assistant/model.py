"""Project B's independent Groq adapter and model prompt."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from .loader import RepositoryChunk

PromptModelFn = Callable[[str], str]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)


def build_prompt(question: str, chunks: Sequence[RepositoryChunk]) -> str:
    """Build a bounded prompt from only the question and retrieved chunks."""

    evidence = "\n\n".join(
        f"[{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}]\n"
        f"{chunk['text']}"
        for chunk in chunks
    )
    return (
        "You are a repository assistant. Answer the question using only the "
        "retrieved repository evidence below. Cite every factual claim with "
        "the exact format path:start_line-end_line. If the evidence does not "
        "answer the question, return exactly INSUFFICIENT_DATA.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved repository evidence:\n{evidence}"
    )


def _ask_groq(prompt: str) -> str:
    """Call Project B's Groq client lazily for the CLI demonstration."""

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"GROQ_API_KEY is not configured in {_ENV_PATH}."
        )

    model_name = os.getenv("GROQ_MODEL", "").strip()
    if not model_name:
        raise RuntimeError(f"GROQ_MODEL is not configured in {_ENV_PATH}.")

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Install the groq package before using the CLI model.") from exc

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content or ""


def ask_model(prompt: str, *, model_fn: PromptModelFn | None = None) -> str:
    """Return model text using an injected callable or Project B's Groq client.

    Tests should always pass ``model_fn``. The real Groq client is created only
    when no callable is injected, which is the path used by the CLI demo.
    """

    answer = model_fn(prompt) if model_fn is not None else _ask_groq(prompt)
    if not isinstance(answer, str):
        raise TypeError("model_fn must return a string")
    return answer


def groq_model(
    question: str,
    chunks: Sequence[RepositoryChunk],
    *,
    model_fn: PromptModelFn | None = None,
) -> str:
    """Adapt the graph model contract to the prompt-based model adapter."""

    return ask_model(build_prompt(question, chunks), model_fn=model_fn)


__all__ = ["PromptModelFn", "ask_model", "build_prompt", "groq_model"]
