"""Provider-free tests for Project B's model adapter."""

from __future__ import annotations

from repo_assistant.model import ask_model, build_prompt, groq_model


def test_injected_fake_model_is_used_instead_of_groq(monkeypatch) -> None:
    def groq_must_not_run(prompt: str) -> str:
        raise AssertionError("Groq must not run for an injected fake model")

    monkeypatch.setattr("repo_assistant.model._ask_groq", groq_must_not_run)
    calls: list[str] = []

    def fake_model(prompt: str) -> str:
        calls.append(prompt)
        return "fake answer"

    assert ask_model("test prompt", model_fn=fake_model) == "fake answer"
    assert calls == ["test prompt"]


def test_ask_model_uses_injected_fake() -> None:
    calls: list[str] = []

    def fake_model(prompt: str) -> str:
        calls.append(prompt)
        return "fake answer"

    assert ask_model("test prompt", model_fn=fake_model) == "fake answer"
    assert calls == ["test prompt"]


def test_groq_model_passes_only_question_and_chunks_to_prompt() -> None:
    chunks = [
        {
            "file_path": "backend/approvals.py",
            "start_line": 1,
            "end_line": 2,
            "text": "approval is required before sensitive tools execute",
        }
    ]
    prompts: list[str] = []

    def fake_model(prompt: str) -> str:
        prompts.append(prompt)
        return "Sensitive tools are paused. [backend/approvals.py:1-2]"

    answer = groq_model(
        "Where is approval enforced?",
        chunks,
        model_fn=fake_model,
    )

    assert answer == "Sensitive tools are paused. [backend/approvals.py:1-2]"
    assert prompts == [build_prompt("Where is approval enforced?", chunks)]
    assert "repo_path" not in prompts[0]
    assert "route" not in prompts[0]
