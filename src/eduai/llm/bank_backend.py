"""No-model backend. It never generates; the app serves validated items from the bank instead."""

from __future__ import annotations

from eduai.llm.base import BackendUnavailable


class BankBackend:
    name = "bank-only"

    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise BackendUnavailable("bank-only mode has no generator")

    def batch_generate(
        self, batch: list[list[dict]], max_tokens: int = 512, temperature: float = 0.0
    ) -> list[str]:
        raise BackendUnavailable("bank-only mode has no generator")


class FakeBackend:
    """Deterministic scripted backend for tests: returns queued outputs in order."""

    name = "fake"

    def __init__(self, outputs: list[str] | None = None, solver_answer: str | None = None):
        self.outputs = list(outputs or [])
        self.solver_answer = solver_answer
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        if self.solver_answer is not None and messages and "taking a science test" in messages[0]["content"]:
            return self.solver_answer
        return self.outputs.pop(0) if self.outputs else "{}"

    def batch_generate(
        self, batch: list[list[dict]], max_tokens: int = 512, temperature: float = 0.0
    ) -> list[str]:
        return [self.generate(m, max_tokens, temperature) for m in batch]
