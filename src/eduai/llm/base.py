from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str: ...

    def batch_generate(
        self, batch: list[list[dict]], max_tokens: int = 512, temperature: float = 0.0
    ) -> list[str]: ...


class BackendUnavailable(RuntimeError):
    pass


class SequentialBatchMixin:
    def batch_generate(
        self, batch: list[list[dict]], max_tokens: int = 512, temperature: float = 0.0
    ) -> list[str]:
        return [self.generate(m, max_tokens, temperature) for m in batch]  # type: ignore[attr-defined]
