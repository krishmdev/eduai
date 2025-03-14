"""Turn a GenerationRequest into model output, with one repair retry for unparseable JSON."""

from __future__ import annotations

import time
from dataclasses import dataclass

from eduai.generation.validate import parse
from eduai.llm.base import LLMBackend
from eduai.prompts import GenerationRequest, build_messages

MAX_TOKENS = 480


@dataclass
class Generation:
    text: str
    seconds: float
    tokens: int | None
    tps: float | None
    retried: bool = False
    first_parsed: bool = True


class Generator:
    def __init__(
        self,
        backend: LLMBackend,
        shots: list[tuple[GenerationRequest, dict]] | None = None,
        temperature: float = 0.0,
        retry: bool = True,
    ):
        self.backend = backend
        self.shots = shots or []
        self.temperature = temperature
        self.retry = retry

    def _call(self, messages: list[dict], temperature: float) -> Generation:
        t0 = time.perf_counter()
        text = self.backend.generate(messages, MAX_TOKENS, temperature)
        stats = getattr(self.backend, "last_stats", {}) or {}
        return Generation(
            text, time.perf_counter() - t0, stats.get("generation_tokens"), stats.get("generation_tps")
        )

    def generate(self, req: GenerationRequest) -> Generation:
        messages = build_messages(req, shots=self.shots)
        gen = self._call(messages, self.temperature)
        gen.first_parsed = parse(gen.text)[0] is not None
        if self.retry and parse(gen.text)[0] is None:
            second = self._call(messages, max(self.temperature, 0.3))
            second.retried = True
            second.first_parsed = False
            second.seconds += gen.seconds
            return second
        return gen
