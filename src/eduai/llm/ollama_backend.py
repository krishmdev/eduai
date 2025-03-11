"""Ollama over its REST API (base model only; Ollama can't load our MLX adapter)."""

from __future__ import annotations

import httpx

from eduai.llm.base import BackendUnavailable, SequentialBatchMixin


class OllamaBackend(SequentialBatchMixin):
    def __init__(
        self, url: str = "http://127.0.0.1:11434", model: str = "llama3.2:3b", timeout: float = 120.0
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.client = httpx.Client(timeout=timeout)
        try:
            tags = self.client.get(f"{self.url}/api/tags", timeout=2.0).json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailable(f"ollama not reachable at {self.url}: {exc}") from exc
        names = {m.get("name") for m in tags.get("models", [])}
        if model not in names:
            raise BackendUnavailable(f"ollama model {model} not pulled")

    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        r = self.client.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
