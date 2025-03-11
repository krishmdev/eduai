"""Backend auto-selection: MLX + adapter -> MLX base -> Ollama -> bank-only."""

from __future__ import annotations

import logging

from eduai.config import Settings, model_path
from eduai.llm.bank_backend import BankBackend
from eduai.llm.base import BackendUnavailable, LLMBackend

log = logging.getLogger(__name__)
ORDER = ("mlx+adapter", "mlx-base", "ollama", "bank")


def _try(kind: str, s: Settings) -> LLMBackend:
    if kind == "mlx+adapter":
        from eduai.llm.mlx_backend import MLXBackend

        return MLXBackend(model_path(s.base_model), s.adapter_path)
    if kind == "mlx-base":
        from eduai.llm.mlx_backend import MLXBackend

        return MLXBackend(model_path(s.base_model), None)
    if kind == "ollama":
        from eduai.llm.ollama_backend import OllamaBackend

        return OllamaBackend(s.ollama_url, s.ollama_model)
    return BankBackend()


def select_backend(s: Settings) -> tuple[LLMBackend, list[str]]:
    """Returns the backend and the reasons earlier candidates were skipped."""
    wanted = {"auto": ORDER, "mlx": ("mlx+adapter", "mlx-base"), "ollama": ("ollama",), "bank": ("bank",)}
    order = wanted.get(s.backend)
    if order is None:
        raise ValueError(f"unknown backend {s.backend!r}")
    skipped = []
    for kind in order:
        try:
            backend = _try(kind, s)
            log.info("using backend %s", backend.name)
            return backend, skipped
        except BackendUnavailable as exc:
            skipped.append(f"{kind}: {exc}")
    if s.backend != "auto":
        raise BackendUnavailable("; ".join(skipped))
    return BankBackend(), skipped
