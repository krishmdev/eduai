"""MLX backend: Llama 3.2 4-bit from the pinned snapshot, optionally with the LoRA adapter."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from eduai.llm.base import BackendUnavailable


class MLXBackend:
    def __init__(self, model_dir: Path, adapter_path: Path | None = None, seed: int = 0):
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as exc:  # non-macOS or the mlx extra is not installed
            raise BackendUnavailable(f"mlx-lm not importable: {exc}") from exc
        if not Path(model_dir).exists():
            raise BackendUnavailable(f"base model not found at {model_dir}")
        if adapter_path is not None and not (Path(adapter_path) / "adapters.safetensors").exists():
            raise BackendUnavailable(f"adapter not found at {adapter_path}")
        mx.random.seed(seed)
        self.model, self.tokenizer = load(
            str(model_dir), adapter_path=str(adapter_path) if adapter_path else None
        )
        self.adapter = adapter_path
        self.name = "mlx+adapter" if adapter_path else "mlx-base"
        self._lock = threading.Lock()
        self.last_stats: dict = {}

    def _prompt(self, messages: list[dict]) -> list[int]:
        out = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        # mlx-lm's TokenizerWrapper returns a list of ids; a raw HF tokenizer may return a mapping.
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=0.95 if temperature > 0 else 0.0)
        prompt = self._prompt(messages)
        text, n_gen, t0 = [], 0, time.perf_counter()
        last = None
        with self._lock:
            for resp in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
            ):
                text.append(resp.text)
                n_gen = resp.generation_tokens
                last = resp
        dt = time.perf_counter() - t0
        self.last_stats = {
            "prompt_tokens": len(prompt),
            "generation_tokens": n_gen,
            "seconds": dt,
            "generation_tps": getattr(last, "generation_tps", None),
            "prompt_tps": getattr(last, "prompt_tps", None),
            "peak_memory_gb": getattr(last, "peak_memory", None),
        }
        return "".join(text)

    def batch_generate(
        self, batch: list[list[dict]], max_tokens: int = 512, temperature: float = 0.0
    ) -> list[str]:
        return [self.generate(m, max_tokens, temperature) for m in batch]

    def choice_logprobs(
        self, messages: list[dict], letters: tuple[str, ...] = ("A", "B", "C", "D")
    ) -> dict[str, float]:
        """Log-probabilities of each answer letter as the next token (used by the solver)."""
        import mlx.core as mx

        prompt = self._prompt(messages)
        with self._lock:
            logits = self.model(mx.array(prompt)[None])[0, -1]
            logp = logits - mx.logsumexp(logits)
        out = {}
        for letter in letters:
            ids = [self.tokenizer.encode(v, add_special_tokens=False) for v in (letter, " " + letter)]
            out[letter] = max(float(logp[i[0]]) for i in ids if i)
        return out
