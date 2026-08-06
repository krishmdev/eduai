"""MLX backend: Llama 3.2 4-bit from the pinned snapshot, optionally with the LoRA adapter."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from eduai.llm.base import BackendUnavailable


def shared_prefix_len(prompts: list[list[int]]) -> int:
    """Length of the token prefix every prompt shares, leaving each prompt at least one token."""
    first, n = prompts[0], 0
    limit = min(len(p) for p in prompts) - 1
    while n < limit and all(p[n] == first[n] for p in prompts):
        n += 1
    return n


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

    def sample_batch(
        self,
        batch: list[list[dict]],
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        completion_batch_size: int = 12,
        cache_limit_gb: float = 2.0,
    ) -> tuple[list[str], dict]:
        """Sampled completions for many prompts at once (continuous batching). Used for training data,
        never for the eval, which decodes one request at a time.

        The token prefix all prompts share (system prompt and any fixed examples) is prefilled once and
        its KV cache reused, which halves the time for 2-shot prompts. MLX's buffer cache is capped,
        since left alone it held 10 GB on top of a 6 GB peak here and ran the 16 GB machine out of
        memory.
        """
        import mlx.core as mx
        from mlx_lm import batch_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        prompts = [self._prompt(m) for m in batch]
        n = shared_prefix_len(prompts)
        mx.set_cache_limit(int(cache_limit_gb * 2**30))
        with self._lock:
            caches = None
            if n >= 32:
                cache = make_prompt_cache(self.model)
                self.model(mx.array(prompts[0][:n])[None], cache=cache)
                mx.eval([c.state for c in cache])
                # batch_generate copies prompt caches into its batch cache, so one can be shared
                caches, prompts = [cache] * len(prompts), [p[n:] for p in prompts]
            resp = batch_generate(
                self.model,
                self.tokenizer,
                prompts,
                prompt_caches=caches,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature, top_p=top_p),
                completion_batch_size=completion_batch_size,
                prefill_batch_size=min(4, completion_batch_size),
            )
            mx.clear_cache()
        st = resp.stats
        stats = {
            "shared_prefix_tokens": n if caches else 0,
            "prompt_tokens": st.prompt_tokens,
            "prompt_tps": st.prompt_tps,
            "generation_tokens": st.generation_tokens,
            "generation_tps": st.generation_tps,
            "peak_memory_gb": st.peak_memory,
        }
        return list(resp.texts), stats

    def _letter_ids(self, letters: tuple[str, ...]) -> dict[str, list[int]]:
        return {
            letter: [
                i[0]
                for i in (self.tokenizer.encode(v, add_special_tokens=False) for v in (letter, " " + letter))
                if i
            ]
            for letter in letters
        }

    def _last_logits(self, x, rows, cols):
        """Logits at one position per row. Llama-style models are projected only at those positions,
        since full (batch, length, 128k) logits for a 16-prompt judge batch take several GB."""
        inner, args = getattr(self.model, "model", None), getattr(self.model, "args", None)
        if inner is None or args is None:
            return self.model(x)[rows, cols]
        h = inner(x)[rows, cols]
        if getattr(args, "tie_word_embeddings", False):
            return inner.embed_tokens.as_linear(h)
        return self.model.lm_head(h)

    def choice_logprobs_batch(
        self, batch: list[list[dict]], letters: tuple[str, ...] = ("A", "B", "C", "D")
    ) -> list[dict[str, float]]:
        """choice_logprobs for several prompts in one right-padded forward pass. Causal attention means
        the padding never affects the last real position. Results differ from the one-prompt path by
        half-precision rounding (up to about 0.03 nats on the 3B), so the eval judge keeps using
        choice_logprobs and this is only used to screen training data."""
        import mlx.core as mx

        prompts = [self._prompt(m) for m in batch]
        width = max(len(p) for p in prompts)
        pad = self.tokenizer.eos_token_id
        x = mx.array([p + [pad] * (width - len(p)) for p in prompts])
        rows, cols = mx.arange(len(prompts)), mx.array([len(p) - 1 for p in prompts])
        with self._lock:
            last = self._last_logits(x, rows, cols)
            logp = last - mx.logsumexp(last, axis=-1, keepdims=True)
            mx.eval(logp)
        ids = self._letter_ids(letters)
        return [{k: max(float(row[i]) for i in v) for k, v in ids.items()} for row in logp]

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
