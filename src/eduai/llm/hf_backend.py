"""transformers + PEFT backend for CUDA/CPU machines (the Colab reference path).

Not exercised on this Mac; the MLX backend is the tested path.
"""

from __future__ import annotations

from eduai.llm.base import BackendUnavailable, SequentialBatchMixin


class HFBackend(SequentialBatchMixin):
    def __init__(self, base: str = "unsloth/Llama-3.2-3B-Instruct", adapter: str | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendUnavailable(str(exc)) from exc
        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise BackendUnavailable("peft is required for adapters") from exc
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(device).eval()
        self.device = device
        self.name = f"hf:{base}{'+adapter' if adapter else ''}"

    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=temperature or None
            )
        return self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
