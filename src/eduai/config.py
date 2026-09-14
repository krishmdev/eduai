from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelPin:
    key: str
    repo: str
    revision: str
    allow: tuple[str, ...]
    optional: bool = False
    # Passed to the tokenizer's apply_chat_template (e.g. to switch a model's thinking mode off).
    template_kwargs: tuple[tuple[str, object], ...] = ()


# Revisions resolved against the Hugging Face API on 2026-09-23. `make models` downloads exactly
# these commits into .models/ and checks every file against models.lock.
EMBED_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
)
MODEL_PINS: dict[str, ModelPin] = {
    "bge-small": ModelPin(
        "bge-small", "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", EMBED_FILES
    ),
    "minilm": ModelPin(
        "minilm",
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        EMBED_FILES,
    ),
    "rerank": ModelPin(
        "rerank",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        (
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
    ),
    "llama-3b": ModelPin(
        "llama-3b",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e",
        (
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        optional=True,
    ),
    "llama-1b": ModelPin(
        "llama-1b",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "08231374eeacb049a0eade7922910865b8fce912",
        (
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        optional=True,
    ),
    # Answer-key verifier (docs/key_verification.md). Not a generator or eval judge of any arm. The
    # snapshot is shared with ../localhost-ai; link its .models/hub entry into .models/hub.
    "qwen3.5-9b": ModelPin(
        "qwen3.5-9b",
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        "938d8919941c6e7efd3c7150eff7fe9d12afa631",
        (
            "chat_template.jinja",
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        optional=True,
        template_kwargs=(("enable_thinking", False),),
    ),
}

SCIQ_DATASET = ("allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUAI_", env_file=None)

    backend: str = "auto"  # auto | mlx (base, 2-shot) | adapter | ollama | bank
    base_model: str = "llama-3b"
    adapter_path: Path = ROOT / "adapters" / "llama32-3b-eduai"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "data" / "eduai.db"
    bank_path: Path | None = None
    egress_canary: bool = False
    # Bank promotion (`eduai bank add-generated`) also requires the answer-key verifier's pass
    # (docs/key_verification.md). On by default for new promotions.
    require_key_verification: bool = True
    key_verifier: str = "qwen3.5-9b"


def models_dir() -> Path:
    return Path(os.environ.get("EDUAI_MODELS_DIR", ROOT / ".models"))


def model_path(key: str) -> Path:
    pin = MODEL_PINS[key]
    org, name = pin.repo.split("/")
    return models_dir() / "hub" / f"models--{org}--{name}" / "snapshots" / pin.revision


def get_settings() -> Settings:
    return Settings()
