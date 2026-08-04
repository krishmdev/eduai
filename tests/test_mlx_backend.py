"""The batched letter-logprob pass against the one-prompt path, on tiny random Llamas (needs mlx)."""

import threading

import pytest

mx = pytest.importorskip("mlx.core")
llama = pytest.importorskip("mlx_lm.models.llama")

from eduai.llm.mlx_backend import MLXBackend  # noqa: E402

LETTER_IDS = {"A": 11, " A": 12, "B": 13, " B": 14, "C": 15, " C": 16, "D": 17, " D": 18}


class Tok:
    """Messages carry their token ids directly; letters map to fixed ids."""

    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return list(messages[0]["ids"])

    def encode(self, text, add_special_tokens=False):
        return [LETTER_IDS[text]]


def _backend(tied: bool) -> MLXBackend:
    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=40,
        tie_word_embeddings=tied,
    )
    b = MLXBackend.__new__(MLXBackend)
    b.model, b.tokenizer, b._lock = llama.Model(args), Tok(), threading.Lock()
    mx.eval(b.model.parameters())
    return b


@pytest.mark.parametrize("tied", [True, False])
def test_batched_letter_logprobs_match_one_prompt_at_a_time(tied):
    b = _backend(tied)
    prompts = [[5], [3, 9, 21, 7, 1, 30, 2], [8, 8, 8], [4, 19, 33, 2, 6]]
    batch = [[{"ids": p}] for p in prompts]
    got = b.choice_logprobs_batch(batch)
    for g, m in zip(got, batch, strict=True):
        want = b.choice_logprobs(m)
        assert g.keys() == want.keys()
        assert all(abs(g[k] - want[k]) < 1e-4 for k in g), (g, want)
