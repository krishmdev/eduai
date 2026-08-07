import json
from collections import Counter

import pytest

from eduai.curriculum.embedder import HashingEmbedder
from eduai.data import rft
from eduai.data.sciq import read_jsonl
from eduai.evaluation.solver import Judge
from eduai.generation.dedup import NoveltyIndex
from eduai.prompts import GenerationRequest

REQ = GenerationRequest(
    "AP Biology",
    "Cells",
    "t",
    "BIO.2.1.b",
    "x",
    "easy",
    "standard",
    "Mitochondria make ATP. The nucleus stores DNA.",
    target_misconception="nucleus",
)
PROMPT = {
    "id": "train-00001",
    "request": REQ.__dict__,
    "group": 1,
    "reference": {"question": "What organelle makes ATP?", "answer": "mitochondria"},
}
GOOD = {
    "stem": "Which structure produces most cellular ATP?",
    "choices": {"A": "mitochondrion", "B": "nucleus", "C": "ribosome", "D": "vacuole"},
    "answer": "A",
    "explanation": "A is correct. Mitochondria make ATP.",
    "lo_id": "BIO.2.1.b",
    "difficulty": "hard",
}


class AlignAll:
    def is_aligned(self, texts, targets):
        return [{"aligned": True} for _ in targets]


class LetterJudge:
    """Prefers a fixed letter slot, so it disagrees with rotations; every rotation lands elsewhere."""

    def choice_logprobs(self, messages, letters=("A", "B", "C", "D")):
        return {x: -0.5 * i for i, x in enumerate(letters)}

    def choice_logprobs_batch(self, batch, letters=("A", "B", "C", "D")):
        return [self.choice_logprobs(m, letters) for m in batch]


class TextJudge:
    """Prefers the option whose text is `favorite`, wherever it sits."""

    def __init__(self, favorite):
        self.favorite = favorite

    def choice_logprobs(self, messages, letters=("A", "B", "C", "D")):
        body = messages[-1]["content"]
        return {x: (0.0 if f"{x}. {self.favorite}" in body else -3.0) for x in letters}

    def choice_logprobs_batch(self, batch, letters=("A", "B", "C", "D")):
        return [self.choice_logprobs(m, letters) for m in batch]


def _sample(idx, text, shots=0):
    return {"id": PROMPT["id"], "shots": shots, "idx": idx, "sha": rft.text_sha(text), "text": text}


def _item(**kw):
    d = json.loads(json.dumps(GOOD))
    d.update(kw)
    return json.dumps(d)


def test_cheap_checks_follow_the_eval_rules():
    assert rft.cheap_checks("no json", REQ) == (None, {"json": False})
    assert rft.cheap_checks(_item(answer="E"), REQ)[1] == {"json": True, "schema": False}
    item, checks = rft.cheap_checks(_item(), REQ)
    assert item and all(checks.values())
    # the requested misconception must be a wrong option
    _, checks = rft.cheap_checks(
        _item(choices={"A": "mitochondrion", "B": "golgi", "C": "ribosome", "D": "x"}), REQ
    )
    assert checks["structure"] is False
    _, checks = rft.cheap_checks(_item(choices={"A": "nucleus", "B": "nucleus", "C": "a", "D": "b"}), REQ)
    assert checks["distinct_key"] is False


@pytest.mark.parametrize("chunk", [1, 3, 4, 7])
@pytest.mark.parametrize("backend", [TextJudge("mitochondrion"), LetterJudge()])
def test_batched_judge_matches_the_eval_judge(chunk, backend):
    base = json.loads(_item())
    items = [
        base,
        json.loads(_item(answer="B")),
        {
            **base,
            "choices": {"A": "vacuole", "B": "nucleus", "C": "mitochondrion", "D": "ribosome"},
            "answer": "C",
        },
    ] * 3
    passages = [REQ.passage] * len(items)
    batched = rft.judge_batch(backend, items, passages, chunk=chunk)
    single = [Judge(backend)(it, p) for it, p in zip(items, passages, strict=True)]
    assert [b["agrees"] for b in batched] == [s["agrees"] for s in single]
    assert [round(b["p_key"], 6) for b in batched] == [round(s["p_key"], 6) for s in single]
    assert [b["majority"] for b in batched] == [s["majority"] for s in single]


def _checked():
    samples = [
        _sample(0, _item(), shots=2),
        _sample(0, _item(stem="What organelle makes ATP?")),
        _sample(1, _item(stem="Where is ATP mostly produced in a cell?")),
        _sample(2, "oops"),
        _sample(3, _item(stem="Which cell part stores most of the DNA in eukaryotes?")),
    ]
    emb = HashingEmbedder(64)
    # the last sample copies a train stem from another prompt of the same group
    nov = NoveltyIndex(
        emb,
        ["unrelated stem about rocks"],
        ["sciq-train-00009"],
        [9],
        ["Which cell part stores most of the DNA in eukaryotes?"],
    )
    return rft.check(samples, {PROMPT["id"]: PROMPT}, AlignAll(), nov)


def _verdicts(rows, p_keys):
    return [
        {**{k: r[k] for k in ("id", "shots", "idx", "sha")}, "agrees": True, "p_key": p}
        for r, p in zip(rows, p_keys, strict=True)
    ]


def test_check_merge_select_and_build(tmp_path):
    prompts = {PROMPT["id"]: PROMPT}
    checked = _checked()
    assert [rft.needs_judge(r) for r in checked] == [True, False, True, False, False]
    assert checked[1]["checks"]["not_source_copy"] is False
    assert checked[4]["checks"]["not_memorized"] is False and checked[4]["checks"]["novel"] is True
    merged = rft.merge(checked, _verdicts([checked[0], checked[2]], [0.9, 0.6]))
    assert [r["first_failure"] for r in merged] == [None, "not_source_copy", None, "json", "not_memorized"]
    chosen = rft.select(merged)
    assert len(chosen) == 1
    # the pick is the passing sample farthest from the source question
    assert chosen[0]["source_cos"] == min(r["source_cos"] for r in merged if r["passed"])
    assert rft.select(merged, min_p_key=0.95) == []
    stats = rft.build(chosen, prompts, tmp_path / "sft_v2", valid_frac=0.0, balance=False)
    assert stats["train"] == 1 and stats["valid"] == 0 and stats["dropped_leak_same_answer_qa"] == 0
    assert stats["mix"] == {"standard/misconception": 1}
    row = read_jsonl(tmp_path / "sft_v2" / "train.jsonl")[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    target = json.loads(row["messages"][2]["content"])
    assert target["difficulty"] == "easy" and target["lo_id"] == "BIO.2.1.b"
    summ = rft.summary(merged, prompts)
    assert summ["prompts_with_a_pass"] == 1
    assert summ["prompts_with_a_pass_by_mix"] == {"standard/misconception": 1}
    # a leak screen drops targets, and too small a holdout is refused
    stats = rft.build(chosen, prompts, tmp_path / "x", valid_frac=0.0, leaks=lambda ts: [True] * len(ts))
    assert stats["train"] == 0 and stats["dropped_leak_same_answer_qa"] == 1
    with pytest.raises(ValueError, match="holdout"):
        rft.build(chosen, prompts, tmp_path / "y", valid_frac=0.01)


def test_select_breaks_source_cos_ties_on_judge_confidence():
    rows = [
        {"id": "p", "passed": True, "source_cos": 0.5, "p_key": 0.6, "idx": 0},
        {"id": "p", "passed": True, "source_cos": 0.5, "p_key": 0.8, "idx": 1},
        {"id": "p", "passed": True, "source_cos": 0.7, "p_key": 0.99, "idx": 2},
        {"id": "p", "passed": False, "source_cos": 0.1, "p_key": 0.99, "idx": 3},
    ]
    assert [r["idx"] for r in rft.select(rows)] == [1]


def test_merge_refuses_verdicts_for_older_texts():
    checked = _checked()
    old = _verdicts([checked[0]], [0.9])
    old[0]["sha"] = "0" * 16
    with pytest.raises(ValueError, match="older sample texts"):
        rft.merge(checked, old)


def test_move_key_swaps_options_and_explanation_letters():
    item = json.loads(_item(explanation="A is correct. Option B is the nucleus (B), not a site of ATP."))
    moved = rft.move_key(item, "B")
    assert moved["answer"] == "B" and moved["choices"]["B"] == "mitochondrion"
    assert moved["choices"]["A"] == "nucleus"
    assert moved["explanation"] == "B is correct. Option A is the nucleus (A), not a site of ATP."
    # a bare article "A" is not a letter reference
    item = json.loads(_item(explanation="A is correct. A mitochondrion makes ATP."))
    assert rft.move_key(item, "C")["explanation"] == "C is correct. A mitochondrion makes ATP."
    item = json.loads(_item(explanation="The correct answer is A. B is incorrect."))
    assert rft.move_key(item, "B")["explanation"] == "The correct answer is B. A is incorrect."
    item = json.loads(_item(explanation="Option A is correct. Choice B is incorrect. (A)"))
    assert rft.move_key(item, "B")["explanation"] == "Option B is correct. Choice A is incorrect. (B)"
    item = json.loads(_item(explanation="The answer is a mitochondrion."))
    assert rft.move_key(item, "B")["explanation"] == "The answer is a mitochondrion."
    assert rft.move_key(item, "A") == item


def test_balanced_letters_are_exact():
    got = rft.balanced_letters([f"train-{i:05d}" for i in range(10)])
    assert sorted(Counter(got.values()).values()) == [2, 2, 3, 3]


class FakeSampler:
    calls = 0

    def sample_batch(self, batch, max_tokens, temperature, completion_batch_size=16):
        FakeSampler.calls += len(batch)
        stats = {"generation_tps": 1.0, "prompt_tps": 1.0, "peak_memory_gb": 1.0}
        return [
            _item(stem=f"Which structure number {FakeSampler.calls + k} makes ATP?")
            for k in range(len(batch))
        ], stats


PARAMS = {"model": "llama-3b", "adapter": None, "seed": 1}


def test_sample_resumes_without_repeating_rows(tmp_path):
    FakeSampler.calls = 0
    quiet = {"log": lambda *_: None, "params": PARAMS}
    out = tmp_path / "s.jsonl"
    p2 = {**PROMPT, "id": "train-00002"}
    rft.sample(FakeSampler(), [PROMPT], out, n=2, **quiet)
    rft.sample(FakeSampler(), [PROMPT, p2], out, n=2, **quiet)
    rows = read_jsonl(out)
    assert FakeSampler.calls == 4 and len(rows) == 4
    assert {(r["id"], r["idx"]) for r in rows} == {
        (i, k) for i in (PROMPT["id"], "train-00002") for k in (0, 1)
    }
    assert all(r["sha"] == rft.text_sha(r["text"]) and r["model"] == "llama-3b" for r in rows)
    # 2-shot samples are their own slots, not resumed 0-shot ones
    shots = [({"x": 1}, {"y": 2}), ({"x": 3}, {"y": 4})]
    msgs = []
    import eduai.data.rft as mod

    real = mod.build_messages
    mod.build_messages = lambda req, shots=None: msgs.append(shots) or real(req)
    try:
        rft.sample(FakeSampler(), [PROMPT], out, shots, n=2, **quiet)
    finally:
        mod.build_messages = real
    rows = read_jsonl(out)
    assert len(rows) == 6 and sum(r["shots"] == 2 for r in rows) == 2 and msgs == [shots, shots]


def test_sample_refuses_other_settings_and_cuts_a_partial_line(tmp_path):
    out = tmp_path / "s.jsonl"
    quiet = {"log": lambda *_: None}
    rft.sample(FakeSampler(), [PROMPT], out, n=1, params=PARAMS, **quiet)
    for params, temp in (
        ({**PARAMS, "adapter": "adapters/x"}, 0.8),
        (PARAMS, 1.0),
        ({**PARAMS, "seed": 2}, 0.8),
    ):
        with pytest.raises(ValueError, match="sampled with"):
            rft.sample(FakeSampler(), [PROMPT], out, n=2, temperature=temp, params=params, **quiet)
    with out.open("a") as fh:
        fh.write('{"id": "train-00001", "shots": 0, "idx": 1, "te')
    assert len(rft.read_rows(out)) == 1
    assert out.read_text().endswith("\n")
    rft.sample(FakeSampler(), [PROMPT], out, n=2, params=PARAMS, **quiet)
    assert [r["idx"] for r in read_jsonl(out)] == [0, 1]
