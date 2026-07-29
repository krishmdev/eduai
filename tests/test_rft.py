import json

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


class TextJudge:
    """Prefers the option whose text is `favorite`, wherever it sits."""

    def __init__(self, favorite):
        self.favorite = favorite

    def choice_logprobs(self, messages, letters=("A", "B", "C", "D")):
        body = messages[-1]["content"]
        return {x: (0.0 if f"{x}. {self.favorite}" in body else -3.0) for x in letters}

    def choice_logprobs_batch(self, batch, letters=("A", "B", "C", "D")):
        return [self.choice_logprobs(m, letters) for m in batch]


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


def test_batched_judge_matches_the_eval_judge():
    items = [json.loads(_item()), json.loads(_item(answer="B"))]
    passages = [REQ.passage] * 2
    backend = TextJudge("mitochondrion")
    batched = rft.judge_batch(backend, items, passages, chunk=1)
    single = [Judge(backend)(it, p) for it, p in zip(items, passages, strict=True)]
    assert [b["agrees"] for b in batched] == [s["agrees"] for s in single] == [True, False]
    assert [round(b["p_key"], 6) for b in batched] == [round(s["p_key"], 6) for s in single]


def test_check_merge_select_and_build(tmp_path):
    prompts = {PROMPT["id"]: PROMPT}
    samples = [
        {"id": PROMPT["id"], "shots": 2, "idx": 0, "text": _item()},
        {"id": PROMPT["id"], "shots": 0, "idx": 0, "text": _item(stem="What organelle makes ATP?")},
        {
            "id": PROMPT["id"],
            "shots": 0,
            "idx": 1,
            "text": _item(stem="Where is ATP mostly produced in a cell?"),
        },
        {"id": PROMPT["id"], "shots": 0, "idx": 2, "text": "oops"},
    ]
    emb = HashingEmbedder(64)
    nov = NoveltyIndex(emb, ["unrelated stem about rocks"], ["sciq-train-00009"], [9], [])
    checked = rft.check(samples, prompts, AlignAll(), nov)
    assert [rft.needs_judge(r) for r in checked] == [True, False, True, False]
    assert checked[1]["checks"]["not_source_copy"] is False
    judged = [
        {"id": r["id"], "shots": r["shots"], "idx": r["idx"], "agrees": True, "p_key": p}
        for r, p in ((checked[0], 0.9), (checked[2], 0.6))
    ]
    merged = rft.merge(checked, judged)
    assert [r["first_failure"] for r in merged] == [None, "not_source_copy", None, "json"]
    chosen = rft.select(merged)
    assert len(chosen) == 1
    # the pick is the passing sample farthest from the source question
    assert chosen[0]["source_cos"] == min(r["source_cos"] for r in merged if r["passed"])
    assert rft.select(merged, min_p_key=0.95) == []
    stats = rft.build(chosen, prompts, tmp_path / "sft_v2", valid_frac=0.0)
    assert stats["train"] == 1 and stats["valid"] == 0
    row = read_jsonl(tmp_path / "sft_v2" / "train.jsonl")[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    target = json.loads(row["messages"][2]["content"])
    assert target["difficulty"] == "easy" and target["lo_id"] == "BIO.2.1.b"
    assert rft.summary(merged)["prompts_with_a_pass"] == 1


def test_sample_resumes_without_repeating_rows(tmp_path):
    class Fake:
        calls = 0

        def sample_batch(self, batch, max_tokens, temperature, completion_batch_size=16):
            Fake.calls += len(batch)
            stats = {"generation_tps": 1.0, "prompt_tps": 1.0, "peak_memory_gb": 1.0}
            return [_item() for _ in batch], stats

    out = tmp_path / "s.jsonl"
    p2 = {**PROMPT, "id": "train-00002"}
    rft.sample(Fake(), [PROMPT], out, n=2, log=lambda *_: None)
    rft.sample(Fake(), [PROMPT, p2], out, n=2, log=lambda *_: None)
    rows = read_jsonl(out)
    assert Fake.calls == 4 and len(rows) == 4
    assert {(r["id"], r["idx"]) for r in rows} == {
        (i, k) for i in (PROMPT["id"], "train-00002") for k in (0, 1)
    }
