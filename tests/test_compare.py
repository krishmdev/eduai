import json
import random

import pytest

from eduai.curriculum.embedder import HashingEmbedder
from eduai.data.sciq import write_jsonl
from eduai.evaluation import compare, report
from eduai.generation.dedup import NoveltyIndex


class AlignAll:
    def is_aligned(self, texts, targets):
        return [{"aligned": True, "target_in_top3": True, "target_score": 1.0, "top1": t} for t in targets]


def _setup(tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "eval"
    prompts = []
    tagged = []
    for i, (q, a) in enumerate(
        [("What organelle makes ATP?", "mitochondria"), ("What gas do plants absorb?", "carbon dioxide")]
    ):
        pid = f"test-{i:05d}"
        prompts.append(
            {
                "id": pid,
                "group": i,
                "reference": {"question": q, "answer": a},
                "request": {
                    "subject": "AP Biology",
                    "unit": "Cells",
                    "topic": "t",
                    "lo_id": "BIO.2.1.b",
                    "lo_text": "x",
                    "difficulty": "easy",
                    "format": "standard",
                    "passage": "Mitochondria make ATP. Plants absorb carbon dioxide.",
                    "target_misconception": None,
                    "avoid": [],
                },
            }
        )
        tagged.append({"id": pid, "question": q, "correct": a, "distractors": ["nucleus", "oxygen", "water"]})
    write_jsonl(data / "items_tagged.jsonl", tagged)
    good = {
        "stem": "Which structure produces most cellular ATP?",
        "choices": {"A": "mitochondrion", "B": "nucleus", "C": "ribosome", "D": "vacuole"},
        "answer": "A",
        "explanation": "A is correct. Mitochondria make ATP.",
        "lo_id": "BIO.2.1.b",
        "difficulty": "easy",
    }
    for arm in compare.ARMS:
        gens = [
            {
                "id": prompts[0]["id"],
                "text": json.dumps(good),
                "seconds": 1.0,
                "generation_tps": 30.0,
                "first_parsed": True,
                "peak_memory_gb": 2.0,
            },
            {
                "id": prompts[1]["id"],
                "text": "not json",
                "seconds": 1.0,
                "generation_tps": 30.0,
                "first_parsed": False,
                "peak_memory_gb": 2.0,
            },
        ]
        write_jsonl(out / f"gen_{arm}.jsonl", gens)
    judged = [{"id": p["id"], "arm": "reference", "agrees": True, "majority": "A"} for p in prompts]
    judged += [
        {"id": prompts[0]["id"], "arm": arm, "agrees": arm == "finetuned", "majority": "A"}
        for arm in compare.ARMS
    ]
    write_jsonl(out / "judge_llama-1b.jsonl", judged)
    return prompts, data, out


def test_score_counts_all_prompts_and_renders(tmp_path):
    prompts, data, out = _setup(tmp_path)
    emb = HashingEmbedder(64)
    # The bank contains the source item itself; it must be excluded, not counted as a duplicate.
    nov = NoveltyIndex(
        emb,
        ["Which structure produces most cellular ATP?", "unrelated stem about rocks"],
        ["sciq-test-00000", "sciq-train-00001"],
        [0, 99],
        [],
    )
    res = compare.score(prompts, AlignAll(), nov, data_dir=data, out_dir=out)
    s = res["summary"]
    assert s["finetuned"]["json"] == 0.5 and s["finetuned"]["all_checks"] == 0.5
    assert s["base-0shot"]["key"] == 0.0 and s["base-0shot"]["all_checks"] == 0.0
    assert s["finetuned"]["novel"] == 0.5 and s["finetuned"]["source_copy"] == 0.0
    assert s["reference"]["aligned"] == 1.0
    assert res["rejections"]["finetuned"] == {"accepted": 1, "no_json": 1}
    card = {
        "off_curriculum": 0,
        "leakage_filter": {"screened": 2, "dropped_passage_containment": 0, "dropped_same_answer_qa": 0},
    }
    md = report.render(res, "m.json", card)
    assert "Base 3B + EduAI LoRA" in md and "95% CI" in md


def test_arms_present_orders_named_arms_then_sweep_arms(tmp_path):
    for arm in ("sweep-b", "finetuned", "base-0shot", "sweep-a"):
        write_jsonl(tmp_path / f"gen_{arm}.jsonl", [])
    assert compare.arms_present(tmp_path) == ["base-0shot", "finetuned", "sweep-a", "sweep-b"]


def test_v1_bootstrap_intervals_do_not_move_when_v2_arms_are_added():
    rng = random.Random(3)

    def rows():
        return [{"usable": rng.random() < 0.3, "key": rng.random() < 0.6} for _ in range(40)]

    v1 = {arm: rows() for arm in ("reference", "base-0shot", "base-2shot", "finetuned")}
    both = {**v1, "finetuned-v2": rows(), "finetuned-v2-2shot": rows()}
    old, new = compare.bootstrap_diffs(v1, n_boot=200), compare.bootstrap_diffs(both, n_boot=200)
    assert old and all(new[k] == v for k, v in old.items())
    assert "finetuned-v2 - base-2shot | usable" in new and "finetuned-v2 - finetuned | key" in new


def test_generate_rejects_an_unknown_arm_without_an_adapter():
    with pytest.raises(ValueError, match="sweep arm"):
        compare.generate_arm("no-such-arm", [])


def test_judge_only_named_arms_keeps_other_rows(tmp_path, monkeypatch):
    prompts, data, out = _setup(tmp_path)
    before = {
        r["arm"]: r for r in compare.read_jsonl(out / "judge_llama-1b.jsonl") if r["arm"] != "reference"
    }

    class FakeBackend:
        def __init__(self, *a, **k):
            pass

        def choice_logprobs(self, messages, letters=("A", "B", "C", "D")):
            body = messages[-1]["content"]
            return {x: (0.0 if f"{x}. nucleus" in body else -5.0) for x in letters}

    import eduai.llm.mlx_backend as mb

    monkeypatch.setattr(mb, "MLXBackend", FakeBackend)
    monkeypatch.setattr(compare, "model_path", lambda key: tmp_path)
    compare.judge_all(prompts, "llama-1b", out, data, arms=["finetuned-v2"])
    rows = compare.read_jsonl(out / "judge_llama-1b.jsonl")
    assert [r["arm"] for r in rows][:2] == ["reference", "reference"]
    after = {r["arm"]: r for r in rows if r["arm"] != "reference"}
    assert set(after) == set(before)
    for arm in before:
        if arm != "finetuned-v2":
            assert after[arm] == before[arm]
    # The fake judge always picks the nucleus option (B) wherever it is rotated to; the key is A.
    assert after["finetuned-v2"]["agrees"] is False and "p_key" in after["finetuned-v2"]


def test_report_labels_versions_and_valid_split_intro(tmp_path):
    prompts, data, out = _setup(tmp_path)
    write_jsonl(out / "gen_sweep-lr1e4.jsonl", compare.read_jsonl(out / "gen_finetuned.jsonl"))
    nov = NoveltyIndex(HashingEmbedder(64), ["unrelated stem about rocks"], ["sciq-train-00001"], [99], [])
    res = compare.score(prompts, AlignAll(), nov, data_dir=data, out_dir=out)
    card = {
        "split": "valid",
        "leakage_filter": {"screened": 2, "dropped_passage_containment": 0, "dropped_same_answer_qa": 0},
    }
    md = report.render(res, "m.json", card)
    assert md.startswith("# Model-selection eval on the valid split")
    assert "| Base 3B + EduAI LoRA v1 |" in md and "| Base 3B + EduAI LoRA v2, 2-shot |" in md
    assert "| sweep-lr1e4 |" in md
    assert "sweep-lr1e4 - base-2shot | usable" in res["bootstrap"]
    # no 3B judge rows, so no all-n/a secondary table; the speed note doesn't quote the test-run load
    assert "3B judge" not in md and "6 running containers" not in md
