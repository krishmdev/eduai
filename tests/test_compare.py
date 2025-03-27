import json

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
