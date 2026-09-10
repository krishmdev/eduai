import hashlib
import json

import pytest

from eduai.data.sciq import read_jsonl, write_jsonl
from eduai.evaluation import key_verify as kv
from eduai.evaluation.solver import Judge


class OptionBackend:
    """Letter log-probabilities from the option texts: 'right' scores `hi`, anything else 0."""

    def __init__(self, hi: float = 5.0):
        self.hi, self.calls = hi, 0

    def choice_logprobs(self, messages):
        self.calls += 1
        body = messages[-1]["content"]
        out = {}
        for line in body.splitlines():
            if len(line) > 2 and line[0] in "ABCD" and line[1] == ".":
                out[line[0]] = self.hi if "right" in line else 0.0
        return out


def item(key_text: str = "right", answer: str = "A", **choices) -> dict:
    ch = {"A": key_text, "B": "wrong one", "C": "wrong two", "D": "wrong three", **choices}
    return {
        "stem": "Which option is the right one here?",
        "choices": ch,
        "answer": answer,
        "explanation": "A is correct because it is right.",
        "lo_id": "BIO.2.1.b",
        "difficulty": "easy",
    }


def test_passes_needs_the_key_on_top_and_p_above_threshold():
    assert kv.passes({"agrees": True, "p_key": 0.51})
    assert not kv.passes({"agrees": True, "p_key": 0.5})
    assert not kv.passes({"agrees": False, "p_key": 0.9})
    assert not kv.passes({"agrees": False, "p_key": None, "duplicate_options": True})


def test_verifier_outcomes():
    v = kv.KeyVerifier(Judge(OptionBackend()), "fake")
    assert v(item())["verified"]
    # key on top but not confident: 'right' only 0.5 nats above three others -> p < 0.5
    weak = kv.KeyVerifier(Judge(OptionBackend(hi=0.5)), "fake")(item())
    assert weak["agrees"] and weak["p_key"] < 0.5 and not weak["verified"]
    # wrong key
    assert not v(item(answer="B"))["verified"]
    # key text duplicated as a distractor
    assert not v(item(B="right"))["verified"]


def _split(tmp_path, arms=("base-2shot", "finetuned-v2")):
    d = tmp_path / "eval"
    d.mkdir()
    prompts = [{"id": f"p{i}", "request": {"passage": "The right answer is right."}} for i in range(3)]
    per_item = []
    for arm in arms:
        gens = [
            {"id": "p0", "arm": arm, "text": json.dumps(item())},
            {"id": "p1", "arm": arm, "text": json.dumps(item(answer="B"))},
            {"id": "p2", "arm": arm, "text": "not json"},
        ]
        write_jsonl(d / f"gen_{arm}.jsonl", gens)
        per_item += [
            {"arm": arm, "id": "p0", "schema": True, "usable": True},
            {"arm": arm, "id": "p1", "schema": True, "usable": True},
            {"arm": arm, "id": "p2", "schema": False, "usable": False},
        ]
    write_jsonl(d / "per_item.jsonl", per_item)
    return d, prompts


def _hashes(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}


def test_verify_split_writes_its_own_file_and_leaves_eval_files_alone(tmp_path):
    d, prompts = _split(tmp_path)
    before = _hashes(d)
    out = kv.verify_split(kv.KeyVerifier(Judge(OptionBackend()), "fake"), prompts, d, ("base-2shot",))
    assert out == d / "verify_fake.jsonl"
    after = _hashes(d)
    assert {k: after[k] for k in before} == before and set(after) - set(before) == {"verify_fake.jsonl"}
    rows = read_jsonl(out)
    assert [(r["arm"], r["id"], r["verified"]) for r in rows] == [
        ("base-2shot", "p0", True),
        ("base-2shot", "p1", False),
    ]
    with pytest.raises(FileExistsError):
        kv.verify_split(kv.KeyVerifier(Judge(OptionBackend()), "fake"), prompts, d, ("base-2shot",))
    # another arm is added and the first arm's rows are kept
    kv.verify_split(kv.KeyVerifier(Judge(OptionBackend()), "fake"), prompts, d, ("finetuned-v2",))
    assert {r["arm"] for r in read_jsonl(out)} == {"base-2shot", "finetuned-v2"}


def test_verify_split_resumes_from_partial(tmp_path):
    d, prompts = _split(tmp_path)
    partial = d / "verify_fake.partial.jsonl"
    row = {"id": "p0", "arm": "base-2shot", "key": "A", "majority": "A", "p_key": 0.9, "verified": True}
    partial.write_text(json.dumps({**row, "duplicate_options": False}) + "\n")
    backend = OptionBackend()
    kv.verify_split(kv.KeyVerifier(Judge(backend), "fake"), prompts, d, ("base-2shot",))
    assert backend.calls == 4  # only p1 was scored, over 4 rotations
    assert not partial.exists()


def test_split_summary_and_bootstrap(tmp_path):
    d, prompts = _split(tmp_path)
    kv.verify_split(
        kv.KeyVerifier(Judge(OptionBackend()), "fake"), prompts, d, ("base-2shot", "finetuned-v2")
    )
    s = kv.split_summary(read_jsonl(d / "per_item.jsonl"), read_jsonl(d / "verify_fake.jsonl"))
    a = s["arms"]["finetuned-v2"]
    assert (a["n"], a["usable"], a["verified_usable"], a["survival"]) == (3, 2, 1, 0.5)
    assert s["arms"]["finetuned-v2-2shot"]["n"] == 0
    assert s["bootstrap_verified_usable"]["finetuned-v2 - base-2shot | verified_usable"]["diff"] == 0.0
    with pytest.raises(ValueError, match="no verification row"):
        kv.split_summary(read_jsonl(d / "per_item.jsonl"), [])


def test_audit_crosscheck_splits_key_verdicts_by_verification(tmp_path):
    audit = tmp_path / "audit"
    (audit / "llm_gemma").mkdir(parents=True)
    key = {"a0": {"arm": "x", "id": "p0"}, "a1": {"arm": "x", "id": "p1"}, "a2": {"arm": "y", "id": "p0"}}
    (audit / "key.json").write_text(json.dumps(key))
    write_jsonl(
        audit / "llm_gemma" / "verdicts.jsonl",
        [
            {"audit_id": "a0", "parsed": True, "key_correct": True},
            {"audit_id": "a1", "parsed": True, "key_correct": False},
            {"audit_id": "a2", "parsed": False, "key_correct": None},
        ],
    )
    verify = [
        {"arm": "x", "id": "p0", "verified": True},
        {"arm": "x", "id": "p1", "verified": False},
        {"arm": "y", "id": "p0", "verified": True},
    ]
    c = kv.audit_crosscheck(verify, audit, "gemma")
    assert c["n_judged"] == 2 and c["n_unparsed"] == 1
    assert c["pass"]["rate"] == 1.0 and c["fail"]["rate"] == 0.0 and c["diff"] == 1.0
    assert 0 < c["fisher_p"] <= 1
    assert kv.audit_crosscheck(verify, audit, "missing") is None


def test_report_is_labelled_post_hoc(tmp_path):
    d, prompts = _split(tmp_path, arms=("base-2shot",))
    kv.verify_split(kv.KeyVerifier(Judge(OptionBackend()), "qwen3.5-9b"), prompts, d, ("base-2shot",))
    res = kv.build_report("qwen3.5-9b", splits={"valid": (d, None)}, audit_dir=tmp_path / "none")
    assert res["post_hoc"] and res["exploratory"] and res["threshold"] == 0.5
    md = kv.render(res)
    assert md.startswith("# Post-hoc: answer-key verification (exploratory)")
    assert "Verified usable" in md and "| Base 3B, 2-shot | 3 |" in md


def test_wilson_interval():
    assert kv.wilson(0, 0) is None
    lo, hi = kv.wilson(5, 10)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9
