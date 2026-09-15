import json

from eduai.generation.validate import extract_json, validate_item
from eduai.prompts import GenerationRequest

REQ = GenerationRequest(
    subject="AP Biology",
    unit="Cells",
    topic="Cell structures",
    lo_id="BIO.2.1.b",
    lo_text="Describe organelles.",
    difficulty="easy",
    format="standard",
    passage="Mitochondria make ATP.",
    target_misconception="ribosome",
)


def item(**kw):
    base = {
        "stem": "Which organelle makes most of a cell's ATP?",
        "choices": {"A": "mitochondrion", "B": "ribosome", "C": "nucleus", "D": "vacuole"},
        "answer": "A",
        "explanation": "A is correct. Mitochondria make ATP.",
        "lo_id": "BIO.2.1.b",
        "difficulty": "easy",
    }
    base.update(kw)
    return base


def test_extract_json_ignores_braces_in_strings_and_prose():
    text = 'Sure! {"stem": "a {weird} stem", "x": {"y": 1}} trailing {"no": 2}'
    assert json.loads(extract_json(text))["x"] == {"y": 1}


def test_valid_item_passes():
    assert validate_item(json.dumps(item()), REQ).ok


def test_rejection_reasons_in_order():
    assert validate_item("no json here", REQ).reason == "no_json"
    assert validate_item("{bad json}", REQ).reason == "bad_json"
    assert validate_item(json.dumps(item(answer="E")), REQ).reason == "schema"
    dup = item(choices={"A": "mitochondrion", "B": "mitochondrions", "C": "nucleus", "D": "ribosome"})
    assert validate_item(json.dumps(dup), REQ).reason == "structure"
    aota = item(choices={"A": "mitochondrion", "B": "ribosome", "C": "nucleus", "D": "all of the above"})
    assert validate_item(json.dumps(aota), REQ).reason == "structure"
    giveaway = item(stem="The mitochondrion is which organelle that makes ATP?")
    assert validate_item(json.dumps(giveaway), REQ).reason == "structure"
    no_mis = item(choices={"A": "mitochondrion", "B": "lysosome", "C": "nucleus", "D": "vacuole"})
    assert validate_item(json.dumps(no_mis), REQ).reason == "misconception_missing"
    wrong_lo = item(lo_id="BIO.1.1.a")
    assert validate_item(json.dumps(wrong_lo), REQ).reason == "structure"


def test_solver_aligner_novelty_hooks():
    disagree = lambda it: {"agrees": False, "majority": "B"}  # noqa: E731
    assert validate_item(json.dumps(item()), REQ, solver=disagree).reason == "key_disagreement"
    agree = lambda it: {"agrees": True, "majority": "A"}  # noqa: E731
    off = lambda it, req: (False, {"target_score": -3.0})  # noqa: E731
    assert validate_item(json.dumps(item()), REQ, solver=agree, aligner=off).reason == "not_aligned"
    on = lambda it, req: (True, {})  # noqa: E731
    dup = lambda it: (False, {"max_bank_cos": 0.97})  # noqa: E731
    res = validate_item(json.dumps(item()), REQ, solver=agree, aligner=on, novelty=dup)
    assert res.reason == "duplicate" and res.checks == {
        "json": True,
        "schema": True,
        "structure": True,
        "misconception": True,
        "key": True,
        "aligned": True,
        "novel": False,
    }


def test_verifier_hook_runs_last_and_rejects_unverified_keys():
    agree = lambda it: {"agrees": True, "majority": "A"}  # noqa: E731
    no = lambda it: {"verified": False, "majority": "A", "p_key": 0.4}  # noqa: E731
    res = validate_item(json.dumps(item()), REQ, solver=agree, verifier=no)
    assert res.reason == "key_unverified" and res.checks["verified"] is False
    assert res.metrics["verifier_p_key"] == 0.4
    yes = lambda it: {"verified": True, "majority": "A", "p_key": 0.9}  # noqa: E731
    assert validate_item(json.dumps(item()), REQ, solver=agree, verifier=yes).ok
    called = []
    dup = lambda it: (False, {"max_bank_cos": 0.97})  # noqa: E731
    validate_item(json.dumps(item()), REQ, novelty=dup, verifier=lambda it: called.append(1))
    assert not called
