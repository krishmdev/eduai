import ast
import json
from pathlib import Path

from eduai import prompts
from eduai.prompts import GenerationRequest, build_messages, render_completion

SRC = Path(__file__).resolve().parents[1] / "src" / "eduai"


def _req(**kw):
    base = dict(
        subject="AP Biology",
        unit="Cells",
        topic="Cell structures",
        lo_id="BIO.2.1.b",
        lo_text="Describe organelles.",
        difficulty="easy",
        format="standard",
        passage="Mitochondria make ATP.",
    )
    base.update(kw)
    return GenerationRequest(**base)


def test_training_and_inference_messages_match():
    req = _req(format="stimulus", target_misconception="ribosome")
    item = {
        "stem": "Based on the excerpt, what makes ATP?",
        "stimulus": "Mitochondria make ATP.",
        "choices": {"A": "mitochondria", "B": "ribosome", "C": "nucleus", "D": "vacuole"},
        "answer": "A",
        "explanation": "A is correct.",
        "lo_id": "BIO.2.1.b",
        "difficulty": "easy",
    }
    train = build_messages(req, item)
    infer = build_messages(req)
    assert train[:-1] == infer
    assert json.loads(train[-1]["content"]) == item
    assert list(json.loads(render_completion(item))) == [
        "stem",
        "stimulus",
        "choices",
        "answer",
        "explanation",
        "lo_id",
        "difficulty",
    ]


def test_sft_builder_uses_shared_prompt():
    sample = Path(__file__).resolve().parents[1] / "data/samples/sft_sample.jsonl"
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["messages"][0]["content"] == prompts.SYSTEM_PROMPT


def test_no_other_module_defines_a_generation_prompt():
    """Only prompts.py may build system/user prompt strings for generation."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "prompts.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                vals = [v.value for v in node.values if isinstance(v, ast.Constant)]
                if "role" in keys and ("system" in vals or "user" in vals):
                    offenders.append(str(path.relative_to(SRC)))
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "multiple-choice science" in node.value
            ):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == []


def test_request_validation():
    import pytest

    with pytest.raises(ValueError):
        _req(difficulty="impossible")
    with pytest.raises(ValueError):
        _req(format="essay")


def test_parse_user_round_trips_sft_rows():
    from eduai.prompts import parse_user, render_user

    sample = Path(__file__).resolve().parents[1] / "data/samples/sft_sample.jsonl"
    for line in sample.read_text().splitlines():
        user = json.loads(line)["messages"][1]["content"]
        assert render_user(parse_user(user)) == user
