"""The blind audit must not turn missing or malformed verdicts into positive scores."""

import json

import pytest

from scripts.blind_audit import draw, score


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_audit_rejects_bad_verdicts_and_sheet_changes(tmp_path):
    gen = tmp_path / "gen"
    gen.mkdir()
    item = {
        "stem": "What produces ATP?",
        "choices": {"A": "mitochondria", "B": "nucleus", "C": "ribosome", "D": "vacuole"},
        "answer": "A",
        "explanation": "Mitochondria produce ATP.",
        "lo_id": "BIO.2.1.b",
        "difficulty": "easy",
    }
    prompts = tmp_path / "prompts.jsonl"
    _write(
        prompts,
        [
            {
                "id": "p1",
                "request": {
                    "passage": "Mitochondria make ATP.",
                    "lo_id": "BIO.2.1.b",
                    "lo_text": "Identify organelles.",
                    "target_misconception": None,
                },
            }
        ],
    )
    _write(gen / "per_item.jsonl", [{"arm": a, "id": "p1", "usable": True} for a in ("v2", "base")])
    for arm in ("v2", "base"):
        _write(gen / f"gen_{arm}.jsonl", [{"id": "p1", "text": json.dumps(item)}])
    out = tmp_path / "audit"
    draw(gen, prompts, ["v2", "base"], 1, 7, out)
    with pytest.raises(SystemExit, match="already has an audit"):
        draw(gen, prompts, ["v2", "base"], 1, 7, out)

    sheet = [json.loads(x) for x in (out / "sheet.jsonl").read_text().splitlines()]
    sheet[0].update(key_correct="false", lo_fit=True)
    sheet[1].update(key_correct=True, lo_fit=False)
    _write(out / "sheet.jsonl", sheet)
    with pytest.raises(SystemExit, match="boolean"):
        score(out)

    sheet[0]["key_correct"] = False
    _write(out / "sheet.jsonl", sheet + [sheet[0]])
    with pytest.raises(SystemExit, match="once each"):
        score(out)
    _write(out / "sheet.jsonl", sheet[:1])
    with pytest.raises(SystemExit, match="exactly"):
        score(out)

    _write(out / "sheet.jsonl", sheet)
    result = score(out)
    assert sorted(v["n"] for v in result.values()) == [1, 1]
    draw(gen, prompts, ["v2", "base"], 1, 7, out, force=True)
    assert not (out / "score.json").exists()
