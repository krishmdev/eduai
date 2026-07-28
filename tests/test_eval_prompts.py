from eduai.curriculum.embedder import HashingEmbedder
from eduai.data.eval_prompts import build_prompts, candidates
from eduai.data.sciq import write_jsonl
from eduai.prompts import GenerationRequest, build_messages

PASSAGES = {
    "a": "Mitochondria are organelles that release energy from food in the form of ATP molecules.",
    "b": "Glaciers carve wide valleys as they slowly move downhill under their own great weight.",
    "c": "Plants take in carbon dioxide through small openings in their leaves called stomata.",
    "d": "Ionic bonds form when one atom gives up electrons and another atom accepts them fully.",
    "e": "Sound waves travel faster through solids than through gases because particles are closer.",
}


def _item(iid, split, key, aligned=True, lo="BIO.2.1.b"):
    return {
        "id": iid,
        "split": split,
        "question": f"Question about {key} number {iid}?",
        "correct": f"answer {key}",
        "distractors": ["x one", "y two", "z three"],
        "support": PASSAGES[key],
        "grounded": True,
        "tags": {"final_split": split, "aligned": aligned, "lo_id": lo, "difficulty": "easy", "group": iid},
    }


def _sft_row(key):
    req = GenerationRequest("AP Biology", "u", "t", "BIO.2.1.b", "x", "easy", "standard", PASSAGES[key])
    out = {
        "stem": f"Question about {key}?",
        "choices": {"A": f"answer {key}", "B": "b", "C": "c", "D": "d"},
        "answer": "A",
        "explanation": "e",
        "lo_id": "BIO.2.1.b",
        "difficulty": "easy",
    }
    return {"messages": build_messages(req, out)}


def _data(tmp_path):
    items = [
        _item("valid-1", "valid", "b"),
        _item("valid-2", "valid", "c", aligned=False),
        _item("valid-3", "valid", "d"),
        _item("valid-4", "valid", "a"),  # SFT valid row: never an eval prompt
        _item("test-1", "test", "d"),
        _item("test-2", "test", "e", aligned=False),
    ]
    write_jsonl(tmp_path / "items_tagged.jsonl", items)
    write_jsonl(tmp_path / "sft_meta.jsonl", [{"id": "valid-4"}])
    write_jsonl(tmp_path / "sft" / "train.jsonl", [_sft_row("d")])
    write_jsonl(tmp_path / "sft" / "valid.jsonl", [_sft_row("b")])
    return tmp_path


def test_valid_candidates_need_alignment_and_screen_against_train_only(tmp_path):
    data = _data(tmp_path)
    kept, stats = candidates(data, HashingEmbedder(64), 10, split="valid")
    # valid-2 is not tagger-aligned; valid-3 copies an SFT train passage; valid-4 is an SFT row.
    # valid-1 shares its passage with an SFT valid row, which the valid split doesn't screen against.
    assert [c["id"] for c in kept] == ["valid-1"]
    assert kept[0]["tagger_lo_id"] == "BIO.2.1.b"
    assert stats["dropped_passage_containment"] == 1


def test_test_candidates_skip_the_tagger_and_screen_against_train_and_valid(tmp_path):
    data = _data(tmp_path)
    kept, stats = candidates(data, HashingEmbedder(64), 10)
    # test-1 copies an SFT train passage; test-2 is kept although the tagger doesn't align it.
    assert [c["id"] for c in kept] == ["test-2"]
    assert "tagger_lo_id" not in kept[0] and stats["dropped_passage_containment"] == 1


def test_build_prompts_records_the_label_source(tmp_path, taxonomy):
    data = _data(tmp_path)
    kept, _ = candidates(data, HashingEmbedder(64), 10, split="valid")
    rows, stats = build_prompts(
        kept, {c["id"]: c["tagger_lo_id"] for c in kept}, taxonomy, 5, 1, label_source="tagger"
    )
    assert stats["used"] == 1 and rows[0]["label_source"] == "tagger"
    assert rows[0]["request"]["lo_id"] == "BIO.2.1.b"
