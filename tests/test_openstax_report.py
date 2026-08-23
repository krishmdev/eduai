import json

from scripts.openstax_report import BASE, by_subject, headline


def test_headline_follows_the_valid_rule_and_ties_go_to_zero_shot(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(
        json.dumps({"summary": {"finetuned-v2": {"usable": 0.38}, "finetuned-v2-2shot": {"usable": 0.34}}})
    )
    assert headline(p) == "finetuned-v2"
    p.write_text(
        json.dumps({"summary": {"finetuned-v2": {"usable": 0.3}, "finetuned-v2-2shot": {"usable": 0.3}}})
    )
    assert headline(p) == "finetuned-v2"
    p.write_text(
        json.dumps({"summary": {"finetuned-v2": {"usable": 0.3}, "finetuned-v2-2shot": {"usable": 0.31}}})
    )
    assert headline(p) == "finetuned-v2-2shot"


def test_by_subject_rates_and_pooled_paired_difference():
    ids = [f"ox-{s}-{i}" for s in ("bio", "phys") for i in range(4)]
    subject_of = {i: "BIO" if "bio" in i else "PHYS1" for i in ids}
    per_item = {
        "finetuned-v2": [{"id": i, "usable": True} for i in ids],
        BASE: [{"id": i, "usable": "bio" in i} for i in ids],
    }
    res = by_subject(per_item, subject_of, "finetuned-v2")
    assert res["BIO"]["diff"]["diff"] == 0.0 and res["PHYS1"]["diff"]["diff"] == 1.0
    assert res["pooled"]["n"] == 8 and res["pooled"][f"{BASE}_usable"] == 0.5
    assert "CHEM" not in res
