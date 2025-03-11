import json
from collections import Counter

from eduai.curriculum.embedder import HashingEmbedder
from eduai.data.build_sft import Builder, Quotas, render_data_card, stratified_sample
from eduai.data.sciq import load_sciq, passage_hash, read_jsonl
from tests.conftest import make_sciq


def _build(tmp_path, taxonomy, tagger, quotas=None):
    src = make_sciq(tmp_path / "sciq")
    items = load_sciq(src)
    q = quotas or Quotas(train=20, valid=4, test=4, eval_prompts=3)
    res = Builder(taxonomy, tagger, HashingEmbedder(256), seed=1, quotas=q).run(items, tmp_path / "out")
    return items, res


def test_data_card_reports_blank_support_exclusions_per_split(tmp_path, taxonomy, keyword_tagger):
    items, res = _build(tmp_path, taxonomy, keyword_tagger)
    card = res.card
    assert card["eligibility"]["train"]["blank_support"] == 7
    assert card["eligibility"]["valid"]["blank_support"] == 2
    assert card["eligibility"]["test"]["blank_support"] == 3
    assert card["per_split"]["train"]["excluded_blank_support"] == 7
    per_subject_blank = sum(v.get("excluded_blank_support", 0) for v in card["per_subject"].values())
    assert per_subject_blank == 12
    md = render_data_card(card)
    # Exclusions are reported before quotas in the card.
    assert md.index("Eligibility gate") < md.index("Quotas")
    assert "| train | 60 | 53 | 7 |" in md


def test_blank_support_items_never_reach_sft_and_stay_in_bank_ungrounded(tmp_path, taxonomy, keyword_tagger):
    items, _ = _build(tmp_path, taxonomy, keyword_tagger)
    blank_ids = {it.id for it in items if not it.grounded}
    meta = read_jsonl(tmp_path / "out" / "sft_meta.jsonl")
    assert blank_ids.isdisjoint({m["id"] for m in meta})
    bank = read_jsonl(tmp_path / "out" / "bank" / "sciq_items.jsonl")
    flagged = [r for r in bank if r["ungrounded"]]
    assert {r["id"].removeprefix("sciq-") for r in flagged} == blank_ids
    assert all(r["explanation"] is None for r in flagged)
    prompts = read_jsonl(tmp_path / "out" / "eval" / "prompts.jsonl")
    assert blank_ids.isdisjoint({p["id"] for p in prompts})


def test_blank_passages_do_not_share_a_split_group():
    assert passage_hash("") is None
    assert passage_hash("   ") is None


def test_quotas_letters_formats_misconceptions(tmp_path, taxonomy, keyword_tagger):
    _, res = _build(tmp_path, taxonomy, keyword_tagger)
    q = res.card["quotas"]["train"]
    assert q["achieved"] == 20
    assert set(q["letters"].values()) == {5}
    assert q["stimulus"] == 8
    assert q["misconception"] == 6
    rows = read_jsonl(tmp_path / "out" / "sft" / "train.jsonl")
    for row in rows:
        user, out = row["messages"][1]["content"], json.loads(row["messages"][2]["content"])
        assert out["choices"][out["answer"]]
        if "Target misconception" in user:
            target = user.split("include '")[1].split("' as one")[0]
            assert target in out["choices"].values() and out["choices"][out["answer"]] != target
        if "Format: stimulus" in user:
            assert out.get("stimulus")


def test_quota_shrinks_and_reports_when_pool_is_small(tmp_path, taxonomy, keyword_tagger):
    _, res = _build(tmp_path, taxonomy, keyword_tagger, Quotas(train=500, valid=4, test=4, eval_prompts=3))
    q = res.card["quotas"]["train"]
    assert q["shrunk"] and q["achieved"] < 500
    md = render_data_card(res.card)
    assert "(shrunk)" in md


def test_no_group_spans_splits(tmp_path, taxonomy, keyword_tagger):
    items, res = _build(tmp_path, taxonomy, keyword_tagger)
    assert res.card["grouping"]["cross_split_groups_after"] == 0
    spans = {}
    for it in items:
        spans.setdefault(it.tags["group"], set()).add(it.tags["final_split"])
    assert all(len(v) == 1 for v in spans.values())


def test_stratified_sample_is_proportional():
    import random

    pool = list(range(100))
    strata = [("a",)] * 70 + [("b",)] * 30
    out = stratified_sample(pool, strata, 10, random.Random(0))
    c = Counter("a" if i < 70 else "b" for i in out)
    assert c == {"a": 7, "b": 3}
