"""Check the committed evaluation snapshot without models or ignored source data.

This verifies prompt membership, raw generation/judge hashes, per-item rates, paired
bootstrap, and report rendering. It cannot replay tagging or novelty scoring: that
requires the ignored SciQ-derived data and pinned local embedding models.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from eduai.evaluation import compare, report

ROOT = Path(__file__).resolve().parents[1]
# Arms with committed generations; the three v1 arms must always be there.
GEN_ARMS = compare.arms_present(ROOT / "reports/eval")
ARMS = ("reference", *GEN_ARMS)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


V1_ARMS = ("reference", "base-0shot", "base-2shot", "finetuned")


def v1_rows_sha256(path: Path) -> str:
    """Hash of the v1 arms' rows only, order-independent, so adding v2 rows to the file doesn't change it."""
    keep = sorted(json.dumps(r, sort_keys=True) for r in rows(path) if r["arm"] in V1_ARMS)
    return hashlib.sha256("\n".join(keep).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"eval snapshot mismatch: {message}")


def main() -> None:
    require({"base-0shot", "base-2shot", "finetuned"} <= set(GEN_ARMS), "v1 arms")
    index = json.loads((ROOT / "reports/eval/prompt_index.json").read_text())
    prompt_rows = index["rows"]
    ids = [r["id"] for r in prompt_rows]
    require(len(ids) == len(set(ids)) == 150, "prompt IDs")
    require(all(r["assigned_split"] == "test" for r in prompt_rows), "assigned split")
    require(
        dict(Counter(r["source_split"] for r in prompt_rows)) == index["source_split_counts"],
        "source split counts",
    )
    require(index["sft_train_valid_group_overlap"] == 0, "SFT group overlap declaration")

    for name, expected in index["raw_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name)
    # Judge and per-item files gain v2 rows after the v2 test run; the v1 rows in them must not change.
    for name, expected in index["v1_rows_sha256"].items():
        require(v1_rows_sha256(ROOT / name) == expected, f"{name} v1 rows")

    result = json.loads((ROOT / "reports/eval.json").read_text())
    require(result["n_prompts"] == len(ids), "report prompt count")
    per_item: dict[str, list[dict]] = {}
    for row in rows(ROOT / "reports/eval/per_item.jsonl"):
        arm = row.pop("arm")
        per_item.setdefault(arm, []).append(row)
    require(set(per_item) == set(ARMS), "per-item arms")
    for arm, arm_rows in per_item.items():
        require([r["id"] for r in arm_rows] == ids, f"{arm} per-item IDs/order")

    require(compare.summarize(per_item) == result["summary"], "summary rates")
    require(compare.bootstrap_diffs(per_item) == result["bootstrap"], "paired bootstrap")
    extras = compare.v2_extras(per_item)
    require({k: result.get(k) for k in extras} == extras, "v2 sensitivity row and aligned split")

    for arm in GEN_ARMS:
        gen = rows(ROOT / f"reports/eval/gen_{arm}.jsonl")
        require([r["id"] for r in gen] == ids, f"{arm} generation IDs/order")
        require(all(r["arm"] == arm for r in gen), f"{arm} generation label")
    for judge in ("llama-1b", "llama-3b"):
        seen: dict[str, set[str]] = {}
        judge_rows = rows(ROOT / f"reports/eval/judge_{judge}.jsonl")
        for row in judge_rows:
            require(row["arm"] in ARMS, f"{judge} arm")
            seen.setdefault(row["arm"], set()).add(row["id"])
        require(sum(map(len, seen.values())) == len(judge_rows), f"{judge} duplicate rows")
        require(seen.get("reference") == set(ids), f"{judge} reference IDs")
        for arm in GEN_ARMS:
            expected = {r["id"] for r in per_item[arm] if r["schema"]}
            require(seen.get(arm) == expected, f"{judge} {arm} schema-valid IDs")

    runs = compare.test_runs(ROOT / "reports/eval")
    require(result.get("test_runs") == runs, "test-run count vs reports/eval/test_runs.jsonl")
    if any(a.startswith("finetuned-v2") for a in GEN_ARMS):
        require(runs is not None and runs["started"] >= 1, "v2 test arms without a logged test run")

    card = json.loads((ROOT / "reports/eval_prompts_card.json").read_text())
    expected_report = report.render(result, "eval_manifest.json", card)
    require((ROOT / "reports/eval_report.md").read_text() == expected_report, "rendered report")
    print("eval snapshot: 150 paired prompts, raw hashes, judges, summary, bootstrap and report OK")
    print("full scoring requires ignored source data and pinned local embedding models")


if __name__ == "__main__":
    main()
