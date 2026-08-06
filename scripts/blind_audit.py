"""Blind audit of usable items: is the key right, and does the item fit the target objective?

The eval's checks are also what picked the v2 training data, so usable can overstate v2. This
draws the same number of usable items from two arms, shuffles them with the arm hidden, and
writes an audit sheet. A reviewer fills in `key_correct` and `lo_fit` (true/false) for each row,
reading only the sheet. `score` then joins the verdicts back to the arms.

    python scripts/blind_audit.py draw --eval-dir reports/valid_eval/gen \
        --prompts data/eval/valid_prompts.jsonl --arm finetuned-v2 --arm base-2shot \
        --out reports/valid_eval/audit
    python scripts/blind_audit.py score --out reports/valid_eval/audit
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from eduai.generation.validate import parse


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def draw(eval_dir: Path, prompts_path: Path, arms: list[str], n: int, seed: int, out: Path) -> None:
    prompts = {p["id"]: p for p in rows(prompts_path)}
    usable = {}
    for r in rows(eval_dir / "per_item.jsonl"):
        if r["arm"] in arms and r.get("usable"):
            usable.setdefault(r["arm"], []).append(r["id"])
    rng = random.Random(seed)
    picked = []
    for arm in arms:
        ids = sorted(usable.get(arm, []))
        gen = {g["id"]: g for g in rows(eval_dir / f"gen_{arm}.jsonl")}
        for pid in rng.sample(ids, min(n, len(ids))):
            picked.append((arm, pid, parse(gen[pid]["text"])[0]))
    rng.shuffle(picked)
    out.mkdir(parents=True, exist_ok=True)
    sheet, key = [], {}
    for k, (arm, pid, item) in enumerate(picked):
        aid = f"a{k:03d}"
        req = prompts[pid]["request"]
        sheet.append(
            {
                "audit_id": aid,
                "passage": req["passage"],
                "lo_id": req["lo_id"],
                "lo_text": req["lo_text"],
                "target_misconception": req.get("target_misconception"),
                "item": {x: item.get(x) for x in ("stimulus", "stem", "choices", "answer", "explanation")},
                "key_correct": None,
                "lo_fit": None,
                "note": "",
            }
        )
        key[aid] = {"arm": arm, "id": pid}
    (out / "sheet.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sheet))
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(
        f"wrote {len(sheet)} rows to {out}/sheet.jsonl ({', '.join(f'{a}: {len(usable.get(a, []))} usable' for a in arms)})"
    )


def score(out: Path) -> dict:
    key = json.loads((out / "key.json").read_text())
    res: dict[str, dict] = {}
    for r in rows(out / "sheet.jsonl"):
        if r["key_correct"] is None or r["lo_fit"] is None:
            raise SystemExit(f"{r['audit_id']} has no verdict yet")
        a = res.setdefault(key[r["audit_id"]]["arm"], {"n": 0, "key_correct": 0, "lo_fit": 0, "both": 0})
        a["n"] += 1
        a["key_correct"] += bool(r["key_correct"])
        a["lo_fit"] += bool(r["lo_fit"])
        a["both"] += bool(r["key_correct"] and r["lo_fit"])
    (out / "score.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--eval-dir", type=Path, required=True)
    d.add_argument("--prompts", type=Path, required=True)
    d.add_argument("--arm", action="append", required=True)
    d.add_argument("--n", type=int, default=40)
    d.add_argument("--seed", type=int, default=20260905)
    d.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "draw":
        draw(a.eval_dir, a.prompts, a.arm, a.n, a.seed, a.out)
    else:
        score(a.out)


if __name__ == "__main__":
    main()
