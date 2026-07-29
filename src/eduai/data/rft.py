"""Rejection-sampling fine-tuning (RFT) data for the v2 adapter.

The v1 targets are the SciQ questions for each passage, so v1 learned to write the source question
back (a quarter of its test items were source copies). v2 targets are instead the base model's own
sampled items for the SFT *train* prompts, kept only when they pass the eval's checks:

  schema -> structure (distinct options, requested LO and format, target misconception present)
  -> key text unique among the options -> tagger alignment -> novelty against the bank (own group
  excluded) and not a copy of the source question -> answer key confirmed by a judge.

The judge here is the base 3B, not the eval's 1B judge, so the adapter isn't trained directly on
the metric's judge. The 3B judge column of the eval is therefore no longer independent for v2.
The tagger and novelty checks are the eval's own; training on items that pass them is the point,
and the report says so. Valid and test groups are never sampled.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from eduai.data.sciq import read_jsonl, write_jsonl
from eduai.generation.validate import misconception_present, parse, schema_errors, structure_problems
from eduai.prompts import GenerationRequest, build_messages, parse_user
from eduai.schema import LETTERS

CHECK_ORDER = ("json", "schema", "structure", "distinct_key", "aligned", "novel", "not_source_copy", "key")


def train_prompts(data_dir: Path, exclude_users: set[str] = frozenset()) -> list[dict]:
    """SFT train rows as eval-style prompt rows (id, request, group, reference).

    `exclude_users` holds the user messages of the fixed few-shot examples, which are left out so the
    2-shot sampler never sees its own target as an example.
    """
    rows = read_jsonl(data_dir / "sft" / "train.jsonl")
    ids = [r["id"] for r in read_jsonl(data_dir / "sft" / "train_stems.jsonl")]
    items = {d["id"]: d for d in read_jsonl(data_dir / "items_tagged.jsonl")}
    out = []
    for row, pid in zip(rows, ids, strict=True):
        user = row["messages"][1]["content"]
        if user in exclude_users:
            continue
        it = items[pid]
        assert it["tags"]["final_split"] == "train", pid
        out.append(
            {
                "id": pid,
                "request": parse_user(user).__dict__,
                "group": it["tags"]["group"],
                "reference": {"question": it["question"], "answer": it["correct"]},
            }
        )
    return out


def cheap_checks(text: str, req: GenerationRequest) -> tuple[dict | None, dict]:
    """The model-free checks, in eval order. Returns the parsed item and the check results so far."""
    item, _ = parse(text)
    checks = {"json": item is not None}
    if item is None:
        return None, checks
    checks["schema"] = not schema_errors(item)
    if not checks["schema"]:
        return None, checks
    checks["structure"] = not structure_problems(item, req) and misconception_present(
        item, req.target_misconception
    )
    texts = [v.strip().lower() for v in item["choices"].values()]
    checks["distinct_key"] = texts.count(item["choices"][item["answer"]].strip().lower()) == 1
    return item, checks


def first_failure(checks: dict) -> str | None:
    for name in CHECK_ORDER:
        if name in checks and not checks[name]:
            return name
        if name not in checks:
            return f"unchecked:{name}"
    return None


def judge_batch(backend, items: list[dict], passages: list[str], chunk: int = 4) -> list[dict]:
    """Four-rotation open-book key check (same rule as evaluation.solver.Judge), batched."""
    import numpy as np

    from eduai.evaluation.solver import rotate
    from eduai.prompts import build_solver_messages

    out = []
    for s in range(0, len(items), chunk):
        part, msgs, maps = items[s : s + chunk], [], []
        for item, passage in zip(part, passages[s : s + chunk], strict=True):
            for k in range(4):
                rot, new_to_old = rotate(item["choices"], k)
                msgs.append(build_solver_messages(item["stem"], rot, item.get("stimulus"), passage))
                maps.append(new_to_old)
        lps = backend.choice_logprobs_batch(msgs)
        for j, item in enumerate(part):
            totals = dict.fromkeys(LETTERS, 0.0)
            for r in range(4):
                for new, old in maps[4 * j + r].items():
                    totals[old] += lps[4 * j + r][new] / 4
            vals = np.array([totals[x] for x in LETTERS])
            probs = np.exp(vals - vals.max())
            probs /= probs.sum()
            choice = LETTERS[int(np.argmax(vals))]
            out.append(
                {
                    "majority": choice,
                    "agrees": choice == item["answer"],
                    "p_key": float(probs[LETTERS.index(item["answer"])]),
                }
            )
    return out


def select(scored: list[dict], min_p_key: float = 0.0) -> list[dict]:
    """One sample per prompt: among samples passing every check, the one farthest from the source
    question (lowest source cosine), then the most confident judge."""
    best: dict[str, dict] = {}
    for r in scored:
        if not r.get("passed") or r.get("p_key", 0.0) < min_p_key:
            continue
        cur = best.get(r["id"])
        rank = (r["source_cos"], -r["p_key"])
        if cur is None or rank < (cur["source_cos"], -cur["p_key"]):
            best[r["id"]] = r
    return [best[k] for k in sorted(best)]


def target(item: dict, req: GenerationRequest) -> dict:
    """The completion the adapter is trained on: the sampled item with the requested LO and difficulty."""
    out = {k: item[k] for k in ("stem", "choices", "answer", "explanation")}
    if req.format == "stimulus":
        out["stimulus"] = item["stimulus"]
    out["choices"] = {k: out["choices"][k] for k in LETTERS}
    out["lo_id"] = req.lo_id
    out["difficulty"] = req.difficulty
    return out


def holdout(pid: str, frac: float = 0.05) -> bool:
    return int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def build(selected: list[dict], prompts: dict[str, dict], out_dir: Path, valid_frac: float = 0.05) -> dict:
    """Write {train,valid}.jsonl in the mlx-lm chat format. Valid rows are train-split prompts held out
    for the training loss only; model selection uses the valid-split eval prompts."""
    train, valid = [], []
    letters: Counter = Counter()
    for r in selected:
        req = GenerationRequest(**prompts[r["id"]]["request"])
        item, _ = parse(r["text"])
        tgt = target(item, req)
        letters[tgt["answer"]] += 1
        row = {"messages": build_messages(req, tgt)}
        (valid if holdout(r["id"], valid_frac) else train).append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    stats = {
        "selected": len(selected),
        "train": len(train),
        "valid": len(valid),
        "key_letters": dict(sorted(letters.items())),
        "from_2shot": sum(1 for r in selected if r.get("shots")),
    }
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _key(r: dict) -> tuple:
    return (r["id"], r["shots"], r["idx"])


def sample(
    backend,
    prompts: list[dict],
    out: Path,
    shots: list[tuple] | None = None,
    n: int = 1,
    temperature: float = 0.8,
    chunk: int = 64,
    batch: int = 16,
    max_tokens: int = 480,
    log=print,
) -> Path:
    """Append n sampled completions per prompt to `out`; rows already there are skipped (resumable)."""
    import time

    done = {_key(r) for r in read_jsonl(out)} if out.exists() else set()
    n_shots = len(shots or [])
    todo = [(p, i) for p in prompts for i in range(n) if (p["id"], n_shots, i) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in range(0, len(todo), chunk):
        part = todo[s : s + chunk]
        msgs = [build_messages(GenerationRequest(**p["request"]), shots=shots) for p, _ in part]
        texts, stats = backend.sample_batch(msgs, max_tokens, temperature, completion_batch_size=batch)
        with out.open("a") as fh:
            for (p, i), text in zip(part, texts, strict=True):
                fh.write(
                    json.dumps(
                        {"id": p["id"], "shots": n_shots, "idx": i, "temperature": temperature, "text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        log(
            f"{s + len(part)}/{len(todo)} samples, {time.time() - t0:.0f} s, "
            f"gen {stats['generation_tps']:.0f} tok/s, prompt {stats['prompt_tps']:.0f} tok/s, "
            f"peak {stats['peak_memory_gb']:.1f} GB"
        )
    return out


def check(samples: list[dict], prompts: dict[str, dict], tagger, novelty) -> list[dict]:
    """Every check except the answer key, for each sample."""
    from eduai.curriculum.tagger import tag_text

    rows, items = [], []
    for s in samples:
        req = GenerationRequest(**prompts[s["id"]]["request"])
        item, checks = cheap_checks(s["text"], req)
        rows.append({**{k: s[k] for k in ("id", "shots", "idx")}, "checks": checks, "text": s["text"]})
        items.append(item)
    todo = [i for i, it in enumerate(items) if it is not None]
    aligned = tagger.is_aligned(
        [tag_text(items[i]["stem"], items[i]["choices"][items[i]["answer"]]) for i in todo],
        [prompts[rows[i]["id"]]["request"]["lo_id"] for i in todo],
    )
    for i, a in zip(todo, aligned, strict=True):
        p = prompts[rows[i]["id"]]
        ok, info = novelty.check(
            items[i],
            exclude_ids=[f"sciq-{p['id']}"],
            exclude_group=p["group"],
            source_stem=p["reference"]["question"],
        )
        novelty.accepted.clear()  # samples are checked independently
        rows[i]["checks"].update(
            aligned=bool(a["aligned"]), novel=bool(ok), not_source_copy=not info.get("source_copy")
        )
        rows[i]["source_cos"] = round(info["source_cos"], 4)
        rows[i]["max_train_cos"] = round(info["max_train_cos"], 4)
    return rows


def needs_judge(row: dict) -> bool:
    return all(row["checks"].get(k) for k in CHECK_ORDER if k != "key")


def judge(
    backend, checked: list[dict], prompts: dict[str, dict], out: Path, chunk: int = 32, log=print
) -> Path:
    """Judge the samples that pass every other check; resumable like `sample`."""
    done = {_key(r) for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in checked if needs_judge(r) and _key(r) not in done]
    for s in range(0, len(todo), chunk):
        part = todo[s : s + chunk]
        items = [parse(r["text"])[0] for r in part]
        res = judge_batch(backend, items, [prompts[r["id"]]["request"]["passage"] for r in part])
        with out.open("a") as fh:
            for r, j in zip(part, res, strict=True):
                fh.write(json.dumps({**{k: r[k] for k in ("id", "shots", "idx")}, **j}) + "\n")
        log(f"judged {s + len(part)}/{len(todo)}")
    return out


def merge(checked: list[dict], judged: list[dict]) -> list[dict]:
    by = {_key(j): j for j in judged}
    out = []
    for r in checked:
        j = by.get(_key(r))
        checks = dict(r["checks"])
        if j is not None:
            checks["key"] = bool(j["agrees"])
        row = {**r, "checks": checks, "p_key": j["p_key"] if j else None}
        row["first_failure"] = first_failure(checks)
        row["passed"] = row["first_failure"] is None
        out.append(row)
    return out


def summary(merged: list[dict]) -> dict:
    out = {}
    for shots in sorted({r["shots"] for r in merged}):
        rows = [r for r in merged if r["shots"] == shots]
        out[f"{shots}-shot"] = {
            "samples": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "first_failure": dict(Counter(r["first_failure"] or "passed" for r in rows).most_common()),
        }
    out["prompts_with_a_pass"] = len({r["id"] for r in merged if r["passed"]})
    return out
