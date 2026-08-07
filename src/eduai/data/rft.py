"""Rejection-sampling fine-tuning (RFT) data for the v2 adapter.

The v1 targets are the SciQ questions for each passage, so v1 learned to write the source question
back (a quarter of its test items were source copies). v2 targets are instead the base model's own
sampled items for the SFT *train* prompts, kept only when they pass the eval's checks:

  schema -> structure (distinct options, requested LO and format, target misconception present)
  -> key text unique among the options -> tagger alignment -> novelty against the bank (own group
  excluded), not a copy of the source question, and not a copy of any SFT train stem (the novelty
  check leaves the prompt's group out, so without this a sibling question from the same group would
  pass) -> answer key confirmed by a judge.

Rows are keyed by (prompt id, shots, sample idx, sha1 of the text), so judge verdicts can't be
attached to a regenerated sample. A samples file is pinned to one model, adapter, seed and
temperature. Sampling uses continuous batching, whose output depends on how prompts are grouped
into batches, so the seed alone doesn't reproduce a samples file; the file itself is the record.

The judge here is the base 3B, not the eval's 1B judge, so the adapter isn't trained directly on
the metric's judge. The 3B judge column of the eval is therefore no longer independent for v2.
The tagger and novelty checks are the eval's own; training on items that pass them is the point,
and the report says so. Valid and test groups are never sampled.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from eduai.data.sciq import read_jsonl, write_jsonl
from eduai.generation.validate import misconception_present, parse, schema_errors, structure_problems
from eduai.prompts import GenerationRequest, build_messages, parse_user
from eduai.schema import LETTERS

CHECK_ORDER = (
    "json",
    "schema",
    "structure",
    "distinct_key",
    "aligned",
    "novel",
    "not_source_copy",
    "not_memorized",
    "key",
)
SAMPLE_PARAMS = ("model", "adapter", "seed", "temperature")


def text_sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def read_rows(path: Path) -> list[dict]:
    """Rows of an append-only jsonl file. A partial last line (a crash mid-write) is cut off the file."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    rows, good = [], 0
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        if line.strip():
            rows.append(json.loads(line))
        good += len(line)
    if good < len(raw):
        with path.open("r+b") as fh:
            fh.truncate(good)
    return rows


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


_LETTER_REF = re.compile(
    r"\b((?i:(?:correct\s+)?(?:answer|option|choice)\s+is\s+))([A-D])\b"
    r"|\b((?i:(?:option|choice|answer)\s+))([A-D])\b"
    r"|\b([A-D])(?=\s+(?i:is\s+(?:in)?correct)\b)"
    r"|\(([A-D])\)"
)


def move_key(item: dict, letter: str) -> dict:
    """Swap the keyed option with the option at `letter`, and follow the swap in the explanation's
    letter references ("B is correct.", "option C", "(D)"). A bare capital A is left alone, since it
    is almost always the article."""
    old = item["answer"]
    if old == letter:
        return item
    swap = {old: letter, letter: old}
    out = json.loads(json.dumps(item))
    out["choices"][letter], out["choices"][old] = item["choices"][old], item["choices"][letter]
    out["answer"] = letter
    exp = item.get("explanation", "")

    def sub(m: re.Match) -> str:
        if m[2]:
            return m[1] + swap.get(m[2], m[2])
        if m[4]:
            return m[3] + swap.get(m[4], m[4])
        if m[5]:
            return swap.get(m[5], m[5])
        return f"({swap.get(m[6], m[6])})"

    out["explanation"] = _LETTER_REF.sub(sub, exp)
    return out


def balanced_letters(ids: list[str]) -> dict[str, str]:
    """An exactly balanced key letter per prompt, assigned in sha1(id) order so it doesn't depend on
    anything the model wrote."""
    order = sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())
    return {pid: LETTERS[k % len(LETTERS)] for k, pid in enumerate(order)}


def mix(ids, prompts: dict[str, dict]) -> dict:
    c = Counter()
    for pid in ids:
        req = prompts[pid]["request"]
        c[f"{req['format']}/{'misconception' if req.get('target_misconception') else 'none'}"] += 1
    return dict(sorted(c.items()))


def build(
    selected: list[dict],
    prompts: dict[str, dict],
    out_dir: Path,
    valid_frac: float = 0.05,
    leaks=None,
    balance: bool = True,
    min_valid: int = 4,
) -> dict:
    """Write {train,valid}.jsonl in the mlx-lm chat format. Valid rows are train-split prompts held out
    for the training loss only; model selection uses the valid-split eval prompts.

    `leaks(targets)` returns one bool per target: True drops it (the CLI passes the same-answer Q+A
    screen against the valid and test prompts' reference questions). With `balance`, each target's key
    is moved to an exactly balanced letter, since the base model's own key letters are skewed.
    """
    train, valid = [], []
    letters: Counter = Counter()
    targets = []
    for r in selected:
        req = GenerationRequest(**prompts[r["id"]]["request"])
        item, _ = parse(r["text"])
        targets.append((r, req, target(item, req)))
    dropped = leaks([t for _, _, t in targets]) if leaks else [False] * len(targets)
    targets = [x for x, d in zip(targets, dropped, strict=True) if not d]
    to = balanced_letters([r["id"] for r, _, _ in targets]) if balance else {}
    original = Counter(t["answer"] for _, _, t in targets)
    for r, req, tgt in targets:
        if balance:
            tgt = move_key(tgt, to[r["id"]])
        letters[tgt["answer"]] += 1
        row = {"messages": build_messages(req, tgt)}
        (valid if holdout(r["id"], valid_frac) else train).append(row)
    if valid_frac > 0 and len(valid) < min_valid:
        raise ValueError(f"only {len(valid)} holdout rows; mlx-lm needs at least one batch")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    kept = [r for r, _, _ in targets]
    stats = {
        "selected": len(selected),
        "dropped_leak_same_answer_qa": int(sum(dropped)),
        "train": len(train),
        "valid": len(valid),
        "sampled_key_letters": dict(sorted(original.items())),
        "key_letters": dict(sorted(letters.items())),
        "from_2shot": sum(1 for r in kept if r.get("shots")),
        "mix": mix([r["id"] for r in kept], prompts),
    }
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _slot(r: dict) -> tuple:
    return (r["id"], r["shots"], r["idx"])


def _key(r: dict) -> tuple:
    return (*_slot(r), r["sha"])


def sample(
    backend,
    prompts: list[dict],
    out: Path,
    shots: list[tuple] | None = None,
    n: int = 1,
    temperature: float = 0.8,
    chunk: int = 64,
    batch: int = 12,
    max_tokens: int = 480,
    params: dict | None = None,
    log=print,
) -> Path:
    """Append n sampled completions per prompt to `out`; rows already there are skipped (resumable).

    `params` (model, adapter, seed) is stored on every row with the temperature, and a file whose rows
    were sampled with other settings is refused rather than mixed.
    """
    import time

    meta = {**dict.fromkeys(SAMPLE_PARAMS), **(params or {}), "temperature": temperature}
    existing = read_rows(out)
    for r in existing:
        old = {k: r.get(k) for k in SAMPLE_PARAMS}
        if old != meta:
            raise ValueError(f"{out} was sampled with {old}, not {meta}; use another --out")
    done = {_slot(r) for r in existing}
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
                row = {"id": p["id"], "shots": n_shots, "idx": i, "sha": text_sha(text), **meta, "text": text}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
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
        if s["sha"] != text_sha(s["text"]):
            raise ValueError(f"sample {_slot(s)} text doesn't match its sha")
        item, checks = cheap_checks(s["text"], req)
        rows.append({**{k: s[k] for k in ("id", "shots", "idx", "sha")}, "checks": checks, "text": s["text"]})
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
            aligned=bool(a["aligned"]),
            novel=bool(ok),
            not_source_copy=not info.get("source_copy"),
            not_memorized=not info["memorized"],
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
    done = {_key(r) for r in read_rows(out)}
    todo = [r for r in checked if needs_judge(r) and _key(r) not in done]
    for s in range(0, len(todo), chunk):
        part = todo[s : s + chunk]
        items = [parse(r["text"])[0] for r in part]
        res = judge_batch(backend, items, [prompts[r["id"]]["request"]["passage"] for r in part])
        with out.open("a") as fh:
            for r, j in zip(part, res, strict=True):
                fh.write(json.dumps({**{k: r[k] for k in ("id", "shots", "idx", "sha")}, **j}) + "\n")
        log(f"judged {s + len(part)}/{len(todo)}")
    return out


def merge(checked: list[dict], judged: list[dict]) -> list[dict]:
    """Attach judge verdicts to checked rows. Verdicts for a slot whose text has changed are stale and
    raise, rather than silently scoring the new text."""
    by = {_key(j): j for j in judged}
    texts = {_slot(r): r["sha"] for r in checked}
    stale = [k for k in by if k[:3] in texts and texts[k[:3]] != k[3]]
    if stale:
        raise ValueError(f"{len(stale)} judge rows are for older sample texts, e.g. {stale[0]}")
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


def summary(merged: list[dict], prompts: dict[str, dict] | None = None) -> dict:
    out = {}
    for shots in sorted({r["shots"] for r in merged}):
        rows = [r for r in merged if r["shots"] == shots]
        out[f"{shots}-shot"] = {
            "samples": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "first_failure": dict(Counter(r["first_failure"] or "passed" for r in rows).most_common()),
        }
    passed = {r["id"] for r in merged if r["passed"]}
    out["prompts_with_a_pass"] = len(passed)
    out["prompts_with_a_pass_by_mix"] = mix(passed, prompts) if prompts else None
    return out
