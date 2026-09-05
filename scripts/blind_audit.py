"""Blind audit of usable items: is the key right, and does the item fit the target objective?

The eval's checks are also what picked the v2 training data, so usable can overstate v2. This
draws the same number of usable items from two arms, shuffles them with the arm hidden, and
writes an audit sheet. A reviewer fills in `key_correct` and `lo_fit` (true/false) for each row,
reading only the sheet. `score` then joins the verdicts back to the arms.

    python scripts/blind_audit.py draw --eval-dir reports/valid_eval/gen \
        --prompts data/eval/valid_prompts.jsonl --arm finetuned-v2 --arm base-2shot \
        --out reports/valid_eval/audit
    python scripts/blind_audit.py score --out reports/valid_eval/audit

The `llm` mode fills the same sheet with verdicts from an LLM judge behind an OpenAI-compatible
endpoint (Localhost AI), one output directory per judge, and `llm-score` summarizes every judge
in the audit directory. See docs/blind_audit_llm.md.

    python scripts/blind_audit.py llm --out reports/valid_eval/audit_llm --judge qwen3.5-9b \
        --model qwen3.5-9b-mlx4 --base-url http://127.0.0.1:8011/v1
    python scripts/blind_audit.py llm-score --out reports/valid_eval/audit_llm \
        --eval-dir reports/valid_eval/gen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eduai.generation.validate import parse


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def draw(
    eval_dir: Path, prompts_path: Path, arms: list[str], n: int, seed: int, out: Path, force: bool = False
) -> None:
    if len(arms) < 2 or len(arms) != len(set(arms)):
        raise SystemExit("draw needs at least two distinct arms")
    if n < 1:
        raise SystemExit("draw needs at least one item per arm")
    if not force and any((out / name).exists() for name in ("sheet.jsonl", "key.json", "score.json")):
        raise SystemExit(f"{out} already has an audit; pass --force to replace it")
    prompts = {p["id"]: p for p in rows(prompts_path)}
    usable = {}
    for r in rows(eval_dir / "per_item.jsonl"):
        if r["arm"] in arms and r.get("usable"):
            usable.setdefault(r["arm"], []).append(r["id"])
    rng = random.Random(seed)
    picked = []
    for arm in arms:
        ids = sorted(usable.get(arm, []))
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{arm} has duplicate usable IDs")
        if len(ids) < n:
            raise SystemExit(f"{arm} has only {len(ids)} usable items; requested {n}")
        gen = {g["id"]: g for g in rows(eval_dir / f"gen_{arm}.jsonl")}
        for pid in rng.sample(ids, n):
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
    if force:
        (out / "score.json").unlink(missing_ok=True)
    print(
        f"wrote {len(sheet)} rows to {out}/sheet.jsonl ({', '.join(f'{a}: {len(usable.get(a, []))} usable' for a in arms)})"
    )


def score(out: Path) -> dict:
    key = json.loads((out / "key.json").read_text())
    sheet = rows(out / "sheet.jsonl")
    ids = [r.get("audit_id") for r in sheet]
    if len(ids) != len(set(ids)) or set(ids) != set(key):
        raise SystemExit("audit sheet IDs must match key.json exactly, once each")
    res: dict[str, dict] = {}
    for r in sheet:
        if type(r.get("key_correct")) is not bool or type(r.get("lo_fit")) is not bool:
            raise SystemExit(f"{r['audit_id']} needs boolean key_correct and lo_fit verdicts")
        a = res.setdefault(key[r["audit_id"]]["arm"], {"n": 0, "key_correct": 0, "lo_fit": 0, "both": 0})
        a["n"] += 1
        a["key_correct"] += bool(r["key_correct"])
        a["lo_fit"] += bool(r["lo_fit"])
        a["both"] += bool(r["key_correct"] and r["lo_fit"])
    (out / "score.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return res


# -- LLM judges --------------------------------------------------------------------------------
RUBRIC_SYSTEM = (
    "You review multiple-choice questions written for AP science courses. You get a source passage, "
    "the learning objective the question is meant to assess, and the question with its keyed answer. "
    "Use the passage and your own subject knowledge. Be strict: a question with a wrong key, two "
    "defensible answers, or a vague stem fails."
)
RUBRIC_USER = """Source passage:
{passage}

Learning objective ({lo_id}): {lo_text}
{misconception}
Question (JSON):
{item}

Decide three things.
1. key_correct: the keyed answer is correct and it is the only correct option of the four.
2. lo_fit: the question assesses this learning objective, not just a word or topic near it.
3. distractors_plausible: every wrong option is plausible to a student who has not mastered the
   objective and clearly wrong to one who has.

End your reply with only this JSON object and nothing after it:
{{"key_correct": true or false, "lo_fit": true or false, "distractors_plausible": true or false, "notes": "one short sentence"}}"""
RETRY_USER = "Reply with only the JSON object described above, with true or false values."
VERDICT_KEYS = ("key_correct", "lo_fit", "distractors_plausible")
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "key_correct": {"type": "boolean"},
        "lo_fit": {"type": "boolean"},
        "distractors_plausible": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [*VERDICT_KEYS, "notes"],
    "additionalProperties": False,
}
# Markers that end a reasoning block: Qwen's </think>, Gemma 4's thought channel.
REASONING_END = ("</think>", "<channel|>")
UNMARKED_REASONING_CHARS = 200
SHEET_FIELDS = ("passage", "lo_id", "lo_text", "target_misconception", "item")


def rubric_sha() -> str:
    return hashlib.sha256((RUBRIC_SYSTEM + RUBRIC_USER + RETRY_USER).encode()).hexdigest()[:16]


def rubric_messages(row: dict) -> list[dict]:
    """Only sheet fields go to the judge; the sheet never carries the arm."""
    mis = row.get("target_misconception")
    return [
        {"role": "system", "content": RUBRIC_SYSTEM},
        {
            "role": "user",
            "content": RUBRIC_USER.format(
                passage=row["passage"],
                lo_id=row["lo_id"],
                lo_text=row["lo_text"],
                misconception=f"Requested misconception to use as a wrong option: {mis}\n" if mis else "",
                item=json.dumps(row["item"], ensure_ascii=False, indent=1),
            ),
        },
    ]


def strip_reasoning(content: str) -> tuple[str, str]:
    """(final answer, reasoning) when the server left the reasoning inside `content`."""
    cut = max((content.rfind(m) + len(m) for m in REASONING_END if m in content), default=-1)
    if cut < 0:
        return content, ""
    return content[cut:], content[:cut]


def extract_verdict(text: str) -> dict | None:
    """The last JSON object in `text` with boolean verdicts, else None."""
    starts = [m.start() for m in re.finditer(r"\{", text)]
    dec = json.JSONDecoder()
    for s in reversed(starts):
        try:
            obj, _ = dec.raw_decode(text[s:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(type(obj.get(k)) is bool for k in VERDICT_KEYS):
            return {
                **{k: obj[k] for k in VERDICT_KEYS},
                "notes": str(obj.get("notes", ""))[:500],
                "_start": s,
            }
    return None


class ChatClient:
    """Minimal OpenAI-compatible /chat/completions client (stdlib only)."""

    def __init__(self, base_url: str, model: str, timeout: float = 1800.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def __call__(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.url,
            data=json.dumps({"model": self.model, **body}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


def judge_row(client, row: dict, thinking: bool, max_tokens: int, thinking_budget: int | None) -> dict:
    messages = rubric_messages(row)
    body: dict = {"messages": messages, "temperature": 0.0, "seed": 0, "max_tokens": max_tokens}
    if thinking:
        # A grammar from the first token would block the reasoning, so no response_format here.
        body["chat_template_kwargs"] = {"enable_thinking": True}
        if thinking_budget:
            body["max_thinking_tokens"] = thinking_budget
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True},
        }
    out = {"audit_id": row["audit_id"], "attempts": 0, "first_parse_failed": False}
    t0 = time.time()
    for attempt in range(2):
        out["attempts"] = attempt + 1
        try:
            resp = client(body)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "response_format" in body:
                body.pop("response_format")  # server without json_schema support: parse instead
                resp = client(body)
            else:
                raise
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content")
        final, stripped = (content, "") if reasoning is not None else strip_reasoning(content)
        verdict = extract_verdict(final)
        start = verdict.pop("_start") if verdict else len(final)
        source = "server" if reasoning is not None else ("client" if stripped else "none")
        if source == "none" and start > UNMARKED_REASONING_CHARS:
            # The server dropped the markers as special tokens; the text before the JSON is the reasoning.
            source, stripped = "unmarked", final[:start]
        out.update(
            reasoning_chars=out.get("reasoning_chars", 0)
            + len(reasoning if reasoning is not None else stripped),
            reasoning_from=source if out.get("reasoning_from", "none") == "none" else out["reasoning_from"],
            completion_tokens=(resp.get("usage") or {}).get("completion_tokens"),
            thinking_tokens=[
                *out.get("thinking_tokens", []),
                (resp.get("usage") or {}).get("thinking_tokens"),
            ],
            finish_reason=resp["choices"][0].get("finish_reason"),
            content=final[-2000:],
        )
        if verdict is not None:
            out.update(verdict)
            break
        if attempt == 0:
            out["first_parse_failed"] = True
            body = {
                **body,
                "messages": [
                    *messages,
                    {"role": "assistant", "content": final[-4000:]},
                    {"role": "user", "content": RETRY_USER},
                ],
            }
    out["parsed"] = all(k in out for k in VERDICT_KEYS)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def preset_info(models_yaml: Path | None, preset: str) -> dict:
    """repo and revision of a Localhost AI preset, read from its models.yaml."""
    if models_yaml is None or not models_yaml.exists():
        return {}
    import yaml

    spec = (yaml.safe_load(models_yaml.read_text()) or {}).get("models", {}).get(preset) or {}
    return {
        k: spec.get(k) for k in ("repo", "revision", "chat_template_kwargs", "max_new_tokens") if k in spec
    }


def git_state(repo: Path | None) -> dict:
    if repo is None or not (repo / ".git").exists():
        return {}

    def g(*a: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()

    return {
        "commit": g("rev-parse", "HEAD"),
        "dirty": bool(g("status", "--porcelain", "--untracked-files=no")),
    }


def thinking_stats(verdicts: list[dict], cap: int | None) -> dict:
    """Thinking-token lengths reported by the server (usage.thinking_tokens), per request, and how
    many items had a request that used the whole budget."""
    per_item = [[t for t in v.get("thinking_tokens", []) if t is not None] for v in verdicts]
    flat = [t for ts in per_item for t in ts]
    hits = sum(bool(cap) and any(t >= cap for t in ts) for ts in per_item)
    return {
        "n_reported": len(flat),
        "p50": float(np.percentile(flat, 50)) if flat else None,
        "p90": float(np.percentile(flat, 90)) if flat else None,
        "max": max(flat) if flat else None,
        "items_at_cap": hits,
        "share_at_cap": hits / len(verdicts) if verdicts and cap else None,
    }


def pilot_cap(lengths: list[int]) -> int:
    """The amended budget rule: max(1024, p90 of the pilot thinking lengths), rounded up to 256."""
    p90 = float(np.percentile(lengths, 90))
    return max(1024, -(-int(np.ceil(p90)) // 256) * 256)


def pilot(
    out: Path,
    judge: str,
    client,
    safety_cap: int = 6000,
    max_tokens: int = 6600,
    concurrency: int = 4,
    meta: dict | None = None,
) -> dict:
    """Thinking-length pilot on items outside the audit draw. Only lengths are kept; the verdicts
    are dropped without being written or printed."""
    sheet = rows(out / "sheet.jsonl")
    dest = out / f"pilot_{judge}.json"
    if dest.exists():
        raise SystemExit(f"{dest} exists; the pilot runs once per judge")
    t0 = time.time()
    first = judge_row(client, sheet[0], True, max_tokens, safety_cap)
    if first.get("reasoning_from") == "none":
        raise SystemExit("thinking was requested but the reply has no reasoning")
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        res = [first, *ex.map(lambda r: judge_row(client, r, True, max_tokens, safety_cap), sheet[1:])]
    lengths = [v["thinking_tokens"][0] for v in res]
    if any(t is None for t in lengths):
        raise SystemExit("the server did not report usage.thinking_tokens")
    rec = {
        "judge": judge,
        "model": getattr(client, "model", None),
        "safety_cap": safety_cap,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "n": len(res),
        "thinking_tokens": lengths,
        "finish_reason": [v["finish_reason"] for v in res],
        "at_safety_cap": sum(t >= safety_cap for t in lengths),
        "p90": float(np.percentile(lengths, 90)),
        "cap": pilot_cap(lengths),
        "wall_seconds": round(time.time() - t0, 1),
        **(meta or {}),
    }
    dest.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("judge", "n", "p90", "cap", "at_safety_cap", "wall_seconds")}))
    return rec


def llm_judge(
    out: Path,
    judge: str,
    client,
    thinking: bool = True,
    max_tokens: int = 3600,
    thinking_budget: int | None = 3000,
    concurrency: int = 4,
    meta: dict | None = None,
    force: bool = False,
    per_arm: int | None = None,
) -> dict:
    sheet = rows(out / "sheet.jsonl")
    if per_arm:
        # A fixed subset: the first per_arm items of each arm in sheet order. key.json is read only
        # to pick the rows; the judge still sees sheet rows alone.
        key = json.loads((out / "key.json").read_text())
        taken: dict[str, int] = {}
        subset = []
        for r in sheet:
            arm = key[r["audit_id"]]["arm"]
            if taken.get(arm, 0) < per_arm:
                taken[arm] = taken.get(arm, 0) + 1
                subset.append(r)
        sheet = subset
    for r in sheet:
        if set(r) & {"arm", "id"}:
            raise SystemExit("the audit sheet must not carry the arm or prompt id")
    jdir = out / f"llm_{judge}"
    if (jdir / "verdicts.jsonl").exists() and not force:
        raise SystemExit(f"{jdir} already has verdicts; this audit runs once (pass --force after a crash)")
    jdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    first = judge_row(client, sheet[0], thinking, max_tokens, thinking_budget)
    if thinking and first.get("reasoning_from") == "none":
        raise SystemExit(
            "thinking was requested but the reply has no reasoning; the server ignored enable_thinking"
        )
    # Each verdict is appended to verdicts.partial.jsonl as it arrives, so a run stopped for time
    # keeps what finished (with a timestamp, for the time projection).
    partial = jdir / "verdicts.partial.jsonl"
    lock = threading.Lock()

    def log(v: dict) -> dict:
        with lock, partial.open("a") as f:
            f.write(json.dumps({**v, "done_at": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
        return v

    partial.write_text("")
    log(first)
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        verdicts = [
            first,
            *ex.map(lambda r: log(judge_row(client, r, thinking, max_tokens, thinking_budget)), sheet[1:]),
        ]
    partial.unlink()
    (jdir / "verdicts.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in verdicts))
    n = len(verdicts)
    manifest = {
        "judge": judge,
        "model": getattr(client, "model", None),
        "thinking": thinking,
        "max_thinking_tokens": thinking_budget if thinking else None,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "response_format": None if thinking else "json_schema (dropped if the server rejects it)",
        "rubric_sha256_16": rubric_sha(),
        "concurrency": concurrency,
        "subset_first_per_arm": per_arm,
        "n": n,
        "first_try_parse_failure_rate": sum(v["first_parse_failed"] for v in verdicts) / n if n else None,
        "unparsed_after_retry": sum(not v["parsed"] for v in verdicts),
        "reasoning_from": {
            k: sum(v.get("reasoning_from") == k for v in verdicts)
            for k in ("server", "client", "unmarked", "none")
        },
        "mean_completion_tokens": (
            float(np.mean([v["completion_tokens"] for v in verdicts if v.get("completion_tokens")]))
            if any(v.get("completion_tokens") for v in verdicts)
            else None
        ),
        "thinking_tokens": thinking_stats(verdicts, thinking_budget if thinking else None),
        "wall_seconds": round(time.time() - t0, 1),
        **(meta or {}),
    }
    (jdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "judge",
                    "n",
                    "first_try_parse_failure_rate",
                    "unparsed_after_retry",
                    "wall_seconds",
                )
            }
        )
    )
    return manifest


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    if pa in (0, 1) or pb in (0, 1):
        return None  # a rater that never varies: kappa says nothing
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def _boot_rate(x: np.ndarray, rng, n_boot: int) -> list[float]:
    b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def llm_score(
    out: Path, eval_dir: Path | None = None, judge_key: str = "llama-1b", n_boot: int = 5000
) -> dict:
    """Per judge and arm: rates of key_correct, lo_fit, both (the precision of "usable") with
    bootstrap CIs, and the first-arm minus second-arm difference; Cohen's kappa between judges;
    agreement with the eval's key judge."""
    key = json.loads((out / "key.json").read_text())
    arms = list(dict.fromkeys(v["arm"] for v in key.values()))
    # The registered difference is v2 minus base-2shot, whatever order the draw put them in.
    arms.sort(key=lambda a: a == "base-2shot")
    judges = sorted(p.name[len("llm_") :] for p in out.glob("llm_*") if (p / "verdicts.jsonl").exists())
    eval_key = {}
    if eval_dir is not None and (eval_dir / f"judge_{judge_key}.jsonl").exists():
        eval_key = {
            (r["arm"], r["id"]): bool(r["agrees"]) for r in rows(eval_dir / f"judge_{judge_key}.jsonl")
        }
    res: dict = {"arms": arms, "judges": {}, "kappa": {}}
    verdicts = {}
    for j in judges:
        vs = {v["audit_id"]: v for v in rows(out / f"llm_{j}" / "verdicts.jsonl")}
        verdicts[j] = vs
        rng = np.random.default_rng(0)
        jr: dict = {"unparsed": sum(not v["parsed"] for v in vs.values()), "by_arm": {}}
        both = {}
        for arm in arms:
            ids = [a for a, k in key.items() if k["arm"] == arm and vs.get(a, {}).get("parsed")]
            ar = {"n": len(ids)}
            for m in (*VERDICT_KEYS, "both"):
                x = np.array(
                    [vs[a]["key_correct"] and vs[a]["lo_fit"] if m == "both" else vs[a][m] for a in ids],
                    float,
                )
                ar[m] = float(x.mean()) if len(x) else None
                if m == "both":
                    ar["both_ci95"] = _boot_rate(x, rng, n_boot) if len(x) else None
                    both[arm] = x
            jr["by_arm"][arm] = ar
        if len(arms) == 2 and all(len(both[a]) for a in arms):
            xa, xb = both[arms[0]], both[arms[1]]
            d = [
                xa[rng.integers(0, len(xa), len(xa))].mean() - xb[rng.integers(0, len(xb), len(xb))].mean()
                for _ in range(n_boot)
            ]
            jr["diff_both"] = {
                "arms": f"{arms[0]} - {arms[1]}",
                "diff": float(xa.mean() - xb.mean()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            }
        if eval_key:
            pairs = [
                (vs[a]["key_correct"], eval_key[(k["arm"], k["id"])])
                for a, k in key.items()
                if vs.get(a, {}).get("parsed") and (k["arm"], k["id"]) in eval_key
            ]
            jr[f"agreement_with_{judge_key}_key"] = {
                "n": len(pairs),
                "agree": sum(x == y for x, y in pairs) / len(pairs) if pairs else None,
                "kappa": kappa([x for x, _ in pairs], [y for _, y in pairs]) if pairs else None,
            }
        jr["manifest"] = json.loads((out / f"llm_{j}" / "manifest.json").read_text())
        res["judges"][j] = jr
    for i, a in enumerate(judges):
        for b in judges[i + 1 :]:
            ids = [
                x
                for x in key
                if verdicts[a].get(x, {}).get("parsed") and verdicts[b].get(x, {}).get("parsed")
            ]
            res["kappa"][f"{a} vs {b}"] = {
                "n": len(ids),
                **{
                    m: kappa(
                        [
                            verdicts[a][x]["key_correct"] and verdicts[a][x]["lo_fit"]
                            if m == "both"
                            else verdicts[a][x][m]
                            for x in ids
                        ],
                        [
                            verdicts[b][x]["key_correct"] and verdicts[b][x]["lo_fit"]
                            if m == "both"
                            else verdicts[b][x][m]
                            for x in ids
                        ],
                    )
                    for m in (*VERDICT_KEYS, "both")
                },
            }
    (out / "llm_score.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({j: {a: r["by_arm"][a]["both"] for a in arms} for j, r in res["judges"].items()}))
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
    d.add_argument("--force", action="store_true", help="Replace an existing audit sheet and key")
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    j = sub.add_parser("llm", help="Fill the sheet with one LLM judge's verdicts")
    j.add_argument("--out", type=Path, required=True)
    j.add_argument("--judge", required=True, help="Name for the output directory, e.g. qwen3.5-9b")
    j.add_argument("--model", required=True, help="Model name sent to the server (the preset)")
    j.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    j.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    j.add_argument("--max-tokens", type=int, default=3600)
    j.add_argument("--thinking-budget", type=int, default=3000)
    j.add_argument("--concurrency", type=int, default=4)
    j.add_argument("--models-yaml", type=Path, help="Localhost AI models.yaml, to record repo and revision")
    j.add_argument("--server-repo", type=Path, help="Localhost AI checkout, to record its commit")
    j.add_argument("--force", action="store_true")
    j.add_argument("--per-arm", type=int, help="Judge only the first N items of each arm, in sheet order")
    pl = sub.add_parser("pilot", help="Thinking-length pilot for one judge; keeps lengths only")
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--judge", required=True)
    pl.add_argument("--model", required=True)
    pl.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    pl.add_argument("--safety-cap", type=int, default=6000)
    pl.add_argument("--max-tokens", type=int, default=6600)
    pl.add_argument("--concurrency", type=int, default=4)
    pl.add_argument("--models-yaml", type=Path)
    pl.add_argument("--server-repo", type=Path)
    ls = sub.add_parser("llm-score")
    ls.add_argument("--out", type=Path, required=True)
    ls.add_argument("--eval-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "draw":
        draw(a.eval_dir, a.prompts, a.arm, a.n, a.seed, a.out, a.force)
    elif a.cmd == "score":
        score(a.out)
    elif a.cmd == "llm":
        meta = {
            "preset": preset_info(a.models_yaml, a.model),
            "server": git_state(a.server_repo),
            "base_url": a.base_url,
        }
        llm_judge(
            a.out,
            a.judge,
            ChatClient(a.base_url, a.model),
            a.thinking,
            a.max_tokens,
            a.thinking_budget,
            a.concurrency,
            meta,
            a.force,
            a.per_arm,
        )
    elif a.cmd == "pilot":
        meta = {"preset": preset_info(a.models_yaml, a.model), "server": git_state(a.server_repo)}
        pilot(
            a.out,
            a.judge,
            ChatClient(a.base_url, a.model),
            a.safety_cap,
            a.max_tokens,
            a.concurrency,
            meta,
        )
    else:
        llm_score(a.out, a.eval_dir)


if __name__ == "__main__":
    main()
