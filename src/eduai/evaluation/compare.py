"""Base 0-shot vs base 2-shot vs fine-tuned adapters, on the same prompts.

Phases (each runs in its own process so only one model is resident at a time):
  generate  one arm -> reports/eval/gen_<arm>.jsonl (raw text, timing)
  judge     a fixed open-book judge that is not any arm's generator (Llama 3.2 1B) scores every
            parsed item and the reference SciQ items; the base 3B scores them again as a secondary,
            "self-judged for base arms" column
  score     independent checks per item, paired bootstrap CIs, reports/eval.json + eval_report.md
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np

from eduai.config import ROOT, model_path
from eduai.data.sciq import read_jsonl, write_jsonl
from eduai.generation.validate import misconception_present, parse, schema_errors, structure_problems
from eduai.prompts import GenerationRequest, parse_user

ARMS = ("base-0shot", "base-2shot", "finetuned", "finetuned-v2", "finetuned-v2-2shot")
ADAPTER = ROOT / "adapters" / "llama32-3b-eduai"
ADAPTER_V2 = ROOT / "adapters" / "llama32-3b-eduai-v2"
# arm -> (adapter or None, whether the prompt carries the two fixed examples)
ARM_SPECS = {
    "base-0shot": (None, False),
    "base-2shot": (None, True),
    "finetuned": (ADAPTER, False),
    "finetuned-v2": (ADAPTER_V2, False),
    "finetuned-v2-2shot": (ADAPTER_V2, True),
}
EVAL_DIR = ROOT / "reports" / "eval"


def load_prompts(path: Path) -> list[dict]:
    return read_jsonl(path)


def arms_present(out_dir: Path) -> list[str]:
    """Arms with generations in out_dir: the named arms in ARMS order, then any sweep arms by name."""
    found = {p.name[len("gen_") : -len(".jsonl")] for p in out_dir.glob("gen_*.jsonl")}
    return [a for a in ARMS if a in found] + sorted(found - set(ARMS))


def request_of(row: dict) -> GenerationRequest:
    return GenerationRequest(**row["request"])


SHOTS_SAMPLE = ROOT / "data" / "samples" / "sft_sample.jsonl"


def fixed_shots(sft_train: Path = SHOTS_SAMPLE) -> list[tuple[GenerationRequest, dict]]:
    """First standard and first stimulus example from SFT train, without a target misconception.

    The committed sample holds the first SFT train rows, so it yields the same two shots the eval used.
    """
    shots, want = [], ["standard", "stimulus"]
    for row in read_jsonl(sft_train):
        req = parse_user(row["messages"][1]["content"])
        if req.format in want and not req.target_misconception and not req.avoid:
            shots.append((req, json.loads(row["messages"][2]["content"])))
            want.remove(req.format)
        if not want:
            break
    return shots


def _rel(path: Path | None) -> str | None:
    """A path for committed metadata: relative to the repo root, even when it lies outside it (a
    worktree using the main checkout's adapter), so no home-directory path is written."""
    if path is None:
        return None
    return os.path.relpath(Path(path).resolve(), ROOT)


def _free_mlx() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


# -- phase 1 ------------------------------------------------------------------------------------
def generate_arm(
    arm: str,
    prompts: list[dict],
    out_dir: Path = EVAL_DIR,
    limit: int | None = None,
    adapter: Path | None = None,
    shots: bool | None = None,
    force: bool = False,
) -> Path:
    """Greedy, one request at a time. Named arms use ARM_SPECS; sweep arms pass `adapter` and `shots`.

    Refuses to replace an existing gen_<arm>.jsonl unless `force`, so committed generations aren't
    overwritten by a rerun."""
    from eduai.generation.generator import Generator
    from eduai.llm.mlx_backend import MLXBackend

    out = out_dir / f"gen_{arm}.jsonl"
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass force=True (--force) to regenerate it")
    spec_adapter, spec_shots = ARM_SPECS.get(arm, (None, False))
    if arm not in ARM_SPECS and adapter is None:
        raise ValueError(f"unknown arm {arm!r}: pass an adapter for a sweep arm")
    adapter = adapter if adapter is not None else spec_adapter
    use_shots = spec_shots if shots is None else shots
    backend = MLXBackend(model_path("llama-3b"), adapter)
    shots = fixed_shots(ROOT / "data" / "sft" / "train.jsonl") if use_shots else []
    gen = Generator(backend, shots=shots, temperature=0.0, retry=True)
    rows = []
    t0 = time.time()
    for row in prompts[:limit]:
        g = gen.generate(request_of(row))
        rows.append(
            {
                "id": row["id"],
                "arm": arm,
                "text": g.text,
                "seconds": round(g.seconds, 3),
                "generation_tokens": g.tokens,
                "generation_tps": g.tps,
                "retried": g.retried,
                "first_parsed": g.first_parsed,
                "peak_memory_gb": backend.last_stats.get("peak_memory_gb"),
            }
        )
    write_jsonl(out, rows)
    (out_dir / f"gen_{arm}.meta.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "n": len(rows),
                "wall_seconds": round(time.time() - t0, 1),
                "shots": len(shots),
                "backend": backend.name,
                "adapter": _rel(adapter),
            },
            indent=2,
        )
        + "\n"
    )
    return out


# -- phase 2 ------------------------------------------------------------------------------------
def reference_items(prompts: list[dict], data_dir: Path) -> dict[str, dict]:
    """SciQ source items as MCQs (same fixed letter shuffle), used as the ceiling row.

    A prompt whose reference already carries "choices" and "key" (the OpenStax set) uses those
    as they are; a prompt with no reference has no ceiling row.
    """
    import random

    tagged = None
    rng = random.Random(0)
    out = {}
    for p in prompts:
        ref = p.get("reference")
        if ref is None:
            continue
        if "choices" in ref:
            out[p["id"]] = {
                "stem": ref["question"],
                "choices": dict(ref["choices"]),
                "answer": ref["key"],
                "lo_id": p["request"]["lo_id"],
                "difficulty": p["request"]["difficulty"],
                "explanation": "reference",
            }
            continue
        if tagged is None:
            tagged = {d["id"]: d for d in read_jsonl(data_dir / "items_tagged.jsonl")}
        src = tagged[p["id"]]
        opts = [src["correct"], *src["distractors"]]
        order = list(range(4))
        rng.shuffle(order)
        letters = "ABCD"
        choices = {letters[i]: opts[order[i]] for i in range(4)}
        answer = letters[order.index(0)]
        out[p["id"]] = {
            "stem": src["question"],
            "choices": choices,
            "answer": answer,
            "lo_id": p["request"]["lo_id"],
            "difficulty": p["request"]["difficulty"],
            "explanation": "reference",
        }
    return out


def judge_all(
    prompts: list[dict],
    judge_key: str,
    out_dir: Path = EVAL_DIR,
    data_dir: Path = ROOT / "data",
    arms: list[str] | None = None,
) -> Path:
    """Judge every arm (arms=None), or only `arms`, keeping the existing rows of the other arms as they are."""
    from eduai.evaluation.solver import Judge
    from eduai.llm.mlx_backend import MLXBackend

    out = out_dir / f"judge_{judge_key}.jsonl"
    kept: dict[str, list[dict]] = {}
    if arms is not None and out.exists():
        for r in read_jsonl(out):
            if r["arm"] not in arms:
                kept.setdefault(r["arm"], []).append(r)
    judge = Judge(MLXBackend(model_path(judge_key), None))
    passages = {p["id"]: p["request"]["passage"] for p in prompts}
    rows = kept.pop("reference", None)
    if rows is None:
        refs = reference_items(prompts, data_dir)
        rows = [{"id": pid, "arm": "reference", **judge(item, passages[pid])} for pid, item in refs.items()]
    for arm in arms_present(out_dir):
        if arm in kept:
            rows += kept.pop(arm)
            continue
        if arms is not None and arm not in arms:
            continue
        for g in read_jsonl(out_dir / f"gen_{arm}.jsonl"):
            item, _ = parse(g["text"])
            if item is None or schema_errors(item):
                continue
            rows.append({"id": g["id"], "arm": arm, **judge(item, passages[g["id"]])})
    write_jsonl(out, rows)
    return out


# -- phase 3 ------------------------------------------------------------------------------------
CHECKS = ("json", "schema", "structure", "key", "aligned", "novel")


def score(
    prompts: list[dict],
    tagger,
    novelty,
    data_dir: Path = ROOT / "data",
    out_dir: Path = EVAL_DIR,
    judge_key: str = "llama-1b",
    secondary_key: str = "llama-3b",
) -> dict:
    from eduai.curriculum.tagger import tag_text

    by_id = {p["id"]: p for p in prompts}
    judge = {(r["arm"], r["id"]): r for r in read_jsonl(out_dir / f"judge_{judge_key}.jsonl")}
    sec_path = out_dir / f"judge_{secondary_key}.jsonl"
    secondary = {(r["arm"], r["id"]): r for r in read_jsonl(sec_path)} if sec_path.exists() else {}
    refs = reference_items(prompts, data_dir)

    per_item: dict[str, list[dict]] = {}
    reasons: dict[str, dict[str, int]] = {}
    ids = [p["id"] for p in prompts]
    ref_ids = [i for i in ids if i in refs]
    ref_align = tagger.is_aligned(
        [tag_text(refs[i]["stem"], refs[i]["choices"][refs[i]["answer"]]) for i in ref_ids],
        [by_id[i]["request"]["lo_id"] for i in ref_ids],
    )
    per_item["reference"] = [
        {
            "id": i,
            "aligned": a["aligned"],
            "key": judge[("reference", i)]["agrees"],
            "key_secondary": secondary.get(("reference", i), {}).get("agrees"),
            "key_letter": refs[i]["answer"],
        }
        for i, a in zip(ref_ids, ref_align, strict=True)
    ]
    timing = {}
    arms = arms_present(out_dir)
    for arm in arms:
        path = out_dir / f"gen_{arm}.jsonl"
        gens = {g["id"]: g for g in read_jsonl(path)}
        rows, hist = [], {}
        parsed = {}
        for pid in ids:
            g = gens.get(pid)
            item, err = parse(g["text"]) if g else (None, "missing")
            parsed[pid] = item if item is not None and not schema_errors(item) else None
        align_ids = [i for i in ids if parsed[i]]
        aligns = (
            dict(
                zip(
                    align_ids,
                    tagger.is_aligned(
                        [
                            tag_text(parsed[i]["stem"], parsed[i]["choices"][parsed[i]["answer"]])
                            for i in align_ids
                        ],
                        [by_id[i]["request"]["lo_id"] for i in align_ids],
                    ),
                    strict=True,
                )
            )
            if align_ids
            else {}
        )
        for pid in ids:
            req = request_of(by_id[pid])
            g = gens.get(pid)
            item, err = parse(g["text"]) if g else (None, "missing")
            r = {
                "id": pid,
                "json": item is not None,
                "first_try_json": bool(g and g.get("first_parsed", True)),
                "schema": False,
                "structure": False,
                "key": False,
                "aligned": False,
                "novel": False,
                "source_copy": False,
                "memorized": False,
                "key_secondary": None,
            }
            first_fail = err if item is None else None
            if item is not None:
                r["schema"] = not schema_errors(item)
                if not r["schema"]:
                    first_fail = "schema"
                else:
                    problems = structure_problems(item, req)
                    r["structure"] = not problems and misconception_present(item, req.target_misconception)
                    j = judge.get((arm, pid))
                    texts = [v.strip().lower() for v in item["choices"].values()]
                    # A key whose text also appears as a distractor can't be told apart by any judge.
                    distinct = texts.count(item["choices"][item["answer"]].strip().lower()) == 1
                    r["key"] = bool(j and j["agrees"]) and distinct
                    s2 = secondary.get((arm, pid))
                    r["key_secondary"] = None if s2 is None else bool(s2["agrees"]) and distinct
                    r["aligned"] = aligns[pid]["aligned"]
                    ok, info = novelty.check(
                        item,
                        exclude_ids=[f"sciq-{pid}"],
                        exclude_group=by_id[pid]["group"],
                        source_stem=(by_id[pid].get("reference") or {}).get("question"),
                    )
                    novelty.accepted.clear()  # arms are compared independently
                    r["novel"] = ok
                    r["source_copy"] = bool(info.get("source_copy"))
                    r["memorized"] = bool(info["memorized"])
                    r["source_cos"] = info.get("source_cos")
                    for name in ("structure", "key", "aligned", "novel"):
                        if not r[name]:
                            first_fail = {
                                "structure": "structure",
                                "key": "key_disagreement",
                                "aligned": "not_aligned",
                                "novel": "duplicate",
                            }[name]
                            break
            r["all_checks"] = all(r[c] for c in CHECKS)
            # Bank-promotion criterion: passes everything and isn't a copy of its source question.
            r["usable"] = r["all_checks"] and not r["source_copy"]
            r["key_letter"] = parsed[pid]["answer"] if parsed.get(pid) else None
            hist[first_fail or "accepted"] = hist.get(first_fail or "accepted", 0) + 1
            rows.append(r)
        per_item[arm] = rows
        reasons[arm] = hist
        tps = [g["generation_tps"] for g in gens.values() if g.get("generation_tps")]
        secs = [g["seconds"] for g in gens.values()]
        mem = [g["peak_memory_gb"] for g in gens.values() if g.get("peak_memory_gb")]
        timing[arm] = {
            "mean_generation_tps": float(np.mean(tps)) if tps else None,
            "mean_seconds_per_item": float(np.mean(secs)) if secs else None,
            "peak_memory_gb": float(max(mem)) if mem else None,
        }

    summary = summarize(per_item)
    structure = {}
    for arm in arms:
        path = out_dir / f"gen_{arm}.jsonl"
        c: dict[str, int] = {}
        for g in read_jsonl(path):
            item, _ = parse(g["text"])
            if item is None or schema_errors(item):
                continue
            req = request_of(by_id[g["id"]])
            kinds = set()
            for pr in structure_problems(item, req):
                kinds.add(
                    "near-identical options"
                    if "near-identical" in pr
                    else "wrong lo_id"
                    if pr.startswith("lo_id")
                    else pr
                )
            if not misconception_present(item, req.target_misconception):
                kinds.add("target misconception missing")
            if len({v.strip().lower() for v in item["choices"].values()}) == 1:
                kinds.add("all four options identical")
            for k in kinds:
                c[k] = c.get(k, 0) + 1
        structure[arm] = dict(sorted(c.items(), key=lambda kv: -kv[1]))
    result = {
        "n_prompts": len(ids),
        "summary": summary,
        "structure_problems": structure,
        "timing": timing,
        "rejections": reasons,
        "bootstrap": bootstrap_diffs(
            per_item, pairs=PAIRS + tuple((a, "base-2shot") for a in arms if a not in ARMS)
        ),
        "judge": judge_key,
        "secondary_judge": secondary_key,
        **v2_extras(per_item),
    }
    write_jsonl(out_dir / "per_item.jsonl", [{"arm": a, **r} for a, rs in per_item.items() for r in rs])
    return result


def summarize(per_item: dict[str, list[dict]]) -> dict:
    out = {}
    for arm, rows in per_item.items():
        n = len(rows)
        s = {"n": n}
        for k in (
            "json",
            "first_try_json",
            "schema",
            "structure",
            "key",
            "aligned",
            "novel",
            "source_copy",
            "memorized",
            "all_checks",
            "usable",
        ):
            if rows and k in rows[0]:
                s[k] = sum(bool(r[k]) for r in rows) / n
        valid = [r for r in rows if r.get("schema", True)]
        s["key_on_valid"] = sum(bool(r["key"]) for r in valid) / len(valid) if valid else None
        s["n_schema_valid"] = len(valid)
        sec = [r["key_secondary"] for r in rows if r.get("key_secondary") is not None]
        s["key_secondary"] = sum(sec) / n if sec else None
        letters = [r.get("key_letter") for r in rows if r.get("key_letter")]
        s["key_letters"] = {x: letters.count(x) for x in "ABCD"}
        s["key_agreement_by_letter"] = {
            x: (
                sum(bool(r["key"]) for r in rows if r.get("key_letter") == x) / letters.count(x)
                if letters.count(x)
                else None
            )
            for x in "ABCD"
        }
        out[arm] = s
    return out


# The first four pairs are the v1 comparisons; new pairs go after them so one seeded generator
# reproduces the v1 intervals exactly.
PAIRS = (
    ("finetuned", "base-0shot"),
    ("finetuned", "base-2shot"),
    ("finetuned", "reference"),
    ("base-2shot", "base-0shot"),
    ("finetuned-v2", "base-0shot"),
    ("finetuned-v2", "base-2shot"),
    ("finetuned-v2", "finetuned"),
    ("finetuned-v2", "reference"),
    ("finetuned-v2-2shot", "base-2shot"),
    ("finetuned-v2-2shot", "finetuned-v2"),
)


def bootstrap_diffs(
    per_item: dict[str, list[dict]], n_boot: int = 5000, seed: int = 0, pairs: tuple = PAIRS
) -> dict:
    """Paired bootstrap over prompts: 95% CI of (arm A rate - arm B rate)."""
    rng = np.random.default_rng(seed)
    out = {}
    for a, b in pairs:
        if not per_item.get(a) or not per_item.get(b):
            continue
        for metric in ("usable", "all_checks", "aligned", "key", "json"):
            if metric not in per_item[a][0] or metric not in per_item[b][0]:
                continue
            ra, rb = per_item[a], per_item[b]
            if len(ra) != len(rb):
                # The reference row can cover only some prompts; pair on the shared ids.
                ids_b = {r["id"] for r in rb}
                ra = [r for r in ra if r["id"] in ids_b]
                ids_a = {r["id"] for r in ra}
                rb = [r for r in rb if r["id"] in ids_a]
            xa = np.array([bool(r[metric]) for r in ra], float)
            xb = np.array([bool(r[metric]) for r in rb], float)
            d = xa - xb
            idx = rng.integers(0, len(d), size=(n_boot, len(d)))
            boots = d[idx].mean(axis=1)
            out[f"{a} - {b} | {metric}"] = {
                "diff": float(d.mean()),
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            }
    return out


# Test prompts whose passage is also a v2 training passage (docs/v2_selection.md, amendment of 2026-09-18).
V2_PASSAGE_OVERLAP = ("test-00916", "train-02955")


def v2_extras(per_item: dict[str, list[dict]]) -> dict:
    """The analyses pre-registered for the v2 test run: the headline pair without the two prompts
    that share a v2 training passage, and (exploratory) usable rates split by whether the SciQ
    reference item is tagger-aligned. Empty without v2 arms."""
    pairs = tuple((a, "base-2shot") for a in ("finetuned-v2", "finetuned-v2-2shot") if a in per_item)
    if not pairs:
        return {}

    def subset(keep) -> dict[str, list[dict]]:
        ok = {r["id"] for r in per_item["reference"] if keep(r)}
        return {a: [r for r in rs if r["id"] in ok] for a, rs in per_item.items()}

    def usable_only(d: dict) -> dict:
        return {k: v for k, v in d.items() if k.endswith("| usable")}

    split = {}
    for name, flag in (("reference_aligned", True), ("reference_not_aligned", False)):
        sub = subset(lambda r, flag=flag: bool(r["aligned"]) == flag)
        split[name] = {
            "n": len(sub["reference"]),
            "usable": {
                a: (float(np.mean([bool(r["usable"]) for r in rs])) if rs else None)
                for a, rs in sub.items()
                if a != "reference"
            },
            "bootstrap": usable_only(bootstrap_diffs(sub, pairs=pairs)),
        }
    excl = subset(lambda r: r["id"] not in V2_PASSAGE_OVERLAP)
    return {
        "v2_sensitivity": {
            "excluded": list(V2_PASSAGE_OVERLAP),
            "n": len(excl["reference"]),
            "bootstrap": usable_only(bootstrap_diffs(excl, pairs=pairs)),
        },
        "v2_aligned_split_exploratory": split,
    }


def test_runs(eval_dir: Path) -> dict | None:
    """Count the logged test-split runs (scripts/run_eval.sh writes a start and a finish line per run).

    Runs started but never finished (crashes, kills) count as runs. None when there is no ledger."""
    path = eval_dir / "test_runs.jsonl"
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    started = sum(e["event"] == "start" for e in events)
    return {"started": started, "unfinished": started - sum(e["event"] == "finish" for e in events)}
