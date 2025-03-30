"""Base 0-shot vs base 2-shot vs fine-tuned, on the same held-out prompts.

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
import time
from pathlib import Path

import numpy as np

from eduai.config import ROOT, model_path
from eduai.data.sciq import read_jsonl, write_jsonl
from eduai.generation.validate import misconception_present, parse, schema_errors, structure_problems
from eduai.prompts import GenerationRequest, parse_user

ARMS = ("base-0shot", "base-2shot", "finetuned")
ADAPTER = ROOT / "adapters" / "llama32-3b-eduai"
EVAL_DIR = ROOT / "reports" / "eval"


def load_prompts(path: Path) -> list[dict]:
    return read_jsonl(path)


def request_of(row: dict) -> GenerationRequest:
    return GenerationRequest(**row["request"])


def fixed_shots(sft_train: Path) -> list[tuple[GenerationRequest, dict]]:
    """First standard and first stimulus example from SFT train, without a target misconception."""
    shots, want = [], ["standard", "stimulus"]
    for row in read_jsonl(sft_train):
        req = parse_user(row["messages"][1]["content"])
        if req.format in want and not req.target_misconception and not req.avoid:
            shots.append((req, json.loads(row["messages"][2]["content"])))
            want.remove(req.format)
        if not want:
            break
    return shots


def _free_mlx() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


# -- phase 1 ------------------------------------------------------------------------------------
def generate_arm(arm: str, prompts: list[dict], out_dir: Path = EVAL_DIR, limit: int | None = None) -> Path:
    from eduai.generation.generator import Generator
    from eduai.llm.mlx_backend import MLXBackend

    backend = MLXBackend(model_path("llama-3b"), ADAPTER if arm == "finetuned" else None)
    shots = fixed_shots(ROOT / "data" / "sft" / "train.jsonl") if arm == "base-2shot" else []
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
    out = out_dir / f"gen_{arm}.jsonl"
    write_jsonl(out, rows)
    (out_dir / f"gen_{arm}.meta.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "n": len(rows),
                "wall_seconds": round(time.time() - t0, 1),
                "shots": len(shots),
                "backend": backend.name,
            },
            indent=2,
        )
        + "\n"
    )
    return out


# -- phase 2 ------------------------------------------------------------------------------------
def reference_items(prompts: list[dict], data_dir: Path) -> dict[str, dict]:
    """SciQ source items as MCQs (same fixed letter shuffle), used as the ceiling row."""
    import random

    tagged = {d["id"]: d for d in read_jsonl(data_dir / "items_tagged.jsonl")}
    rng = random.Random(0)
    out = {}
    for p in prompts:
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
    prompts: list[dict], judge_key: str, out_dir: Path = EVAL_DIR, data_dir: Path = ROOT / "data"
) -> Path:
    from eduai.evaluation.solver import Judge
    from eduai.llm.mlx_backend import MLXBackend

    judge = Judge(MLXBackend(model_path(judge_key), None))
    passages = {p["id"]: p["request"]["passage"] for p in prompts}
    rows = []
    refs = reference_items(prompts, data_dir)
    for pid, item in refs.items():
        rows.append({"id": pid, "arm": "reference", **judge(item, passages[pid])})
    for arm in ARMS:
        path = out_dir / f"gen_{arm}.jsonl"
        if not path.exists():
            continue
        for g in read_jsonl(path):
            item, _ = parse(g["text"])
            if item is None or schema_errors(item):
                continue
            rows.append({"id": g["id"], "arm": arm, **judge(item, passages[g["id"]])})
    out = out_dir / f"judge_{judge_key}.jsonl"
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
    ref_align = tagger.is_aligned(
        [tag_text(refs[i]["stem"], refs[i]["choices"][refs[i]["answer"]]) for i in ids],
        [by_id[i]["request"]["lo_id"] for i in ids],
    )
    per_item["reference"] = [
        {
            "id": i,
            "aligned": a["aligned"],
            "key": judge[("reference", i)]["agrees"],
            "key_secondary": secondary.get(("reference", i), {}).get("agrees"),
            "key_letter": refs[i]["answer"],
        }
        for i, a in zip(ids, ref_align, strict=True)
    ]
    timing = {}
    for arm in ARMS:
        path = out_dir / f"gen_{arm}.jsonl"
        if not path.exists():
            continue
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
                    r["key"] = bool(j and j["agrees"])
                    s2 = secondary.get((arm, pid))
                    r["key_secondary"] = None if s2 is None else bool(s2["agrees"])
                    r["aligned"] = aligns[pid]["aligned"]
                    ok, info = novelty.check(
                        item,
                        exclude_ids=[f"sciq-{pid}"],
                        exclude_group=by_id[pid]["group"],
                        source_stem=by_id[pid]["reference"]["question"],
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
    for arm in ARMS:
        path = out_dir / f"gen_{arm}.jsonl"
        if not path.exists():
            continue
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
        "bootstrap": bootstrap_diffs(per_item),
        "judge": judge_key,
        "secondary_judge": secondary_key,
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


def bootstrap_diffs(per_item: dict[str, list[dict]], n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap over prompts: 95% CI of (arm A rate - arm B rate)."""
    rng = np.random.default_rng(seed)
    pairs = [
        ("finetuned", "base-0shot"),
        ("finetuned", "base-2shot"),
        ("finetuned", "reference"),
        ("base-2shot", "base-0shot"),
    ]
    out = {}
    for a, b in pairs:
        if a not in per_item or b not in per_item:
            continue
        for metric in ("usable", "all_checks", "aligned", "key", "json"):
            if metric not in per_item[a][0] or metric not in per_item[b][0]:
                continue
            xa = np.array([bool(r[metric]) for r in per_item[a]], float)
            xb = np.array([bool(r[metric]) for r in per_item[b]], float)
            d = xa - xb
            idx = rng.integers(0, len(d), size=(n_boot, len(d)))
            boots = d[idx].mean(axis=1)
            out[f"{a} - {b} | {metric}"] = {
                "diff": float(d.mean()),
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            }
    return out
