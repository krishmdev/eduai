"""Post-hoc answer-key verification with a stronger judge (docs/key_verification.md).

The verifier is the eval's `Judge` (open-book, next-token letter log-probabilities over the 4
cyclic rotations) on a bigger model from another family, Qwen3.5 9B with thinking off. An item is
verified when the verifier's top option is the key and the key's probability exceeds a fixed
threshold. Scores go to verify_<model>.jsonl next to the eval files, which are never rewritten;
reports/key_verification.{json,md} is a separate, exploratory report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from eduai.config import MODEL_PINS, ROOT, model_path
from eduai.data.sciq import read_jsonl, write_jsonl
from eduai.generation.validate import parse, schema_errors

THRESHOLD = 0.5  # pre-registered, not tuned
DEFAULT_VERIFIER = "qwen3.5-9b"
VERIFY_ARMS = ("base-2shot", "finetuned-v2", "finetuned-v2-2shot")
PAIRS = (("finetuned-v2", "base-2shot"), ("finetuned-v2-2shot", "base-2shot"))
SPLITS = {
    "valid": (ROOT / "reports" / "valid_eval" / "gen", ROOT / "data" / "eval" / "valid_prompts.jsonl"),
    "test": (ROOT / "reports" / "eval", ROOT / "data" / "eval" / "prompts.jsonl"),
}
AUDIT_DIR = ROOT / "reports" / "eval" / "audit_llm"
INDEPENDENT_AUDITOR = "gemma-4-12b"
# Same family as the verifier, so agreement with it is partly circular; reported as a note only.
CIRCULAR_AUDITOR = "qwen3.5-9b"
REPORT_JSON = ROOT / "reports" / "key_verification.json"
REPORT_MD = ROOT / "reports" / "key_verification.md"


def passes(judged: dict, threshold: float = THRESHOLD) -> bool:
    p = judged.get("p_key")
    return bool(judged.get("agrees")) and p is not None and p > threshold


class KeyVerifier:
    def __init__(self, judge, model_key: str, threshold: float = THRESHOLD):
        self.judge, self.model_key, self.threshold = judge, model_key, threshold

    def __call__(self, item: dict, passage: str | None = None) -> dict:
        j = self.judge(item, passage)
        return {**j, "verified": passes(j, self.threshold)}


def load_verifier(model_key: str = DEFAULT_VERIFIER, threshold: float = THRESHOLD) -> KeyVerifier:
    from eduai.evaluation.solver import Judge
    from eduai.llm.mlx_backend import MLXBackend

    pin = MODEL_PINS[model_key]
    backend = MLXBackend(model_path(model_key), None, chat_template_kwargs=dict(pin.template_kwargs))
    return KeyVerifier(Judge(backend), model_key, threshold)


def verify_path(eval_dir: Path, model_key: str) -> Path:
    return eval_dir / f"verify_{model_key}.jsonl"


def verify_split(
    verifier,
    prompts: list[dict],
    eval_dir: Path,
    arms: tuple[str, ...] = VERIFY_ARMS,
    force: bool = False,
    progress=None,
) -> Path:
    """Verify every schema-valid item of `arms` in eval_dir. Rows of other arms already in the file
    are kept. Each row is appended to a .partial file as it's scored, so a stopped run resumes."""
    model_key = verifier.model_key
    out = verify_path(eval_dir, model_key)
    partial = out.with_suffix(".partial.jsonl")
    old = read_jsonl(out) if out.exists() else []
    if not force and (clash := sorted({r["arm"] for r in old} & set(arms))):
        raise FileExistsError(f"{out} already has rows for {clash}; pass force to redo them")
    kept = [r for r in old if r["arm"] not in arms]
    done = {(r["arm"], r["id"]): r for r in (read_jsonl(partial) if partial.exists() else [])}
    passages = {p["id"]: p["request"]["passage"] for p in prompts}
    rows = []
    with open(partial, "a") as fh:
        for arm in arms:
            for g in read_jsonl(eval_dir / f"gen_{arm}.jsonl"):
                item, _ = parse(g["text"])
                if item is None or schema_errors(item):
                    continue
                row = done.get((arm, g["id"]))
                if row is None:
                    v = verifier(item, passages[g["id"]])
                    row = {
                        "id": g["id"],
                        "arm": arm,
                        "key": item["answer"],
                        "majority": v["majority"],
                        "p_key": v["p_key"],
                        "verified": v["verified"],
                        "duplicate_options": bool(v.get("duplicate_options")),
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                rows.append(row)
                if progress:
                    progress(arm, len(rows))
    write_jsonl(out, kept + rows)
    partial.unlink()
    return out


def promotable(per_item: list[dict], arm: str, verify: list[dict] | None) -> set[str]:
    """Bank-promotion gate: usable (every check passed, not a copy of its source) and, when `verify`
    is given, verified. Every usable item of the arm needs a verification row."""
    ids = {r["id"] for r in per_item if r["arm"] == arm and r.get("all_checks") and not r.get("source_copy")}
    if verify is None:
        return ids
    ver = {r["id"]: r["verified"] for r in verify if r["arm"] == arm}
    if missing := sorted(ids - set(ver)):
        raise ValueError(f"{len(missing)} usable {arm} items have no verification row, e.g. {missing[0]}")
    return {i for i in ids if ver[i]}


# -- post-hoc report -----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def split_summary(per_item: list[dict], verify: list[dict], arms: tuple[str, ...] = VERIFY_ARMS) -> dict:
    from eduai.evaluation.compare import bootstrap_diffs

    ver = {(r["arm"], r["id"]): r for r in verify}
    by_arm: dict[str, list[dict]] = {}
    for r in per_item:
        if r["arm"] not in arms:
            continue
        if r.get("schema") and (r["arm"], r["id"]) not in ver:
            raise ValueError(f"no verification row for {r['arm']} {r['id']}")
        v = ver.get((r["arm"], r["id"]))
        verified = bool(v and v["verified"])
        by_arm.setdefault(r["arm"], []).append(
            {"id": r["id"], "usable": bool(r["usable"]), "verified_usable": bool(r["usable"]) and verified}
        )
    arms_out = {}
    for arm in arms:
        rs = by_arm.get(arm, [])
        n, u = len(rs), sum(r["usable"] for r in rs)
        vu = sum(r["verified_usable"] for r in rs)
        scored = [v for (a, _), v in ver.items() if a == arm]
        arms_out[arm] = {
            "n": n,
            "usable": u,
            "verified_usable": vu,
            "usable_rate": u / n if n else None,
            "verified_usable_rate": vu / n if n else None,
            "survival": vu / u if u else None,
            "n_scored": len(scored),
            "verified_rate_on_scored": (sum(v["verified"] for v in scored) / len(scored)) if scored else None,
        }
    boot_in = {a: [{"id": r["id"], "usable": r["verified_usable"]} for r in rs] for a, rs in by_arm.items()}
    boot = bootstrap_diffs(boot_in, pairs=PAIRS)
    return {
        "arms": arms_out,
        "bootstrap_verified_usable": {k.replace("| usable", "| verified_usable"): v for k, v in boot.items()},
    }


def audit_crosscheck(verify_test: list[dict], audit_dir: Path, auditor: str) -> dict | None:
    """The auditor's key_correct on audited items, split by whether they pass verification."""
    path = audit_dir / f"llm_{auditor}" / "verdicts.jsonl"
    if not path.exists():
        return None
    from scipy.stats import fisher_exact

    key = json.loads((audit_dir / "key.json").read_text())
    ver = {(r["arm"], r["id"]): r for r in verify_test}
    groups: dict[str, list[bool]] = {"pass": [], "fail": []}
    by_arm: dict[str, dict[str, list[bool]]] = {}
    unparsed = 0
    for v in read_jsonl(path):
        if not v.get("parsed") or v.get("key_correct") is None:
            unparsed += 1
            continue
        k = key[v["audit_id"]]
        g = "pass" if ver[(k["arm"], k["id"])]["verified"] else "fail"
        groups[g].append(bool(v["key_correct"]))
        by_arm.setdefault(k["arm"], {"pass": [], "fail": []})[g].append(bool(v["key_correct"]))

    def cell(xs: list[bool]) -> dict:
        k = sum(xs)
        return {
            "n": len(xs),
            "key_correct": k,
            "rate": k / len(xs) if xs else None,
            "ci95": wilson(k, len(xs)),
        }

    p, f = groups["pass"], groups["fail"]
    out = {
        "auditor": auditor,
        "n_judged": len(p) + len(f),
        "n_unparsed": unparsed,
        "pass": cell(p),
        "fail": cell(f),
        "all": cell(p + f),
        "by_arm": {a: {g: cell(xs) for g, xs in d.items()} for a, d in sorted(by_arm.items())},
    }
    if p and f:
        out["diff"] = sum(p) / len(p) - sum(f) / len(f)
        table = [[sum(p), len(p) - sum(p)], [sum(f), len(f) - sum(f)]]
        out["fisher_p"] = float(fisher_exact(table, alternative="two-sided").pvalue)
    return out


def build_report(
    model_key: str = DEFAULT_VERIFIER, splits: dict = SPLITS, audit_dir: Path = AUDIT_DIR
) -> dict:
    res: dict = {
        "post_hoc": True,
        "exploratory": True,
        "preregistration": "docs/key_verification.md",
        "verifier": model_key,
        "revision": MODEL_PINS[model_key].revision,
        "threshold": THRESHOLD,
        "arms": list(VERIFY_ARMS),
        "splits": {},
    }
    verify_test = None
    for split, (eval_dir, _) in splits.items():
        vp = verify_path(eval_dir, model_key)
        if not vp.exists():
            continue
        verify = read_jsonl(vp)
        res["splits"][split] = split_summary(read_jsonl(eval_dir / "per_item.jsonl"), verify)
        if split == "test":
            verify_test = verify
    if verify_test is not None:
        res["audit_independent"] = audit_crosscheck(verify_test, audit_dir, INDEPENDENT_AUDITOR)
        res["audit_circular_note"] = audit_crosscheck(verify_test, audit_dir, CIRCULAR_AUDITOR)
    return res


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _ci(c) -> str:
    return "n/a" if c is None else f"{c[0]:.0%} to {c[1]:.0%}"


def render(res: dict) -> str:
    from eduai.evaluation.report import label

    lines = [
        "# Post-hoc: answer-key verification (exploratory)",
        "",
        f"Pre-registered in `{res['preregistration']}` after the test audit, so this is post-hoc and "
        "exploratory. The registered eval result in `reports/eval_report.md` is unchanged.",
        "",
        f"Verifier: `{res['verifier']}` (revision {res['revision'][:7]}, thinking off, open-book, 4 "
        f"rotations). An item is verified when the verifier's top option is the key and p(key) > "
        f"{res['threshold']}. Verified usable = usable (the eval's criterion) and verified.",
        "",
    ]
    for split, s in res["splits"].items():
        lines += [
            f"## {split.capitalize()} split",
            "",
            "| Arm | n | Usable | Verified usable | Usable items that survive | Verified, of schema-valid |",
            "|---|---|---|---|---|---|",
        ]
        for arm, a in s["arms"].items():
            lines.append(
                f"| {label(arm)} | {a['n']} | {_pct(a['usable_rate'])} ({a['usable']}) | "
                f"{_pct(a['verified_usable_rate'])} ({a['verified_usable']}) | {_pct(a['survival'])} | "
                f"{_pct(a['verified_rate_on_scored'])} of {a['n_scored']} |"
            )
        lines += ["", "Paired bootstrap over prompts, verified usable (descriptive):", ""]
        for k, v in s["bootstrap_verified_usable"].items():
            pair = k.split(" | ")[0]
            lines.append(
                f"- {pair}: {v['diff'] * 100:+.1f} points (95% CI {v['ci95'][0] * 100:+.1f} to "
                f"{v['ci95'][1] * 100:+.1f})"
            )
        lines.append("")

    def audit_table(c: dict) -> list[str]:
        out = [
            "| Verification | Items | Key judged correct | 95% CI |",
            "|---|---|---|---|",
        ]
        for g, name in (("pass", "Pass"), ("fail", "Fail"), ("all", "All audited")):
            x = c[g]
            out.append(f"| {name} | {x['n']} | {_pct(x['rate'])} ({x['key_correct']}) | {_ci(x['ci95'])} |")
        return out

    if "audit_independent" in res:
        c = res["audit_independent"]
        lines += ["## Independent check: Gemma 4 12B audit subset (test)", ""]
        if c is None:
            lines += ["The Gemma verdicts are not available yet.", ""]
        else:
            lines += [
                f"Gemma 4 12B (thinking on) judged {c['n_judged']} usable test items blind "
                f"({c['n_unparsed']} without a parsed verdict are left out). Its key verdicts, split by "
                "whether the item passes verification:",
                "",
                *audit_table(c),
                "",
            ]
            if "diff" in c:
                lines += [
                    f"Pass minus fail: {c['diff'] * 100:+.1f} points (Fisher exact two-sided p = "
                    f"{c['fisher_p']:.3f}).",
                    "",
                ]
    if res.get("audit_circular_note"):
        c = res["audit_circular_note"]
        lines += [
            "## Note: Qwen3.5 9B audit (same family as the verifier, circular)",
            "",
            f"On the {c['n_judged']} items of the Qwen audit, key judged correct was "
            f"{_pct(c['pass']['rate'])} of {c['pass']['n']} verified items and {_pct(c['fail']['rate'])} of "
            f"{c['fail']['n']} unverified ones. The verifier and this auditor are the same model family, "
            "so this isn't evidence that verification works.",
            "",
        ]
    lines += [
        "The auditors are language models, not people; their verdicts are a second machine opinion.",
        "",
    ]
    return "\n".join(lines)


def write_report(res: dict, json_path: Path = REPORT_JSON, md_path: Path = REPORT_MD) -> None:
    json_path.write_text(json.dumps(res, indent=2) + "\n")
    md_path.write_text(render(res))
