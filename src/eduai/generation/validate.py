"""Validation checks for a generated item, run in order. The first failure is the rejection reason."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import jsonschema
from rapidfuzz import fuzz

from eduai.prompts import GenerationRequest
from eduai.schema import ITEM_JSON_SCHEMA, LETTERS

REASONS = (
    "no_json",
    "bad_json",
    "schema",
    "structure",
    "misconception_missing",
    "key_disagreement",
    "not_aligned",
    "duplicate",
)
BANNED = re.compile(r"\b(all|none|both) of the (above|these|following)\b", re.I)
_VALIDATOR = jsonschema.Draft202012Validator(ITEM_JSON_SCHEMA)


@dataclass
class CheckResult:
    ok: bool
    reason: str | None = None
    detail: str = ""
    item: dict | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def extract_json(text: str) -> str | None:
    """First balanced {...} block, ignoring braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse(text: str) -> tuple[dict | None, str | None]:
    block = extract_json(text)
    if block is None:
        return None, "no_json"
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(obj, dict):
        return None, "bad_json"
    return obj, None


def schema_errors(item: dict) -> list[str]:
    return [e.message for e in _VALIDATOR.iter_errors(item)]


def structure_problems(item: dict, req: GenerationRequest | None = None) -> list[str]:
    problems = []
    choices = item["choices"]
    texts = [choices[k].strip() for k in LETTERS]
    for i in range(4):
        for j in range(i + 1, 4):
            if fuzz.ratio(texts[i].lower(), texts[j].lower()) >= 90:
                problems.append(f"options {LETTERS[i]} and {LETTERS[j]} are near-identical")
    if any(BANNED.search(t) for t in texts):
        problems.append("uses all/none of the above")
    key = choices[item["answer"]].strip().lower()
    stem = item["stem"].lower()
    if len(key) > 3 and re.search(rf"\b{re.escape(key)}\b", stem):
        problems.append("stem gives away the answer")
    if not (10 <= len(item["stem"]) <= 600):
        problems.append("stem length out of bounds")
    if any(len(t) == 0 or len(t) > 200 for t in texts):
        problems.append("option length out of bounds")
    if req is not None:
        if item["lo_id"] != req.lo_id:
            problems.append(f"lo_id {item['lo_id']} != requested {req.lo_id}")
        if req.format == "stimulus" and not item.get("stimulus"):
            problems.append("stimulus format without a stimulus")
    return problems


def misconception_present(item: dict, target: str | None) -> bool:
    if not target:
        return True
    wrong = [item["choices"][k] for k in LETTERS if k != item["answer"]]
    return any(fuzz.ratio(target.lower(), w.lower()) >= 85 for w in wrong)


def validate_item(
    text: str,
    req: GenerationRequest,
    *,
    solver: Callable[[dict], dict] | None = None,
    aligner: Callable[[dict, GenerationRequest], tuple[bool, dict]] | None = None,
    novelty: Callable[[dict], tuple[bool, dict]] | None = None,
    verifier: Callable[[dict], dict] | None = None,
) -> CheckResult:
    """Run every check in order; stop at the first failure but record which checks ran. The
    answer-key verifier (the slowest check) runs last."""
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    item, err = parse(text)
    checks["json"] = item is not None
    if item is None:
        return CheckResult(False, err, checks=checks)
    errors = schema_errors(item)
    checks["schema"] = not errors
    if errors:
        return CheckResult(False, "schema", "; ".join(errors[:3]), item, checks)
    problems = structure_problems(item, req)
    checks["structure"] = not problems
    if problems:
        return CheckResult(False, "structure", "; ".join(problems), item, checks)
    checks["misconception"] = misconception_present(item, req.target_misconception)
    if not checks["misconception"]:
        return CheckResult(False, "misconception_missing", req.target_misconception or "", item, checks)
    if solver is not None:
        sol = solver(item)
        metrics.update({k: v for k, v in sol.items() if isinstance(v, int | float)})
        checks["key"] = bool(sol["agrees"])
        if not checks["key"]:
            return CheckResult(
                False, "key_disagreement", f"solver chose {sol.get('majority')}", item, checks, metrics
            )
    if aligner is not None:
        ok, info = aligner(item, req)
        metrics.update(info)
        checks["aligned"] = ok
        if not ok:
            return CheckResult(False, "not_aligned", str(info), item, checks, metrics)
    if novelty is not None:
        ok, info = novelty(item)
        metrics.update(info)
        checks["novel"] = ok
        if not ok:
            return CheckResult(
                False, "duplicate", f"max cosine {info.get('max_bank_cos', 0):.3f}", item, checks, metrics
            )
    if verifier is not None:
        ver = verifier(item)
        if ver.get("p_key") is not None:
            metrics["verifier_p_key"] = float(ver["p_key"])
        checks["verified"] = bool(ver["verified"])
        if not checks["verified"]:
            return CheckResult(
                False, "key_unverified", f"verifier chose {ver.get('majority')}", item, checks, metrics
            )
    return CheckResult(True, None, "", item, checks, metrics)
