"""Single source of truth for the generation prompt.

Both the SFT builder (training data) and every inference path (MLX, Ollama, eval) build their
messages through `build_messages`, and tests/test_prompts.py enforces that nothing else
constructs a system or user prompt for item generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You write AP-style multiple-choice science questions for high school students. "
    "Each question has exactly four options labeled A-D and one correct answer. "
    "Base the question and the explanation on the source passage. Distractors must be plausible "
    "but clearly wrong according to the passage. Never use 'all of the above' or 'none of the above'. "
    "Reply with a single JSON object and nothing else."
)

OUTPUT_KEYS = ("stem", "choices", "answer", "explanation", "lo_id", "difficulty")
FORMATS = ("standard", "stimulus")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class GenerationRequest:
    subject: str
    unit: str
    topic: str
    lo_id: str
    lo_text: str
    difficulty: str
    format: str
    passage: str
    target_misconception: str | None = None
    avoid: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}")


def render_user(req: GenerationRequest) -> str:
    lines = [
        f"Subject: {req.subject}",
        f"Unit: {req.unit}",
        f"Topic: {req.topic}",
        f"Learning objective ({req.lo_id}): {req.lo_text}",
        f"Difficulty: {req.difficulty}",
        f"Format: {req.format}",
    ]
    if req.format == "stimulus":
        lines.append(
            "Include a short 'stimulus' excerpt from the passage and write a stem that asks about it."
        )
    if req.target_misconception:
        lines.append(
            f"Target misconception: include '{req.target_misconception}' as one of the wrong options."
        )
    if req.avoid:
        lines.append("Do not repeat these existing questions:")
        lines += [f"- {a}" for a in req.avoid]
    keys = (
        '"stem", '
        + ('"stimulus", ' if req.format == "stimulus" else "")
        + ('"choices" (object with keys A-D), "answer" (one letter), "explanation", "lo_id", "difficulty"')
    )
    lines += ["", "Source passage:", req.passage.strip(), "", f"Return JSON with keys {keys}."]
    return "\n".join(lines)


def build_messages(
    req: GenerationRequest, completion: dict | None = None, shots: list[tuple] | None = None
) -> list[dict]:
    """Chat messages for one generation. `shots` are (request, completion) pairs for few-shot."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_req, shot_out in shots or []:
        msgs.append({"role": "user", "content": render_user(shot_req)})
        msgs.append({"role": "assistant", "content": render_completion(shot_out)})
    msgs.append({"role": "user", "content": render_user(req)})
    if completion is not None:
        msgs.append({"role": "assistant", "content": render_completion(completion)})
    return msgs


def render_completion(item: dict) -> str:
    ordered = {"stem": item["stem"]}
    if item.get("stimulus"):
        ordered["stimulus"] = item["stimulus"]
    ordered.update({k: item[k] for k in OUTPUT_KEYS if k != "stem"})
    return json.dumps(ordered, ensure_ascii=False)


SOLVER_SYSTEM = (
    "You are taking a science test. Read the question and reply with only the letter of the best answer."
)


def build_solver_messages(stem: str, choices: dict[str, str], stimulus: str | None = None) -> list[dict]:
    body = (f"{stimulus.strip()}\n\n" if stimulus else "") + stem.strip() + "\n"
    body += "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
    body += "\nAnswer:"
    return [{"role": "system", "content": SOLVER_SYSTEM}, {"role": "user", "content": body}]
