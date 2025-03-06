from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

LETTERS = ("A", "B", "C", "D")

ITEM_JSON_SCHEMA = {
    "type": "object",
    "required": ["stem", "choices", "answer", "explanation", "lo_id", "difficulty"],
    "properties": {
        "stem": {"type": "string", "minLength": 10, "maxLength": 600},
        "stimulus": {"type": "string", "maxLength": 1500},
        "choices": {
            "type": "object",
            "required": list(LETTERS),
            "additionalProperties": False,
            "properties": {k: {"type": "string", "minLength": 1, "maxLength": 200} for k in LETTERS},
        },
        "answer": {"enum": list(LETTERS)},
        "explanation": {"type": "string", "minLength": 10, "maxLength": 1200},
        "lo_id": {"type": "string"},
        "difficulty": {"enum": ["easy", "medium", "hard"]},
    },
    "additionalProperties": False,
}


class Item(BaseModel):
    id: str
    stem: str
    choices: dict[str, str]
    answer: str
    explanation: str | None = None
    stimulus: str | None = None
    lo_id: str
    subject: str
    unit_id: str
    difficulty: str
    b: float = 0.0
    source: str = "sciq"
    grounded: bool = True
    aligned: bool = True
    misconceptions: dict[str, str] = Field(default_factory=dict)

    @field_validator("choices")
    @classmethod
    def _four_choices(cls, v: dict[str, str]) -> dict[str, str]:
        if sorted(v) != list(LETTERS):
            raise ValueError("choices must have keys A-D")
        return v

    @field_validator("answer")
    @classmethod
    def _answer_letter(cls, v: str) -> str:
        if v not in LETTERS:
            raise ValueError("answer must be A-D")
        return v
