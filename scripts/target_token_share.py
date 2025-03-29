"""What the SFT loss is spent on: share of completion tokens per JSON field, and the share of
completion tokens that sit inside a span copied verbatim from the prompt (source passage,
learning objective, misconception, etc.). Tokenizer only; no model is loaded."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOK = (
    ROOT
    / ".models/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e/tokenizer.json"
)
_W = re.compile(r"\S+")


def main(path: str, out: str) -> None:
    tok = Tokenizer.from_file(str(TOK))
    fields: Counter = Counter()
    copied = total = 0
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    for r in rows:
        user, comp = r["messages"][1]["content"], r["messages"][2]["content"]
        d = json.loads(comp)
        cat = ["json_syntax"] * len(comp)
        values = {
            "stem": d["stem"],
            "stimulus": d.get("stimulus"),
            "explanation": d["explanation"],
            "lo_id": d["lo_id"],
            "difficulty": d["difficulty"],
            "answer": d["answer"],
        }
        for name, val in [*values.items(), *(("choices", v) for v in d["choices"].values())]:
            if not val:
                continue
            s = json.dumps(val, ensure_ascii=False)[1:-1]
            i = comp.find('"' + s + '"')
            if i >= 0:
                cat[i + 1 : i + 1 + len(s)] = [name] * len(s)
        # characters inside a run of >= 4 words that also appears verbatim in the prompt
        in_copy = [False] * len(comp)
        words = [(m.start(), m.end(), m.group()) for m in _W.finditer(comp)]
        for k in range(len(words) - 3):
            a, b = words[k][0], words[k + 3][1]
            if comp[a:b].strip('",{}') in user:
                for j in range(a, b):
                    in_copy[j] = True
        enc = tok.encode(comp, add_special_tokens=False)
        for a, b in enc.offsets:
            if b <= a:
                continue
            fields[Counter(cat[a:b]).most_common(1)[0][0]] += 1
            copied += any(in_copy[a:b])
            total += 1
    res = {
        "rows": len(rows),
        "completion_tokens": total,
        "tokens_per_row": round(total / len(rows), 1),
        "field_share": {k: round(v / total, 4) for k, v in fields.most_common()},
        "copied_from_prompt_share": round(copied / total, 4),
        "note": "copied = token inside a >=4-word span that appears verbatim in the prompt",
    }
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
