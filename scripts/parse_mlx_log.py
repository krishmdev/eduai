"""Parse an mlx_lm.lora log into JSON: loss curves, throughput, peak memory, wall time."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRAIN = re.compile(
    r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+), It/sec ([\d.]+), "
    r"Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB"
)
VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+), Val took ([\d.]+)s")
TEST = re.compile(r"Test loss ([\d.]+), Test ppl ([\d.]+)")
TRAINABLE = re.compile(r"Trainable parameters: ([\d.]+)% \(([\d.]+)M/([\d.]+)M\)")
WALL = re.compile(r"wall_seconds=(\d+) exit=(\d+)")


def parse(text: str) -> dict:
    train = [
        {
            "iter": int(m[1]),
            "loss": float(m[2]),
            "lr": float(m[3]),
            "it_per_sec": float(m[4]),
            "tokens_per_sec": float(m[5]),
            "trained_tokens": int(m[6]),
            "peak_mem_gb": float(m[7]),
        }
        for m in TRAIN.finditer(text)
    ]
    val = [{"iter": int(m[1]), "loss": float(m[2]), "seconds": float(m[3])} for m in VAL.finditer(text)]
    out: dict = {"train": train, "val": val}
    if t := TEST.search(text):
        out["test"] = {"loss": float(t[1]), "ppl": float(t[2])}
    if t := TRAINABLE.search(text):
        out["trainable_params_m"] = float(t[2])
        out["total_params_m"] = float(t[3])
        out["trainable_pct"] = float(t[1])
    if t := WALL.search(text):
        out["wall_seconds"] = int(t[1])
        out["exit"] = int(t[2])
    if train:
        steady = train[1:] or train
        out["peak_mem_gb"] = max(r["peak_mem_gb"] for r in train)
        out["mean_it_per_sec"] = round(sum(r["it_per_sec"] for r in steady) / len(steady), 4)
        out["mean_tokens_per_sec"] = round(sum(r["tokens_per_sec"] for r in steady) / len(steady), 1)
        out["trained_tokens"] = train[-1]["trained_tokens"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--project-iters", type=int, default=600)
    args = ap.parse_args()
    data = parse(Path(args.log).read_text())
    data["log"] = args.log
    data["config"] = args.config
    if data.get("mean_it_per_sec"):
        data["projected_hours_for_iters"] = {
            "iters": args.project_iters,
            "hours": round(args.project_iters / data["mean_it_per_sec"] / 3600, 2),
            "note": "estimate from pilot throughput, excludes validation passes",
        }
    Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k not in ("train", "val")}, indent=2))


if __name__ == "__main__":
    main()
