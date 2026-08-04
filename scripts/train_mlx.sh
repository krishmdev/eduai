#!/usr/bin/env bash
# Run an MLX LoRA fine-tune and record the log, a run manifest, and parsed metrics.
# Usage: scripts/train_mlx.sh <config.yaml> [--pilot] [extra mlx_lm.lora args...]
# Run it on an otherwise idle machine; timing and peak memory are recorded.
# RUN_NAME sets the report file prefix (default "training", the v1 run), so a v2 run doesn't
# overwrite the v1 log: RUN_NAME=training_v2 scripts/train_mlx.sh configs/lora_llama32_3b_v2.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
config="$1"; shift
name="${RUN_NAME:-training}"
extra=()
if [[ "${1:-}" == "--pilot" ]]; then
  shift
  name="pilot"
  extra=(--iters 20 --steps-per-report 5 --steps-per-eval 1000 --val-batches 2 --save-every 1000
         --adapter-path adapters/pilot)
fi
mkdir -p reports adapters
log="reports/${name}_log.txt"
manifest_tool=${RUN_MANIFEST_TOOL:-}
if [[ -n "$manifest_tool" && -x "$manifest_tool" ]]; then
  python3 "$manifest_tool" --out "reports/${name}_manifest_start.json" config="$config" phase="$name" \
    device=mps dtype=4bit-base+lora-fp16 model="$(grep '^model:' "$config" | awk '{print $2}')"
fi
start=$(date +%s)
set +e
uv run python -m mlx_lm lora -c "$config" ${extra[@]+"${extra[@]}"} "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
end=$(date +%s)
echo "wall_seconds=$((end - start)) exit=$status" | tee -a "$log"
if [[ -n "$manifest_tool" && -x "$manifest_tool" ]]; then
  python3 "$manifest_tool" --out "reports/${name}_manifest_end.json" config="$config" phase="$name" \
    wall_seconds="$((end - start))" exit="$status"
fi
uv run python scripts/parse_mlx_log.py "$log" --config "$config" --out "reports/${name}.json"
exit "$status"
