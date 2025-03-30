#!/usr/bin/env bash
# Generation + judging for the eval, one model per process. Run it on an otherwise idle machine
# (timings are recorded); `make eval RUN_WRAPPER=...` can prefix a job queue. RUN_MANIFEST_TOOL optionally points at a script
# that records host state as JSON.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.models" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p reports/eval
tool=${RUN_MANIFEST_TOOL:-}
[[ -n "$tool" && -x "$tool" ]] && python3 "$tool" --out reports/eval_manifest.json task=eval phase=start device=mps \
  model=mlx-community/Llama-3.2-3B-Instruct-4bit judge=mlx-community/Llama-3.2-1B-Instruct-4bit
start=$(date +%s)
for arm in ${ARMS:-base-0shot base-2shot finetuned}; do
  uv run eduai eval generate "$arm" ${LIMIT:+--limit $LIMIT}
done
uv run eduai eval judge --judge llama-1b
uv run eduai eval judge --judge llama-3b
end=$(date +%s)
[[ -n "$tool" && -x "$tool" ]] && python3 "$tool" --out reports/eval_manifest_end.json task=eval wall_seconds=$((end - start))
echo "eval generation+judging took $((end - start)) s"
