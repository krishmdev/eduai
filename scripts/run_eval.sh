#!/usr/bin/env bash
# Generation + judging for the eval, one model per process. Run it on an otherwise idle machine
# (timings are recorded); `make eval RUN_WRAPPER=...` can prefix a job queue. RUN_MANIFEST_TOOL optionally points at a script
# that records host state as JSON.
#
# Defaults reproduce the test-set eval. Model selection uses the valid split instead:
#   PROMPTS=data/eval/valid_prompts.jsonl OUT=reports/valid_eval JUDGES=llama-1b ARMS="..." scripts/run_eval.sh
# JUDGE_ARMS limits judging to those arms and keeps the other arms' judge rows.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.models" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
prompts=${PROMPTS:-data/eval/prompts.jsonl}
out=${OUT:-reports}
if [[ "$out" == "reports" ]]; then eval_dir=reports/eval; else eval_dir="$out/gen"; fi
mkdir -p "$eval_dir"
tool=${RUN_MANIFEST_TOOL:-}
tag=${MANIFEST_TAG:-eval}
# Don't overwrite another run's manifest (the default tag is the v1 test run's): pick a new MANIFEST_TAG.
if [[ -n "$tool" && -x "$tool" && -e "$out/${tag}_manifest.json" && -z "${FORCE_MANIFEST:-}" ]]; then
  echo "$out/${tag}_manifest.json exists; set MANIFEST_TAG to a new tag (or FORCE_MANIFEST=1)" >&2
  exit 1
fi
# The v2 named arms must use the pinned adapter (the 300-iter checkpoint chosen on valid).
adapter_pin=()
if [[ " ${ARMS:-} " == *" finetuned-v2"* ]]; then
  shasum -a 256 -c configs/adapter_v2.sha256 >&2
  adapter_pin=(adapter_v2_sha256="$(awk '{printf "%s%s=%s", sep, $2, $1; sep=","}' configs/adapter_v2.sha256)")
fi
[[ -n "$tool" && -x "$tool" ]] && python3 "$tool" --out "$out/${tag}_manifest.json" task=eval phase=start device=mps \
  ${adapter_pin[@]+"${adapter_pin[@]}"} \
  prompts="$prompts" arms="${ARMS:-base-0shot base-2shot finetuned}" \
  model=mlx-community/Llama-3.2-3B-Instruct-4bit judge=mlx-community/Llama-3.2-1B-Instruct-4bit
start=$(date +%s)
# Every test-split run is logged before it starts, so crashed attempts count too (reports/eval/test_runs.jsonl).
ledger="$eval_dir/test_runs.jsonl"
if [[ "$out" == "reports" ]]; then
  printf '{"event": "start", "at": "%s", "arms": "%s", "judges": "%s", "tag": "%s"}\n' \
    "$(date +%Y-%m-%dT%H:%M:%S%z)" "${ARMS:-base-0shot base-2shot finetuned}" "${JUDGES:-llama-1b llama-3b}" "$tag" >> "$ledger"
fi
for arm in ${ARMS:-base-0shot base-2shot finetuned}; do
  # NAME=ADAPTER_DIR[:shots] defines a sweep arm; plain names are the fixed arms.
  if [[ "$arm" == *=* ]]; then
    name=${arm%%=*}; spec=${arm#*=}; shots=--no-shots
    [[ "$spec" == *:shots ]] && { shots=--shots; spec=${spec%:shots}; }
    uv run eduai eval generate "$name" --prompts "$prompts" --eval-dir "$eval_dir" --adapter "$spec" $shots \
      ${LIMIT:+--limit $LIMIT}
  else
    uv run eduai eval generate "$arm" --prompts "$prompts" --eval-dir "$eval_dir" ${LIMIT:+--limit $LIMIT}
  fi
done
judge_args=()
for a in ${JUDGE_ARMS:-}; do judge_args+=(--arm "$a"); done
for judge in ${JUDGES:-llama-1b llama-3b}; do
  uv run eduai eval judge --judge "$judge" --prompts "$prompts" --eval-dir "$eval_dir" ${judge_args[@]+"${judge_args[@]}"}
done
end=$(date +%s)
if [[ "$out" == "reports" ]]; then
  printf '{"event": "finish", "at": "%s", "tag": "%s", "wall_seconds": %d}\n' \
    "$(date +%Y-%m-%dT%H:%M:%S%z)" "$tag" "$((end - start))" >> "$ledger"
fi
[[ -n "$tool" && -x "$tool" ]] && python3 "$tool" --out "$out/${tag}_manifest_end.json" task=eval wall_seconds=$((end - start))
echo "eval generation+judging took $((end - start)) s"
