#!/usr/bin/env bash
# One LLM judge for the blind audit (docs/blind_audit_llm.md): start a Localhost AI server for the
# preset, fill the audit sheet, stop the server. Localhost AI is a separate local OpenAI-compatible
# inference server (MLX backend) with pinned model presets; any server that serves the preset's model
# at /v1 and answers /readyz would do. Run the whole script under the GPU lock you use (this project
# used a local compute-lease wrapper) so it is held for the server's lifetime, one judge at a time:
#   <lease-wrapper> run eduai-openstax -- \
#     scripts/llm_audit_run.sh qwen3.5-9b-mlx4 qwen3.5-9b reports/valid_eval/audit_llm
# Env: LHAI_DIR (Localhost AI checkout, default ../localhost-ai next to this repo), PORT (8011), CONCURRENCY (4), THINKING_BUDGET (3000),
# MAX_TOKENS (3600), READY_TIMEOUT seconds (1200), LHAI_MAX_CONTEXT (9216; the server default of
# 2048 is too short for prompt + thinking + answer). MODE=pilot runs the thinking-length pilot
# (blind_audit.py pilot, safety cap 6000) instead of the audit. PER_ARM=N judges only the first N
# items of each arm in sheet order.
set -euo pipefail
cd "$(dirname "$0")/.."
preset=$1 judge=$2 audit=$3
lhai=${LHAI_DIR:-$(cd .. && pwd)/localhost-ai}
port=${PORT:-8011}
log="$audit/llm_${judge}_server.log"
mkdir -p "$audit"
[[ -f "$audit/sheet.jsonl" ]] || { echo "no $audit/sheet.jsonl; run blind_audit.py draw first" >&2; exit 1; }
mode=${MODE:-audit}
if [[ $mode == pilot ]]; then
  [[ -f "$audit/pilot_${judge}.json" ]] && { echo "$judge pilot already in $audit" >&2; exit 1; }
else
  [[ -f "$audit/llm_${judge}/verdicts.jsonl" ]] && { echo "$judge already judged $audit" >&2; exit 1; }
fi
export LHAI_MAX_CONTEXT=${LHAI_MAX_CONTEXT:-9216}

(cd "$lhai" && LHAI_MODEL="$preset" exec uv run --frozen lhai serve --port "$port") >"$log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null || true' EXIT
deadline=$(( $(date +%s) + ${READY_TIMEOUT:-1200} ))
until curl -sf "http://127.0.0.1:$port/readyz" >/dev/null; do
  kill -0 "$server" 2>/dev/null || { echo "server exited; see $log" >&2; exit 1; }
  (( $(date +%s) < deadline )) || { echo "server not ready in time; see $log" >&2; exit 1; }
  sleep 5
done
if [[ $mode == pilot ]]; then
  uv run python scripts/blind_audit.py pilot --out "$audit" --judge "$judge" --model "$preset" \
    --base-url "http://127.0.0.1:$port/v1" --concurrency "${CONCURRENCY:-4}" \
    --models-yaml "$lhai/models.yaml" --server-repo "$lhai" ${PER_ARM:+--per-arm "$PER_ARM"}
  exit
fi
uv run python scripts/blind_audit.py llm --out "$audit" --judge "$judge" --model "$preset" \
  --base-url "http://127.0.0.1:$port/v1" --thinking --thinking-budget "${THINKING_BUDGET:-3000}" \
  --max-tokens "${MAX_TOKENS:-3600}" --concurrency "${CONCURRENCY:-4}" \
  --models-yaml "$lhai/models.yaml" --server-repo "$lhai" ${PER_ARM:+--per-arm "$PER_ARM"}
