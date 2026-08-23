#!/usr/bin/env bash
# OpenStax OOD check (docs/openstax_ood.md): base 2-shot and the v2 headline arm on
# data/openstax/prompts.jsonl, judged by llama-1b only, then scored per subject. Pre-registered to
# run once, so it refuses to start if generations already exist. Wrap it in the compute lease:
#   ~/Developer/portfolio/.tools/compute_lease.py run eduai-openstax -- scripts/openstax_ood.sh
set -euo pipefail
cd "$(dirname "$0")/.."
out=reports/openstax_ood
if compgen -G "$out/gen/gen_*.jsonl" >/dev/null; then
  echo "$out/gen already has generations; this check runs once" >&2
  exit 1
fi
headline=$(uv run python scripts/openstax_report.py headline)
echo "arms: base-2shot $headline"
PROMPTS=data/openstax/prompts.jsonl OUT=$out JUDGES=llama-1b ARMS="base-2shot $headline" \
  MANIFEST_TAG=openstax RUN_MANIFEST_TOOL=${RUN_MANIFEST_TOOL:-$HOME/Developer/portfolio/.tools/run_manifest.py} \
  scripts/run_eval.sh
HF_HOME="$PWD/.models" HF_HUB_OFFLINE=1 uv run python scripts/openstax_report.py score
