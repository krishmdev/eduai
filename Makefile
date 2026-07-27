PY := uv run
# Offline targets must not let uv touch the network (no sync, no index lookups).
PYOFF := uv run --frozen --offline
SCIQ_DIR ?= $(HOME)/Downloads/SciQ dataset-2 3
EMBED_MODELS := bge-small minilm rerank
# Optional command prefix for heavy runs (training, eval), e.g. a job queue. Empty by default.
RUN_WRAPPER ?=
PORT ?= 8001

export HF_HOME := $(CURDIR)/.models
export TOKENIZERS_PARALLELISM := false

.PHONY: setup-demo setup models models-llm lint test data tagger-eval pilot train sim eval eval-score eval-check demo e2e-offline egress-open-check serve clean

# Bank-only demo needs the core application dependencies, not embedding or LLM extras.
setup-demo:
	uv sync --frozen --no-default-groups

setup:
	uv sync --all-extras --frozen
	$(MAKE) models

models:
	$(PY) eduai models fetch $(EMBED_MODELS)
	$(PY) eduai models verify $(EMBED_MODELS)

models-llm:
	$(PY) eduai models fetch llama-3b llama-1b
	$(PY) eduai models verify llama-3b llama-1b

lint:
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .

test:
	uv run --extra dev pytest

data:
	$(PY) eduai data build --sciq "$(SCIQ_DIR)" --out data

tagger-eval:
	$(PY) eduai tagger eval --gold data/gold/tagging_gold.jsonl --out reports/tagger_eval.md

pilot:
	$(RUN_WRAPPER) scripts/train_mlx.sh configs/lora_llama32_3b.yaml --pilot

train:
	$(RUN_WRAPPER) scripts/train_mlx.sh configs/lora_llama32_3b.yaml

sim:
	$(PY) eduai simulate --students 500 --out reports

eval:
	$(RUN_WRAPPER) scripts/run_eval.sh
	$(MAKE) eval-score

eval-score:
	$(PY) eduai eval score

eval-check:
	$(PYOFF) python scripts/check_eval_snapshot.py

demo:
	EDUAI_BACKEND=bank $(PYOFF) eduai serve --port $(PORT)

e2e-offline:
	$(PYOFF) python scripts/e2e_offline.py --port $(PORT)

# Companion to e2e-offline: the same canary must connect when not sandboxed, or the check is vacuous.
egress-open-check:
	$(PYOFF) python -m eduai.egress open

serve:
	$(PY) eduai serve --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data/eduai.db
