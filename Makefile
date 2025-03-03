PY := uv run
SCIQ_DIR ?= $(HOME)/Downloads/SciQ dataset-2 3
EMBED_MODELS := bge-small minilm rerank
LEASE := <local> run eduai-train --
OFFLINE ?= $(firstword $(wildcard ../.tools/offline-run))
PORT ?= 8001

export HF_HOME := $(CURDIR)/.models
export TOKENIZERS_PARALLELISM := false

.PHONY: setup models models-llm lint test data sft tagger-eval pilot train sim bank eval report demo e2e-offline serve clean

setup:
	uv sync --all-extras --frozen
	$(MAKE) models

models:
	$(PY) eduai models fetch $(EMBED_MODELS)
	$(PY) eduai models verify $(EMBED_MODELS)

models-llm:
	$(PY) eduai models fetch llama-3b
	$(PY) eduai models verify llama-3b

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .

test:
	$(PY) pytest

data:
	$(PY) eduai data build --sciq "$(SCIQ_DIR)" --out data

tagger-eval:
	$(PY) eduai tagger eval --gold data/gold/tagging_gold.jsonl --out reports/tagger_eval.md

pilot:
	$(LEASE) scripts/train_mlx.sh configs/lora_llama32_3b.yaml --pilot

train:
	$(LEASE) scripts/train_mlx.sh configs/lora_llama32_3b.yaml

sim:
	$(PY) eduai simulate --students 500 --out reports

eval:
	$(LEASE) $(PY) eduai eval --n 150 --out reports

report:
	$(PY) eduai report --out reports

demo:
	EDUAI_BACKEND=bank $(PY) eduai serve --port $(PORT)

e2e-offline:
	$(PY) python scripts/e2e_offline.py --port $(PORT)

serve:
	$(PY) eduai serve --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data/eduai.db
