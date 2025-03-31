# Verification log

All runs were on an Apple M1 Pro (16 GB, macOS). Training, eval and other timing-sensitive work
ran exclusively on the machine, with no other heavy job at the same time. Each result file has a
run manifest next to it.

| Date | What | How | Result |
|---|---|---|---|
| 2026-09-23 | SciQ blank-support counts | `eduai data build` eligibility gate | 1,198 / 113 / 116 blank (train / valid / test), matching the dataset readme |
| 2026-09-23 | Pinned model downloads | `make models`, `make models-llm`, `eduai models verify llama-1b` | every file matches the sha256 in `models.lock` |
| 2026-09-23 | MLX pilot, 20 iterations | `make pilot` | 56.2 tok/s, peak 6.60 GB, projected 1.68 h for 600 iterations (`reports/pilot.json`) |
| 2026-09-23 | Full LoRA run, 600 iterations | `make train` | 6,995 s, peak 7.65 GB, val loss 0.937 to 0.293 (`reports/training.json`) |
| 2026-09-23 | Adapter packaging | `eduai fetch-adapter --source <local tarball>` | sha256 of the tarball and of every file match `adapters/MANIFEST.json`; unpacked weights are byte-identical to the trained ones |
| 2026-09-23 | SFT rebuild determinism | `eduai data build` into a scratch dir, sha256 compared | train/valid/test identical to the files the adapter was trained on (`configs/sft_v1.sha256`) |
| 2026-09-23 | Offline e2e (macOS) | `tools/offline-run make e2e-offline` | server and driver canaries blocked (EPERM) for 1.1.1.1, api.openai.com, huggingface.co; practice and assessment sessions completed |
| 2026-09-23 | Canary companion | `make egress-open-check`, unsandboxed | all three targets connect |
| 2026-09-23 | Linux, no network | CI steps run locally: `uv sync` in `ghcr.io/astral-sh/uv:0.9.28-python3.11-bookworm-slim`, then `--network none` for ruff, `e2e_offline.py` and pytest | ruff clean, e2e OK (canary: network unreachable), 51 tests pass (the count at the time) |
| 2026-09-24 | Generation eval, 3 arms x 150 prompts | `scripts/run_eval.sh`, then `eduai eval score` | 2,755 s for generation and judging; results in `reports/eval_report.md` |
| 2026-09-25 | Re-judge with four option rotations | `eduai eval judge --judge llama-1b`, `--judge llama-3b`, then `eduai eval score` | 2,286 s; eval tables regenerated |
| 2026-09-23 | UI | Playwright screenshots at 1360x900 and 390x844 | fixed the missing sidebar (animation fill), the strike-through on the answer tag, and wrapping in unit rows; screenshots in `docs/screenshots/` |

Not verified here:
- the Colab/CUDA notebook and `HFBackend` (no NVIDIA GPU);
- the Ollama backend with a pulled model;
- GitHub Actions itself (the repo hasn't been pushed; the steps were run locally in the same containers).

No paid API or provider key was used anywhere in this project.
