# Verification log

All runs were on an Apple M1 Pro with 16 GB of memory, running macOS. Training, evaluation and
other timing-sensitive work never overlapped with another training or benchmark job, but the
machine was not otherwise idle. The eval manifests record a load average of 7 to 9.5, 6 running
containers and about 12 GB of swap in use; load wasn't recorded for the training runs. Each result
file has a run manifest beside it.

Manifest caveats:
- Commit hashes inside some manifests no longer resolve, because the repository history was
  rewritten after those runs. `sim_manifest.json` names 0747337, `data_card_manifest.json` names
  32f0ff7, and `tagger_eval_manifest.json` names 2474271. The results files are identified by the
  sha256 values below instead.
- `host.python` in the manifests (3.14) is the interpreter that ran the host-recording tool. The
  pipeline itself ran in the project's Python 3.11 virtualenv; the manifests' `packages` field comes
  from that environment.

| File | sha256 |
|---|---|
| `reports/data_card.json` | `09bd0024314030fb04b5ae482e6f0daad6c32a86d51a57c3f2034e72f1bb62b5` |
| `reports/tagger_eval.json` | `deae2a071c3daa5f6f3e3b8c08af4d9291db57e1bf57efc2d9b15398938aeb52` |
| `reports/sim_results.json` | `9db0ae4a1f95e55e41730c0fd3b37f958c2a6054374e5d3932c687f21f8d403f` |
| `reports/pilot.json` | `eafbc900d54ee25a358bfb0b425668243b3d771119e2961a24fed28a6d8a18c0` |
| `reports/training.json` | `086bcee324448e65d5f8461e3ec40d3d54192ff070dd0c69231574ea72dc8b67` |
| `reports/eval.json` | `d352435797992e59f7f9f00671200d2f3f8528bc78857607a9a54c0db9b79006` |
| `reports/eval/gen_base-0shot.jsonl` | `f4f7a2c51a8ac370dd176aca1a6af5889f6487735e545d0e35e088b0b11bb1ab` |
| `reports/eval/gen_base-2shot.jsonl` | `9121bee470fca9a347ddcbd27d03f4ded914f0f05d27cff8067074bda641046c` |
| `reports/eval/gen_finetuned.jsonl` | `e8361fac96773e819de10f99fd39ee0de639e6d7a28a7f50f1747c827cd63a81` |
| `reports/eval/judge_llama-1b.jsonl` | `6a4c670f566bb30486e25363874bf0ce0e18f4c282e7a03af444ac0b3b944da1` |
| `reports/eval/judge_llama-3b.jsonl` | `cc269eb930f9db2e0cfa64f03084e43506e9296b05e16e7ce257e58896887021` |

| Date | What | How | Result |
|---|---|---|---|
| 2026-07-23 | SciQ blank-support counts | `eduai data build` eligibility gate | 1,198 / 113 / 116 blank (train / valid / test), matching the dataset readme |
| 2026-07-23 | Pinned model downloads | `make models`, `make models-llm`, `eduai models verify llama-1b` | every file matches the sha256 in `models.lock` |
| 2026-07-23 | MLX pilot, 20 iterations | `make pilot` | 56.2 tok/s, peak 6.60 GB, projected 1.68 h for 600 iterations (`reports/pilot.json`) |
| 2026-07-23 | Full LoRA run, 600 iterations | `make train` | 6,995 s, peak 7.65 GB, val loss 0.937 to 0.293 (`reports/training.json`) |
| 2026-07-23 | Adapter packaging | `eduai fetch-adapter --source <local tarball>` | sha256 of the tarball and of every file match `adapters/MANIFEST.json`; unpacked weights are byte-identical to the trained ones |
| 2026-07-23 | SFT rebuild determinism | `eduai data build` into a scratch dir, sha256 compared | train/valid/test identical to the files the adapter was trained on (`configs/sft_v1.sha256`) |
| 2026-07-23 | Offline e2e (macOS) | `tools/offline-run make e2e-offline` | server and driver canaries blocked (EPERM) for 1.1.1.1, api.openai.com, huggingface.co; practice and assessment sessions completed |
| 2026-07-23 | Canary companion | `make egress-open-check`, unsandboxed | all three targets connect |
| 2026-07-23 | Linux, no network | CI steps run locally: `uv sync` in `ghcr.io/astral-sh/uv:0.9.28-python3.11-bookworm-slim`, then `--network none` for ruff, `e2e_offline.py` and pytest | ruff clean, e2e OK (canary: network unreachable), 51 tests pass (the count at the time) |
| 2026-08-05 | Generation eval, 3 arms x 150 prompts | `scripts/run_eval.sh`, then `eduai eval score` | 2,755 s for generation and judging; results in `reports/eval_report.md` |
| 2026-08-20 | Re-judge with four option rotations | `eduai eval judge --judge llama-1b`, `--judge llama-3b`, then `eduai eval score` | 2,286 s (`reports/rejudge_manifest.json`); eval tables regenerated |
| 2026-07-23 | UI | Playwright screenshots at 1360x900 and 390x844 | fixed the missing sidebar (animation fill), the strike-through on the answer tag, and wrapping in unit rows; screenshots in `docs/screenshots/` |
| 2026-08-20 | Evaluation snapshot audit | `.venv/bin/python scripts/check_eval_snapshot.py` with `PYTHONPATH=src`; cached-model `eduai eval score` with Hugging Face offline flags | 150 paired IDs, five raw generation/judge hashes, summary rates, paired bootstrap, and report rendering agree; rescoring left `eval.json` and numeric report unchanged. The scorer produced only small floating-point `source_cos` differences in `per_item.jsonl`, so that derived file was retained at its committed bytes. |
| 2026-08-20 | No-key source-archive demo | In a disposable `git archive` plus these working-tree changes: `UV_CACHE_DIR=<writable temp cache> uv sync --frozen --no-default-groups`, then `UV_CACHE_DIR=<same cache> tools/offline-run make e2e-offline PORT=18084` | Fresh Python 3.11 environment installed 47 locked core packages without model extras; 666 sample-bank items loaded; 8-answer practice and assessment sessions completed; server and driver egress canaries blocked all three external targets. The same fresh environment passed `scripts/check_eval_snapshot.py`. |
| 2026-08-20 | Code checks | `.venv/bin/python -m pytest -q`; `.venv/bin/ruff check .`; `.venv/bin/ruff format --check .` | 54 tests passed; lint and format checks passed. |

These checks were not run:
- the Colab/CUDA notebook and `HFBackend` (no NVIDIA GPU);
- the Ollama backend with a pulled model;
- GitHub Actions itself (the repo hasn't been pushed; the steps were run locally in the same containers).

No paid API or provider key was used for this project.
