# Answer-key verification with Qwen3.5 9B: pre-registration

Written and committed before the verifier has scored any eval item.

## Why, and what this is not

The blind LLM audit on test (`docs/blind_audit_llm.md`, `reports/eval/audit_llm/`) had Qwen3.5 9B,
with thinking on, read 80 items the pipeline had marked usable. It judged the key correct on 57.5%
of them in both arms. The eval's key check is an open-book Llama 3.2 1B judge (the 3B is a
secondary column), and those small judges pass many wrong keys.

This adds a stronger key check and measures what it does. It is post-hoc and exploratory. The
idea came from looking at test audit results, so nothing here is a confirmatory test.

- The pre-registered v2 test result (`reports/eval.json`, `reports/eval_report.md`) stays as it
  is. None of the existing eval files are rewritten, and `scripts/check_eval_snapshot.py` must
  still pass.
- Nothing is selected on test. The verifier, prompt, rotations and threshold are all fixed here,
  before any item is scored. Valid runs first, but its results can't change any of these
  settings.

## Verifier

- Model: `qwen3.5-9b`, mlx-community/Qwen3.5-9B-MLX-4bit, revision
  938d8919941c6e7efd3c7150eff7fe9d12afa631. The files are hashed in `models.lock`, and the
  snapshot is the one `../localhost-ai` uses. It runs in-process through `MLXBackend`, not through
  a server.
- Thinking is off: `enable_thinking=False` goes to the chat template, so the generation prompt
  ends with an empty `<think></think>` block and the next token is the answer letter.
- Method: the same `Judge` the eval uses (`src/eduai/evaluation/solver.py`). It is open-book, with
  the source passage the generator saw. It reads next-token log-probabilities for A to D over the
  4 cyclic rotations of the options, maps them back to the original options, averages them, and
  takes a softmax over the four averages to get `p_key`.
- An item is verified if the verifier's top option is the keyed option AND `p_key > 0.5`. An item
  whose key text also appears as a distractor fails, as it does in the eval.
- The threshold is 0.5. It is fixed here and is not tuned on valid or test.

## Items and arms

- Arms: `base-2shot`, `finetuned-v2`, `finetuned-v2-2shot`.
- Splits: valid (`reports/valid_eval/gen`, `data/eval/valid_prompts.jsonl`) first, then test
  (`reports/eval`, `data/eval/prompts.jsonl`).
- The verifier scores every schema-valid item in those arms, the same set the Llama judges score.
  That is about 900 items. The reference SciQ row is not scored.
- Output goes to `verify_qwen3.5-9b.jsonl` in each eval directory, one row per (arm, id), plus a
  manifest. The post-hoc report is `reports/key_verification.{json,md}`.
- There is one GPU chain under the compute lease: valid, then test. The chain doesn't start until
  the Gemma audit judge has finished.

## What is reported (all exploratory)

1. For each split and arm: usable (unchanged, from `per_item.jsonl`), verified usable (usable AND
   verified), and the share of usable items that survive verification.
2. Paired bootstrap 95% CIs over prompts, with the same resampler and seed as the eval, for
   verified usable in `finetuned-v2 - base-2shot` and `finetuned-v2-2shot - base-2shot`. These are
   descriptive and aren't a replacement for the registered comparison.
3. The independent check is the Gemma 4 12B audit subset on test
   (`reports/eval/audit_llm/llm_gemma-4-12b/verdicts.jsonl`: the first 20 items of each arm, per
   amendment 3 of the audit, or however many finished before its hard stop). Gemma is a
   different model family from both the Llama judges and the verifier. For the items that pass
   verification and the items that fail it, the report gives Gemma's `key_correct` share, with
   counts and 95% Wilson intervals, and the difference with a two-sided Fisher exact p-value. The
   question is whether items that pass verification have their key judged correct more often than
   items that fail it. With about 40 items, that interval will be wide.
4. The Qwen3.5 9B audit verdicts come from the same model family as the verifier, so agreement
   between them is partly circular. They are reported only as a note, never as evidence that
   verification works.

The judges are language models, not people. Gemma's verdicts are a second machine opinion, not
ground truth.

## Bank promotion

`eduai bank add-generated` now needs the item to be verified as well as usable. This is on by
default for new promotions (`--require-verified`, `EDUAI_REQUIRE_KEY_VERIFICATION`). It reads
`verify_<model>.jsonl` from the eval directory and stops if that file doesn't exist. Passing
`--no-require-verified` gives the old behaviour. The live `Pipeline` accepts an optional
`verifier` and, when one is given, rejects unverified items with the reason `key_unverified`.
