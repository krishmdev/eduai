# Blind audit with LLM judges: pre-registration

Written and committed before either judge has seen any audit item.

## What this is, and what it isn't

The eval's own checks helped pick the v2 training data, so "usable" can overstate v2 (see
`docs/v2_selection.md`). The blind audit reads usable items with the arm hidden and asks whether
the key is right and whether the item fits its objective. No human reviewer is available, so the
audit is done by two LLM judges from model families other than Llama: Qwen3.5 9B and Gemma 4 12B.
They are language models, not people. Their verdicts are a second machine opinion that doesn't
share the Llama judge's blind spots, and they can be wrong. The report states this next to every
number.

## Items

- Valid: `reports/valid_eval/audit_llm/sheet.jsonl`, drawn and committed now by `blind_audit.py
  draw` with seed 20260905: 40 usable items from `finetuned-v2` (the headline v2 arm) and 40 from
  `base-2shot`, shuffled, with the arm only in `key.json`. The valid split had 57 and 48 usable items
  in those arms.
- Test: after the single test run, the same command with the same seed on `reports/eval`
  (`data/eval/prompts.jsonl`) into `reports/eval/audit_llm/`.

The judge sees only the sheet fields: passage, objective id and text, the requested misconception
if any, and the item (stimulus, stem, choices, keyed answer, explanation). Never the arm or the
prompt id. The test for the script checks that no arm name reaches the server.

## Judges

| Name | Localhost AI preset | Model | Thinking |
|---|---|---|---|
| qwen3.5-9b | `qwen3.5-9b-mlx4` | mlx-community/Qwen3.5-9B-MLX-4bit, revision 938d8919941c6e7efd3c7150eff7fe9d12afa631 | on |
| gemma-4-12b | `gemma-4-12b-mlx4` | mlx-community/gemma-4-12B-it-4bit (revision recorded from the preset at run time) | on |

- Thinking is turned on per request with `chat_template_kwargs: {"enable_thinking": true}`. The
  default Qwen preset turns thinking off and is left as it is; the request overrides it. Gemma 4's
  template has thinking off unless `enable_thinking` is true.
- Thinking budget: `max_thinking_tokens` 3000, `max_tokens` 3600, temperature 0, seed 0.
- No `response_format` with thinking on, because a grammar from the first token would block the
  reasoning. The final answer is read from `content`. If the server doesn't return
  `reasoning_content`, the client strips the reasoning itself (Qwen's `<think>...</think>`,
  Gemma's thought channel ending in `<channel|>`, or unmarked text before the JSON).
- The verdict is the last JSON object in the answer with boolean `key_correct`, `lo_fit` and
  `distractors_plausible`. On a parse failure the judge is asked once more for the JSON. The
  first-try parse failure rate and the number still unparsed are recorded per judge.
- The run stops before the batch if thinking was requested and the first reply shows no reasoning,
  so a server that ignores `enable_thinking` can't produce a thinking-off run by accident.
- Each judge's manifest records the preset, model repo and revision, the Localhost AI commit and
  whether its tree was dirty, the thinking setting and budget, the rubric hash (`3dd2fbda1e71c86b`),
  concurrency and wall time.

## Rubric

Fixed in `scripts/blind_audit.py` (`RUBRIC_SYSTEM`, `RUBRIC_USER`, `RETRY_USER`), the same for
both judges. It asks for three verdicts:

1. `key_correct`: the keyed answer is correct and the only correct option.
2. `lo_fit`: the item assesses the objective, not just a nearby word or topic.
3. `distractors_plausible`: every wrong option is tempting to a student who hasn't mastered the
   objective and clearly wrong to one who has.

## Outputs (`blind_audit.py llm-score`)

- Per judge and arm: the share of items judged `key_correct`, `lo_fit`, `distractors_plausible`,
  and both `key_correct` and `lo_fit`. The last is the precision of "usable". It comes with a
  bootstrap 95% interval (5,000 resamples within the arm, seed 0), and the v2 minus base-2shot
  difference comes with a bootstrap interval that resamples each arm independently, since the two
  arms' items are different prompts.
- Items still unparsed after the retry are left out of the rates, and their count is reported.
- Cohen's kappa between the two judges for each verdict and for "both".
- Agreement with the eval's llama-1b key judge. Every usable item already passed the 1B key check,
  so this agreement equals the judge's `key_correct` rate, and kappa against the 1B is undefined
  (reported as null).

## Rules

- Run once on valid now, and once on test after the single test run. Nothing is rerun because of
  its result. A crash may be rerun with `--force`, and the report says so.
- The rubric, the parsing and the item draws are not changed after any verdict has been seen.
  `max_tokens` may be raised before the valid run if the dry run below shows the reasoning being
  cut off; that is recorded in the manifest.
- A dry run on 4 items from other arms (`v2-it450`, `finetuned`) measures the typical reasoning
  length first. Those verdicts are thrown away and never reported.
- The judges run one at a time, each under the compute lease for the whole life of its server:
  Qwen at concurrency 4, Gemma 12B at 2.

## Cost

Estimated, not measured: about 1.3 hours for Qwen3.5 9B and 1.5 to 2 hours for Gemma 4 12B per
split, with reasoning on, for 80 items.
