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

## Amendment, 2026-09-18 (before any audit item was judged)

Committed before either judge has seen an audit item or a pilot item. It replaces the fixed budget,
the dry run and the Qwen concurrency above; everything else stands.

- Thinking budget from a pilot, per judge. Each judge first runs with a 6,000-token safety cap on
  thinking (`max_tokens` 6,600) over 10 items that are not in either audit draw:
  `reports/valid_eval/audit_pilot/sheet.jsonl`, 5 usable items each from the valid `base-0shot`
  and `finetuned` (v1) arms, seed 20260905. Six of them come from prompts that also appear in the
  valid draw, but they are other arms' items. Only the thinking-token lengths
  (`usage.thinking_tokens`) and finish reasons are kept; the verdicts are dropped unseen
  (`blind_audit.py pilot`). The judge's cap is max(1024, p90 of its 10 pilot lengths), rounded up
  to a multiple of 256, and `max_tokens` is the cap plus 600. The pilot lengths and caps are
  committed before any audit item is judged, and the same caps are used on valid and test.
- The dry run on 4 `v2-it450`/`finetuned` items is dropped; the pilot replaces it.
- Concurrency: Qwen3.5 9B at 8 (it ran cleanly at batch 16), Gemma 4 12B at 2 or less because of
  memory. Localhost AI runs with `LHAI_MAX_CONTEXT=9216` so prompt, thinking and answer fit.
- Batched greedy decoding drifts with the batch size, so a verdict is not bitwise reproducible if
  the same item is judged again at another concurrency. The audit is a judgment, not a
  batch-invariance check; the concurrency used is in each manifest.
- Reported with the audit: per judge, the share of audit items whose thinking reached the cap.
- Run order: valid audit (both judges), then the test audit, which is allowed now because the single
  v2 test run has happened (`reports/eval/test_runs.jsonl`, 1 run).

## Amendment 2, 2026-09-18 (after the pilots, before any audit item was judged)

The pilots ran as registered above; their lengths are committed in full in
`reports/valid_eval/audit_pilot/pilot_{qwen3.5-9b,gemma-4-12b}.json` (thinking tokens per item,
6,000-token safety cap):

- Qwen3.5 9B: 6000, 6000, 6000, 5514, 5339, 6000, 6000, 6000, 6000, 5615. 7 of 10 at the safety
  cap; 2,012 s for 10 items at concurrency 8.
- Gemma 4 12B: 1576, 6000, 1618, 2313, 1537, 6000, 6000, 6000, 6000, 342. 5 of 10 at the safety
  cap; 2,325 s for 10 items at concurrency 2.

Both p90s equal the safety cap, so the p90 rule gives 6,144 for both judges. At the measured speed
that is about 2.5 hours per split for Qwen alone, which doesn't fit the GPU time available. This
amendment replaces the first one's cap rule and split plan:

- Split: the audit runs on the test draw only (`reports/eval/audit_llm/`), with both judges. The
  headline v2 claim is on test. The valid audit is dropped for GPU time; the valid draw stays
  committed and unjudged.
- Thinking cap: fixed at 3,072 tokens for Qwen3.5 9B and 2,048 for Gemma 4 12B (`max_tokens` is
  the cap plus 600). Gemma's is lower because it runs about half as fast at concurrency 2 or less.
  Most audit items will likely reach the cap, so the verdicts come from truncated reasoning: when
  the budget runs out the server closes the thinking block and the model must answer.
- Reported per judge: the share of audit items whose thinking reached the cap, and, separately,
  the first-try parse failures and items still unparsed after the retry (a forced close still
  normally yields a verdict).
- If Gemma's first 10 items project its run past 3 hours, it is stopped, the verdicts that
  finished (`verdicts.partial.jsonl`) are reported as a partial run, and the report says so.
- Everything else is unchanged: rubric, blindness, seed, the test draw, one run per judge,
  Qwen at concurrency 8 and Gemma at 2.

## Amendment 3, 2026-09-18 (while Qwen judged the test draw, before Gemma judged anything)

Qwen3.5 9B judges all 80 test items as planned; its results are the primary audit numbers. At the
measured Gemma speed (about 16 tokens/s at concurrency 2), 80 items would take about 3.5 hours, so:

- Gemma 4 12B judges a fixed 40-item subset: the first 20 items of each arm in the committed
  blinded sheet order (`blind_audit.py llm --per-arm 20`). The rule reads `key.json` only to split
  by arm; the judge sees sheet rows alone. It depends neither on any verdict nor on Qwen's results.
- Same 2,048-token cap, concurrency 2.
- Reported for Gemma: per-arm precision on n = 20 + 20, and Qwen-vs-Gemma kappa on those 40 items.
- Hard stop at 2.25 hours of Gemma judging. If it triggers, the report says exactly how many items
  were judged, from `verdicts.partial.jsonl`.

## Note (2026-09-28, after the audit; nothing above is changed)

Found in the final review, and reported in the README next to the audit numbers:

- The sheet hides the arm name but not every cue. finetuned-v2 keys A on 20 of its 40 sheet
  items against 8 of 40 for base-2shot, and 14 of its items carry a requested misconception
  against 5. Qwen judged the key correct on 7 of 14 v2 items with a misconception and 16 of 26
  without.
- Gemma's one unparsed verdict (a018, finetuned-v2) failed because of a misspelled field,
  `distractors_plplausible`. Its other fields read key correct, not on objective. Counted as
  written, Gemma's v2 precision would be 50.0% (10 of 20) and the v2 minus base difference +5.0
  points instead of +7.6. The parser and committed scores stay as they are.
