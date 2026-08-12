# Model-selection eval on the valid split

The report covers 150 prompts from groups assigned to the valid split that are not SFT valid rows. It is used for choosing training settings and adapters; the test prompts are not. Target objectives are the tagger's own labels (the test prompts use an independent labeling pass), so alignment here is easier than on the test set. The leakage filter against SFT train rows screened 163 prompts and dropped 3 for >= 50% 8-gram passage containment and 10 for a same-answer question with Q+A cosine >= 0.88. Manifest: `eval_manifest.json`.

Every percentage uses all prompts as the denominator. Checks are scored independently:

- JSON: a JSON object could be extracted and parsed (after one retry at T=0.3 if the greedy output did not parse). First-try JSON is the greedy output alone.
- Structure: four distinct options, no all/none of the above, key not in the stem, the requested LO id and format, and the target misconception present as a wrong option when one was requested.
- Key agreement: a fixed open-book judge (llama-1b, Llama 3.2 1B 4-bit, not any arm's generator) picks the keyed option from A-D log-probabilities averaged over all four cyclic rotations of the options (so every option is scored in every position), with the source passage in its prompt.
- Aligned: the target LO is in the tagger's top 3 and its own score is at least tau. The tagger is a noisy filter (about 70% of in-curriculum gold items pass), so the reference row is the ceiling.
- Novel: stem cosine < 0.92 against the whole SciQ bank, excluding the prompt's own source item and its near-duplicate group. Closeness to the source is reported separately as source copy.
- Usable: all checks and not a copy of the source question. This is the bank-promotion criterion.
- Memorized: stem cosine >= 0.92 with an SFT training stem. Reported only; it doesn't reject items.
- v2 caveat: the v2 adapters' training targets are base 3B samples kept only when they passed these same checks (structure, tagger alignment, novelty, not a source copy), with the base 3B as the key judge. Usable is partly what v2 was trained to pass, so it overstates v2 more than the other arms; the blind audit in the README is the check on that. Memorized is measured against the v1 SFT train stems, which the v2 targets were screened against at build time.

- Key agreement (valid): the same judge result over schema-valid items only, which removes the effect of JSON failures. Items whose key text also appears as a distractor never count as agreeing.

| | JSON | First-try JSON | Schema | Structure | Key agreement | Key agreement (valid) | Aligned | Novel | All checks | Source copy | Usable | Memorized (train) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | | | | | 92.0% | 92.0% | 100.0% | | | | | |
| Base 3B, 0-shot | 100.0% | 100.0% | 98.0% | 48.7% | 68.7% | 70.1% | 80.7% | 96.0% | 24.7% | 0.7% | 24.0% | 0.0% |
| Base 3B, 2-shot | 98.0% | 95.3% | 83.3% | 67.3% | 54.0% | 64.8% | 64.7% | 82.0% | 36.0% | 6.7% | 32.0% | 0.7% |
| Base 3B + EduAI LoRA v1 | 100.0% | 100.0% | 100.0% | 52.7% | 68.7% | 68.7% | 73.3% | 97.3% | 34.0% | 25.3% | 16.0% | 0.7% |
| Base 3B + EduAI LoRA v2 | 100.0% | 100.0% | 99.3% | 71.3% | 71.3% | 71.8% | 76.0% | 98.7% | 41.3% | 5.3% | 38.0% | 0.0% |
| Base 3B + EduAI LoRA v2, 2-shot | 100.0% | 100.0% | 98.0% | 77.3% | 69.3% | 70.7% | 77.3% | 96.0% | 40.7% | 8.7% | 34.0% | 0.7% |
| v2-it150 | 100.0% | 100.0% | 99.3% | 74.0% | 68.7% | 69.1% | 77.3% | 98.7% | 39.3% | 5.3% | 34.7% | 0.0% |
| v2-it300 | 100.0% | 100.0% | 99.3% | 71.3% | 71.3% | 71.8% | 76.0% | 98.7% | 41.3% | 5.3% | 38.0% | 0.0% |
| v2-it450 | 100.0% | 100.0% | 100.0% | 75.3% | 68.0% | 68.0% | 82.0% | 99.3% | 40.7% | 6.7% | 36.0% | 0.0% |

Paired bootstrap (5,000 resamples over prompts), difference in rate with 95% CI. With n = 150, one arm's rate has a standard error around 4 points, so differences under about 10 points should be read as noise unless the interval excludes zero.

| Comparison | Metric | Difference | 95% CI |
|---|---|---|---|
| finetuned - base-0shot | usable | -8.0% | [-16.7%, +0.7%] |
| finetuned - base-0shot | all_checks | +9.3% | [-0.7%, +19.3%] |
| finetuned - base-0shot | aligned | -7.3% | [-14.7%, +0.0%] |
| finetuned - base-0shot | key | +0.0% | [-10.0%, +10.7%] |
| finetuned - base-0shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned - base-2shot | usable | -16.0% | [-24.7%, -7.3%] |
| finetuned - base-2shot | all_checks | -2.0% | [-12.0%, +8.0%] |
| finetuned - base-2shot | aligned | +8.7% | [+0.7%, +17.3%] |
| finetuned - base-2shot | key | +14.7% | [+2.7%, +26.0%] |
| finetuned - base-2shot | json | +2.0% | [+0.0%, +4.7%] |
| finetuned - reference | aligned | -26.7% | [-34.0%, -20.0%] |
| finetuned - reference | key | -23.3% | [-32.0%, -14.7%] |
| base-2shot - base-0shot | usable | +8.0% | [-1.3%, +17.3%] |
| base-2shot - base-0shot | all_checks | +11.3% | [+2.7%, +20.0%] |
| base-2shot - base-0shot | aligned | -16.0% | [-24.7%, -7.3%] |
| base-2shot - base-0shot | key | -14.7% | [-24.7%, -4.7%] |
| base-2shot - base-0shot | json | -2.0% | [-4.7%, +0.0%] |
| finetuned-v2 - base-0shot | usable | +14.0% | [+3.3%, +24.7%] |
| finetuned-v2 - base-0shot | all_checks | +16.7% | [+6.7%, +26.7%] |
| finetuned-v2 - base-0shot | aligned | -4.7% | [-12.0%, +2.7%] |
| finetuned-v2 - base-0shot | key | +2.7% | [-6.7%, +12.7%] |
| finetuned-v2 - base-0shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned-v2 - base-2shot | usable | +6.0% | [-4.7%, +16.0%] |
| finetuned-v2 - base-2shot | all_checks | +5.3% | [-5.3%, +16.0%] |
| finetuned-v2 - base-2shot | aligned | +11.3% | [+2.7%, +20.0%] |
| finetuned-v2 - base-2shot | key | +17.3% | [+6.7%, +28.0%] |
| finetuned-v2 - base-2shot | json | +2.0% | [+0.0%, +4.7%] |
| finetuned-v2 - finetuned | usable | +22.0% | [+12.0%, +32.0%] |
| finetuned-v2 - finetuned | all_checks | +7.3% | [-2.7%, +17.3%] |
| finetuned-v2 - finetuned | aligned | +2.7% | [-5.3%, +10.7%] |
| finetuned-v2 - finetuned | key | +2.7% | [-7.3%, +12.7%] |
| finetuned-v2 - finetuned | json | +0.0% | [+0.0%, +0.0%] |
| finetuned-v2 - reference | aligned | -24.0% | [-30.7%, -17.3%] |
| finetuned-v2 - reference | key | -20.7% | [-28.7%, -12.0%] |
| finetuned-v2-2shot - base-2shot | usable | +2.0% | [-8.0%, +12.7%] |
| finetuned-v2-2shot - base-2shot | all_checks | +4.7% | [-5.3%, +15.3%] |
| finetuned-v2-2shot - base-2shot | aligned | +12.7% | [+4.0%, +21.3%] |
| finetuned-v2-2shot - base-2shot | key | +15.3% | [+4.7%, +26.0%] |
| finetuned-v2-2shot - base-2shot | json | +2.0% | [+0.0%, +4.7%] |
| finetuned-v2-2shot - finetuned-v2 | usable | -4.0% | [-14.0%, +6.0%] |
| finetuned-v2-2shot - finetuned-v2 | all_checks | -0.7% | [-10.7%, +10.0%] |
| finetuned-v2-2shot - finetuned-v2 | aligned | +1.3% | [-5.3%, +8.0%] |
| finetuned-v2-2shot - finetuned-v2 | key | -2.0% | [-12.0%, +8.0%] |
| finetuned-v2-2shot - finetuned-v2 | json | +0.0% | [+0.0%, +0.0%] |
| v2-it150 - base-2shot | usable | +2.7% | [-7.3%, +12.7%] |
| v2-it150 - base-2shot | all_checks | +3.3% | [-6.7%, +14.0%] |
| v2-it150 - base-2shot | aligned | +12.7% | [+4.0%, +21.3%] |
| v2-it150 - base-2shot | key | +14.7% | [+4.7%, +24.7%] |
| v2-it150 - base-2shot | json | +2.0% | [+0.0%, +4.7%] |
| v2-it300 - base-2shot | usable | +6.0% | [-4.0%, +16.0%] |
| v2-it300 - base-2shot | all_checks | +5.3% | [-5.3%, +16.0%] |
| v2-it300 - base-2shot | aligned | +11.3% | [+2.7%, +20.0%] |
| v2-it300 - base-2shot | key | +17.3% | [+7.3%, +28.0%] |
| v2-it300 - base-2shot | json | +2.0% | [+0.0%, +4.7%] |
| v2-it450 - base-2shot | usable | +4.0% | [-6.0%, +14.0%] |
| v2-it450 - base-2shot | all_checks | +4.7% | [-5.3%, +14.7%] |
| v2-it450 - base-2shot | aligned | +17.3% | [+8.7%, +26.0%] |
| v2-it450 - base-2shot | key | +14.0% | [+4.0%, +24.0%] |
| v2-it450 - base-2shot | json | +2.0% | [+0.0%, +4.7%] |

First failing check per item (checks in validator order):

| Arm | accepted | duplicate | key_disagreement | no_json | not_aligned | schema | structure |
|---|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 37 | 1 | 27 | 0 | 8 | 3 | 74 |
| Base 3B, 2-shot | 54 | 2 | 32 | 3 | 13 | 22 | 24 |
| Base 3B + EduAI LoRA v1 | 51 | 2 | 8 | 0 | 18 | 0 | 71 |
| Base 3B + EduAI LoRA v2 | 62 | 1 | 29 | 0 | 15 | 1 | 42 |
| Base 3B + EduAI LoRA v2, 2-shot | 61 | 3 | 34 | 0 | 18 | 3 | 31 |
| v2-it150 | 59 | 1 | 33 | 0 | 18 | 1 | 38 |
| v2-it300 | 62 | 1 | 29 | 0 | 15 | 1 | 42 |
| v2-it450 | 61 | 1 | 37 | 0 | 14 | 0 | 37 |

Answer-key letter distribution (schema-valid items) and judge key agreement by keyed letter. A skewed key position is a generation defect in its own right; four-rotation judging removes the judge's own position bias from the comparison.

| Arm | A | B | C | D | Agree when key=A | B | C | D |
|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | 32 | 37 | 31 | 50 | 90.6% | 94.6% | 96.8% | 88.0% |
| Base 3B, 0-shot | 20 | 61 | 44 | 22 | 70.0% | 75.4% | 65.9% | 63.6% |
| Base 3B, 2-shot | 25 | 51 | 21 | 28 | 72.0% | 58.8% | 66.7% | 67.9% |
| Base 3B + EduAI LoRA v1 | 95 | 30 | 6 | 19 | 70.5% | 70.0% | 50.0% | 63.2% |
| Base 3B + EduAI LoRA v2 | 47 | 54 | 32 | 16 | 66.0% | 68.5% | 87.5% | 68.8% |
| Base 3B + EduAI LoRA v2, 2-shot | 30 | 54 | 27 | 36 | 66.7% | 77.8% | 74.1% | 61.1% |
| v2-it150 | 44 | 42 | 44 | 19 | 52.3% | 83.3% | 75.0% | 63.2% |
| v2-it300 | 47 | 54 | 32 | 16 | 66.0% | 68.5% | 87.5% | 68.8% |
| v2-it450 | 43 | 52 | 39 | 16 | 55.8% | 78.8% | 74.4% | 50.0% |

Structure problems among schema-valid items (an item can have several):

| Arm | all four options identical | near-identical options | stem gives away the answer | stimulus format without a stimulus | target misconception missing | wrong lo_id |
|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 0 | 44 | 1 | 0 | 37 | 11 |
| Base 3B, 2-shot | 0 | 7 | 7 | 2 | 9 | 0 |
| Base 3B + EduAI LoRA v1 | 13 | 60 | 19 | 0 | 2 | 0 |
| Base 3B + EduAI LoRA v2 | 1 | 14 | 28 | 0 | 9 | 0 |
| Base 3B + EduAI LoRA v2, 2-shot | 0 | 7 | 18 | 0 | 7 | 0 |
| v2-it150 | 1 | 11 | 19 | 0 | 14 | 0 |
| v2-it300 | 1 | 14 | 28 | 0 | 9 | 0 |
| v2-it450 | 0 | 11 | 23 | 0 | 7 | 0 |

Generation speed (greedy, one request at a time, MLX on the M1 Pro). Generation held the compute lease, so no other training or benchmark job ran at the same time, but the machine wasn't idle; host load for each run is in the `*_manifest.json` files next to this report. Treat these speeds, and the gap between the adapter and base arms, as rough.

| Arm | Mean generation tok/s | Mean seconds per item | Peak memory (GB) |
|---|---|---|---|
| Base 3B, 0-shot | 75.9 | 3.2 | 2.50 |
| Base 3B, 2-shot | 74.0 | 3.9 | 2.74 |
| Base 3B + EduAI LoRA v1 | 47.4 | 3.8 | 3.07 |
| Base 3B + EduAI LoRA v2 | 47.7 | 3.8 | 3.07 |
| Base 3B + EduAI LoRA v2, 2-shot | 45.9 | 4.9 | 3.24 |
| v2-it150 | 47.9 | 3.9 | 3.07 |
| v2-it300 | 48.1 | 3.7 | 3.07 |
| v2-it450 | 48.1 | 3.8 | 3.07 |
