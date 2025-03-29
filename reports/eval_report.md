# Generation eval: base vs few-shot vs fine-tuned

150 held-out prompts from the SciQ test split, each with a target learning objective assigned by an independent labeling pass that never saw tagger output (`data/gold/eval_lo_labels.jsonl`; AI-labeled, pending human spot-check). 59 candidates labeled off-curriculum were dropped. Leakage filter against the SFT train/valid rows: screened 244, dropped 5 for >= 50% 8-gram passage containment and 9 for a same-answer question with Q+A cosine >= 0.88. Manifest: `eval_manifest.json`.

Every percentage uses all prompts as the denominator. Checks are scored independently:

- JSON: a JSON object could be extracted and parsed (after one retry at T=0.3 if the greedy output did not parse). First-try JSON is the greedy output alone.
- Structure: four distinct options, no all/none of the above, key not in the stem, the requested LO id and format, and the target misconception present as a wrong option when one was requested.
- Key agreement: a fixed open-book judge (llama-1b, Llama 3.2 1B 4-bit, not any arm's generator) picks the keyed option from A-D log-probabilities averaged over two cyclic rotations of the options, with the source passage in its prompt.
- Aligned: the target LO is in the tagger's top 3 and its own score is at least tau. The tagger is a noisy filter (about 70% of in-curriculum gold items pass), so the reference row is the ceiling.
- Novel: stem cosine < 0.92 against the whole SciQ bank, excluding the prompt's own source item and its near-duplicate group. Closeness to the source is reported separately as source copy.

| | JSON | First-try JSON | Schema | Structure | Key agreement | Aligned | Novel | All checks | Source copy | Memorized (train) |
|---|---|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | | | | | 92.7% | 70.7% | | | | |
| Base 3B, 0-shot | 100.0% | 99.3% | 93.3% | 40.7% | 70.7% | 74.7% | 92.7% | 25.3% | 4.7% | 0.7% |
| Base 3B, 2-shot | 100.0% | 98.0% | 80.7% | 60.7% | 56.7% | 64.0% | 80.0% | 33.3% | 2.7% | 0.0% |
| Base 3B + EduAI LoRA | 100.0% | 100.0% | 100.0% | 54.0% | 74.7% | 72.0% | 96.0% | 32.7% | 25.3% | 1.3% |

Paired bootstrap (5,000 resamples over prompts), difference in rate with 95% CI. With n = 150, one arm's rate has a standard error around 4 points, so differences under about 10 points should be read as noise unless the interval excludes zero.

| Comparison | Metric | Difference | 95% CI |
|---|---|---|---|
| finetuned - base-0shot | all_checks | +7.3% | [-2.7%, +17.3%] |
| finetuned - base-0shot | aligned | -2.7% | [-10.7%, +5.3%] |
| finetuned - base-0shot | key | +4.0% | [-6.7%, +14.7%] |
| finetuned - base-0shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned - base-2shot | all_checks | -0.7% | [-10.7%, +10.0%] |
| finetuned - base-2shot | aligned | +8.0% | [-1.3%, +17.3%] |
| finetuned - base-2shot | key | +18.0% | [+8.0%, +28.0%] |
| finetuned - base-2shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned - reference | aligned | +1.3% | [-5.3%, +8.0%] |
| finetuned - reference | key | -18.0% | [-26.0%, -10.0%] |
| base-2shot - base-0shot | all_checks | +8.0% | [-1.3%, +16.7%] |
| base-2shot - base-0shot | aligned | -10.7% | [-18.7%, -2.7%] |
| base-2shot - base-0shot | key | -14.0% | [-24.0%, -4.0%] |
| base-2shot - base-0shot | json | +0.0% | [+0.0%, +0.0%] |

Secondary key agreement with the base 3B as judge (self-judged for the base arms, so biased in their favor):

| | Key agreement (3B judge) |
|---|---|
| SciQ reference item (ceiling) | 97.3% |
| Base 3B, 0-shot | 76.0% |
| Base 3B, 2-shot | 67.3% |
| Base 3B + EduAI LoRA | 86.0% |

First failing check per item (checks in validator order):

| Arm | accepted | duplicate | key_disagreement | not_aligned | schema | structure |
|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 38 | 0 | 12 | 11 | 10 | 79 |
| Base 3B, 2-shot | 50 | 1 | 24 | 16 | 29 | 30 |
| Base 3B + EduAI LoRA | 49 | 3 | 7 | 22 | 0 | 69 |

Structure problems among schema-valid items (an item can have several):

| Arm | all four options identical | near-identical options | stem gives away the answer | stimulus format without a stimulus | target misconception missing | wrong lo_id |
|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 0 | 52 | 1 | 0 | 32 | 13 |
| Base 3B, 2-shot | 0 | 11 | 11 | 1 | 11 | 0 |
| Base 3B + EduAI LoRA | 9 | 63 | 12 | 0 | 1 | 0 |

Generation speed (greedy, one request at a time, MLX on the M1 Pro, under the compute lease):

| Arm | Mean generation tok/s | Mean seconds per item | Peak memory (GB) |
|---|---|---|---|
| Base 3B, 0-shot | 80.3 | 3.0 | 2.50 |
| Base 3B, 2-shot | 76.1 | 3.8 | 2.74 |
| Base 3B + EduAI LoRA | 48.0 | 3.6 | 3.05 |
