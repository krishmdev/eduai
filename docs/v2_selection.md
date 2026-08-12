# v2 adapter: model-selection protocol

Written and committed before the first v2 run on the valid split, so the choices below can't be
bent to fit the results.

## Why a protocol

The valid split has 150 prompts. A paired difference in usable rate there has a 95% interval about
17 points wide, so if enough configurations or checkpoints are tried, one of them will beat base
2-shot (32.0% usable on valid) by chance alone. The budget and the rule are fixed here to limit
that.

## Candidate budget

- At most 2 training runs. Each run is one data build (`eduai rft build`) plus one training config.
  The second run is only for fixing something the first one showed on valid, such as a collapse
  of the key letter or a loss that is still falling.
- At most 3 checkpoints per run are evaluated on valid, all 0-shot. That is 6 adapter candidates
  at most.
- Every candidate evaluated keeps its generations in `reports/valid_eval/gen/` and its row in the
  valid report, including the ones that lose. Nothing is deleted from that ledger.

## Selection rule

1. The adapter is the candidate with the highest valid usable rate (0-shot).
2. Ties (the same number of usable items) go to higher key agreement, then to the earlier
   checkpoint.
3. The chosen adapter is then evaluated once with the eval's two fixed examples (the v2 2-shot
   arm) on valid.
4. The headline v2 arm is whichever of v2 0-shot and v2 2-shot has the higher valid usable rate.
   A tie goes to 0-shot, which has the shorter prompt.

## Test set

- The test set is run once, after the choice above is committed. That run generates the two v2
  arms (0-shot and 2-shot) with the chosen adapter and judges them with the same 1B primary and
  3B secondary judges as the v1 arms. The base and v1 arms keep their committed test generations.
- The report records how many times the test set was run.
- v2 is said to beat base 2-shot only if the paired bootstrap 95% interval for the usable
  difference on test excludes zero. Otherwise the report says it did not.

## Things the metric can't see

The v2 training targets were picked with the eval's own checks (structure, tagger alignment,
novelty) and with the base 3B as the key judge, which agrees closely with the 1B eval judge. So
usable is partly what v2 was trained to pass. To look past that, 40 v2-usable and 40
base-2shot-usable items from valid are pooled, shuffled with the arm hidden, and checked one at a time
for a correct key and a fit to the target objective. That audit is done on test too at the end.
No human reviewer is available for this project, so the audit is done by the AI assistant that ran
the pipeline, reading each passage and item. The report says so next to the numbers.

## Selection (recorded after the valid runs, before any v2 test run)

One training run (configs/lora_llama32_3b_v2.yaml, 450 iters). Valid usable, 0-shot, 150 prompts:
150 iters 34.7%, 300 iters 38.0%, 450 iters 36.0%. The 300-iter checkpoint is the adapter
(adapters/llama32-3b-eduai-v2). With the two fixed examples it scored 34.0%, so the headline v2
arm is finetuned-v2 (0-shot). Against base 2-shot (32.0%) on valid that is +6.0 points, 95%
interval -4.7 to +16.0. The val loss was flat after 150 iters and the key letters did not
collapse, so the second training run the budget allows was not used.
