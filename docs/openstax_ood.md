# OpenStax out-of-distribution check: pre-registration

Written and committed before any model has generated anything for these prompts.

## Why

SciQ passages, and so every eval prompt so far, are middle school and early high school level. This
is a second look at whether the v2 adapter's gain over base 2-shot holds on AP-level text it
never saw. It is a secondary check. The headline numbers stay those of the SciQ test set, and
nothing here is used to choose or tune anything.

## Prompts

`data/openstax/prompts.jsonl`, built by `eduai data openstax-prompts` (card:
`reports/openstax_ood/prompts_card.json`). The sources are pinned in `data/openstax/sources.json`:

| Subject | Passages from | Reference questions from |
|---|---|---|
| AP Biology | Biology for AP Courses | Biology 2e review questions |
| AP Physics 1 | College Physics for AP Courses 2e | the same book's Test Prep for AP Courses |
| AP Chemistry | Chemistry 2e | the same book's end-of-chapter exercises |

AP Environmental Science is left out because OpenStax has no book for it.

- 50 prompts per subject, 150 in all. Passages are body paragraphs from the book sections. Figures,
  equations, tables, boxed features and end-of-section material are removed. Paragraphs that lead
  into or refer back to removed material are dropped, and each chunk is cut at a sentence boundary
  to a length drawn from the SciQ eval passages. Median length is 268 characters, against 297 for
  SciQ; the quartiles are 203 and 375, against 174 and 524.
- At most one passage per section in each selection round, so the set spreads across the books.
- Leakage screen: a passage is dropped when 8-gram containment is 0.5 or more against any passage
  in the v1 or v2 SFT train and valid rows (27 dropped), or against any SciQ support passage (46
  dropped; some SciQ supports quote OpenStax). A reference question is dropped when a training item
  has the same answer and Q+A cosine of 0.88 or more (2 dropped). These are the eval's own leakage
  utilities and thresholds.
- Target objectives come from the existing tagger over the 123 objectives, as on the valid split:
  its own top-1, kept only when that objective is in the book's subject and clears tau. A prompt
  with a reference question is labeled from that question and its answer, which is how the valid
  split was labeled (43 prompts). The rest have no question and are labeled from the passage text
  (107 prompts). Every passage cleared tau when tagged, so tau does no filtering here and the
  passage labels are noisier than the question labels.
- Formats follow the main eval quotas: 40% stimulus in each subject. Target misconceptions (30% in
  the main eval) need a human-written wrong option, so they come from the reference questions' wrong
  options: 15 in biology, 15 in physics, and none in chemistry, which has no usable reference.
- Difficulty is not measured. It is assigned by a seeded, balanced shuffle, about a third each.

## References

The reference row is the "ceiling": real questions written by the textbook authors, keyed by the
book's own solutions. A prompt gets one when a four-option, single-answer question with a
published key exists in the same section and its embedding cosine to the passage is at least 0.6.
Each question is used once. Provenance (book, repo, commit, module, exercise id, cosine) is stored
on each row.

Coverage is 62 of 150: 42 biology, 20 physics, 0 chemistry. The AP biology edition's own review
questions are served from the OpenStax exercise service without answer keys, so biology references
come from Biology 2e sections with the same title. Chemistry 2e has almost no four-option
single-answer questions with a published key (one in the whole book). The reference row is scored
over the 62 prompts that have one, and the report gives its n.

## Arms

- `base-2shot`: base Llama 3.2 3B with the eval's two fixed examples.
- The v2 headline arm, chosen on valid by the rule in `docs/v2_selection.md`: the higher valid
  usable rate of `finetuned-v2` (0-shot) and `finetuned-v2-2shot`, a tie going to 0-shot.
  `scripts/openstax_report.py headline` reads this from `reports/valid_eval/eval.json`. As recorded
  on main in 9ef8bad, it is `finetuned-v2` (the 300-iteration checkpoint, 0-shot): 38.0% usable on
  valid, against 34.0% with the two examples.

## Checks and metric

The same as the main eval, with nothing changed: greedy generation with one retry, the same
structure checks, key agreement from the llama-1b open-book judge (log-probabilities over all four
rotations), tagger alignment to the target objective, novelty against the SciQ bank, and usable as
all checks passed and not a copy of the source question. The 3B secondary judge is not run.

Reported: usable rate for each arm per subject and pooled, and the v2 minus base-2shot difference
with a paired bootstrap 95% interval (5,000 resamples over prompts, seed 0), per subject and pooled.
The other check rates and the rejection histogram are reported alongside. With 50 prompts per
subject the per-subject intervals will be wide, about 25 points, so they are descriptive; the
pooled interval (n = 150) is the one read.

## Rules

- Run once: `scripts/openstax_ood.sh` refuses to start if generations already exist.
- No tuning of prompts, checks, thresholds, arms or adapters on these results, and no rerun if the
  result is disappointing. A failed run (crash, not a bad score) may be restarted from scratch, and
  the report says so.
- v2 is said to do better than base 2-shot out of distribution only if the pooled interval excludes
  zero. Otherwise the report says it did not show a difference.

## Cost

About 25 minutes on the M1 Pro under the compute lease: generation took 587 s (base 2-shot) and
564 s (finetuned-v2) for 150 valid prompts, and the 1B judge about 5 minutes for two arms.
