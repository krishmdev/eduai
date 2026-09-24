# EduAI

EduAI generates AP-style (format; SciQ source text) multiple-choice science questions with
Llama 3.2 3B, either the base model with two fixed examples or one of two LoRA adapters trained
here (the eval compares them).
An embedding-based tagger maps questions to learning objectives, and an adaptive test chooses
each next question from the student's earlier responses. It covers Biology, Chemistry,
Physics 1 and Environmental Science using SciQ source material. The pipeline runs on one
M1 Pro laptop.

<p>
<img src="docs/screenshots/desktop_feedback.png" alt="A practice question after answering: the wrong choice struck through, the correct one highlighted, an explanation from the source passage, and mastery by unit in a sidebar" width="68%">
<img src="docs/screenshots/mobile_question.png" alt="The same question card on a phone" width="24%">
</p>

```mermaid
flowchart LR
  S[SciQ 13,679 items] --> G[eligibility gate<br/>blank passages out]
  G --> T[curriculum tagger<br/>bge-small + rerank]
  C[curriculum YAML<br/>123 objectives] --> T
  T --> SP[leak-free splits<br/>passage, Q+A, cosine groups]
  SP --> SFT[SFT chat data<br/>3,000 / 300 / 200]
  SFT --> L[MLX LoRA<br/>Llama 3.2 3B 4-bit]
  L --> GEN[generate] --> V[validate<br/>schema, structure, judge key,<br/>alignment, novelty]
  V --> B[(item bank<br/>SQLite)]
  SP --> B
  B --> A[adaptive session<br/>EAP + Fisher info, BKT]
  A -- responses --> B
```

## Quickstart (no model or API key)

```bash
make setup-demo   # installs only core app dependencies; package download may need network
make demo         # bank-only app on http://127.0.0.1:8001
```

The demo serves a committed sample of 634 SciQ-derived items (`data/samples/bank_sample.jsonl`)
and 32 generated items. The generated ones come from the v1 adapter (source `generated:finetuned`)
and passed the eval's automatic checks. They were promoted before the answer-key verification gate
existed and haven't been through it. The demo needs neither the raw dataset nor a model. In the browser,
choose Biology, start a 10-question practice session, answer a few questions to see source-passage feedback and unit progress, then
open the report. Assessment mode withholds the answer until the session ends and shows an
illustrative ability estimate. The sample questions and AI-derived labels await human review;
the session does not measure learning in real students.

`make setup` is for the full training/evaluation workflow and fetches pinned embedding models.
Once dependencies are installed, confirm the whole demo process tree runs without
network access:

```bash
tools/offline-run make e2e-offline   # sandbox-exec profile that denies outbound network (macOS)
make egress-open-check               # the same canary connects when unsandboxed
```

`e2e-offline` starts the API server and turns on an egress canary inside the server process. It
then runs a practice session and an assessment over localhost and fails if any process could
reach 1.1.1.1, api.openai.com or huggingface.co. CI runs the same script in a Linux container
started with `--network none`.

## Full pipeline

```bash
make data SCIQ_DIR=~/Downloads/SciQ\ dataset-2\ 3   # SciQ JSON -> tagged items, splits, SFT data, data card
make models-llm                                      # Llama 3.2 3B 4-bit (1.8 GB) + 1B judge, pinned revisions
make train                                           # about 2 hours; the adapter release isn't published yet
make eval                                            # 3 arms x 150 prompts, then judge and score (about 45 min)
make serve                                           # full app; auto-selects MLX base 2-shot, then the adapter
make test
```

The app tries backends in this order: the MLX base model with the eval's two fixed examples
(it beat the v1 adapter on usable items, and the v2 adapter didn't beat it; v2 isn't released),
then MLX with the v1 LoRA adapter, then Ollama
(`llama3.2:3b`, base only; Ollama can't load the MLX adapter), then bank-only. You can force one
with `EDUAI_BACKEND=mlx|adapter|ollama|bank`.

## Curriculum

`curriculum/*.yaml` defines 123 learning objectives, organized as unit, then topic, then
objective. Each objective has keywords, a Bloom level, and one of our own blueprint weights. The
objective sentences and unit titles were written for this project, loosely following each AP
course's scope. See [curriculum/README.md](curriculum/README.md). This project isn't affiliated
with the College Board.

## Data

The [data card](reports/data_card.md) follows the pipeline in order:

1. **Eligibility gate.** 1,427 SciQ records have a blank `support` passage: 1,198 train, 113
   valid and 116 test.
   - These can't produce passage-grounded explanations, so they are kept out of SFT and eval.
   - They stay in the item bank flagged `ungrounded`, with no explanation.
   - Their blank passage never gets a hash, so they don't collapse into one split group.
2. **Curriculum alignment.** The tagger's score has to clear tau. The data card counts
   exclusions per split and per subject before any quota is applied.
3. **Split grouping.** Union-find links items that share a passage, have the same normalized
   question and answer, or have a bge cosine above 0.92 on question plus answer.
   - Each group goes to one split, with test taking priority, then valid, then train.
   - 813 items moved. A post-assignment check (`splits.leakage`) confirms no group spans two splits.
4. **Quotas.** SFT is 3,000 / 300 / 200. The answer letter is exactly 25% each of A through D, 40% of
   items are stimulus style, and 30% carry a target misconception. All quotas were met.

The prompt lives in `prompts.py` and is shared by training and inference.
`tests/test_prompts.py` checks that no other module builds a generation prompt.

Here is an example SFT pair, abridged from `data/samples/sft_sample.jsonl`:

```
user:  Subject: AP Biology / Unit: Energy in cells / Topic: Photosynthesis
       Learning objective (BIO.3.2.a): Describe how the light reactions and the Calvin cycle ...
       Difficulty: easy / Format: stimulus / Source passage: The primary function of leaves is ...
assistant: {"stem": "Based on the excerpt, the process by which leaves collect sunlight and make food
       is called this?", "stimulus": "...", "choices": {"A": "glycolysis", "B": "photosynthesis", ...},
       "answer": "B", "explanation": "B is correct. The primary function of leaves is ...", ...}
```

## Fine-tuning

The fine-tune is mlx-lm 0.31.3 LoRA on `mlx-community/Llama-3.2-3B-Instruct-4bit`
([configs/lora_llama32_3b.yaml](configs/lora_llama32_3b.yaml)). Settings: rank 8, the last 16
layers, lr 2e-5, 600 iterations at batch 4, prompt masked, and gradient checkpointing. No other
training or benchmark job ran at the same time; background load during training wasn't recorded.

| Run | Iterations | Wall time | Peak memory | Train tokens/s | Source |
|---|---|---|---|---|---|
| Pilot | 20 | 240 s | 6.60 GB | 56.2 | `reports/pilot.json` |
| Full | 600 | 6,995 s (1.94 h) | 7.65 GB | 56.7 | `reports/training.json` |

The pilot projected 1.68 h, below the 2.5 h fallback threshold. The full run used the 3B model
with 16 layers and trained 6.95M parameters on 348K target tokens.

![loss](reports/figures/training_loss.png)

Validation loss fell from 0.937 to 0.293. Almost all of that drop happens in the first 100
iterations, and the loss is mostly about the output format. In the validation targets, 37% of
completion tokens are JSON syntax, and 41% sit inside spans copied verbatim from the prompt
([reports/target_token_share.json](reports/target_token_share.json)). The training
explanations are extracted passage sentences, so the model learns to extract, not to explain.

The adapter is not committed. It is packaged for a GitHub Release asset `adapter-v1`
(`llama-3.2-3b-eduai-lora-v1.tar.gz`, with the Llama 3.2 license and use policy inside), and its
sha256 and size are listed in [adapters/MANIFEST.json](adapters/MANIFEST.json). That release isn't
published yet, so `eduai fetch-adapter` fails for now; train with `make train`, or pass a local
tarball with `eduai fetch-adapter --source`. Once the release is up, `eduai fetch-adapter`
downloads and verifies it. The CUDA route is
[notebooks/colab_qlora_peft.ipynb](notebooks/colab_qlora_peft.ipynb) (QLoRA with PEFT and TRL).
That notebook is a reference and hasn't been run.

## Curriculum tagging

The tagger works in four steps. A subject gate compares the item against subject centroids.
bge-small then retrieves the top 10 objectives, and the ms-marco MiniLM cross-encoder reranks
them. Finally, tau is calibrated as the 20th percentile of gold-objective scores. Here are the
results on the 150-item gold set ([reports/tagger_eval.md](reports/tagger_eval.md)):

| Mode | Top-1 | Top-3 | Unit | Subject | Aligned (CV) | Off-curriculum rejected (CV) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 49.1% | 75.4% | 64.9% | 87.7% | 65.8% | 61.1% |
| bge-small-en-v1.5 | 36.8% | 71.9% | 64.0% | 84.2% | 65.8% | 52.8% |
| bge-small + rerank | 50.9% | 77.2% | 66.7% | 87.7% | 68.4% | 55.6% |

The tagger is a noisy filter.
- It wrongly rejects about 30% of in-curriculum items, and it passes about 44% of off-curriculum
  ones.
- The reranker adds little over plain MiniLM.
- Because of this, LO conditioning in generation is weakly supervised. The app only serves
  aligned items, and it reports mastery at the unit level.

The gold labels, and the independent eval-prompt labels described below, came from an AI
labeling pass that never saw tagger output. They still need a human spot-check.

## Generation eval

<!-- eval:start -->
There are 150 prompts from groups assigned to the held-out test split (including SciQ rows originally labeled train or valid), and every arm gets the same prompts ([reports/eval_report.md](reports/eval_report.md), raw generations in `reports/eval/`). This section is written by `scripts/readme_eval.py`.

- **Target objectives are independent.** Each prompt's target LO comes from an independent labeling pass, not from the tagger. 59 off-curriculum candidates were dropped.
- **Leaky prompts are filtered out.** A prompt was dropped if its passage shares 50% or more of its 8-grams with a training passage, or if a training item has the same answer and a question-plus-answer cosine of 0.88 or more. That removed 14 of 244 screened candidates.
- **The judge is fixed, open-book, and not one of the generators.** Llama 3.2 1B sees the passage and scores the options by log-probability, averaged over all four rotations of the options.
- **Novelty excludes the prompt's own source item and its group.** Copies of the source question are counted separately. **Usable** means passing all checks and not being a source copy; it is the criterion for promoting an item into the bank.
- **Memorization is reported only.** It counts items within 0.92 cosine of an SFT training stem, and it does not reject anything.

| Arm | Schema valid | Structure | Key agreement (schema-valid items) | Aligned | Novel | All checks | Source copy | Usable | Gen tok/s |
|---|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | | | 93.3% | 70.7% | | | | | |
| Base 3B, 0-shot | 93.3% | 40.7% | 75.7% | 74.7% | 92.7% | 26.7% | 4.7% | 23.3% | 80.3 |
| Base 3B, 2-shot | 80.7% | 60.7% | 75.2% | 64.0% | 80.0% | 36.7% | 2.7% | 34.7% | 76.1 |
| Base 3B + EduAI LoRA v1 | 100.0% | 54.0% | 66.0% | 72.0% | 96.0% | 30.7% | 25.3% | 21.3% | 48.0 |
| Base 3B + EduAI LoRA v2 | 100.0% | 66.0% | 75.3% | 79.3% | 99.3% | 38.0% | 8.7% | 32.0% | 46.4 |
| Base 3B + EduAI LoRA v2, 2-shot | 99.3% | 73.3% | 77.2% | 78.0% | 99.3% | 44.0% | 9.3% | 37.3% | 44.9 |

The v1 fine-tuned model does not generate better questions overall. The paired bootstrap over prompts gives these differences:

- v1 fine-tuned vs 2-shot on usable items: -13.3 points (95% CI -23.3 to -3.3).
- v1 fine-tuned vs 0-shot on usable items: -2.0 points (95% CI -12.0 to +7.3).
- 2-shot vs 0-shot on usable items: +11.3 points (95% CI +1.3 to +20.7).

v1 fine-tuning fixed the output format:

- It produced schema-valid JSON on 100.0% of prompts, against 80.7% for 2-shot.
- It almost never drops the requested misconception.

Key agreement on schema-valid items is 66.0% for the fine-tune, 75.2% for 2-shot and 75.7% for 0-shot. An item whose key text is repeated among its options counts as not agreeing, and many of the fine-tune's items have repeated options.

It also learned the wrong things from its targets, which were the SciQ source questions:

- It copies the source question 25.3% of the time.
- It writes near-identical options on 63 of 150 items, including 9 where all four options are the same string.
- It puts the key at A on 104 of 150 valid items, even though the SFT answer letters were exactly 25% each.

Before items enter the bank, their options are reshuffled and the key is remapped. Alignment is at the reference ceiling (70.7%) for 0-shot and fine-tuned; 2-shot is lower at 64.0%, but that counts its schema failures as misses (79.3% on its schema-valid items). Generation with the unfused adapter ran at 48 tok/s, against 80 for the base model, on a machine with other background load (see the eval manifest), so treat the speeds as rough.

In short, the v1 LoRA fine-tune of Llama 3.2 3B taught format reliability, but the 3,000 SciQ-derived targets also taught copying and a key-position bias.

### v2 adapter

The v2 LoRA was trained on items the base 3B wrote itself: samples drawn with the eval's two fixed examples, kept only when they passed the checks above with the base 3B as the key judge, one per training prompt (1,606 train rows; [reports/rft_card.json](reports/rft_card.json)). The checkpoint, and the choice between v2 with and without the two examples, were made on the valid split and written down before the test run ([docs/v2_selection.md](docs/v2_selection.md)). The v2 arms were run on the test set once. The base and v1 rows above are the committed ones from before v2.

- Pre-registered headline, v2 0-shot vs 2-shot base on usable items: -2.7 points (95% CI -13.3 to +8.0). The interval includes zero, so v2 did not beat 2-shot prompting of the base model.
- Without the 2 test prompts whose passages are also v2 training passages (test-00916, train-02955): -4.1 points (95% CI -14.9 to +6.8).
- v2 with the two fixed examples (37.3% usable) vs 2-shot base: +2.7 points (95% CI -6.7 to +12.0). This arm scored higher than v2 0-shot on test but lower on valid, where the choice was made, so it isn't the headline.
- v2 vs v1 on usable items: +10.7 points (95% CI +1.3 to +20.0).

Compared with v1, v2 copies the source question less (8.7% against 25.3%), puts the key at A less often (61 of 150 valid items, against 104), and writes near-identical options on 22 items instead of 63. It keeps v1's schema-valid rate (100.0%). Its key agreement on schema-valid items (75.3%) is about the same as 2-shot base (75.2%). Its most common structure problem is a stem that gives away the answer (27 items, against 11 for 2-shot base).

Usable overstates v2 more than the other arms. v2's targets were picked with these same checks and with the base 3B as key judge, and the 3B gives the same key verdict as the 1B eval judge on 78.7% of v2's schema-valid items (74.4% for 2-shot base, 90.7% for v1). The manual blind audit planned in the protocol was not done. Instead, two LLM judges from other model families audited a sample of usable test items blind (see the blind audit below). No person has checked these items.
<!-- eval:end -->

`make eval-check` audits the committed evaluation snapshot without models: it checks the 150
paired prompt IDs, generation and judge hashes, per-item rates, paired bootstrap, and rendered
report. The [prompt index](reports/eval/prompt_index.json) records original SciQ split IDs,
assigned held-out groups, and hashes of the local selection inputs without publishing their
passages. It was checked against the local SFT train/valid groups (zero group overlap); a fresh
clone can verify the committed snapshot but cannot independently repeat that source-data check.
Full `make eval-score` recomputation requires the ignored SciQ-derived data and pinned embedding
models described in the full pipeline. Neither audit validates the AI-generated gold labels;
they still need human review.

### OpenStax out-of-distribution check

A secondary check, run once as pre-registered in [docs/openstax_ood.md](docs/openstax_ood.md):
150 prompts built from passages of pinned OpenStax textbooks (Biology for AP Courses, College
Physics for AP Courses 2e, Chemistry 2e; 50 per subject), none of which the models trained on.
It compares 2-shot base with v2 (0-shot), judged by the 1B key judge only, with the same checks as
the main eval. Full table: [reports/openstax_ood/report.md](reports/openstax_ood/report.md).

| Subject | n | 2-shot base usable | v2 usable | v2 minus base | 95% CI |
|---|---|---|---|---|---|
| Biology | 50 | 18.0% | 40.0% | +22.0 | +8.0 to +38.0 |
| Physics 1 | 50 | 34.0% | 30.0% | -4.0 | -22.0 to +14.0 |
| Chemistry | 50 | 28.0% | 30.0% | +2.0 | -14.0 to +18.0 |
| Pooled | 150 | 26.7% | 33.3% | +6.7 | -2.7 to +16.0 |

Intervals are paired bootstraps over prompts (5,000 resamples). The pre-registered reading uses
the pooled interval, and it includes zero, so this check didn't show a difference between v2 and
2-shot base out of distribution. The per-subject intervals are descriptive only. Biology's is
the one that excludes zero. v2 had higher key agreement (62.0% vs 49.3%) and alignment (80.7% vs
59.3%) but failed the structure checks more often (49 items vs 19 rejected there first). The run
briefly overlapped another process using the GPU without the lease, so the scores were checked
again on CPU and 10 generations were regenerated: every verdict matched, and all 10 generations
were byte-identical ([verification.json](reports/openstax_ood/verification.json)). That was a
check, not a second run.

### Blind audit with LLM judges

No person reviewed the generated items. Instead, two language models from families other than
Llama read usable test items with the arm hidden and judged whether the key is correct, whether
the item fits its objective, and whether the distractors are plausible. The two judges are
Qwen3.5 9B and Gemma 4 12B, served locally through Localhost AI with thinking turned on. Their
verdicts are another machine opinion, and they can be wrong. The plan and its three dated
amendments are in [docs/blind_audit_llm.md](docs/blind_audit_llm.md). Results are in
[reports/eval/audit_llm/](reports/eval/audit_llm/).

What the audit covers, and where it departs from the original plan:

- Test split only. The draw is 40 usable items each from v2 (0-shot) and 2-shot base. The
  valid audit was dropped for lack of GPU time.
- Truncated thinking. A pilot showed both models wanting more than 6,000 thinking tokens on most
  items, so the caps were fixed at 3,072 tokens for Qwen and 2,048 for Gemma. Qwen reached its
  cap on 79 of 80 items and Gemma on 34 of 40, so most verdicts come from cut-off reasoning.
- Gemma judged a subset, the first 20 items of each arm in the blinded sheet order. It ran at
  concurrency 1 to leave the host memory headroom. One Gemma verdict never parsed, which leaves
  39 items.

Precision here means the share of "usable" items a judge rates as both correctly keyed and on
objective. Intervals are bootstrap 95% intervals, resampling each arm separately.

| Judge | Items | v2 | 2-shot base | v2 minus base |
|---|---|---|---|---|
| Qwen3.5 9B | 40 + 40 | 5.0% (0.0 to 12.5) | 7.5% (0.0 to 15.0) | -2.5 (-12.5 to +7.5) |
| Gemma 4 12B | 19 + 20 | 52.6% (31.6 to 73.7) | 45.0% (25.0 to 65.0) | +7.6 (-23.2 to +38.7) |

| Judge | Key correct, v2 | Key correct, base | On objective, v2 | On objective, base |
|---|---|---|---|---|
| Qwen3.5 9B | 57.5% | 57.5% | 10.0% | 7.5% |
| Gemma 4 12B | 78.9% | 75.0% | 68.4% | 65.0% |

- Neither judge separates the two arms: both intervals for the difference include zero.
- The judges disagree on the objective much more than on the key. Going by its notes, Qwen
  reads "fits the objective" narrowly: when an objective lists several ideas, it often rejects a
  question that tests only one of them. On the 39 items both judges rated, Cohen's kappa is 0.60 for the key, 0.08 for
  the objective, 0.14 for distractors, and 0.16 for the combined verdict. Raw agreement is 82% on
  the key and 41% on the objective.
- Agreement with the eval's Llama 3.2 1B key judge equals each judge's key-correct rate (57.5% for
  Qwen, 76.9% for Gemma), because every usable item had already passed the 1B check. Kappa against
  the 1B is undefined for the same reason.

The two judges don't agree on how many usable items are real, so this audit can't put a number on
how much "usable" overstates either arm. It gives no sign that v2's usable items are better or
worse than 2-shot base's.

### Post-hoc: answer-key verification

This part is exploratory. I planned it after seeing the audit, which found wrong keys among items
the eval had marked usable, so it doesn't change the registered v2 result above. The plan was
written down before any item was scored:
[docs/key_verification.md](docs/key_verification.md). Results are in
[reports/key_verification.md](reports/key_verification.md).

The verifier is Qwen3.5 9B (4-bit, thinking off) running the eval's own key check: open-book,
answer-letter probabilities averaged over four rotations of the options. An item passes when the
verifier's top option is the key and the key gets more than 0.5 of the probability. The threshold
was fixed in advance. It scored every schema-valid item of the three arms on valid and then on
test, 841 items in one run.

| Split | Arm | Usable | Usable and verified | Usable items kept |
|---|---|---|---|---|
| Valid | 2-shot base | 32.0% | 28.0% | 87.5% |
| Valid | v2 | 38.0% | 30.7% | 80.7% |
| Valid | v2, 2-shot | 34.0% | 28.7% | 84.3% |
| Test | 2-shot base | 34.7% | 28.7% | 82.7% |
| Test | v2 | 32.0% | 26.7% | 83.3% |
| Test | v2, 2-shot | 37.3% | 34.0% | 91.1% |

Verification removes 9% to 19% of the usable items, depending on the arm and split. On test,
v2 minus 2-shot base on verified usable is -2.0 points (95% CI -12.0 to +8.0). These are
descriptive numbers, not a new test.

Gemma 4 12B is the independent check, since it comes from neither the Llama nor the Qwen family.
Of the 39 audit items it rated, 35 pass verification and 4 fail. Gemma judged the key correct on
80.0% of the passing items (95% CI 64% to 90%) and on 2 of the 4 failing ones, against 76.9% for
all 39. So verification nudges Gemma-judged key correctness up by about three points on this
subset, but with 4 failures the difference can't be told from chance (Fisher p = 0.22). The Qwen
audit shows a much larger gap (69% of verified items judged correct against 7% of unverified). The
verifier and that auditor are the same model family, though, so the gap is expected and doesn't
count as evidence.

Bank promotion (`eduai bank add-generated`) now requires the verifier's pass by default. Use
`--no-require-verified` or `EDUAI_REQUIRE_KEY_VERIFICATION=false` to get the old gate back.

## Adaptive testing and feedback

The whole system uses one response model, p = c + (1 − c)·σ(θ − b) with c = 0.25 for four
options (`kt/irt.py`).

- **Assessment.** The ability estimate is the EAP posterior on a 161-point grid with a N(0, 1)
  prior.
  - Each next item maximizes Fisher information I = p′² / (p(1 − p)). Under this model the peak
    falls at p* = (1 + √(1 + 8c)) / 4 ≈ 0.683, not at 0.5.
  - The only stopping rule is posterior SD below a threshold or a cap on the number of items.
  - Tests check the information against a finite-difference p′, check that the argmax matches p*
    for c ∈ {0, 0.2, 0.25}, and check the grid posterior against a fine-grid reference.
- **Practice.** Items are chosen to target p = 0.7. That target is a desirable-difficulty
  choice for teaching, not an information-maximizing one.
  - The next learning objective comes from a UCB score over BKT mastery, within the unit
    blueprint quotas.
- **Item calibration.** An Elo-style step on the same p runs inside a SQLite transaction. It
  uses θ from the student's EAP estimate before the response. An item keeps its label
  difficulty until it has 5 responses.
- **Mastery.** Per-objective BKT with difficulty-adjusted guess and slip, plus a Baum-Welch EM
  fit. Unit mastery is averaged over the objectives the student has actually seen.

These numbers are from 500 simulated students per condition ([reports/sim_report.md](reports/sim_report.md)):

![A](reports/figures/sim_a_rmse.png)

| Stop when SD < | Coverage of true θ by θ̂ ± 1.96 SD | RMSE | Mean length | Stopped by precision (cap 80) |
|---|---|---|---|---|
| 0.3 | 95.8% | 0.292 | 71.6 | 93.2% |
| 0.4 | 95.0% | 0.377 | 38.7 | 99.8% |
| 0.5 | 95.2% | 0.495 | 22.9 | 100% |

Posterior SD is well calibrated at every threshold. Reaching SD < 0.3 takes about 72 items even
when items are well targeted, so the web assessment stops at SD < 0.5 or 30 items.

Adaptive item selection gives lower RMSE than random items at every test length. At 30 items it is
0.444 against 0.476, and at 60 items 0.318 against 0.366. The gap is modest, though. Every policy
sees the same simulated students, and on paired squared error the 95% bootstrap interval excludes
zero only at 60 items; up to 40 items it includes zero. A fixed form does worse, with intervals
excluding zero at 20, 40 and 60 items.

In the practice simulation, a student learns with the highest probability when p is close to 0.7.
Practice mode targets exactly that value, so the adaptive arm is built to benefit from the
assumption. To separate the two effects, one arm keeps the p = 0.7 item targeting but picks the
objective at random:

| Practice policy, 300 questions | Mean true mastery | Reached 80% | Adaptive minus this (95% CI) |
|---|---|---|---|
| Adaptive (UCB objective + p = 0.7 items) | 82.9% | 74.6% | |
| Random objective + p = 0.7 items | 79.5% | 61.6% | +3.3 pts [+2.6, +4.0] |
| Random objective and item | 75.5% | 53.8% | +7.4 pts [+6.6, +8.2] |
| Round-robin objective | 76.7% | 56.8% | +6.1 pts [+5.3, +6.9] |

Other simulation results:

- Item calibration beats the label difficulty once responses come in. The RMSE of b is 0.514 from
  labels alone, 0.407 after 20 responses, and 0.246 after 160. A step size of 0.6 made b worse
  (0.582 at 5 responses), so the step size is 0.15.
- Evidence sharing between neighboring objectives didn't help. The Brier score was 0.2062 with it
  and 0.2038 without, so it is off in the app.
- The EM fit's guess parameter hit its 0.49 bound. This is model mismatch: two-state BKT can only
  explain correct answers that come from ability as guessing. It is reported rather than tuned
  away.

All of this is simulation, and the responses come from the same model the system assumes.
The estimators behave as designed in this simulation. These results say nothing about real students.

## App

The app is FastAPI with Jinja and htmx, plus a JSON API under `/api/` and SQLite storage.

Practice mode gives feedback and the source-passage explanation after each answer, with a
mastery-by-unit sidebar. Assessment mode gives no feedback until the end: the page shows only how
close the estimate is to its precision target, rounded to 0.1 so the size of each step doesn't
reveal whether an answer was right. The report page and the JSON API hold back correctness,
ability and unit results until the session finishes. The report leads with the ability estimate and its
95% interval, then the trajectory chart, unit mastery with counts, wrong answers with the key, and
the full response log. Its 1 to 5 score is an illustrative mapping, labeled as simulated. An
unfinished practice session's report is marked as in progress, with a link back to the session.

<img src="docs/screenshots/desktop_report.png" alt="Report page with the ability estimate and interval, trajectory chart, unit mastery table, wrong answers, and answer log" width="70%">

## Limitations

- The data is narrow. SciQ questions are short crowdworker recall items, Physics 1 and APES
  coverage is thin, and many items are off-curriculum (astronomy, anatomy trivia). "AP-style"
  describes the prompt and format; the source material isn't at AP level.
- Neither fine-tune beat 2-shot prompting on usable items (see the eval). v1 copies source
  questions, often repeats options, and favors key A. v2, trained on the base model's own
  checked samples, fixed much of that, but on test it was 2.7 points behind 2-shot base, with an
  interval that includes zero. Its targets were chosen with the eval's own checks, so usable
  flatters it. The only audit of its items is by two LLM judges, on the test split only and with
  truncated thinking, and the judges disagree with each other.
- Explanations are extracted passage sentences, not reasoning.
- All judges are small: Llama 3.2 1B for the key check and the noisy tagger for alignment.
  Neither replaces human review.
- Both gold label sets came from an AI labeling pass and are waiting for a human spot-check.
- The adaptive-testing results come from simulated students only.
- Not verified on this machine: the CUDA/Colab notebook and `HFBackend` (no NVIDIA GPU), and the
  Ollama backend with a pulled model.
- The v1 SFT splits were built before the passage-containment link
  (`eduai data build --containment-link`) and before the alignment rule compared the target
  objective's own score. The eval prompts are filtered against the training rows the adapter
  actually saw, and `configs/sft_v1.sha256` pins those files.

## Credits and licenses

The code is MIT. SciQ is CC BY-NC 3.0, so the samples, eval outputs and adapter are
non-commercial. The models are Llama 3.2 under the Llama 3.2 Community License ("Built with
Llama"). The embedders are bge-small (MIT) and MiniLM (Apache-2.0). See
[DATA_LICENSES.md](DATA_LICENSES.md). AP is a registered trademark of the College Board, which
isn't affiliated with this project.

The OpenStax prompts for the out-of-distribution check are adapted from OpenStax textbooks
(Rice University) under CC BY-NC-SA 4.0, so the prompt file and everything generated from it are
shared under that license. The raw book files aren't committed. Attribution and the list of
changes are in [DATA_LICENSES.md](DATA_LICENSES.md).
