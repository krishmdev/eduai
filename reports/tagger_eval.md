# Curriculum tagger evaluation

Gold set: 150 SciQ items, 114 mapped to a learning objective and 36 marked off-curriculum. Labels were written by an AI labeling pass that did not see the tagger's output (53 flagged low-confidence); they still need a human spot-check.

Text tagged: question plus correct answer. Top-1 strict counts only the primary gold LO; lenient also accepts the listed alternates. Aligned means the gold LO is in the top 3 and the top score is at least tau (the 20th percentile of gold-LO scores). The CV columns fit tau on one half of the gold set and score the other half.

| Mode | Top-1 strict | Top-1 lenient | Top-3 | Subject gate | tau | Aligned (CV) | Off-curriculum rejected (CV) |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 49.1% | 59.6% | 75.4% | 87.7% | 0.297 | 67.5% | 61.1% |
| bge-small-en-v1.5 | 36.8% | 50.0% | 71.9% | 84.2% | 0.588 | 70.2% | 52.8% |
| bge-small-en-v1.5 (no query prefix) | 41.2% | 50.9% | 75.4% | 86.0% | 0.629 | 71.1% | 55.6% |
| bge-small-en-v1.5+rerank | 50.9% | 61.4% | 77.2% | 87.7% | -10.799 | 70.2% | 55.6% |
| bge-small-en-v1.5+rerank (no gate) | 52.6% | 63.2% | 77.2% | 88.6% | -10.799 | 71.9% | 55.6% |

Run manifest: `tagger_eval_manifest.json`
