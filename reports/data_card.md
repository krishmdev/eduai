# Data card

Source: SciQ (Welbl, Liu and Gardner, 2017), CC BY-NC 3.0. Counts below are produced by `eduai data build`; the machine-readable version is `reports/data_card.json`.

## 1. Eligibility gate (before any split sizes or quotas)

Records with a blank `support` passage cannot give passage-grounded explanations. They are excluded from SFT and stimulus pools and kept only in the bank, flagged `ungrounded`, with no explanation.

| SciQ split | Records | With support | Blank support (excluded from SFT) |
|---|---|---|---|
| train | 11679 | 10481 | 1198 |
| valid | 1000 | 887 | 113 |
| test | 1000 | 884 | 116 |
| all | 13679 | 12252 | 1427 |

## 2. Exclusions per split

Applied in order: blank support, support with no usable sentence, then curriculum alignment (tagger score below tau). `moved` counts grounded, aligned items whose near-duplicate group touched a higher-priority split (test > valid > train) and were reassigned there.

| SciQ split | Total | Blank support | No usable sentence | Not aligned | Moved by grouping |
|---|---|---|---|---|---|
| train | 11679 | 1198 | 0 | 2204 | 631 |
| valid | 1000 | 113 | 0 | 176 | 37 |
| test | 1000 | 116 | 0 | 198 | 0 |

## 3. Exclusions per subject (subject = tagger's top-1)

| Subject | Total | Blank support | No usable sentence | Not aligned | Eligible |
|---|---|---|---|---|---|
| APES | 2322 | 243 | 0 | 466 | 1613 |
| BIO | 6481 | 968 | 0 | 1433 | 4080 |
| CHEM | 2965 | 133 | 0 | 317 | 2515 |
| PHYS1 | 1911 | 83 | 0 | 362 | 1466 |

## 4. Near-duplicate grouping

Union-find over shared passage hash (118 links), identical normalized question+answer (62 links) and bge cosine > 0.92 on question+answer (1848 links). 11651 groups, largest 21 items. Reassigned items: {'train->test': 466, 'train->valid': 304, 'valid->test': 43}. Groups spanning more than one split after assignment: 0.

## 5. Eligible pools after exclusions

| Split | Eligible |
|---|---|
| train | 7646 |
| valid | 923 |
| test | 1105 |

Difficulty tertile cut points (from the grounded pool): -0.26 and 0.261.

## 6. Quotas (applied last)

Targets: SFT 3,000/300/200, answer letters 25% each, 40% stimulus-style, 30% with a target misconception. When the available data cannot meet a target, the achieved count is reported as-is.

| Split | Target | Achieved | A | B | C | D | Stimulus | Misconception |
|---|---|---|---|---|---|---|---|---|
| train | 3000 | 3000 | 750 | 750 | 750 | 750 | 1200 (40%) | 900 (30%) |
| valid | 300 | 300 | 75 | 75 | 75 | 75 | 120 (40%) | 90 (30%) |
| test | 200 | 200 | 50 | 50 | 50 | 50 | 80 (40%) | 60 (30%) |

Held-out eval prompts (from the SciQ test split, disjoint from SFT test rows): 150.
