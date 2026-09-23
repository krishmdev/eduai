# OpenStax out-of-distribution check

Secondary check, run once, pre-registered in `docs/openstax_ood.md`. 150 prompts from pinned OpenStax AP textbooks (50 each for biology, physics and chemistry; AP Environmental Science has no OpenStax book). Target objectives are the tagger's own top-1 over the passage. Judge: llama-1b. Usable is the main eval's definition. Manifest: `openstax_manifest.json`.

Reference coverage: 62 of 150 prompts have a human-written book question from the same section; the reference row is over those prompts only.

| | n | JSON | Structure | Key agreement | Aligned | Novel | Source copy | Usable |
|---|---|---|---|---|---|---|---|---|
| Book question (ceiling) | 62 | | | 41.9% | 74.2% | | | |
| base-2shot | 150 | 98.0% | 64.7% | 49.3% | 59.3% | 77.3% | 0.0% | 26.7% |
| finetuned-v2 | 150 | 100.0% | 67.3% | 62.0% | 80.7% | 98.7% | 0.0% | 33.3% |

Key agreement and alignment above count every prompt, so an item that isn't schema-valid counts as neither. On schema-valid items only:

| | Schema-valid items | Key agreement | Aligned |
|---|---|---|---|
| base-2shot | 116 | 63.8% | 76.7% |
| finetuned-v2 | 150 | 62.0% | 80.7% |

Usable by subject, finetuned-v2 minus base-2shot, paired bootstrap (5,000 resamples over prompts), 95% CI.

| Subject | n | base-2shot | finetuned-v2 | Difference | 95% CI |
|---|---|---|---|---|---|
| BIO | 50 | 18.0% | 40.0% | +22.0 | [+8.0, +38.0] |
| PHYS1 | 50 | 34.0% | 30.0% | -4.0 | [-22.0, +14.0] |
| CHEM | 50 | 28.0% | 30.0% | +2.0 | [-14.0, +18.0] |
| pooled | 150 | 26.7% | 33.3% | +6.7 | [-2.7, +16.0] |

Rejections (first failed check): {"base-2shot": {"key_disagreement": 37, "accepted": 40, "schema": 31, "not_aligned": 20, "structure": 19, "no_json": 3}, "finetuned-v2": {"structure": 49, "accepted": 50, "key_disagreement": 38, "not_aligned": 11, "duplicate": 2}}

Prompt build: {"BIO": {"modules": 167, "chunks": 2526, "screened": 113, "dropped_tag_off_subject": 25, "dropped_sciq_containment": 27, "dropped_sft_containment": 11, "ref_dropped_same_answer_qa": 2, "with_reference": 42, "labeled_from_reference": 28, "no_section_mcq": 3, "ref_below_min_cos": 5, "kept": 50, "mcqs_parsed_in_reference_book": 800}, "PHYS1": {"modules": 242, "chunks": 1021, "screened": 124, "dropped_tag_off_subject": 56, "dropped_sciq_containment": 14, "dropped_sft_containment": 4, "no_section_mcq": 20, "ref_dropped_same_answer_qa": 0, "with_reference": 20, "labeled_from_reference": 15, "ref_below_min_cos": 10, "kept": 50, "mcqs_parsed_in_reference_book": 177}, "CHEM": {"modules": 114, "chunks": 957, "screened": 71, "dropped_sciq_containment": 5, "dropped_tag_off_subject": 4, "dropped_sft_containment": 12, "no_section_mcq": 50, "kept": 50, "mcqs_parsed_in_reference_book": 1}}

## Verification (not a second run)

- A verification of the single OpenStax run, not a second run. Another process briefly used the GPU without the compute lease between about 18:02 and 18:05 on 2026-09-18, overlapping the start of base-2shot generation (lease acquired 18:02:50; scoring ran at about 18:24-18:27).
- No .cache/embeddings entries were written between 17:59 and 18:07 (worktree and main tree), so none were deleted.
- All 362 per-item rows (150 per arm plus 62 references) rescored from the committed generations and judge file with EDUAI_EMBED_DEVICE=cpu and an empty embedding cache. Every alignment, novelty, source-copy and usable verdict matched the MPS scoring; only source_cos differed, by at most 2.4e-7. Summary and per-subject tables identical.
- The first 10 base-2shot prompts (the ones generated during the overlap) regenerated under the lease: 10 of 10 byte-identical to the committed generations.
