# Data and model licenses

The code in this repository is MIT licensed (see `LICENSE`). Data and model artifacts have their
own terms.

## SciQ

Welbl, Liu and Gardner, "Crowdsourcing Multiple Choice Science Questions", W-NUT 2017.
Licensed under Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0).

SciQ-derived material is for non-commercial use and must keep this attribution:
- the committed samples in `data/samples/`, the gold sets in `data/gold/` (question and answer text),
  and the eval outputs in `reports/eval/`;
- the SFT data built locally by `eduai data build`;
- the LoRA adapter trained on that data.

The raw dataset is not committed. `make data` builds everything from a local copy.

## OpenStax (secondary out-of-distribution check)

`data/openstax/prompts.jsonl` holds passages and multiple-choice questions adapted from these
OpenStax textbooks, downloaded as CNXML from the commits pinned in `data/openstax/sources.json`:

- Biology for AP Courses, Julianne Zedalis and John Eggebrecht, OpenStax (Rice University).
  https://openstax.org/details/books/biology-ap-courses
- Biology 2e, Mary Ann Clark, Matthew Douglas and Jung Choi, OpenStax (Rice University). Used only
  for its review questions and their answer letters. https://openstax.org/details/books/biology-2e
- College Physics for AP Courses 2e, Gregg Wolfe, Erika Gasper, John Stoke, Julie Kretchman, David
  Anderson, Nathan Czuba, Sudhi Oberoi, Liza Pujji, Irina Lyublinskaya and Douglas Ingram, OpenStax
  (Rice University). https://openstax.org/details/books/college-physics-ap-courses-2e
- Chemistry 2e, Paul Flowers, Klaus Theopold, Richard Langley and William R. Robinson, OpenStax
  (Rice University). https://openstax.org/details/books/chemistry-2e

The pinned sources are licensed Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0), as stated in each collection file and repository LICENSE.
Changes: paragraphs were extracted from the section text, figures, equations, cross-references and
most math were removed or flattened to plain text, and paragraphs were merged or cut at sentence
boundaries. Questions keep their wording, except where the parser lost part of a stem: in the
committed prompt file, the book question for ox-phys-010 is missing its I-III statement list,
which later parser versions keep. The answer letter comes from the book's solution. The
derived prompt file is shared under the same license. The raw CNXML is not committed;
`scripts/fetch_openstax.py` downloads it.

Generated items and reports produced from these prompts are also derived from CC BY-NC-SA material
and stay non-commercial, like everything derived from SciQ.

## Llama 3.2

The base model is `mlx-community/Llama-3.2-3B-Instruct-4bit`, a 4-bit MLX conversion of Meta's
Llama 3.2 3B Instruct. The judge is the 1B variant. Both are under the Llama 3.2 Community License.

License text: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE
Acceptable Use Policy: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md
Both files are included in the adapter release tarball.

Built with Llama. The adapter distributed as a release asset (`Llama-3.2-3B-EduAI-AP-LoRA`) is a derivative of
Llama 3.2. It is subject to the Llama 3.2 Community License and Acceptable Use Policy. Its
training data also limits it to non-commercial use.

## Audit and verification models

These models judged items for the blind audit (`reports/eval/audit_llm/`) and the answer-key
verification (`reports/**/verify_qwen3.5-9b.jsonl`). Their verdicts and scores are committed; the
weights are not redistributed.

- Qwen3.5 9B (`mlx-community/Qwen3.5-9B-MLX-4bit`, pinned in `src/eduai/config.py`), a 4-bit MLX
  conversion of Qwen's Qwen3.5 9B: Apache-2.0.
- Gemma 4 12B (`mlx-community/gemma-4-12B-it-4bit`, revision 73bcf09 in the audit manifest), a
  4-bit MLX conversion of Google's Gemma 4 12B instruction-tuned model. Google's model card lists
  Apache-2.0 under the Gemma 4 license page; the pinned mlx-community card has no license field.

## Embedding and reranking models

- `BAAI/bge-small-en-v1.5`: MIT.
- `sentence-transformers/all-MiniLM-L6-v2`: Apache-2.0.
- `cross-encoder/ms-marco-MiniLM-L6-v2`: Apache-2.0.

## htmx

`src/eduai/web/static/htmx.min.js` is htmx 2.0.11, BSD Zero-Clause license.

## AP and the College Board

AP and Advanced Placement are registered trademarks of the College Board. This project isn't
affiliated with or endorsed by the College Board. The curriculum files in `curriculum/` use short,
generic unit titles and objective sentences written for this project; they aren't copied from
any College Board course framework. Unit weights are our own rough numbers.
