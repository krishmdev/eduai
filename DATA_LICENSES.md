# Data and model licenses

The code in this repository is MIT licensed (see `LICENSE`). The data and model artifacts it uses
or produces carry their own terms.

## SciQ

Welbl, Liu and Gardner, "Crowdsourcing Multiple Choice Science Questions", W-NUT 2017.
Licensed under Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0).

Everything derived from SciQ is non-commercial and must keep this attribution:
- the committed samples in `data/samples/`, the gold sets in `data/gold/` (question and answer text),
  and the eval outputs in `reports/eval/`;
- the SFT data built locally by `eduai data build`;
- the LoRA adapter trained on that data.

The raw dataset is not committed. `make data` builds everything from a local copy.

## Llama 3.2

The base model is `mlx-community/Llama-3.2-3B-Instruct-4bit`, a 4-bit MLX conversion of Meta's
Llama 3.2 3B Instruct. The judge is the 1B variant. Both are under the Llama 3.2 Community License.

License text: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE
Acceptable Use Policy: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md
Both files are included in the adapter release tarball.

Built with Llama. The adapter distributed as a release asset (`Llama-3.2-3B-EduAI-AP-LoRA`) is a
derivative of Llama 3.2 and is subject to the Llama 3.2 Community License and Acceptable Use
Policy. Because of its training data it is also limited to non-commercial use.

## Embedding and reranking models

- `BAAI/bge-small-en-v1.5`: MIT.
- `sentence-transformers/all-MiniLM-L6-v2`: Apache-2.0.
- `cross-encoder/ms-marco-MiniLM-L6-v2`: Apache-2.0.

## htmx

`src/eduai/web/static/htmx.min.js` is htmx 2.0.11, BSD Zero-Clause license.

## AP and the College Board

AP and Advanced Placement are registered trademarks of the College Board. This project isn't
affiliated with or endorsed by the College Board. The curriculum files in `curriculum/` use short
generic unit titles and objective sentences written for this project; they aren't copied from
any College Board course framework. Unit weights are our own rough numbers.
