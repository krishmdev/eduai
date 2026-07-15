Adapter weights are not committed. Run `uv run eduai fetch-adapter` to download the release asset
listed in `MANIFEST.json`. The command checks its sha256 and per-file hashes, then unpacks it into
`adapters/llama32-3b-eduai/`. To train the adapter yourself, run `make train` (about 2 hours on an
M1 Pro).
