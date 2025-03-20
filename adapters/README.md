Adapter weights are not committed. `uv run eduai fetch-adapter` downloads the release asset listed
in `MANIFEST.json`, checks its sha256 and the per-file hashes, and unpacks it into
`adapters/llama32-3b-eduai/`. Training it yourself: `make train` (about 2 hours on an M1 Pro).
