"""Download the pinned OpenStax CNXML used by the OpenStax OOD check into data/openstax/raw/.

Only each book's collection file and its modules' index.cnxml are fetched (no media), from
raw.githubusercontent.com at the commit pinned in data/openstax/sources.json. Re-running skips
files that are already present.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "openstax" / "sources.json"
RAW = ROOT / "data" / "openstax" / "raw"


def _get(url: str, dest: Path) -> int:
    if dest.exists():
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return len(data)


def main() -> int:
    books = json.loads(SOURCES.read_text())["books"]
    total = 0
    for b in books:
        base = f"https://raw.githubusercontent.com/{b['repo']}/{b['sha']}"
        repo_dir = RAW / b["repo"].split("/")[1]
        col = repo_dir / "collections" / f"{b['collection']}.collection.xml"
        total += _get(f"{base}/collections/{b['collection']}.collection.xml", col)
        mods = re.findall(r'document="(m\d+)"', col.read_text())
        with ThreadPoolExecutor(8) as ex:
            sizes = list(
                ex.map(
                    lambda m, base=base, repo_dir=repo_dir: _get(
                        f"{base}/modules/{m}/index.cnxml", repo_dir / "modules" / m / "index.cnxml"
                    ),
                    mods,
                )
            )
        total += sum(sizes)
        print(f"{b['key']}: {len(mods)} modules at {b['sha'][:7]}")
    print(f"{total / 1e6:.1f} MB under {RAW.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
