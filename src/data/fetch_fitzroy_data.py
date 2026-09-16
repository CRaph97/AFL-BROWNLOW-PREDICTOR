"""
Download the pre-scraped, community-maintained fitzRoy data files that back the
fitzRoy R package (https://github.com/jimmyday12/fitzRoy), from its companion
data repository (https://github.com/jimmyday12/fitzroy_data), which is updated
nightly by that project's own CI and sourced from AFL Tables and Footywire
"with permission" (per that repo's README).

This is a deliberate choice NOT to scrape afltables.com / footywire.com directly
ourselves (see docs/DATA_SOURCE_AUDIT.md section 5): reusing an already-permitted,
actively-maintained aggregator is lower legal and engineering risk than writing
our own scraper against a structured source that already exists.

Every downloaded file is logged with its size and a SHA-256 hash to
data/raw/fitzroy_data/PROVENANCE.json so later runs can detect upstream changes.
Raw files are never modified in place.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request

REPO_RAW_BASE = "https://raw.githubusercontent.com/jimmyday12/fitzroy_data/main"

FILES = {
    "afldata.rda": "data-raw/afl_tables_playerstats/afldata.rda",
    "player_ids.csv": "data-raw/afl_tables_playerstats/player_ids.csv",
    "player_mapping_afltables.csv": "data-raw/afl_tables_playerstats/player_mapping_afltables.csv",
    "player_stats.rda": "data-raw/player_stats/player_stats.rda",
}

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fitzroy_data"


def _download(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "afl-brownlow-predictor-research (contact: local project owner)"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_all(dest_dir: Path = RAW_DIR) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for local_name, repo_path in FILES.items():
        url = f"{REPO_RAW_BASE}/{repo_path}"
        data = _download(url)
        out_path = dest_dir / local_name
        out_path.write_bytes(data)
        sha256 = hashlib.sha256(data).hexdigest()
        provenance[local_name] = {
            "source_url": url,
            "source_repo": "https://github.com/jimmyday12/fitzroy_data",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(data),
            "sha256": sha256,
        }
        print(f"downloaded {local_name}: {len(data):,} bytes, sha256={sha256[:12]}...")

    prov_path = dest_dir / "PROVENANCE.json"
    existing = {}
    if prov_path.exists():
        existing = json.loads(prov_path.read_text())
    existing.setdefault("history", []).append(provenance)
    existing["latest"] = provenance
    prov_path.write_text(json.dumps(existing, indent=2))
    return provenance


if __name__ == "__main__":
    fetch_all()
