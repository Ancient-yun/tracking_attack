"""Fetch official St4RTrack Seq weights and WorldTrack release; never substitute weights.

Public folder snapshots and per-file sizes/SHA256 are saved for reproducibility.
Downloads use resumable gdown streams, local cookies, and refuse HTML/truncated files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import gdown
import requests

from inspect_drive import entries

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    "po_mini": "1ToXkVJKlHs6xhCBaY4LB29whViH4RAHn",
    "ds_mini": "1jl9Og6MWLj1q3runKkpkE5iH14ba6_3T",
}
CHECKPOINT = {
    "id": "1ElgLYxWNHmps7-xvmHz2D6w4B0kZbBd1",
    "name": "St4RTrack_Seqmode_reweightMax5.pth",
    "bytes": 4411404231,
    "dataset": "checkpoint",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(dataset: str, refresh: bool) -> list[dict]:
    snapshot = ROOT / "assets" / f"{dataset}_folder.html"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not snapshot.exists():
        response = requests.get(f"https://drive.google.com/drive/folders/{FOLDERS[dataset]}", timeout=60)
        response.raise_for_status()
        snapshot.write_text(response.text, encoding="utf-8")
    listing = entries(snapshot.read_text(encoding="utf-8"))
    if not listing or any(not x["name"].endswith(".npz") for x in listing):
        raise RuntimeError(f"Unexpected WorldTrack folder listing for {dataset}")
    # Release currently contains 50 clips per folder; reject a possibly paginated snapshot.
    if len(listing) != 50:
        raise RuntimeError(f"Expected 50 author-provided {dataset} files, found {len(listing)}; inspect pagination")
    for entry in listing:
        entry["dataset"] = dataset
    (ROOT / "assets" / f"{dataset}_files.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    return listing


def acquire(entry: dict) -> dict:
    if entry["dataset"] == "checkpoint":
        target = ROOT / "assets" / "checkpoints" / entry["name"]
    else:
        target = ROOT / "data" / "worldtrack_release" / entry["dataset"] / entry["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={entry['id']}"
    if target.exists() and target.stat().st_size != entry["bytes"]:
        # gdown skips an existing final path with resume=True. Preserve an invalid
        # final file for diagnosis before retrying rather than silently accepting it.
        target.rename(target.with_name(target.name + f".invalid-{time.time_ns()}"))
    if not target.exists() or target.stat().st_size != entry["bytes"]:
        last_error = None
        for attempt in range(3):
            try:
                result = gdown.download(
                    url=url, output=str(target), quiet=True, resume=True,
                    cookies_file=str(ROOT / ".cache" / "gdown_cookies" / f"{entry['id']}.txt"),
                    timeout=(30, 120), retries=2,
                )
                if result is None:
                    raise RuntimeError("gdown returned no downloaded file")
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Download failed: {entry['name']}") from last_error
    actual = target.stat().st_size
    if actual != entry["bytes"]:
        raise RuntimeError(f"Truncated/wrong file {target}: {actual} bytes, expected {entry['bytes']}")
    if not zipfile.is_zipfile(target):
        raise RuntimeError(f"Invalid ZIP/NPZ/PyTorch archive: {target}")
    result = dict(entry, source_url=url, path=target.relative_to(ROOT).as_posix(), sha256=sha256(target))
    print(f"Verified {entry['dataset']}/{entry['name']} ({actual:,} bytes)", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", choices=list(FOLDERS), default=list(FOLDERS))
    parser.add_argument("--limit", type=int, default=0, help="First N files in author's listing, 0 = all 50")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--refresh-listing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    (ROOT / ".cache").mkdir(exist_ok=True)
    (ROOT / ".cache" / "gdown_cookies").mkdir(exist_ok=True)
    targets = [] if args.skip_checkpoint else [CHECKPOINT]
    if not args.checkpoint_only:
        for dataset in args.datasets:
            listed = discover(dataset, args.refresh_listing)
            targets += listed[:args.limit] if args.limit else listed
    completed, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire, entry): entry for entry in targets}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                entry = futures[future]
                failed.append(dict(entry, error=str(exc)))
                print(f"ERROR {entry['name']}: {exc}", file=sys.stderr, flush=True)
    manifest_path = ROOT / "assets" / "download_manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = {item["path"]: item for item in json.loads(manifest_path.read_text(encoding="utf-8")).get("completed", [])}
    previous.update({item["path"]: item for item in completed})
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_policy": "Exact author-distributed Seqmode_reweightMax5.pth; no automatic fallback",
        "selection_policy": "All 50 author-provided files; --limit selects first N in captured listing for development only",
        "official_data_folder": "https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV",
        "official_checkpoint_folder": "https://drive.google.com/drive/folders/1uSfnZbzqa8pfIb6k383-BerLQ0m9-R1l",
        "completed": sorted(previous.values(), key=lambda item: item["path"]),
        "failed": failed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
