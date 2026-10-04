"""Freeze sequence selection instead of silently choosing new evaluation clips."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re

from .common import resolve_path, sha256, write_json


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def build_manifest(data_root: Path, datasets: list[str], count: int, seed: int, num_frames: int,
                   sequence_list: Path | None = None, hash_files: bool = True) -> dict:
    if count <= 0 or num_frames <= 0:
        raise ValueError("count and num_frames must be positive")
    requested = None
    if sequence_list:
        requested = [line.strip() for line in sequence_list.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        if len(requested) != len(set(requested)):
            raise ValueError("The supplied sequence list contains duplicates")
    entries, inventory = [], {}
    for dataset in datasets:
        files = sorted((data_root / dataset).glob("*.npz"), key=natural_key)
        if not files:
            raise FileNotFoundError(f"No WorldTrack NPZ files in {data_root / dataset}; run scripts/download_assets.py first")
        inventory[dataset] = len(files)
        if requested is not None:
            selected = [item for item in files if f"{dataset}/{item.name}" in requested]
            if not selected:
                raise ValueError(f"No entries for {dataset} in {sequence_list}")
        else:
            if len(files) < count:
                raise ValueError(f"{dataset}: requested {count} sequences, only {len(files)} available")
            selected = sorted(random.Random(f"{seed}:{dataset}").sample(files, count), key=natural_key)
        for path in selected:
            entry = {"dataset": dataset, "sequence": path.stem, "path": str(path.resolve()), "bytes": path.stat().st_size}
            if hash_files:
                entry["sha256"] = sha256(path)
            entries.append(entry)
    if requested is not None:
        actual = {f"{entry['dataset']}/{Path(entry['path']).name}" for entry in entries}
        if actual != set(requested):
            raise ValueError(f"Unknown supplied sequence entries: {sorted(set(requested) - actual)}")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root), "datasets": datasets, "num_frames": num_frames, "seed": seed,
            "selection": "supplied_sequence_list" if requested is not None else "seeded_sample",
            "paper_sequence_identity_verified": False,
            "sequence_list": str(sequence_list) if sequence_list else None,
            "available_sequences": inventory, "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/worldtrack_release")
    parser.add_argument("--datasets", nargs="+", default=["po_mini", "ds_mini"])
    parser.add_argument("--count", type=int, default=50, help="Sequences per dataset; insufficient data is an error")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--sequence-list", help="Author/user list, one dataset/filename.npz per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args()
    target = resolve_path(args.output)
    if target.exists():
        raise FileExistsError(f"Manifest already exists: {target}; choose a new name to preserve selection")
    manifest = build_manifest(resolve_path(args.data_root), args.datasets, args.count, args.seed, args.num_frames,
                              resolve_path(args.sequence_list) if args.sequence_list else None, not args.no_hash)
    write_json(target, manifest)
    print(f"Saved {len(manifest['entries'])} sequences to {target}")


if __name__ == "__main__":
    main()
