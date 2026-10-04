"""Keep the frozen sequence roster/hashes; rewrite only host-specific paths."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

ROOT = Path(__file__).resolve().parents[1]


def convert(source: Path, data_root: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    result = copy.deepcopy(payload)
    result["data_root"] = "data/worldtrack_release"
    identities = set()
    for entry in result["entries"]:
        dataset, sequence = entry["dataset"], entry["sequence"]
        for component in (dataset, sequence):
            if not component or component in (".", "..") or "/" in component or "\\" in component:
                raise ValueError(f"Unsafe manifest identity: {dataset}/{sequence}")
        identity = (dataset, sequence)
        if identity in identities:
            raise ValueError(f"Duplicate manifest identity: {identity}")
        identities.add(identity)
        filename = PureWindowsPath(entry["path"]).name
        if filename != sequence + ".npz":
            raise ValueError(f"Source path and sequence differ: {entry}")
        local = data_root / dataset / filename
        if not local.is_file() or local.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or size-mismatched data file: {local}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError(f"Missing frozen SHA-256: {identity}")
        entry["path"] = f"data/worldtrack_release/{dataset}/{filename}"
    result["path_conversion"] = {
        "source_manifest": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mode": "project_relative_posix",
        "roster_and_file_hashes_preserved": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "manifests")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "worldtrack_release")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docker" / "manifests")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.source_dir.glob("*.json"))
    if not sources:
        raise FileNotFoundError(f"No frozen manifests in {args.source_dir}")
    for source in sources:
        result = convert(source, args.data_root)
        target = args.output_dir / source.name
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            if previous != result:
                raise FileExistsError(f"Frozen Docker manifest differs: {target}; use a new output directory")
        else:
            target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{target}: {len(result['entries'])} preserved entries")


if __name__ == "__main__":
    main()
