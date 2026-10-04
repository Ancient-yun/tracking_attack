"""Paths, atomic artifacts, and version provenance for reproducible runs."""
from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = ROOT / "vendor" / "St4RTrack"
OFFICIAL_COMMIT = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_official_metrics(vendor_root: str | Path = VENDOR_ROOT):
    """Load the actual standalone evaluator, avoiding training-only imports."""
    source = resolve_path(vendor_root) / "dust3r" / "track_eval_util.py"
    spec = importlib.util.spec_from_file_location("_st4rtrack_official_metrics", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provenance(checkpoint: str | Path, vendor_root: str | Path = VENDOR_ROOT) -> dict:
    import torch

    vendor = resolve_path(vendor_root)
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Official source changed: expected {OFFICIAL_COMMIT}, got {commit}")
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("Official tracked source has local changes; restore or explicitly pin a reviewed source version.")
    checkpoint = resolve_path(checkpoint)
    files = sorted(checkpoint.glob("*.safetensors")) + sorted(checkpoint.glob("*.json")) if checkpoint.is_dir() else [checkpoint]
    if not files or any(not item.is_file() for item in files):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return {
        "official_repository": "https://github.com/HavenFeng/St4RTrack",
        "official_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_files": {item.name: {"sha256": sha256(item), "bytes": item.stat().st_size} for item in files},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "Pillow", "torch", "torchvision", "scipy", "huggingface-hub", "einops", "roma", "opencv-python")},
        "submodule_commits": subprocess.check_output(["git", "-C", str(vendor), "submodule", "status", "--recursive"], text=True).strip(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "official_metric_source_sha256": sha256(vendor / "dust3r" / "track_eval_util.py"),
        "experiment_source_sha256": {item.name: sha256(item) for item in sorted((ROOT / "src" / "st4rtrack_pgd").glob("*.py"))},
    }
