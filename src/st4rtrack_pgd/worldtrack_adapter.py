"""WorldTrack inputs using the pinned St4RTrack loader and evaluation rules.

The attack space is the official resized RGB tensor before normalization.
Tracking queries are visible in frame zero, sampled by integer truncation at
that frame's pixel coordinates for *every* pair. As in ``track_eval.py``, later
occlusion does not remove a query from evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


DEFAULT_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "St4RTrack"


def ensure_vendor_importable(vendor_root: str | Path | None = None) -> Path:
    """Select the official checkout, rejecting an already imported other copy."""
    root = Path(vendor_root or DEFAULT_VENDOR_ROOT).resolve()
    if not (root / "dust3r" / "model.py").is_file():
        raise FileNotFoundError(f"St4RTrack checkout is missing at {root}")
    if not (root / "croco" / "models" / "croco.py").is_file():
        raise FileNotFoundError(f"CroCo submodule is missing at {root / 'croco'}; clone recursively")
    loaded = sys.modules.get("dust3r")
    if loaded is not None:
        package_paths = list(getattr(loaded, "__path__", ()))
        if package_paths and Path(package_paths[0]).resolve() != root / "dust3r":
            raise RuntimeError(f"Another dust3r checkout is already imported: {package_paths[0]}")
    for candidate in (root, root / "croco"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root


@dataclass
class WorldTrackSequence:
    rgb: torch.Tensor                 # [T, 3, H, W], float32 in [0, 1]
    gt_tracks: torch.Tensor           # [T, N, 3], first-camera world frame
    valid: torch.Tensor               # [T, N], official evaluation membership
    dynamic: torch.Tensor             # [N], cumulative world motion > 0.01 m
    query_xy: torch.Tensor            # [N, 2], floating output-image coordinates
    metadata: dict[str, Any]
    official_views: tuple[dict[str, Any], ...] = field(default_factory=tuple, repr=False)

    def to(self, device: str | torch.device) -> "WorldTrackSequence":
        """Move attack/evaluation tensors while keeping clean reference views."""
        return replace(self, **{
            name: getattr(self, name).to(device)
            for name in ("rgb", "gt_tracks", "valid", "dynamic", "query_xy")
        })


# Convenient name for callers which do not need the dataset-specific prefix.
Sequence = WorldTrackSequence


def select_queries(
    tracks_uv: np.ndarray,
    tracks_world: np.ndarray,
    visibility: np.ndarray,
    original_wh: tuple[int, int],
    output_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the tracking branch of the official evaluator exactly.

    Returns scaled floating xy, selected world tracks, dynamic flags, and the
    original point indices. Floating xy remains float64 until integer sampling,
    preserving the evaluator's NumPy multiplication and ``astype(int)`` rule.
    """
    uv = np.asarray(tracks_uv)
    world = np.asarray(tracks_world)
    vis = np.asarray(visibility)
    if world.ndim != 3 or world.shape[-1] != 3:
        raise ValueError(f"Expected world tracks [T,N,3], received {world.shape}")
    if uv.shape != world.shape[:2] + (2,) or vis.shape != world.shape[:2]:
        raise ValueError("UV, world tracks, and visibility shapes do not agree")
    if world.shape[0] == 0:
        raise ValueError("The sequence contains no frames")
    if min(*original_wh, *output_wh) <= 0:
        raise ValueError("Image dimensions must be positive")
    if not np.isfinite(world).all() or not np.isfinite(uv).all():
        raise ValueError("WorldTrack contains NaN or Inf coordinates")
    indices = np.flatnonzero(vis[0].astype(bool))
    xy = uv[0, indices] * np.array(output_wh) / np.array(original_wh)
    inside = (
        (xy[:, 0] >= 0) & (xy[:, 0] < output_wh[0])
        & (xy[:, 1] >= 0) & (xy[:, 1] < output_wh[1])
    )
    xy, indices = xy[inside], indices[inside]
    if len(indices) == 0:
        raise ValueError("No initially visible in-bounds tracking queries")
    selected_world = world[:, indices]
    total_motion = np.linalg.norm(np.diff(selected_world, axis=0), axis=-1).sum(axis=0)
    dynamic = total_motion > 0.01
    return xy, selected_world, dynamic, indices


def sample_query_tracks(point_map: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
    """Sample [H,W,3] or [T,H,W,3] maps by official integer truncation.

    Sampling is differentiable with respect to point maps. Queries are fixed
    throughout clean/noise/FGSM/PGD runs; no bilinear interpolation is applied.
    """
    if point_map.ndim not in (3, 4) or point_map.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3] or [T,H,W,3], received {tuple(point_map.shape)}")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must have shape [N,2]")
    xy = query_xy.to(device=point_map.device)
    h, w = point_map.shape[-3:-1]
    if not torch.isfinite(xy).all():
        raise ValueError("Queries contain NaN or Inf")
    if not ((xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)).all():
        raise ValueError("Queries fall outside the prediction map")
    pixels = xy.to(dtype=torch.long)  # truncate toward zero, as np.astype(int)
    if point_map.ndim == 3:
        return point_map[pixels[:, 1], pixels[:, 0]]
    return point_map[:, pixels[:, 1], pixels[:, 0]]


def load_sequence(
    path: str | Path,
    num_frames: int = 64,
    image_size: int = 512,
    vendor_root: str | Path | None = None,
) -> WorldTrackSequence:
    """Load the first frames and apply official no-crop aspect-ratio resize.

    ``load_npz_data(normalize_cam=True)`` changes W2C extrinsics to
    ``E_t @ inverse(E_0)`` and transforms camera tracks by their inverse. Thus
    the world coordinate basis is the frame-zero camera, also used by pred1.
    ``load_images(size=512, square_ok=True, crop=False)`` first resizes the
    longest edge, then resizes both dimensions down to multiples of sixteen.
    Queries scale independently in x/y to that final resolution.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if image_size != 512:
        raise ValueError("Official tracking evaluation uses image_size=512")
    npz_path = Path(path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    root = ensure_vendor_importable(vendor_root)
    # Lazy imports avoid importing visualization/training dependencies for
    # simple coordinate and attack tests.
    loader = importlib.import_module("dust3r.datasets.tapvid3d").load_npz_data
    load_images = importlib.import_module("dust3r.utils.image").load_images
    (images, _cam, uv, intrinsics, world, visibility, name, extrinsics) = loader(
        str(npz_path), num_frames=num_frames, normalize_cam=True
    )
    if not images:
        raise ValueError(f"No decoded frames in {npz_path}")
    views = load_images(
        images, size=image_size, num_frames=num_frames, step_size=1,
        verbose=False, square_ok=True, crop=False,
    )
    expected_t = len(images)
    if len(views) != expected_t or world.shape[0] != expected_t:
        raise ValueError("Decoded frames and GT trajectories have inconsistent lengths")
    shapes = {tuple(view["img"].shape) for view in views}
    if len(shapes) != 1:
        raise ValueError("All frames must have the same preprocessed resolution")
    # This inverse normalization exactly preserves the official normalized
    # pixel values when pair_fn applies its corresponding normalization.
    rgb = torch.cat([view["img"] for view in views], dim=0).mul(0.5).add(0.5)
    h, w = rgb.shape[-2:]
    xy, selected_world, dynamic, indices = select_queries(
        uv, world, visibility, images[0].size, (w, h),
    )
    # The official evaluator never consults visibility after choosing frame 0.
    valid = torch.ones(selected_world.shape[:2], dtype=torch.bool)
    metadata = {
        "sequence_name": name,
        "source_path": str(npz_path),
        "vendor_root": str(root),
        "num_frames": expected_t,
        "requested_num_frames": num_frames,
        "num_queries": len(indices),
        "original_wh": list(images[0].size),
        "preprocessed_wh": [w, h],
        "image_size": image_size,
        "preprocessing": "official load_images(size=512, square_ok=True, crop=False, step_size=1)",
        "attack_space": "preprocessed RGB [0,1] before mean=.5/std=.5 normalization",
        "coordinate_frame": "first-camera world: normalized E_t = E_t @ inverse(E_0)",
        "query_rule": "frame-0 visible and in-bounds; scaled xy truncated to integers",
        "evaluation_mask_rule": "all frames of initial queries, including later occlusions",
        "dynamic_rule": "sum_t norm(world[t+1]-world[t]) > 0.01 m",
        "selected_point_indices": indices.tolist(),
        "intrinsics": np.asarray(intrinsics).tolist(),
        "intrinsics_fx_fy_cx_cy": np.asarray(intrinsics).tolist(),
        "first_extrinsic_w2c": np.asarray(extrinsics[0]).tolist(),
        "official_reference_batch_size": 1,
        "official_clean_batch_size": 1,
    }
    return WorldTrackSequence(
        rgb=rgb.contiguous().float(),
        gt_tracks=torch.from_numpy(np.ascontiguousarray(selected_world)).float(),
        valid=valid,
        dynamic=torch.from_numpy(dynamic.astype(bool)),
        query_xy=torch.from_numpy(np.ascontiguousarray(xy)).double(),
        metadata=metadata,
        official_views=tuple(views),
    )
