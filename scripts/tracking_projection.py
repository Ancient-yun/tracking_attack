"""CPU-only diagnostic projection of immutable saved WorldTrack trajectories.

This is a GT camera diagnostic projection, not a new model inference or a
prediction of camera poses. The stored pred1 trajectories share the first
camera world frame with GT. Each condition is aligned using its recorded
official all-point scale before the original dataset cameras project it.
No coordinates are clipped to image boundaries; behind-camera coordinates
are marked invalid and represented by NaN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _assert(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def _camera_to_xy(camera: np.ndarray, intrinsics: np.ndarray, resize: np.ndarray) -> np.ndarray:
    # Mirror the author's project_points_to_video_frame, including its epsilon.
    xy = camera[..., :2] / (camera[..., 2:3] + 1e-8)
    return (xy * intrinsics[:2] + intrinsics[2:]) * resize


def _project(world: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray,
             resize: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = np.einsum("tij,tnj->tni", extrinsics[:, :3, :3], world.astype(np.float64))
    camera += extrinsics[:, None, :3, 3]
    valid = np.isfinite(camera).all(axis=-1) & (camera[..., 2] > 0)
    xy = _camera_to_xy(camera, intrinsics, resize)
    valid &= np.isfinite(xy).all(axis=-1)
    behind = np.isfinite(camera).all(axis=-1) & (camera[..., 2] <= 0)
    xy[~valid] = np.nan
    in_frame = valid & (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height)
    return xy, valid, in_frame, behind


def load_projected_tracks(sequence_dir: str | Path, project_root: str | Path) -> dict:
    """Load verified saved clean/PGD tracks and project into resized images.

    Arrays use [T,N,2] for xy, [T,N] for masks/errors, and [N] for dynamic.
    Errors and per-frame means include all official queries at all times,
    including later occlusions. ``visibility`` is only for overlay styling.
    ``in_frame`` and ``behind_camera`` are separate from positive-depth validity.
    The source NPZ numeric fields and saved tracks are read with pickle disabled.
    Original RGB/JPEG frames are never decoded by this helper.
    """
    sequence_dir, project_root = Path(sequence_dir).resolve(), Path(project_root).resolve()
    metadata = _json(sequence_dir / "sequence.json")
    run_dir = sequence_dir.parents[2]
    run = _json(run_dir / "run.json")
    dataset, sequence = sequence_dir.parent.name, sequence_dir.name
    _assert(metadata["sequence_name"] == sequence, "Saved sequence identity does not match directory")
    entries = [entry for entry in run["manifest"]["entries"]
               if entry["dataset"] == dataset and entry["sequence"] == sequence]
    _assert(len(entries) == 1, "Sequence must occur exactly once in the run manifest")
    entry = entries[0]
    source = project_root / "data" / "worldtrack_release" / dataset / f"{sequence}.npz"
    _assert(source.is_file(), f"Missing original source NPZ: {source}")
    source_hash = _sha256(source)
    _assert(source_hash == entry["sha256"], "Original NPZ SHA-256 differs from immutable run manifest")
    _assert(source.stat().st_size == entry["bytes"], "Original NPZ size differs from immutable run manifest")
    t = int(metadata["num_frames"])
    indices = np.asarray(metadata["selected_point_indices"], dtype=np.int64)
    width, height = map(int, metadata["preprocessed_wh"])
    resize = np.asarray(metadata["preprocessed_wh"], dtype=np.float64) / np.asarray(metadata["original_wh"], dtype=np.float64)
    with np.load(source, allow_pickle=False) as raw:
        source_num_frames = raw["tracks_XYZ"].shape[0]
        camera_gt = raw["tracks_XYZ"][:t].copy()
        visibility_raw = raw["visibility"][:t].astype(bool)
        intrinsics = raw["fx_fy_cx_cy"].copy()
        if "extrinsics_w2c" in raw.files:
            extrinsics = raw["extrinsics_w2c"][:t].copy()
            first_inv = np.linalg.inv(extrinsics[0])
            for frame in range(t):
                extrinsics[frame] = extrinsics[frame] @ first_inv
            inverse = np.linalg.inv(extrinsics)
            world_gt = np.zeros_like(camera_gt)
            for frame in range(t):
                world_gt[frame] = (inverse[frame, :3, :3] @ camera_gt[frame].T).T + inverse[frame, :3, 3]
        else:
            extrinsics = np.tile(np.eye(4), (t, 1, 1))
            world_gt = camera_gt.copy()
    _assert(camera_gt.shape[0] == t, "Source NPZ contains fewer frames than saved experiment")
    _assert(np.array_equal(intrinsics, np.asarray(metadata["intrinsics"])), "Source and saved intrinsics differ")
    # Preserve the official adapter's multiplication then division order so
    # float64 query coordinates remain bit-identical to the saved arrays.
    original_xy = (_camera_to_xy(camera_gt, intrinsics, np.ones(2))
                   * np.asarray(metadata["preprocessed_wh"])
                   / np.asarray(metadata["original_wh"]))
    initial_indices = np.flatnonzero(visibility_raw[0])
    initial_xy = original_xy[0, initial_indices]
    in_initial_frame = ((initial_xy[:, 0] >= 0) & (initial_xy[:, 0] < width)
                        & (initial_xy[:, 1] >= 0) & (initial_xy[:, 1] < height))
    _assert(np.array_equal(indices, initial_indices[in_initial_frame]), "Saved point IDs differ from official initial-visible/in-bounds selection")
    world_gt_selected = world_gt[:, indices].astype(np.float32)
    query_xy = original_xy[0, indices]
    motion = np.linalg.norm(np.diff(world_gt[:, indices], axis=0), axis=-1).sum(axis=0)
    dynamic = motion > 0.01
    conditions = {}
    sources = {"source_npz": str(source), "source_npz_sha256": source_hash,
               "recorded_source_npz_sha256": entry["sha256"], "source_frames": source_num_frames,
               "used_source_frame_indices": list(range(t)), "conditions": {}}
    for label, directory in (("clean", "clean"), ("attack", "pgd_eps4")):
        condition_dir = sequence_dir / directory
        record = _json(condition_dir / "result.json")
        tracks_file = condition_dir / "tracks.npz"
        tracks_hash = _sha256(tracks_file)
        _assert(tracks_hash == record["artifact_sha256"]["tracks.npz"], f"{label} saved trajectory SHA-256 mismatch")
        _assert(record["signature"] == run["signature"], f"{label} result belongs to another run")
        _assert(record["dataset"] == dataset and record["sequence"] == sequence, f"{label} result identity mismatch")
        _assert(record["saved_input_verification"]["passed"], f"{label} saved-input model replay did not pass")
        with np.load(tracks_file, allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        _assert(values["pred"].shape == world_gt_selected.shape, f"{label} pred/GT shape mismatch")
        _assert(np.array_equal(values["gt"], world_gt_selected), f"{label} saved GT differs from official source reconstruction")
        _assert(np.array_equal(values["query_xy"], query_xy), f"{label} saved query xy differs from source projection")
        _assert(np.array_equal(values["dynamic"], dynamic), f"{label} dynamic IDs differ from source rule")
        _assert(values["valid"].all(), f"{label} official all-time query mask is not all true")
        _assert(np.array_equal(values["intrinsics"], intrinsics), f"{label} saved intrinsics differ from source")
        scale = float(record["metrics"]["scale_all"])
        _assert(np.isfinite(scale) and scale > 0, f"{label} scale is not finite/positive")
        # Preserve float32 multiplication used by the official metric evaluator.
        aligned = values["pred"] * np.float32(scale)
        errors = np.linalg.norm(aligned - values["gt"], axis=-1)
        mean_error = float(errors.mean())
        _assert(np.isclose(mean_error, record["metrics"]["epe_all_m"], rtol=1e-6, atol=1e-6),
                f"{label} reconstructed error disagrees with recorded official EPE")
        xy, valid, in_frame, behind = _project(aligned, extrinsics, intrinsics, resize, width, height)
        conditions[label] = {"aligned": aligned, "xy": xy, "valid": valid, "in_frame": in_frame,
                             "behind": behind, "errors": errors, "record": record,
                             "rgb_path": condition_dir / "rgb_float32.npy"}
        sources["conditions"][label] = {"tracks_path": str(tracks_file), "tracks_sha256": tracks_hash,
                                        "recorded_tracks_sha256": record["artifact_sha256"]["tracks.npz"],
                                        "result_path": str(condition_dir / "result.json"),
                                        "result_sha256": _sha256(condition_dir / "result.json"),
                                        "official_scale_all": scale,
                                        "saved_input_replay_passed": record["saved_input_verification"]["passed"],
                                        "saved_input_replay_max_track_error": record["saved_input_verification"]["max_track_error"]}
    gt_xy, gt_valid, gt_in_frame, gt_behind = _project(world_gt_selected, extrinsics, intrinsics, resize, width, height)
    source_selected_xy = original_xy[:, indices]
    uv_error = np.linalg.norm(gt_xy[gt_valid] - source_selected_xy[gt_valid], axis=-1)
    max_uv_error = float(uv_error.max(initial=0))
    # Allows accumulated float32 pose/GT storage rounding, well below one pixel.
    _assert(max_uv_error < 0.01, f"GT 3D projection does not match original camera-track UV: max {max_uv_error:g}px")
    projection_checks = {"passed": True, "projection_description": "GT camera diagnostic projection",
                         "coordinate_basis": "first-camera world; E_t = raw_E_t @ inverse(raw_E_0)",
                         "intrinsics_resize_xy": resize.tolist(),
                         "gt_source_saved_bit_exact": True, "query_source_saved_bit_exact": True,
                         "gt_uv_error_max_px": max_uv_error, "gt_uv_error_mean_px": float(uv_error.mean()),
                         "gt_uv_error_tolerance_px": 0.01,
                         "pred_alignment": "per-condition recorded official scale_all; no rotation/translation fit",
                         "overlay_occlusion": "source GT visibility controls styling only; errors include all queries",
                         "coordinate_clamping": False, "behind_camera_xy": "NaN",
                         "epe_mean_matches_recorded": True}
    return {"gt_xy": gt_xy, "clean_xy": conditions["clean"]["xy"], "attack_xy": conditions["attack"]["xy"],
            "projection_valid": {"gt": gt_valid, **{key: item["valid"] for key, item in conditions.items()}},
            "in_frame": {"gt": gt_in_frame, **{key: item["in_frame"] for key, item in conditions.items()}},
            "behind_camera": {"gt": gt_behind, **{key: item["behind"] for key, item in conditions.items()}},
            "visibility": visibility_raw[:, indices], "dynamic": dynamic, "query_xy": query_xy,
            "clean_error_m": conditions["clean"]["errors"], "attack_error_m": conditions["attack"]["errors"],
            "clean_epe_per_frame_m": conditions["clean"]["errors"].mean(axis=1),
            "attack_epe_per_frame_m": conditions["attack"]["errors"].mean(axis=1),
            "gt_world": world_gt_selected, "clean_aligned_world": conditions["clean"]["aligned"],
            "attack_aligned_world": conditions["attack"]["aligned"],
            "clean_result": conditions["clean"]["record"], "attack_result": conditions["attack"]["record"],
            "clean_rgb_path": conditions["clean"]["rgb_path"], "attack_rgb_path": conditions["attack"]["rgb_path"],
            "metadata": metadata, "sources": sources, "projection_checks": projection_checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    reports = []
    for sequence_file in sorted((args.run_dir / "sequences").glob("*/*/sequence.json")):
        result = load_projected_tracks(sequence_file.parent, args.project_root)
        reports.append({"dataset": sequence_file.parent.parent.name, "sequence": sequence_file.parent.name,
                        "shape": list(result["gt_xy"].shape), "sources": result["sources"],
                        "projection_checks": result["projection_checks"]})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
