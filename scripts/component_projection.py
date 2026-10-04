"""CPU-only, immutable-array adapter for the full-frame component study.

The projection deliberately reuses the pinned experiment's camera convention.
Saved metric scales are applied once; no pose fit or model inference occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tracking_projection import _camera_to_xy, _project

OBJECTIVES = ("tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training")
THRESHOLDS = (0.1, 0.3, 0.5, 1.0)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def metric_check(actual, recorded, description):
    require(np.isfinite(actual) and np.isfinite(recorded)
            and np.isclose(actual, recorded, rtol=1e-6, atol=1e-6),
            f"{description}: reconstructed={actual!r}, recorded={recorded!r}")


def safe_component(value):
    require(isinstance(value, str) and value not in ("", ".", "..")
            and "/" not in value and "\\" not in value and ":" not in value,
            f"Unsafe identity component: {value!r}")
    return value


def residual_metrics(error, valid):
    """Fixed-metre APD and EPE; retain all GT-valid support."""
    values = np.asarray(error)[np.asarray(valid, dtype=bool)]
    require(values.size > 0 and np.isfinite(values).all(), "Empty/nonfinite metric support")
    return {"epe_m": float(values.mean()),
            "apd3d": float(np.mean([np.mean(values <= threshold) for threshold in THRESHOLDS]) * 100)}


class ComponentContext:
    """One invocation's checked source/query context; arrays stay read-only."""

    def __init__(self, run_dir, project_root, *, metadata=None, records=None):
        self.root, self.project = Path(run_dir).resolve(), Path(project_root).resolve()
        self.run = metadata if metadata is not None else read_json(self.root / "run.json")
        self.records = records or {}
        self.sources, self.conditions, self.hashes = {}, {}, {}

    def fingerprint(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            self.hashes[key] = sha256(path)
        return self.hashes[key]

    def checked_array(self, directory, name, inventory, *, mmap=True):
        path = Path(directory) / name
        digest = self.fingerprint(path)
        require(inventory.get(name) == digest, f"Artifact SHA-256 mismatch: {path}")
        result = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        return result, {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}

    def source(self, dataset, sequence):
        dataset, sequence = safe_component(dataset), safe_component(sequence)
        key = dataset, sequence
        if key in self.sources:
            return self.sources[key]
        directory = self.root / "sequences" / dataset / sequence
        metadata = read_json(directory / "sequence.json")
        inventory = read_json(directory / "sequence_manifest.json")
        require(inventory["signature"] == self.run["signature"], "Sequence signature differs")
        for name in ("sequence.json", "targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy"):
            require(self.fingerprint(directory / name) == inventory["artifact_sha256"][name],
                    f"Sequence target SHA mismatch: {directory / name}")
        t = int(metadata["num_frames"])
        require(t == 128 and metadata.get("original_frame_count") == t
                and metadata.get("all_frames_used") is True
                and metadata.get("used_frame_indices") == list(range(t))
                and metadata.get("source_frame_counts")
                and all(value == t for value in metadata["source_frame_counts"].values()),
                "Expected verified use of all 128 original frames")
        require(metadata["sequence_name"] == sequence, "Sequence metadata identity differs")
        entries = [entry for entry in self.run["manifest"]["entries"]
                   if (entry["dataset"], entry["sequence"]) == key]
        require(len(entries) == 1, "Sequence must occur exactly once in run manifest")
        entry = entries[0]
        original = self.project / "data" / "worldtrack_release" / dataset / f"{sequence}.npz"
        require(self.fingerprint(original) == entry["sha256"] and original.stat().st_size == entry["bytes"],
                f"Original NPZ hash/size differs: {original}")
        point_ids = np.asarray(metadata["selected_point_indices"], dtype=np.int64)
        width, height = map(int, metadata["preprocessed_wh"])
        resize = np.asarray(metadata["preprocessed_wh"], dtype=np.float64) / np.asarray(metadata["original_wh"], dtype=np.float64)
        with np.load(original, allow_pickle=False) as raw:
            camera_gt = raw["tracks_XYZ"].copy()
            visibility = raw["visibility"].astype(bool)
            intrinsics = raw["fx_fy_cx_cy"].copy()
            extrinsics = raw["extrinsics_w2c"].copy()
        require(camera_gt.shape[0] == visibility.shape[0] == extrinsics.shape[0] == t,
                "Original numeric temporal fields do not have all 128 frames")
        first_inv = np.linalg.inv(extrinsics[0])
        for frame in range(t):
            extrinsics[frame] = extrinsics[frame] @ first_inv
        inverse = np.linalg.inv(extrinsics)
        world_gt = np.zeros_like(camera_gt)
        for frame in range(t):
            world_gt[frame] = (inverse[frame, :3, :3] @ camera_gt[frame].T).T + inverse[frame, :3, 3]
        require(np.array_equal(intrinsics, np.asarray(metadata["intrinsics"])), "Saved/source intrinsics differ")
        original_xy = (_camera_to_xy(camera_gt, intrinsics, np.ones(2))
                       * np.asarray(metadata["preprocessed_wh"]) / np.asarray(metadata["original_wh"]))
        initially_visible = np.flatnonzero(visibility[0])
        initial = original_xy[0, initially_visible]
        inside = ((initial[:, 0] >= 0) & (initial[:, 0] < width)
                  & (initial[:, 1] >= 0) & (initial[:, 1] < height))
        require(np.array_equal(point_ids, initially_visible[inside]), "Saved/source query IDs differ")
        gt = world_gt[:, point_ids].astype(np.float32)
        dynamic = np.linalg.norm(np.diff(world_gt[:, point_ids], axis=0), axis=-1).sum(axis=0) > 0.01
        gt_xy, gt_valid, gt_inside, gt_behind = _project(gt, extrinsics, intrinsics, resize, width, height)
        uv_error = np.linalg.norm(gt_xy[gt_valid] - original_xy[:, point_ids][gt_valid], axis=-1)
        require(uv_error.size > 0 and float(uv_error.max()) < 0.01, "GT camera projection differs from original UV")
        dense_gt, gt_evidence = self.checked_array(directory, "reconstruction_gt.npy", inventory["artifact_sha256"])
        dense_valid, valid_evidence = self.checked_array(directory, "reconstruction_valid.npy", inventory["artifact_sha256"])
        require(dense_gt.shape == (t, height, width, 3) and dense_valid.shape == (t, height, width)
                and dense_valid.dtype == np.bool_, "Dense GT/mask shape/dtype differs")
        with np.load(directory / "targets.npz", allow_pickle=False) as targets:
            gt_norm = targets["reconstruction_gt_norm"].copy()
        require(gt_norm.shape == (t,) and np.isfinite(gt_norm).all() and (gt_norm > 0).all(), "Invalid GT normalization")
        source = {"dataset": dataset, "sequence": sequence, "directory": directory, "metadata": metadata,
                  "gt": gt, "query_xy": original_xy[0, point_ids], "dynamic": dynamic,
                  "visibility": visibility[:, point_ids], "point_ids": point_ids,
                  "intrinsics": intrinsics, "extrinsics": extrinsics, "resize": resize,
                  "gt_xy": gt_xy, "gt_valid": gt_valid, "gt_inside": gt_inside, "gt_behind": gt_behind,
                  "dense_gt": dense_gt, "dense_valid": dense_valid, "gt_norm": gt_norm,
                  "evidence": {"source_npz": {"path": str(original), "sha256": entry["sha256"], "bytes": entry["bytes"]},
                               "sequence.json": {"path": str(directory / "sequence.json"), "sha256": self.fingerprint(directory / "sequence.json")},
                               "targets.npz": {"path": str(directory / "targets.npz"), "sha256": self.fingerprint(directory / "targets.npz")},
                               "reconstruction_gt.npy": gt_evidence, "reconstruction_valid.npy": valid_evidence},
                  "checks": {"passed": True, "gt_source_bit_exact": True, "query_source_bit_exact": True,
                             "source_frames": t, "used_source_frame_indices": list(range(t)),
                             "gt_uv_error_max_px": float(uv_error.max()), "gt_uv_tolerance_px": 0.01,
                             "coordinate_basis": "E_t=raw_E_t @ inverse(raw_E_0)", "coordinate_clamping": False}}
        self.sources[key] = source
        return source

    def condition(self, dataset, sequence, objective):
        require(objective in ("clean", *OBJECTIVES), "Unknown component objective")
        key = dataset, sequence, objective
        if key in self.conditions:
            return self.conditions[key]
        source = self.source(dataset, sequence)
        directory = self.root / "conditions" / objective / dataset / sequence
        record = read_json(directory / "result.json")
        require((record["dataset"], record["sequence"], record["objective"]) == key
                and record["signature"] == self.run["signature"], "Condition signature/identity differs")
        require(record["saved_input_verification"]["passed"] is True, "Recorded model replay did not pass")
        require(record["epsilon_255"] == (0 if objective == "clean" else 4)
                and record["steps"] == (0 if objective == "clean" else 20), "Attack settings differ")
        if key in self.records:
            require(record == self.records[key], "Condition scalar changed after completion gate")
        hashes = record["artifact_sha256"]
        tracks_file = directory / "tracks.npz"
        require(self.fingerprint(tracks_file) == hashes["tracks.npz"], "Saved trajectories changed")
        with np.load(tracks_file, allow_pickle=False) as archive:
            tracks = {name: archive[name].copy() for name in archive.files}
        require(tracks["pred"].dtype == np.float32 and tracks["pred"].shape == source["gt"].shape
                and np.isfinite(tracks["pred"]).all(), "Invalid saved trajectories")
        for name, expected in (("gt", source["gt"]), ("query_xy", source["query_xy"]),
                               ("dynamic", source["dynamic"]), ("intrinsics", source["intrinsics"])):
            require(np.array_equal(tracks[name], expected), f"Saved/source {name} differs")
        require(tracks["valid"].shape == source["gt"].shape[:2] and tracks["valid"].all(),
                "Official evaluation mask must include all query times")
        metrics = record["tracking_metrics"]
        scale = float(metrics["scale_all"])
        require(np.isfinite(scale) and scale > 0, "Invalid trajectory scale")
        aligned = tracks["pred"] * np.float32(scale)
        error = np.linalg.norm(aligned - tracks["gt"], axis=-1)
        reproduced = residual_metrics(error, tracks["valid"])
        metric_check(reproduced["epe_m"], metrics["epe_all_m"], "Tracking EPE")
        metric_check(reproduced["apd3d"], metrics["apd3d_all"], "Tracking APD")
        width, height = source["metadata"]["preprocessed_wh"]
        xy, valid, inside, behind = _project(aligned, source["extrinsics"], source["intrinsics"], source["resize"], width, height)
        arrays, evidence = {}, {"result.json": {"path": str(directory / "result.json"), "sha256": self.fingerprint(directory / "result.json")},
                                "tracks.npz": {"path": str(tracks_file), "sha256": hashes["tracks.npz"]}}
        for name in ("rgb_float32.npy", "delta_float32.npy", "reconstruction_float32.npy", "reconstruction_confidence_float32.npy"):
            arrays[name], evidence[name] = self.checked_array(directory, name, hashes)
            require(arrays[name].dtype == np.float32, f"Expected saved float32: {name}")
        t = source["metadata"]["num_frames"]
        require(arrays["rgb_float32.npy"].shape == arrays["delta_float32.npy"].shape == (t, 3, height, width), "RGB/delta shape differs")
        require(arrays["reconstruction_float32.npy"].shape == source["dense_gt"].shape
                and arrays["reconstruction_confidence_float32.npy"].shape == source["dense_valid"].shape, "Dense output shape differs")
        components_file = directory / "components.npz"
        require(self.fingerprint(components_file) == hashes["components.npz"], "Compact components changed")
        with np.load(components_file, allow_pickle=False) as archive:
            components = {name: archive[name].copy() for name in archive.files}
        evidence["components.npz"] = {"path": str(components_file), "sha256": hashes["components.npz"]}
        require(self.fingerprint(directory / "history.json") == hashes["history.json"], "PGD history changed")
        history = read_json(directory / "history.json")
        evidence["history.json"] = {"path": str(directory / "history.json"), "sha256": hashes["history.json"]}
        result = {"record": record, "directory": directory, "tracks": tracks, "aligned": aligned, "error": error,
                  "epe_per_frame": error.mean(axis=1), "xy": xy, "projection_valid": valid,
                  "in_frame": inside, "behind_camera": behind, "arrays": arrays, "components": components,
                  "history": history, "evidence": evidence,
                  "checks": {"tracking_epe_reproduced_m": reproduced["epe_m"], "tracking_apd_reproduced_percent": reproduced["apd3d"],
                             "tracking_metrics_reproduced": True, "recorded_model_replay_passed": True}}
        self.conditions[key] = result
        return result

    def pair(self, dataset, sequence, objective):
        require(objective in OBJECTIVES, "Pair requires an attack objective")
        source = self.source(dataset, sequence)
        clean, attack = self.condition(dataset, sequence, "clean"), self.condition(dataset, sequence, objective)
        require(clean["record"]["seed"] == attack["record"]["seed"], "Paired derived seeds differ")
        maximum = 0.0
        for t in range(source["metadata"]["num_frames"]):
            c, a = clean["arrays"]["rgb_float32.npy"][t], attack["arrays"]["rgb_float32.npy"][t]
            cd, ad = clean["arrays"]["delta_float32.npy"][t], attack["arrays"]["delta_float32.npy"][t]
            require(all(np.isfinite(value).all() for value in (c, a, cd, ad)), "Nonfinite saved RGB/delta")
            require(min(float(c.min()), float(a.min())) >= 0 and max(float(c.max()), float(a.max())) <= 1, "RGB outside [0,1]")
            require(not np.any(cd) and np.array_equal(a - c, ad), "Delta is not exact attacked minus clean")
            maximum = max(maximum, float(np.abs(ad).max()))
        require(maximum <= 4 / 255 + 1e-7, "Saved perturbation exceeds epsilon")
        require(np.isclose(maximum, attack["record"]["perturbation_norms"]["linf"], rtol=0, atol=1e-7), "Recorded perturbation norm differs")
        projected = {"gt_xy": source["gt_xy"], "visibility": source["visibility"], "dynamic": source["dynamic"], "query_xy": source["query_xy"],
                     "metadata": source["metadata"], "sources": {"sequence": source["evidence"], "conditions": {"clean": clean["evidence"], "attack": attack["evidence"]}},
                     "projection_valid": {"gt": source["gt_valid"], "clean": clean["projection_valid"], "attack": attack["projection_valid"]},
                     "in_frame": {"gt": source["gt_inside"], "clean": clean["in_frame"], "attack": attack["in_frame"]},
                     "behind_camera": {"gt": source["gt_behind"], "clean": clean["behind_camera"], "attack": attack["behind_camera"]}}
        for label, condition in (("clean", clean), ("attack", attack)):
            projected[f"{label}_xy"] = condition["xy"]
            projected[f"{label}_epe_per_frame_m"] = condition["epe_per_frame"]
            # Compatibility alias is confined to this in-memory plotting view.
            projected[f"{label}_result"] = {**condition["record"], "metrics": condition["record"]["tracking_metrics"]}
        projected["projection_checks"] = {**source["checks"], "epsilon_linf": maximum,
                                           "rgb_delta_exact": True, "recorded_replay_only": True,
                                           "tracking_metrics_reproduced": True}
        return projected

    def reconstruction_checks(self, dataset, sequence):
        """Scan one frame at a time; use common clean+five objective color limits."""
        source = self.source(dataset, sequence)
        results, maximum, conf_min, conf_max = {}, 0.0, float("inf"), 0.0
        for objective in ("clean", *OBJECTIVES):
            condition = self.condition(dataset, sequence, objective)
            scale = np.float32(condition["record"]["reconstruction_metrics"]["scale"])
            require(np.isfinite(scale) and scale > 0, "Invalid reconstruction scale")
            count, summed, within, per_frame = 0, 0.0, np.zeros(len(THRESHOLDS), np.int64), []
            for t in range(source["metadata"]["num_frames"]):
                pred = condition["arrays"]["reconstruction_float32.npy"][t]
                conf = condition["arrays"]["reconstruction_confidence_float32.npy"][t]
                valid = source["dense_valid"][t]
                require(np.isfinite(pred).all() and np.isfinite(conf).all() and (conf >= 1).all()
                        and np.isfinite(source["dense_gt"][t][valid]).all(), "Invalid dense output/GT/confidence")
                error = np.linalg.norm(pred * scale - source["dense_gt"][t], axis=-1)[valid]
                require(error.size > 0 and np.isfinite(error).all(), "Invalid/empty dense residual")
                count += error.size
                summed += float(error.sum(dtype=np.float64))
                within += [np.count_nonzero(error <= threshold) for threshold in THRESHOLDS]
                per_frame.append(float(error.mean()))
                maximum = max(maximum, float(error.max()))
                conf_min = min(conf_min, float(conf.min()))
                conf_max = max(conf_max, float(conf.max()))
            metrics = {"epe_m": summed / count, "apd3d": float((within / count).mean() * 100), "valid_pixels": count,
                       "epe_per_frame_m": per_frame, "passed": True}
            recorded = condition["record"]["reconstruction_metrics"]
            metric_check(metrics["epe_m"], recorded["epe_m"], "Reconstruction EPE")
            metric_check(metrics["apd3d"], recorded["apd3d"], "Reconstruction APD")
            require(count == recorded["valid_pixels"], "Reconstruction valid support differs")
            results[objective] = metrics
        return {"passed": True, "conditions": results, "vmax_m": max(maximum, 1e-9),
                "vmax_policy": "maximum fixed-GT-valid residual over clean and all five objectives, all 128 frames",
                "confidence_vmin": conf_min, "confidence_vmax": max(conf_max, conf_min + 1e-9),
                "confidence_unit": "model score, not probability", "clipped_metric_values": False}


def load_component_projection(run_dir, project_root, dataset, sequence, objective):
    return ComponentContext(run_dir, project_root).pair(dataset, sequence, objective)
