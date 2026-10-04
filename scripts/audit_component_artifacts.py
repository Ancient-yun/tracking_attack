"""CPU-only audit of a component-study run, with no experiment/model imports.

Direct checks use saved NumPy arrays. Model replay and official clean parity
are explicitly recorded evidence; this script does not repeat inference or
execute the official metric implementation. Missing outputs never pass.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import zipfile

import numpy as np

# The shared helper itself imports only NumPy and the standard library.
_spec = importlib.util.spec_from_file_location(
    "_component_audit_helpers", Path(__file__).with_name("audit_attack_artifacts.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
Audit = _helpers.Audit
read_json = _helpers.read_json
fingerprint = _helpers.fingerprint
close = _helpers.close
bit_equal = _helpers.bit_equal

OBJECTIVES = ("tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training")
OBJECTIVE_TERMS = dict(zip(OBJECTIVES, ("tracking_mse", "tracking_regression", "reconstruction_regression", "confidence", "total")))
OUTPUT_FIELDS = {
    "tracks", "native_tracks", "track_conf", "reconstruction_norm",
    "reconstruction_regression_sum", "reconstruction_log_conf_sum",
    "reconstruction_l21_sum", "reconstruction_conf_sum",
}
TARGET_FIELDS = {"native_tracks", "native_track_valid", "training_dynamic", "tracks", "valid",
                 "reconstruction_gt_norm", "reconstruction_valid_counts"}
TRACK_FIELDS = {"pred", "gt", "valid", "dynamic", "query_xy", "intrinsics"}
ARTIFACTS = {"rgb_float32.npy", "delta_float32.npy", "reconstruction_float32.npy",
             "reconstruction_confidence_float32.npy", "components.npz", "tracks.npz",
             "history.json", "perturbation_norms.json"}
NATIVE_SOURCES = {"dust3r/losses.py", "dust3r/utils/geometry.py", "dust3r/utils/misc.py",
                  "scripts_run/train_seq_reweight.sh", "dust3r/datasets/pointodyssey.py",
                  "dust3r/datasets/dynamic_replica.py"}


def tensor_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype}:{value.shape}:".encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def json_safe(value):
    """Keep failure reports writable even when a corrupted NPY contains NaN."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return isinstance(value, (int, float)) and math.isfinite(value)


def archive(path, fields, audit):
    with np.load(path, allow_pickle=False) as data:
        audit.check("NPZ has exactly the required fields", set(data.files) == fields, path)
        return {name: data[name] for name in fields}


def mmap(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def stable_seed(base, dataset, sequence, epsilon):
    key = f"{base}:{dataset}:{sequence}:pgd:{epsilon:g}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def campaign_size(metadata):
    configured = metadata["config"].get("campaign_expected_clips")
    declared = metadata["manifest"].get("campaign_expected_clips")
    if configured is None and declared is None:
        return 14
    if (type(configured) is not int or type(declared) is not int or configured != declared
            or configured <= 0 or configured % 2):
        raise ValueError("config/manifest campaign_expected_clips must be matching positive even integers")
    return configured


def campaign_frames(metadata):
    """Historical64 runs remain readable but cannot claim released-all128 coverage."""
    config, manifest = metadata["config"], metadata["manifest"]
    declared = (config.get("campaign_expected_frames"), manifest.get("campaign_expected_frames"))
    flags = (config.get("all_frames"), manifest.get("all_frames"))
    if declared == (None, None) and flags == (None, None):
        if config.get("num_frames") == 128 or manifest.get("num_frames") == 128:
            raise ValueError("A128-frame run must explicitly declare all_frames and campaign_expected_frames in config/manifest")
        return 64, False
    if (any(type(value) is not int or value != 128 for value in declared)
            or flags != (True, True) or any(type(value) is not bool for value in flags)
            or any(type(value) is not int or value != 128
                   for value in (config.get("num_frames"), manifest.get("num_frames")))):
        raise ValueError("All-frame campaign must declare matching integer128 frames and boolean all_frames=true in both config/manifest")
    return 128, True


def all_frame_sequence_metadata(metadata):
    return all(type(metadata.get(name)) is int and metadata[name] == 128
               for name in ("num_frames", "requested_num_frames", "original_frame_count"))


def raw_all_frame_headers(path):
    """Read only archive headers; no JPEG decode or full depth array allocation."""
    shapes = {}
    names = ("images_jpeg_bytes", "tracks_XYZ", "visibility", "depth_map", "extrinsics_w2c")
    with zipfile.ZipFile(path) as archive:
        for name in names:
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported released NPY header version")
                shapes[name] = shape
    return shapes


def all_frame_raw_shapes(shapes):
    tracks, depth = shapes.get("tracks_XYZ", ()), shapes.get("depth_map", ())
    return (shapes.get("images_jpeg_bytes") == (128,) and len(tracks) == 3 and tracks[0] == 128
            and tracks[-1] == 3 and shapes.get("visibility") == tracks[:2]
            and len(depth) == 3 and depth[0] == 128
            and shapes.get("extrinsics_w2c") == (128, 4, 4))


def audit_sources(audit, metadata, project, experiment_source_dir):
    if experiment_source_dir is None:
        _helpers.audit_source_files(audit, metadata, project)
        return
    # An explicitly selected full image snapshot replaces the host package as
    # the source being audited. There is no per-file fallback to another copy.
    directory = Path(experiment_source_dir).resolve()
    digests = metadata["provenance"]["experiment_source_sha256"]
    found = {path.name for path in directory.glob("*.py") if path.is_file()}
    audit.check("explicit experiment source snapshot has exact recorded Python file set", found == set(digests), directory)
    source_metadata = {**metadata, "provenance": {**metadata["provenance"], "experiment_source_sha256": {}}}
    _helpers.audit_source_files(audit, source_metadata, project)
    for name, digest in digests.items():
        audit.check("snapshot Python source name safe", Path(name).name == name and name.endswith(".py"), directory, name)
        path = directory / name
        if audit.require(path):
            audit.hash(path, digest)


def audit_native_sources(audit, component_metadata, project):
    sources = component_metadata["source_sha256"]
    audit.check("native official source identities complete", set(sources) == NATIVE_SOURCES)
    vendor = Path(project) / "vendor/St4RTrack"
    for name in sorted(NATIVE_SOURCES & set(sources)):
        path = vendor / name
        if not audit.require(path):
            continue
        raw = path.read_bytes()
        raw_hash = audit.hash(path)
        blob = subprocess.check_output(["git", "-C", str(vendor), "show", f"HEAD:{name}"])
        canonical = raw.replace(b"\r\n", b"\n")
        blob_hash = hashlib.sha256(blob).hexdigest()
        audit.check("native host exact CRLF-to-LF bytes match pinned Git blob", canonical == blob, path)
        audit.check("native Git blob SHA256 matches recorded source", blob_hash == sources[name], path)
        audit.canonical_comparisons.append({"path": str(path), "scope": "official native source only",
            "raw_host_sha256": raw_hash, "git_blob_sha256": blob_hash,
            "canonical_host_sha256": hashlib.sha256(canonical).hexdigest(),
            "recorded_docker_sha256": sources[name],
            "canonicalization": "Exact bytes.replace(b'\\r\\n', b'\\n'); no other whitespace changes",
            "canonical_bytes_equal_git_blob": canonical == blob})


def audit_sequence(audit, entry, config, project, verify_sources, signature):
    directory = audit.root / "sequences" / entry["dataset"] / entry["sequence"]
    files = ("sequence.json", "targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy", "official_clean_parity.json")
    present = [audit.require(directory / name) for name in (*files, "sequence_manifest.json")]
    if not all(present):
        return None
    inventory = read_json(directory / "sequence_manifest.json")
    audit.hash(directory / "sequence_manifest.json")
    audit.check("sequence inventory signature and required SHA set agree", inventory.get("signature") == signature
                and set(inventory.get("artifact_sha256", {})) == set(files), directory)
    for name in files:
        audit.hash(directory / name, inventory.get("artifact_sha256", {}).get(name, "missing"))
    metadata = read_json(directory / "sequence.json")
    component = metadata["component_targets"]
    parity = read_json(directory / "official_clean_parity.json")
    parity_batch = parity.get("batch_size", metadata.get("official_reference_batch_size"))
    audit.check("recorded official clean parity passed", parity.get("passed") is True
                and parity_batch == 1 and close(parity.get("max_abs_error"), parity.get("max_abs_error"))
                and parity["max_abs_error"] >= 0, directory)
    frames = config["num_frames"]
    audit.check("sequence frame count agrees with frozen run", metadata["num_frames"] == frames, directory)
    if config.get("all_frames") is True:
        audit.check("all-frame sequence consumes exactly every released raw frame", all_frame_sequence_metadata(metadata), directory)
        if verify_sources:
            raw_path = Path(project) / entry["path"]
            if audit.require(raw_path):
                audit.check("raw NPZ image/depth/track/visibility/pose headers all have128 frames",
                            all_frame_raw_shapes(raw_all_frame_headers(raw_path)), raw_path)
    targets = archive(directory / "targets.npz", TARGET_FIELDS, audit)
    gt, valid = mmap(directory / "reconstruction_gt.npy"), mmap(directory / "reconstruction_valid.npy")
    width, height = metadata["preprocessed_wh"]
    audit.check("dense target shape/dtype matches RGB model grid", gt.shape == (frames, height, width, 3)
                and gt.dtype == np.float32 and valid.shape == (frames, height, width) and valid.dtype == np.bool_, directory)
    counts, norms = [], []
    for frame in range(frames):
        audit.check("dense GT geometry is finite including invalid padding", np.isfinite(gt[frame]).all(), directory, frame)
        counts.append(int(valid[frame].sum()))
        norms.append(max(float(np.linalg.norm(gt[frame], axis=-1).sum(dtype=np.float32) / (3 * height * width)), 1e-8))
    audit.check("each frame has nonempty dense GT membership", all(count > 0 for count in counts), directory)
    audit.check("fixed dense mask counts match saved targets", np.array_equal(targets["reconstruction_valid_counts"], counts), directory)
    audit.check("dense native GT norm is sum(all XYZ norms)/(3HW)", np.allclose(
        targets["reconstruction_gt_norm"], norms, rtol=3e-5, atol=1e-6), directory)
    hashes = component["tensor_sha256"]
    for name, value in {**targets, "reconstruction_gt": gt, "reconstruction_valid": valid}.items():
        audit.check("persisted target tensor hash matches sequence metadata", tensor_hash(value) == hashes.get(name), directory, name)
    audit.check("target arrays finite and native masks fixed", all(np.isfinite(value).all() for value in targets.values())
                and targets["valid"].dtype == targets["native_track_valid"].dtype == np.bool_
                and targets["valid"].all() and targets["native_track_valid"].all(), directory)
    audit.check("target geometry/norm float32 and counts int64", targets["tracks"].dtype == targets["native_tracks"].dtype
                == targets["reconstruction_gt_norm"].dtype == np.float32
                and targets["reconstruction_valid_counts"].dtype == np.int64, directory)
    m = targets["native_tracks"].shape[1]
    n = targets["tracks"].shape[1]
    audit.check("fixed native/evaluation target shapes", targets["tracks"].shape == (frames, n, 3) and n > 0
                and targets["valid"].shape == (frames, n) and targets["native_tracks"].shape == (frames, m, 3) and m > 0
                and targets["native_track_valid"].shape == (frames, m)
                and targets["training_dynamic"].shape == (m,) and targets["training_dynamic"].dtype == np.bool_, directory)
    movement = np.abs(targets["native_tracks"][-1] - targets["native_tracks"][0]).sum(-1)
    movement /= np.linalg.norm(targets["native_tracks"][0], axis=-1) + 1e-8
    audit.check("native endpoint relative-L1 dynamic mask agrees", np.array_equal(
        movement > movement.mean(), targets["training_dynamic"]), directory)
    audit.check("native confidence loss has static support", (~targets["training_dynamic"]).any(), directory)
    if verify_sources:
        audit_native_sources(audit, component, project)
    return {"metadata": metadata, "component": component, "targets": targets, "gt": gt, "valid": valid,
            "directory": directory, "counts": np.asarray(counts), "shape": (frames, height, width),
            "recorded_only_target_hashes": sorted(set(hashes) - set(targets) - {
                "reconstruction_gt", "reconstruction_valid", "native_query_xy", "native_point_indices"})}


def audit_membership(audit, sequence, tracks, rgb):
    metadata, component, targets = sequence["metadata"], sequence["component"], sequence["targets"]
    path = sequence["directory"]
    n = targets["tracks"].shape[1]
    query = tracks["query_xy"]
    audit.check("query count/coordinates/selected indices agree", metadata["num_queries"] == component["eval_query_count"] == n
                and query.shape == (n, 2) and len(metadata["selected_point_indices"]) == n, path)
    audit.check("evaluation queries finite and in bounds", np.isfinite(query).all()
                and (query[:, 0] >= 0).all() and (query[:, 0] < rgb.shape[-1]).all()
                and (query[:, 1] >= 0).all() and (query[:, 1] < rgb.shape[-2]).all(), path)
    audit.check("evaluation query tensor hash matches sequence metadata", tensor_hash(query) == component["evaluation_query_xy_sha256"], path)
    if "intrinsics" in metadata:
        audit.check("evaluation intrinsics agree with fixed sequence metadata", bit_equal(tracks["intrinsics"],
                    np.asarray(metadata["intrinsics"], dtype=tracks["intrinsics"].dtype)), path)
    movement = np.linalg.norm(np.diff(tracks["gt"], axis=0), axis=-1).sum(axis=0)
    # GT was stored as float32, whereas official membership was chosen from
    # the loader's float64 tracks. Do not silently reclassify a boundary case.
    unambiguous = np.abs(movement - .01) > 1e-5
    audit.check("evaluation dynamic mask matches unambiguous cumulative world motion", np.array_equal(
        tracks["dynamic"][unambiguous], (movement > .01)[unambiguous]), path)
    pixels = np.rint(query).astype(np.int64)
    pixels[:, 0] = pixels[:, 0].clip(0, rgb.shape[-1] - 1)
    pixels[:, 1] = pixels[:, 1].clip(0, rgb.shape[-2] - 1)
    flat = pixels[:, 1] * rgb.shape[-1] + pixels[:, 0]
    last = {int(pixel): index for index, pixel in enumerate(flat)}
    selected = np.asarray([last[pixel] for pixel in sorted(last)], dtype=np.int64)
    native_xy = pixels[selected]
    point_indices = np.asarray(metadata["selected_point_indices"], dtype=np.int64)[selected]
    audit.check("native raster membership is round/clamp/last-writer row-major", bit_equal(
        targets["native_tracks"], tracks["gt"][:, selected]), path)
    audit.check("native query count and collisions agree", component["native_query_count"] == len(selected)
                and component["native_collision_count"] == n - len(selected), path)
    for name, value in (("native_query_xy", native_xy), ("native_point_indices", point_indices)):
        audit.check("reconstructed native membership tensor hash agrees", tensor_hash(value) == component["tensor_sha256"][name], path, name)
    return selected


def audit_history(audit, path, config, objective):
    history = read_json(path)
    frames, steps = config["num_frames"], config["steps"]
    clean = objective == "clean"
    epsilon = 0 if clean else config["epsilon_255"] / 255
    label = "tracking_mse" if clean else objective
    audit.check("history nonempty", isinstance(history, list) and bool(history), path)
    if not isinstance(history, list) or not history:
        return {}, []
    gradients, zero_frames = [], []
    for state in history:
        audit.check("history objective/loss/scale/terms valid", state["objective"] == label
                    and finite_tree(state["terms"]) and np.isfinite([state["loss"], state["scale"]]).all()
                    and state["scale"] > 0, path)
        audit.check("history objective equals its recorded loss term", close(state["loss"], state["terms"][OBJECTIVE_TERMS[label]]), path)
        linf = np.asarray(state["delta_linf_per_frame"])
        l2 = np.asarray(state["delta_l2_per_frame"])
        audit.check("history delta norms cover every frame and budget", linf.shape == l2.shape == (frames,)
                    and np.isfinite(linf).all() and np.isfinite(l2).all() and (linf >= 0).all() and (l2 >= 0).all()
                    and linf.max() <= epsilon + 1e-7 and close(state["delta_linf"], linf.max(), atol=1e-7, rtol=0), path)
        if "gradient_linf_per_frame" in state or "gradient_l2_per_frame" in state:
            glinf = np.asarray(state.get("gradient_linf_per_frame", []))
            gl2 = np.asarray(state.get("gradient_l2_per_frame", []))
            audit.check("recorded RGB gradient norms cover all frames and are finite", glinf.shape == gl2.shape == (frames,)
                        and np.isfinite(glinf).all() and np.isfinite(gl2).all(), path)
            zeros = np.flatnonzero((glinf <= 0) | (gl2 <= 0)).tolist() if glinf.shape == gl2.shape == (frames,) else []
            audit.check("recorded RGB gradient norms nonnegative", (glinf >= 0).all() and (gl2 >= 0).all(), path)
            if zeros:
                zero_frames.append({"restart": state["restart"], "step": state["step"], "frames": zeros})
            gradients.append((state["restart"], state["step"]))
    if clean:
        audit.check("clean history has baseline only", len(history) == 1 and not gradients
                    and (history[0]["restart"], history[0]["step"]) == (-1, 0), path)
    else:
        audit.check("PGD history has every requested restart-zero gradient step", gradients == [(0, step) for step in range(steps)], path)
        audit.check("PGD history includes clean incumbent and final candidate", len(history) == steps + 2
                    and (history[0]["restart"], history[0]["step"]) == (-1, 0)
                    and (history[-1]["restart"], history[-1]["step"]) == (0, steps), path)
    return {"history_entries": len(history), "gradient_steps": len(gradients),
            "gradient_frames_per_step": frames if gradients else 0, "zero_gradient_frames": zero_frames,
            "zero_gradient_frame_step_count": sum(len(item["frames"]) for item in zero_frames),
            "all_recorded_gradient_norms_nonzero": not zero_frames if gradients else None}, history


def audit_oracle(audit, path, config, terms, component_metadata, project, verify_sources):
    evidence = read_json(path)
    errors = evidence.get("head_gradient_max_abs_error", {})
    fields = {"head1_xyz", "head1_conf", "head2_xyz", "head2_conf"}
    audit.check("recorded native official AST oracle passed on all four head derivatives", evidence.get("passed") is True
                and evidence.get("device") == "cpu" and evidence.get("num_frames") == config["num_frames"]
                and set(errors) == fields and all(isinstance(error, (int, float)) and math.isfinite(error) and error >= 0 for error in errors.values()), path)
    oracle = evidence["oracle"]
    audit.check("recorded oracle used unchanged official active-loss AST", oracle["mode"].startswith("unchanged official active-loss AST")
                and oracle["full_training_loss_module_imported"] is False, path)
    audit.check("oracle official source hashes match native target provenance",
                oracle["losses_source_sha256"] == component_metadata["source_sha256"]["dust3r/losses.py"]
                and oracle["geometry_source_sha256"] == component_metadata["source_sha256"]["dust3r/utils/geometry.py"]
                and oracle["misc_source_sha256"] == component_metadata["source_sha256"]["dust3r/utils/misc.py"], path)
    expected = {"tracking": terms["tracking_regression"] - config["alpha"] * terms["diagnostics"]["track_log_conf_mean"],
                "reconstruction": terms["reconstruction_regression"] - config["alpha"] * terms["diagnostics"]["reconstruction_log_conf_mean"],
                "joint": terms["total"], "tracking_l21": terms["diagnostics"]["tracking_l21"],
                "reconstruction_l21": terms["diagnostics"]["reconstruction_l21"]}
    for name, value in expected.items():
        audit.check("recorded oracle component agrees with direct saved-array terms", close(evidence["components"].get(name), value, atol=2e-5, rtol=8e-5), path, name)
    loss_error = evidence["loss_abs_error"]
    audit.check("recorded oracle loss error finite/nonnegative within documented tolerance", isinstance(loss_error, (int, float))
                and math.isfinite(loss_error) and 0 <= loss_error <= evidence["atol"] + evidence["rtol"] * abs(expected["joint"]), path)
    if verify_sources:
        source = Path(project) / "vendor/St4RTrack/dust3r/losses.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        names = {"Sum", "BaseCriterion", "LLoss", "L21Loss", "Criterion", "MultiLoss", "Regr3D", "ConfLoss"}
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        selected_ast = ast.Module(body=selected, type_ignores=[])
        expected_hash = hashlib.sha256(ast.dump(selected_ast, include_attributes=False).encode()).hexdigest()
        audit.check("recorded oracle AST matches exact pinned source nodes", {node.name for node in selected} == names
                    and oracle["selected_nodes"] == [node.name for node in selected]
                    and oracle["selected_ast_sha256"] == expected_hash, path)


def native_terms(outputs, targets, config):
    mask = targets["native_track_valid"]
    dynamic = np.broadcast_to(targets["training_dynamic"], mask.shape)[mask]
    confidence = outputs["track_conf"][mask]
    maximum = confidence[~dynamic].max()
    weights = np.where(dynamic, maximum * config["reweight_scale"], confidence)
    norm = np.maximum(outputs["reconstruction_norm"], 1e-8)[:, None, None]
    gt_norm = np.maximum(targets["reconstruction_gt_norm"], 1e-8)[:, None, None]
    distances = np.linalg.norm((outputs["native_tracks"] / norm - targets["native_tracks"] / gt_norm)[mask], axis=-1)
    tracking = float((distances * weights).mean())
    count = targets["reconstruction_valid_counts"].sum()
    reconstruction = float(outputs["reconstruction_regression_sum"].sum() / count)
    track_log = float(np.log(np.maximum(weights, 1)).mean())
    reconstruction_log = float(outputs["reconstruction_log_conf_sum"].sum() / count)
    conf_term = -config["alpha"] * (track_log + reconstruction_log)
    pred, gt, valid = outputs["tracks"], targets["tracks"], targets["valid"]
    scale = np.median(np.maximum(np.linalg.norm(gt[valid], axis=-1), 1e-12)) / np.median(
        np.maximum(np.linalg.norm(pred[valid], axis=-1), 1e-12))
    mse = float(((pred[valid] * scale - gt[valid]) ** 2).sum(-1).mean())
    return {"tracking_mse": mse, "tracking_regression": tracking, "reconstruction_regression": reconstruction,
            "confidence": conf_term, "total": tracking + reconstruction + conf_term,
            "diagnostics": {"tracking_l21": float(distances.mean()),
                "reconstruction_l21": float(outputs["reconstruction_l21_sum"].sum() / count),
                "track_conf_mean": float(weights.mean()), "track_raw_conf_mean": float(confidence.mean()),
                "static_conf_max_detached": float(maximum), "reconstruction_conf_mean": float(outputs["reconstruction_conf_sum"].sum() / count),
                "track_log_conf_mean": track_log, "reconstruction_log_conf_mean": reconstruction_log,
                "native_track_count": int(mask.sum()), "reconstruction_count": int(count)}}


def check_terms(audit, actual, expected, path, prefix=""):
    audit.check("loss term names match native formula", set(actual) == set(expected), path, prefix)
    for name in set(actual) & set(expected):
        if isinstance(expected[name], dict):
            check_terms(audit, actual[name], expected[name], path, prefix + name + ".")
        else:
            audit.check("loss term agrees with saved component arrays", close(actual[name], expected[name], atol=2e-5, rtol=8e-5), path, prefix + name)


def audit_condition(audit, directory, entry, objective, metadata, sequence, clean=None, *, project=None, verify_sources=False):
    path = directory / "result.json"
    if not audit.require(path):
        return None
    record = read_json(path)
    audit.hash(path)
    config = metadata["config"]
    frames = config["num_frames"]
    is_clean = objective == "clean"
    epsilon = 0 if is_clean else config["epsilon_255"] / 255
    seed = stable_seed(config["seed"], entry["dataset"], entry["sequence"], config["epsilon_255"])
    audit.check("result identity/signature/seed matches frozen run", record["dataset"] == entry["dataset"]
                and record["sequence"] == entry["sequence"] and record["objective"] == objective
                and record["signature"] == metadata["signature"] and record["seed"] == seed
                and record["steps"] == (0 if is_clean else config["steps"])
                and record["epsilon_255"] == (0 if is_clean else config["epsilon_255"]), path)
    audit.check("result loss/timing/terms finite", finite_tree(record["loss_terms"])
                and np.isfinite([record["attack_loss"], record["elapsed_seconds"], record["condition_wall_seconds"]]).all()
                and record["elapsed_seconds"] >= 0 and record["condition_wall_seconds"] >= record["elapsed_seconds"], path)
    if not is_clean:
        attack = record["attack_config"]
        audit.check("attack configuration matches requested PGD scope", attack["method"] == "pgd"
                    and close(attack["eps"], epsilon, atol=1e-12, rtol=0) and attack["steps"] == config["steps"]
                    and attack["restarts"] == config["restarts"] == 1 and attack["seed"] == seed
                    and close(attack["step_size"], epsilon * config["step_size_fraction"], atol=1e-12, rtol=0)
                    and attack["gradient_mode"] == "recompute" and attack["check_replay"] is True
                    and attack["record_per_frame"] is True and attack["random_start"] is True, path)
    digests = record["artifact_sha256"]
    required_artifacts = ARTIFACTS | ({"native_loss_oracle.json"} if is_clean else set())
    actual_files = {item.name for item in directory.iterdir() if item.is_file() and item.name != "result.json"}
    audit.check("every saved artifact has SHA256 and all required artifacts are listed", set(digests) == actual_files == required_artifacts, path)
    present = True
    for name in sorted(required_artifacts):
        present = audit.require(directory / name) and present
        if (directory / name).is_file():
            audit.hash(directory / name, digests.get(name, "missing"))
    if not present or sequence is None:
        return None
    outputs = archive(directory / "components.npz", OUTPUT_FIELDS, audit)
    tracks = archive(directory / "tracks.npz", TRACK_FIELDS, audit)
    rgb, delta = mmap(directory / "rgb_float32.npy"), mmap(directory / "delta_float32.npy")
    reconstruction = mmap(directory / "reconstruction_float32.npy")
    confidence = mmap(directory / "reconstruction_confidence_float32.npy")
    t, h, w = sequence["shape"]
    audit.check("saved RGB/delta dtype and grid shape", rgb.dtype == delta.dtype == np.float32
                and rgb.shape == delta.shape == (frames, 3, h, w), directory)
    audit.check("saved dense output shapes and dtype", reconstruction.shape == (t, h, w, 3)
                and reconstruction.dtype == np.float32 and confidence.shape == (t, h, w) and confidence.dtype == np.float32, directory)
    audit.check("compact outputs finite float32", all(value.dtype == np.float32 and np.isfinite(value).all() for value in outputs.values()), directory)
    n, m = sequence["targets"]["tracks"].shape[1], sequence["targets"]["native_tracks"].shape[1]
    audit.check("native compact outputs have exact target membership shapes", outputs["tracks"].shape == (frames, n, 3)
                and outputs["native_tracks"].shape == (frames, m, 3) and outputs["track_conf"].shape == (frames, m)
                and all(outputs[name].shape == (frames,) for name in OUTPUT_FIELDS - {"tracks", "native_tracks", "track_conf"}), directory)
    audit.check("compact native confidence >= checkpoint floor one", (outputs["track_conf"] >= 1).all(), directory)
    audit.check("query prediction matches saved compact prediction bit-exactly", bit_equal(tracks["pred"], outputs["tracks"]), directory)
    for name, target in (("gt", "tracks"), ("valid", "valid")):
        audit.check("condition GT/membership equals fixed shared target bit-exactly", bit_equal(tracks[name], sequence["targets"][target]), directory, name)
    audit.check("evaluation trajectory arrays finite and masks valid", tracks["pred"].shape == tracks["gt"].shape == (frames, n, 3)
                and tracks["valid"].shape == (frames, n) and tracks["valid"].dtype == tracks["dynamic"].dtype == np.bool_
                and tracks["dynamic"].shape == (n,) and tracks["valid"].all()
                and all(np.isfinite(tracks[name]).all() for name in ("pred", "gt", "query_xy", "intrinsics")), directory)
    if is_clean:
        audit_membership(audit, sequence, tracks, rgb)
        audit.check("clean RGB tensor hash matches sequence metadata", tensor_hash(rgb) == sequence["component"]["rgb_sha256"], directory)
    elif clean is not None:
        for name in ("gt", "valid", "dynamic", "query_xy", "intrinsics"):
            audit.check("objective and clean share bit-identical evaluation GT/query/masks/intrinsics", bit_equal(tracks[name], clean["tracks"][name]), directory, name)
    linf, l2 = [], []
    for frame in range(frames):
        audit.check("saved RGB finite in [0,1]", np.isfinite(rgb[frame]).all() and (rgb[frame] >= 0).all() and (rgb[frame] <= 1).all(), directory, frame)
        audit.check("saved delta finite", np.isfinite(delta[frame]).all(), directory, frame)
        linf.append(float(np.abs(delta[frame]).max()))
        l2.append(float(np.linalg.norm(delta[frame].reshape(-1))))
        audit.check("RGB delta respects epsilon including clean", linf[-1] <= epsilon + 1e-7, directory, frame)
        if not is_clean and clean is not None:
            difference = np.subtract(rgb[frame], clean["rgb"][frame], dtype=np.float32)
            audit.check("attacked minus clean RGB equals saved delta bit-exactly", bit_equal(difference, delta[frame]), directory, frame)
        audit.check("dense geometry finite and confidence >= one", np.isfinite(reconstruction[frame]).all()
                    and np.isfinite(confidence[frame]).all() and (confidence[frame] >= 1).all(), directory, frame)
        mask = sequence["valid"][frame]
        norm = max(float(np.linalg.norm(reconstruction[frame], axis=-1).sum(dtype=np.float32) / (3*h*w)), 1e-8)
        audit.check("native normalization agrees with full saved pointmap", close(outputs["reconstruction_norm"][frame], norm, atol=1e-6, rtol=3e-5), directory, frame)
        residual = np.linalg.norm(reconstruction[frame][mask] / outputs["reconstruction_norm"][frame]
            - sequence["gt"][frame][mask] / sequence["targets"]["reconstruction_gt_norm"][frame], axis=-1)
        conf = confidence[frame][mask]
        stats = {"reconstruction_regression_sum": (conf * residual).sum(), "reconstruction_log_conf_sum": np.log(conf).sum(),
                 "reconstruction_l21_sum": residual.sum(), "reconstruction_conf_sum": conf.sum()}
        for name, expected in stats.items():
            audit.check("native sufficient statistic matches saved dense arrays and fixed mask", close(outputs[name][frame], expected, atol=2e-4, rtol=8e-5), directory, {"frame": frame, "field": name})
    if is_clean:
        audit.check("clean delta exactly zero", all(value == 0 for value in linf), directory)
    if record["selected_state"]["restart"] == -1:
        audit.check("selected clean incumbent has exactly zero saved delta", all(value == 0 for value in linf), directory)
    norms = read_json(directory / "perturbation_norms.json")
    audit.check("norms result/file agree", record["perturbation_norms"] == norms, path)
    audit.check("saved perturbation norms agree with direct arrays", norms["epsilon_255"] == (0 if is_clean else config["epsilon_255"])
                and close(norms["linf"], max(linf), atol=1e-7, rtol=0)
                and np.allclose(norms["linf_per_frame"], linf, atol=1e-7, rtol=0)
                and np.allclose(norms["l2_per_frame"], l2, atol=2e-5, rtol=8e-5)
                and norms["attacked_frames"] == sum(value > 0 for value in linf), directory)
    verification = record["saved_input_verification"]
    audit.check("recorded saved-input replay passed for every compact field", verification.get("passed") is True
                and set(verification.get("capture_output_errors", {})) == set(verification.get("compact_output_errors", {})) == OUTPUT_FIELDS, path)
    for stage in ("capture_output_errors", "compact_output_errors"):
        for name in OUTPUT_FIELDS:
            error = verification.get(stage, {}).get(name)
            audit.check("recorded compact replay maximum error exactly zero", close(error, 0, atol=0, rtol=0), path, stage + "." + name)
    for name in ("max_reconstruction_error", "max_reconstruction_confidence_error"):
        audit.check("recorded dense replay maximum error exactly zero", close(verification.get(name), 0, atol=0, rtol=0), path, name)
    info, history = audit_history(audit, directory / "history.json", config, objective)
    if history:
        maximum = max(state["loss"] for state in history)
        audit.check("saved result is strongest recorded incumbent", close(record["attack_loss"], maximum, atol=1e-6, rtol=1e-6), path)
        selected = record["selected_state"]
        matches = [state for state in history if (state["restart"], state["step"]) == (selected["restart"], selected["step"])]
        audit.check("selected state exists and matches result loss/terms", len(matches) == 1
                    and close(matches[0]["loss"], record["attack_loss"], atol=1e-6, rtol=1e-6)
                    and matches[0]["terms"] == record["loss_terms"], path)
    expected_terms = native_terms(outputs, sequence["targets"], config)
    check_terms(audit, record["loss_terms"], expected_terms, path)
    if is_clean:
        audit_oracle(audit, directory / "native_loss_oracle.json", config, expected_terms, sequence["component"], project, verify_sources)
    label = "tracking_mse" if is_clean else objective
    audit.check("selected attack loss matches direct saved-array objective", close(record["attack_loss"], expected_terms[OBJECTIVE_TERMS[label]], atol=2e-5, rtol=8e-5), path)
    metrics, recon_metrics = record["tracking_metrics"], record["reconstruction_metrics"]
    audit.check("recorded metrics use actual saved populations", metrics["num_frames"] == frames and metrics["num_queries"] == n
                and metrics["num_dynamic_queries"] == int(tracks["dynamic"].sum()) and recon_metrics["num_frames"] == frames
                and recon_metrics["valid_pixels"] == int(sequence["counts"].sum()), path)
    for group in (metrics, recon_metrics):
        for name, value in group.items():
            if isinstance(value, (int, float)):
                audit.check("recorded metric finite", math.isfinite(value), path, name)
                if "apd" in name or "within_" in name:
                    audit.check("recorded percentage metric in [0,100]", 0 <= value <= 100, path, name)
                if "epe" in name:
                    audit.check("recorded distance metric nonnegative", value >= 0, path, name)
    return {"record": record, "tracks": tracks, "rgb": rgb, "details": {
        "dataset": entry["dataset"], "sequence": entry["sequence"], "objective": objective,
        "rgb_shape": list(rgb.shape), "num_queries": n, "native_queries": m,
        "delta_linf": max(linf), "nonzero_delta_frames": sum(value > 0 for value in linf),
        "selected_clean_incumbent": record["selected_state"]["restart"] == -1,
        "stationary_saved_input": all(value == 0 for value in linf), **info}}


def summary_rows(conditions):
    records = [condition["record"] for condition in conditions.values()]
    baselines = {(r["dataset"], r["sequence"]): r for r in records if r["objective"] == "clean"}
    rows = []
    for record in records:
        row = {key: record[key] for key in ("dataset", "sequence", "objective", "epsilon_255", "steps", "seed", "elapsed_seconds")}
        row["condition_wall_seconds"] = record.get("condition_wall_seconds")
        for prefix, group in (("tracking_", record["tracking_metrics"]), ("reconstruction_", record["reconstruction_metrics"]),
                              ("loss_", record["loss_terms"]), ("diagnostic_", record["loss_terms"].get("diagnostics", {}))):
            row.update({prefix + key: value for key, value in group.items() if isinstance(value, (int, float))})
        clean = baselines.get((record["dataset"], record["sequence"]))
        if clean:
            row["tracking_apd_drop_pp"] = clean["tracking_metrics"]["apd3d_all"] - record["tracking_metrics"]["apd3d_all"]
            row["tracking_epe_increase_m"] = record["tracking_metrics"]["epe_all_m"] - clean["tracking_metrics"]["epe_all_m"]
            row["reconstruction_apd_drop_pp"] = clean["reconstruction_metrics"]["apd3d"] - record["reconstruction_metrics"]["apd3d"]
            row["reconstruction_epe_increase_m"] = record["reconstruction_metrics"]["epe_m"] - clean["reconstruction_metrics"]["epe_m"]
        rows.append(row)
    return rows


def audit_summaries(audit, conditions, metadata, expected_keys):
    rows = summary_rows(conditions)
    entries, objectives = metadata["manifest"]["entries"], metadata["config"]["objectives"]
    aggregate = []
    for dataset in ("po_mini", "ds_mini", "all"):
        for objective in ("clean", *objectives):
            group = [row for row in rows if row["objective"] == objective and (dataset == "all" or row["dataset"] == dataset)]
            if not group:
                continue
            item = {"dataset": dataset, "objective": objective, "completed_sequences": len(group),
                    "expected_sequences": sum(dataset == "all" or row["dataset"] == dataset for row in entries),
                    "attack_seconds_sum": sum(row["elapsed_seconds"] for row in group)}
            for field in group[0]:
                values = [row.get(field) for row in group]
                if field not in item and all(isinstance(value, (int, float)) for value in values):
                    item[field] = float(np.mean(values))
            aggregate.append(item)
    summary_path = audit.root / "summary.json"
    if audit.require(summary_path):
        summary = read_json(summary_path)
        audit.hash(summary_path)
        missing = sorted(expected_keys - set(conditions))
        audit.check("summary counts/missing identities match saved conditions", summary["expected_conditions"] == len(expected_keys)
                    and summary["completed_conditions"] == len(conditions)
                    and summary["missing_conditions"] == [list(item) for item in missing]
                    and summary["complete"] == (not missing), summary_path)
        audit.check("summary reports no failed attempts", summary["failed_attempts"] == 0 and summary["failures"] == [], summary_path)
        actual = {(row["dataset"], row["objective"]): row for row in summary["aggregate"]}
        reference = {(row["dataset"], row["objective"]): row for row in aggregate}
        audit.check("summary aggregate groups complete and unique", len(actual) == len(summary["aggregate"])
                    and set(actual) == set(reference), summary_path)
        for identity in set(actual) & set(reference):
            audit.check("summary aggregate fields complete", set(actual[identity]) == set(reference[identity]), summary_path, list(identity))
            for field, value in reference[identity].items():
                recorded = actual[identity].get(field)
                ok = close(value, recorded, atol=1e-9, rtol=1e-9) if isinstance(value, (int, float)) else value == recorded
                audit.check("summary equal-weight mean agrees with condition records", ok, summary_path,
                            {"identity": list(identity), "field": field})
        audit.check("summary declares equal-weight sequence aggregation", summary["aggregation"].startswith("equal_weight_per_sequence"), summary_path)
    for filename, actual_reference, keys in (("sequence_metrics.csv", rows, ("dataset", "sequence", "objective")),
                                            ("aggregate_metrics.csv", aggregate, ("dataset", "objective"))):
        path = audit.root / filename
        if not audit.require(path):
            continue
        audit.hash(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            actual = list(csv.DictReader(stream))
        found = {tuple(row[key] for key in keys): row for row in actual}
        reference = {tuple(row[key] for key in keys): row for row in actual_reference}
        audit.check("CSV identities complete and unique", len(found) == len(actual) and set(found) == set(reference), path)
        for key in set(found) & set(reference):
            for field, value in reference[key].items():
                recorded = found[key].get(field)
                ok = close(value, recorded, atol=1e-9, rtol=1e-9) if isinstance(value, (int, float)) else recorded == ("" if value is None else value)
                audit.check("CSV field agrees with condition records or equal-weight aggregate", ok, path, {"identity": list(key), "field": field})
    failure_path = audit.root / "failures.jsonl"
    if failure_path.exists():
        audit.hash(failure_path)
        audit.check("no recorded failed attempts", not failure_path.read_text(encoding="utf-8").strip(), failure_path)
    return aggregate


def audit_run(run_dir, project_root, *, expected_clips=None, expected_frames=None, expected_steps=None,
              verify_sources=True, experiment_source_dir=None):
    audit = Audit(run_dir)
    explicit_clip_scope = expected_clips is not None
    explicit_frame_scope = expected_frames is not None
    expected_clips = 14 if expected_clips is None else expected_clips
    expected_frames = 64 if expected_frames is None else expected_frames
    expected_steps = 20 if expected_steps is None else expected_steps
    conditions, sequence_details, aggregate = {}, [], []
    expected_keys, metadata, full_scope, advertised_clips = set(), None, False, 14
    advertised_frames, all_frames = 64, False
    try:
        path = audit.root / "run.json"
        if audit.require(path):
            metadata = read_json(path)
            audit.hash(path)
            config, manifest = metadata["config"], metadata["manifest"]
            advertised_clips = campaign_size(metadata)
            advertised_frames, all_frames = campaign_frames(metadata)
            if not explicit_clip_scope:
                expected_clips = advertised_clips
            if not explicit_frame_scope:
                expected_frames = advertised_frames
            audit.check("run signature equals exact canonical config/manifest/provenance fingerprint", metadata["signature"] == fingerprint(
                {"config": config, "manifest": manifest, "provenance": metadata["provenance"]}), path)
            objectives = config["objectives"]
            audit.check("requested audit scope matches run metadata", len(manifest["entries"]) == expected_clips
                        and manifest["num_frames"] == config["num_frames"] == expected_frames and config["steps"] == expected_steps, path)
            audit.check("PGD settings and fixed weight/GT contract", config["epsilon_255"] == 4 and config["restarts"] == 1
                        and config["step_size_fraction"] == .25 and config["alpha"] == .2 and config["reweight_scale"] == 5
                        and config["image_size"] == 512 and config["gradient_mode"] == "recompute"
                        and config["save_inputs"] is True and config["verify_saved_inputs"] is True and config["verify_official_clean"] is True
                        and metadata["tta"] is False and metadata["frozen_weights"] is True, path)
            audit.check("objectives are unique active components", len(set(objectives)) == len(objectives) > 0 and set(objectives) <= set(OBJECTIVES), path)
            entries = manifest["entries"]
            identities = [(entry["dataset"], entry["sequence"]) for entry in entries]
            audit.check("manifest identities unique and safe", len(set(identities)) == len(entries) and all(
                name not in ("", ".", "..") and Path(name).name == name and "/" not in name and "\\" not in name
                for identity in identities for name in identity), path)
            limited_markers = " ".join(str(value) for value in (
                manifest.get("purpose", ""), manifest.get("scope", ""), config.get("scope", ""),
                metadata.get("scope", ""), metadata.get("analysis_scope", ""))).lower()
            limited = any(marker in limited_markers for marker in ("limited", "preflight", "smoke", "pilot"))
            full_scope = len(entries) == advertised_clips and Counter(entry["dataset"] for entry in entries) == {
                "po_mini": advertised_clips // 2, "ds_mini": advertised_clips // 2}
            full_scope = full_scope and config["num_frames"] == advertised_frames and config["steps"] == 20 and set(objectives) == set(OBJECTIVES)
            full_scope = full_scope and not limited
            if (expected_clips, expected_frames, expected_steps) == (advertised_clips, advertised_frames, 20) and not limited:
                audit.check("full campaign has declared balanced roster and all five objectives", full_scope, path)
            expected_keys = {(entry["dataset"], entry["sequence"], objective) for entry in entries for objective in ("clean", *objectives)}
            if verify_sources:
                audit_sources(audit, metadata, project_root, experiment_source_dir)
            for entry in entries:
                try:
                    sequence = audit_sequence(audit, entry, config, project_root, verify_sources, metadata["signature"])
                    if sequence:
                        sequence_details.append({"dataset": entry["dataset"], "sequence": entry["sequence"],
                            "num_queries": sequence["targets"]["tracks"].shape[1],
                            "native_queries": sequence["targets"]["native_tracks"].shape[1],
                            "dense_valid_counts_per_frame": sequence["counts"].tolist(),
                            "unpersisted_target_tensor_hashes_recorded_only": sequence["recorded_only_target_hashes"]})
                    baseline = None
                    for objective in ("clean", *objectives):
                        directory = audit.root / "conditions" / objective / entry["dataset"] / entry["sequence"]
                        try:
                            condition = audit_condition(audit, directory, entry, objective, metadata, sequence, baseline,
                                                        project=project_root, verify_sources=verify_sources)
                            if condition:
                                conditions[entry["dataset"], entry["sequence"], objective] = condition
                                if objective == "clean":
                                    baseline = condition
                        except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError) as error:
                            audit.check("condition arrays and schema readable", False, directory, repr(error))
                    del sequence, baseline
                except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
                    audit.check("sequence target arrays and metadata readable", False, audit.root, {"entry": entry, "error": repr(error)})
            actual = {(path.parents[1].name, path.parent.name, path.parents[2].name)
                      for path in (audit.root / "conditions").glob("*/*/*/result.json")}
            audit.check("no unexpected condition results", actual <= expected_keys, audit.root)
            aggregate = audit_summaries(audit, conditions, metadata, expected_keys)
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
        audit.check("run metadata/provenance/summary readable", False, audit.root, repr(error))
    expected_count = len(expected_keys) if metadata is not None else expected_clips * 6
    auditor_hashes = {"audit_component_artifacts.py": audit.hash(Path(__file__)),
                     "audit_attack_artifacts.py": audit.hash(Path(__file__).with_name("audit_attack_artifacts.py"))}
    status = "incomplete" if audit.missing else "failed" if audit.errors else "incomplete" if len(conditions) != expected_count else "passed"
    return json_safe({"schema_version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(audit.root),
        "status": status, "passed": status == "passed", "full_campaign_completed": status == "passed" and full_scope,
        "all_frames_completed": status == "passed" and full_scope and all_frames,
        "scope": "CPU saved-array/file integrity and independent NumPy component formulas; no model inference or official evaluator execution",
        "direct_array_checks": ["RGB/delta/epsilon and exact clean subtraction", "GT/query/native raster membership and tensor hashes",
            "finite geometry and confidence floor", "dense statistics and native component formulas on fixed masks"],
        "recorded_evidence_only": ["model replay errors, official clean parity and native official AST head-derivative oracle", "per-step RGB gradient norms (gradient tensors are not saved)",
            "official tracking/reconstruction metric values; aggregate means are recomputed from condition records",
            "unpersisted camera/depth target tensor hashes listed per sequence"],
        "gradient_zero_policy": "Finite nonnegative zero norms are reported separately as stationarity; this audit does not assert nonzero gradients or independently prove model graph connectivity",
        "source_files_verified": verify_sources, "auditor_source_sha256": auditor_hashes,
        "experiment_source_directory": str(Path(experiment_source_dir).resolve() if experiment_source_dir is not None
                                           else Path(project_root).resolve() / "src/st4rtrack_pgd"),
        "experiment_source_selection": "explicit full snapshot; no per-file fallback" if experiment_source_dir is not None else "current host package",
        "expected_conditions": expected_count, "completed_conditions_checked": len(conditions),
        "campaign_expected_clips": advertised_clips, "full_campaign_expected_conditions": advertised_clips * 6,
        "campaign_expected_frames": advertised_frames, "all_frames": all_frames,
        "requested_scope": {"clips": expected_clips, "frames": expected_frames, "steps": expected_steps},
        "checked_assertions": audit.checked, "errors": audit.errors, "missing": sorted(set(audit.missing)),
        "sequences": sequence_details, "conditions": [condition["details"] for condition in conditions.values()],
        "recomputed_equal_weight_aggregate_from_condition_records": aggregate,
        "file_sha256": audit.hashes, "canonical_source_comparisons": audit.canonical_comparisons})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-clips", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--experiment-source-dir", type=Path,
                        help="Explicit complete package snapshot from the run image; every recorded .py file must match exactly")
    args = parser.parse_args()
    for name in ("expected_clips", "expected_frames", "expected_steps"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    report = audit_run(args.run_dir, args.project_root, expected_clips=args.expected_clips,
                       expected_frames=args.expected_frames, expected_steps=args.expected_steps,
                       experiment_source_dir=args.experiment_source_dir)
    if not args.run_dir.is_dir():
        parser.error("run-dir must already exist; no run outputs were created")
    target = args.run_dir / "artifact_validation.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": report["passed"], "full_campaign_completed": report["full_campaign_completed"],
        "completed_conditions_checked": report["completed_conditions_checked"], "expected_conditions": report["expected_conditions"],
        "errors": len(report["errors"]), "missing": len(report["missing"]), "report": str(target.resolve())}, indent=2))
    return 0 if report["passed"] else 2 if report["status"] == "incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
