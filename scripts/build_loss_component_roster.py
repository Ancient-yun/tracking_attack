"""Extend the frozen 14-clip loss roster with balanced deterministic GT-only clips.

No model, Torch, CUDA or Docker is imported. The original 14 manifest/config
are never rewritten. An explicit even --clips N creates new named files;
--dry-run performs the same selected-file/depth checks without writing them.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import struct
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("po_mini", "ds_mini")
SEED = 20261002
FRAMES = 64


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_asset(root, entry):
    expected = f"data/worldtrack_release/{entry['dataset']}/{entry['sequence']}.npz"
    path = PurePosixPath(entry["path"])
    if (entry["dataset"] not in DATASETS or path.is_absolute() or ".." in path.parts
            or "\\" in entry["path"] or ":" in entry["path"] or path.as_posix() != expected):
        raise ValueError(f"Unsafe/noncanonical frozen data path: {entry}")
    return root / entry["path"]


def npy_header(stream):
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Bad NPY magic in depth_map")
    major = stream.read(2)[0]
    if major not in (1, 2, 3):
        raise ValueError("Unsupported NPY version")
    size = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(stream.read(size).decode("utf-8" if major == 3 else "latin1"))
    if header["fortran_order"]:
        raise ValueError("Expected released C-order depth maps")
    return header


def gt_validation(root, entry, frames=FRAMES):
    if frames not in (64, 128):
        raise ValueError("Only explicit64 or released-all128 frame scopes are supported")
    path = local_asset(root, entry)
    with zipfile.ZipFile(path) as archive, archive.open("depth_map.npy") as stream:
        header = npy_header(stream)
        with archive.open("images_jpeg_bytes.npy") as image_stream:
            image_header = npy_header(image_stream)
    shape = header["shape"]
    if shape != ((128, 540, 960) if entry["dataset"] == "po_mini" else (128, 720, 1280)):
        raise ValueError(f"Released frame/resolution contract changed: {path}: {shape}")
    if image_header["shape"] != (128,):
        raise ValueError(f"Released raw image frame count changed: {path}: {image_header['shape']}")
    with np.load(path, allow_pickle=False) as data:
        raw_cam = data["tracks_XYZ"]
        raw_visibility = data["visibility"]
        intrinsics = data["fx_fy_cx_cy"]
        raw_extrinsics = data["extrinsics_w2c"]
    if (raw_cam.ndim != 3 or raw_cam.shape[0] != 128 or raw_cam.shape[-1] != 3
            or raw_visibility.shape != raw_cam.shape[:2] or raw_extrinsics.shape != (128, 4, 4)):
        raise ValueError(f"Released128-frame GT contract changed: {path}")
    cam, visibility, extrinsics = raw_cam[:frames], raw_visibility[:frames], raw_extrinsics[:frames].copy()
    if (cam.shape[0] != frames or visibility.shape != cam.shape[:2]
            or intrinsics.shape != (4,) or extrinsics.shape != (frames, 4, 4)):
        raise ValueError(f"GT shapes changed: {path}")
    if not all(np.isfinite(value).all() for value in (cam, visibility, intrinsics, extrinsics)):
        raise ValueError(f"Nonfinite candidate GT: {path}")
    original_wh = np.array([shape[2], shape[1]])
    xy = (cam[0, :, :2] / (cam[0, :, 2:3] + 1e-8) * intrinsics[:2] + intrinsics[2:])
    xy = xy * np.array([512, 288]) / original_wh
    selected = np.flatnonzero(visibility[0].astype(bool) & (xy[:, 0] >= 0) & (xy[:, 0] < 512)
                             & (xy[:, 1] >= 0) & (xy[:, 1] < 288))
    first_inverse = np.linalg.inv(extrinsics[0])
    for frame in range(frames):
        extrinsics[frame] = extrinsics[frame] @ first_inverse
    camera_to_world = np.linalg.inv(extrinsics)
    world = np.zeros_like(cam)
    for frame in range(frames):
        world[frame] = (camera_to_world[frame, :3, :3] @ cam[frame].T).T + camera_to_world[frame, :3, 3]
    if not np.isfinite(world).all():
        raise ValueError(f"Nonfinite normalized world GT: {path}")
    dynamic = np.linalg.norm(np.diff(world[:, selected], axis=0), axis=-1).sum(axis=0) > .01
    public = {"raw_track_count": int(cam.shape[1]), "num_queries": len(selected),
        "dynamic_queries": int(dynamic.sum()), "static_queries": int((~dynamic).sum()),
        f"first{frames}_gt_finite": True, "validated_frames": frames, "original_frame_count": 128,
        "raw_image_frame_count": image_header["shape"][0],
        "visibility_finite": True, "initial_query_coordinates_finite": bool(np.isfinite(xy).all()),
        "original_wh": original_wh.tolist(), "preprocessed_wh": [512, 288]}
    if not public["initial_query_coordinates_finite"]:
        raise ValueError(f"Nonfinite query coordinates: {path}")
    # Dense native targets use the upstream adapter's batched float32 transform.
    with np.load(path, allow_pickle=False) as data:
        dense_extrinsics = np.array(data["extrinsics_w2c"][:frames], dtype=np.float32, copy=True)
    dense_extrinsics = dense_extrinsics @ np.linalg.inv(dense_extrinsics[0])
    return public, intrinsics.astype(np.float64), np.linalg.inv(dense_extrinsics)


def extend_entries(full_entries, previous_four, base14, eligible_ids, clips, seed=SEED):
    """One frozen whole-roster prefix for every N, including the original 14."""
    if not isinstance(clips, int) or isinstance(clips, bool) or clips < 16 or clips % 2:
        raise ValueError("clips must be an even integer >=16; the original14 is immutable")
    rng = random.Random(seed)
    ordered, original_draws = {}, {}
    for dataset in DATASETS:
        preserved = [entry for entry in previous_four if entry["dataset"] == dataset]
        if len(preserved) != 2:
            raise ValueError("Expected the two original measured clips in each dataset")
        used = {entry["sequence"] for entry in preserved}
        candidates = [entry for entry in full_entries if entry["dataset"] == dataset
                      and entry["sequence"] in eligible_ids[dataset] and entry["sequence"] not in used]
        drawn = rng.sample(candidates, 6)
        original_draws[dataset] = [entry["sequence"] for entry in drawn]
        if [entry for entry in base14 if entry["dataset"] == dataset] != preserved + drawn[:5]:
            raise ValueError("The existing14 no longer reproduces the original frozen draws")
        ordered[dataset] = preserved + drawn
    # Crucially, finish both original six-item draws before continuing either
    # dataset. Draw whole remaining permutations so N cannot change RNG use.
    continuation = {}
    for dataset in DATASETS:
        used = {entry["sequence"] for entry in ordered[dataset]}
        remaining = [entry for entry in full_entries if entry["dataset"] == dataset
                     and entry["sequence"] in eligible_ids[dataset] and entry["sequence"] not in used]
        drawn = rng.sample(remaining, len(remaining))
        continuation[dataset] = [entry["sequence"] for entry in drawn]
        ordered[dataset].extend(drawn)
    if clips // 2 > min(len(ordered[dataset]) for dataset in DATASETS):
        raise ValueError("Requested balanced roster exceeds the frozen eligible candidates")
    result = list(base14)
    for index in range(7, clips // 2):
        result.extend(ordered[dataset][index] for dataset in DATASETS)
    if result[:14] != base14 or Counter(entry["dataset"] for entry in result) != {dataset: clips // 2 for dataset in DATASETS}:
        raise AssertionError("Prefix or balanced-roster invariant failed")
    return result, original_draws, continuation


def depth_validation(path, intrinsics, camera_to_world, frames=FRAMES):
    if frames not in (64, 128):
        raise ValueError("Only explicit64 or released-all128 depth scopes are supported")
    raw_counts, model_counts = [], []
    with zipfile.ZipFile(path) as archive, archive.open("depth_map.npy") as stream:
        header = npy_header(stream)
        raw_frames, oh, ow = header["shape"]
        dtype = np.dtype(header["descr"])
        if raw_frames != 128 or dtype != np.float32 or camera_to_world.shape != (frames, 4, 4):
            raise ValueError("Released depth frame/dtype contract changed")
        source_y = np.floor(np.arange(288) * oh / 288).astype(np.int64)
        source_x = np.floor(np.arange(512) * ow / 512).astype(np.int64)
        u, v = np.meshgrid(np.arange(512), np.arange(288))
        fx, fy, cx, cy = intrinsics * np.array([512/ow, 288/oh, 512/ow, 288/oh])
        for frame in range(frames):
            raw = np.frombuffer(stream.read(oh * ow * dtype.itemsize), dtype=dtype)
            if raw.size != oh * ow or not np.isfinite(raw).all():
                raise ValueError(f"Truncated/nonfinite depth at frame{frame}: {path}")
            raw = raw.reshape(oh, ow)
            raw_counts.append(int((raw > 0).sum()))
            depth = raw[source_y[:, None], source_x[None, :]]
            camera = np.stack(((u-cx)*depth/fx, (v-cy)*depth/fy, depth), axis=-1).astype(np.float32)
            transform = camera_to_world[frame]
            world = np.einsum("ij,hwj->hwi", transform[:3, :3], camera) + transform[None, None, :3, 3]
            if not np.isfinite(world).all():
                raise ValueError(f"Nonfinite model-grid world GT at frame{frame}: {path}")
            model_counts.append(int(((depth > 0) & np.isfinite(world).all(-1)).sum()))
    if min(raw_counts) <= 0 or min(model_counts) <= 0:
        raise ValueError(f"A selected frame has empty depth/mask membership: {path}")
    return {"shape": [frames, oh, ow], "original_frame_count": raw_frames, "dtype": "float32", "finite": True,
        "valid_rule": "isfinite(depth_map) & (depth_map > 0)", "positive_counts_per_frame": raw_counts,
        "minimum_positive_count": min(raw_counts), "positive_count_total": sum(raw_counts), "pixels_per_frame": oh*ow,
        "model_grid_wh": [512, 288], "model_grid_depth_resize": "nearest source pixel: floor(output_index * original_size / output_size)",
        "model_grid_gt_finite": True, "model_grid_valid_rule": "depth > 0 and finite normalized world point",
        "model_grid_valid_counts_per_frame": model_counts, "model_grid_minimum_valid_count": min(model_counts)}


def campaign_config(root, runtime_optimization=False):
    """An explicit speed switch cannot change the experiment or roster scope."""
    config_path = root / "configs/loss_components.json"
    original = read_json(config_path)
    source = {config_path.relative_to(root).as_posix(): file_sha256(config_path)}
    if not runtime_optimization:
        return original, source
    optimized_path = root / "configs/loss_components_optimized.json"
    optimized = read_json(optimized_path)
    if original.get("runtime_optimization", False) is not False:
        raise ValueError("Frozen original config unexpectedly enables runtime optimization")
    if (optimized.get("runtime_optimization") is not True or
            {key: value for key, value in optimized.items() if key != "runtime_optimization"} !=
            {key: value for key, value in original.items() if key != "runtime_optimization"}):
        raise ValueError("Optimized config must differ only by runtime_optimization=true")
    source[optimized_path.relative_to(root).as_posix()] = file_sha256(optimized_path)
    return optimized, source


def all_frame_entries(full_entries, previous_four, base14, eligible_ids, clips, seed=SEED):
    """Keep eligible old per-dataset prefix slots; replace ineligible slots by GT-only draws."""
    if not isinstance(clips, int) or isinstance(clips, bool) or clips < 4 or clips % 2:
        raise ValueError("All-frame clips must be an even integer >=4")
    rng = random.Random(seed)
    ordering, fallbacks, missing_old, replacements = {}, {}, {}, {}
    for dataset in DATASETS:
        preserved = [entry for entry in previous_four if entry["dataset"] == dataset]
        preferred = [entry for entry in base14 if entry["dataset"] == dataset]
        if len(preserved) != 2 or len(preferred) != 7 or preferred[:2] != preserved:
            raise ValueError("Original14 per-dataset prefix must begin with the original two")
        eligible = set(eligible_ids[dataset])
        preferred_ids = {entry["sequence"] for entry in preferred}
        remaining = [entry for entry in full_entries if entry["dataset"] == dataset
                     and entry["sequence"] in eligible and entry["sequence"] not in preferred_ids]
        drawn = rng.sample(remaining, len(remaining))
        fallbacks[dataset] = [entry["sequence"] for entry in drawn]
        missing_old[dataset] = [entry["sequence"] for entry in preferred if entry["sequence"] not in eligible]
        replacements[dataset], ordering[dataset], next_fallback = [], [], 0
        for entry in preferred:
            if entry["sequence"] in eligible:
                ordering[dataset].append(entry)
            elif next_fallback < len(drawn):
                replacement = drawn[next_fallback]
                next_fallback += 1
                ordering[dataset].append(replacement)
                replacements[dataset].append({"ineligible_old_sequence": entry["sequence"],
                                              "replacement_sequence": replacement["sequence"]})
        ordering[dataset].extend(drawn[next_fallback:])
    if clips // 2 > min(len(ordering[dataset]) for dataset in DATASETS):
        raise ValueError("Requested balanced roster exceeds full128-frame eligible candidates")
    entries = [ordering[dataset][index] for index in range(clips//2) for dataset in DATASETS]
    if len({(entry["dataset"], entry["sequence"]) for entry in entries}) != clips:
        raise AssertionError("All-frame roster duplicated a clip")
    selected = {(entry["dataset"], entry["sequence"]) for entry in entries}
    if any(entry["sequence"] in eligible_ids[entry["dataset"]] and
           (entry["dataset"], entry["sequence"]) not in selected for entry in previous_four):
        raise AssertionError("An eligible original clip was not preserved")
    return entries, {"ineligible_original14_ids": missing_old, "replaced_original14_slots": replacements,
        "full_seeded_fallback_permutation": fallbacks,
        "ineligible_original_four_ids": [f"{entry['dataset']}/{entry['sequence']}" for entry in previous_four
                                        if entry["sequence"] not in eligible_ids[entry["dataset"]]],
        "original_four_preserved_if_eligible": True,
        "original14_per_dataset_prefix_preserved_where_eligible": True,
        "original14_preserved_as_literal_prefix": False,
        "sampling_algorithm": "rng=Random(20261002); independently recheck all128 GT; preserve old14 per-dataset prefix slots that remain eligible; draw the full PO then DR permutation of eligible candidates outside old14; replace each ineligible old slot with the next sampled candidate; append unused draws; interleave the first N/2 PO and DR slots. No metrics or timings influence selection.",
        "nested_roster_rule": "All all-frame even-N rosters use the same pair-interleaved deterministic literal prefix"}


def build_all_frames(root, clips, runtime_optimization=False):
    root = Path(root).resolve()
    paths = [root / name for name in ("docker/manifests/synthetic_full.json", "docker/manifests/pgd_4clips.json",
                                     "docker/manifests/loss_components_14clips.json")]
    full, four, base = (read_json(path) for path in paths)
    config, config_source = campaign_config(root, runtime_optimization)
    if config["seed"] != SEED or base["seed"] != SEED:
        raise ValueError("Original deterministic seed changed")
    if (len(full["entries"]) != 100 or Counter(e["dataset"] for e in full["entries"]) != {d: 50 for d in DATASETS}
            or len({(e["dataset"], e["sequence"]) for e in full["entries"]}) != 100):
        raise ValueError("Expected the frozen unique official50+50 candidate inventory")
    checks = {(entry["dataset"], entry["sequence"]): gt_validation(root, entry, 128) for entry in full["entries"]}
    eligible = {dataset: [entry["sequence"] for entry in full["entries"] if entry["dataset"] == dataset
                         and checks[dataset, entry["sequence"]][0]["dynamic_queries"] > 0
                         and checks[dataset, entry["sequence"]][0]["static_queries"] > 0] for dataset in DATASETS}
    entries, sampling = all_frame_entries(full["entries"], four["entries"], base["entries"], eligible, clips)
    source = {(entry["dataset"], entry["sequence"]): entry for entry in full["entries"]}
    validation = []
    for entry in entries:
        if source[entry["dataset"], entry["sequence"]] != entry:
            raise ValueError("Original clip metadata differs from full inventory")
        path = local_asset(root, entry)
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Selected NPZ size/SHA256 mismatch: {path}")
        record, intrinsics, camera_to_world = checks[entry["dataset"], entry["sequence"]]
        depth = depth_validation(path, intrinsics, camera_to_world, 128)
        validation.append({"dataset": entry["dataset"], "sequence": entry["sequence"], **record, "depth_all_frames": depth})
    excluded = [{"dataset": entry["dataset"], "sequence": entry["sequence"],
                 "reason": "dynamic_queries==0 or static_queries==0 over all128-frame normalized GT",
                 **checks[entry["dataset"], entry["sequence"]][0]}
                for entry in full["entries"] if entry["sequence"] not in eligible[entry["dataset"]]]
    details = {"purpose": "Matched balanced all128-frame development subset for all five objectives; not a verified paper roster",
        "same_roster_for_all_objectives": True, "candidate_order": "Frozen synthetic_full.json order within each dataset",
        "source_manifests": {path.relative_to(root).as_posix(): file_sha256(path) for path in paths},
        "source_config": config_source, "filter_rule": "All128 tracks/visibility/poses/intrinsics/query coordinates finite; initial visible in-bounds query membership; dynamic_queries>0 and static_queries>0 over all128; selected every raw128 depth frame and model-grid mask finite with positive support",
        "query_rule": base["selection_details"]["query_rule"], "coordinate_rule": base["selection_details"]["coordinate_rule"],
        "dynamic_rule": "sum_t=0..126 ||normalized_world_GT[t+1]-normalized_world_GT[t]||_2 > 0.01 metres",
        "eligibility_inputs": "Only full128-frame released GT/query geometry;64-frame eligibility is never reused; no attack outcomes, clean metrics or runtime measurements",
        "candidate_GT_checked": 100, "candidate_all128_GT_finite": 100,
        "candidate_validation": [{"dataset": entry["dataset"], "sequence": entry["sequence"],
                                  **checks[entry["dataset"], entry["sequence"]][0]} for entry in full["entries"]],
        "eligible_counts": {dataset: len(eligible[dataset]) for dataset in DATASETS},
        "eligible_sequences_in_source_order": eligible, "excluded": excluded,
        "selected_counts": {dataset: clips//2 for dataset in DATASETS}, "python_version": sys.version.split()[0],
        "file_bytes_and_sha256_preserved": True, "selected_file_sha256_verified": True, "path_mode": "project_relative_posix",
        "selected_validation": validation, "validation_implementation": "CPU NumPy official coordinates; stream every raw128 depth frame and nearest model-grid depth/world masks; no model inference", **sampling}
    common = {"num_frames": 128, "campaign_expected_frames": 128, "all_frames": True, "campaign_expected_clips": clips}
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": "data/worldtrack_release", "datasets": list(DATASETS), "seed": SEED, **common,
        "selection": "all128_GT_revalidated_old_per_dataset_prefix_with_seeded_fallback",
        "paper_sequence_identity_verified": False, "sequence_list": None,
        "available_sequences": {dataset: 50 for dataset in DATASETS}, "entries": entries, "selection_details": details,
        "runtime_planning_note": "N is explicitly selected after separate full128 timing calibration; this GT-only generator guarantees no per-objective runtime"}
    return manifest, {**config, **common}


def build(root, clips, runtime_optimization=False, frames=FRAMES):
    if frames == 128:
        return build_all_frames(root, clips, runtime_optimization)
    if frames != FRAMES:
        raise ValueError("frames must be64 or128")
    root = Path(root).resolve()
    full_path = root / "docker/manifests/synthetic_full.json"
    four_path = root / "docker/manifests/pgd_4clips.json"
    base_path = root / "docker/manifests/loss_components_14clips.json"
    full, four, base = (read_json(path) for path in (full_path, four_path, base_path))
    config, config_source = campaign_config(root, runtime_optimization)
    if base["seed"] != SEED or base["num_frames"] != FRAMES or config["seed"] != SEED or config["num_frames"] != FRAMES:
        raise ValueError("Original seed/frame contract changed")
    if len(full["entries"]) != 100 or Counter(e["dataset"] for e in full["entries"]) != {dataset: 50 for dataset in DATASETS}:
        raise ValueError("Expected the frozen official50+50 candidate inventory")
    if len({(e["dataset"], e["sequence"]) for e in full["entries"]}) != 100:
        raise ValueError("Duplicate full-candidate identity")
    checks = {(entry["dataset"], entry["sequence"]): gt_validation(root, entry) for entry in full["entries"]}
    eligible = {dataset: [e["sequence"] for e in full["entries"] if e["dataset"] == dataset
                         and checks[dataset, e["sequence"]][0]["dynamic_queries"] > 0
                         and checks[dataset, e["sequence"]][0]["static_queries"] > 0] for dataset in DATASETS}
    if eligible != base["selection_details"]["eligible_sequences_in_source_order"]:
        raise ValueError("GT eligibility differs from the frozen14 source population")
    entries, drawn, continuation = extend_entries(full["entries"], four["entries"], base["entries"], eligible, clips)
    if drawn != base["selection_details"]["sampled_six_additions"]:
        raise ValueError("Original six-item seeded draws changed")
    full_index = {(entry["dataset"], entry["sequence"]): entry for entry in full["entries"]}
    validation = []
    for entry in entries:
        if full_index[entry["dataset"], entry["sequence"]] != entry:
            raise ValueError("Preserved clip source metadata differs from full inventory")
        path = local_asset(root, entry)
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Selected NPZ size/SHA256 mismatch: {path}")
        record, intrinsics, camera_to_world = checks[entry["dataset"], entry["sequence"]]
        depth = depth_validation(path, intrinsics, camera_to_world)
        validation.append({"dataset": entry["dataset"], "sequence": entry["sequence"], **record, "depth_first64": depth})
    details = {"purpose": "Matched balanced development subset for all five loss objectives; not a verified paper roster",
        "same_roster_for_all_objectives": True, "candidate_order": "Existing order within each dataset in docker/manifests/synthetic_full.json",
        "source_manifests": {p.relative_to(root).as_posix(): file_sha256(p) for p in (full_path, four_path, base_path)},
        "source_config": config_source,
        "filter_rule": base["selection_details"]["filter_rule"], "query_rule": base["selection_details"]["query_rule"],
        "coordinate_rule": base["selection_details"]["coordinate_rule"], "dynamic_rule": base["selection_details"]["dynamic_rule"],
        "eligibility_inputs": "Only first64-frame released GT/query geometry; no attack outcomes, clean metrics or runtime measurements",
        "candidate_GT_checked": 100, "candidate_first64_GT_finite": 100,
        "eligible_counts": {dataset: len(eligible[dataset]) for dataset in DATASETS},
        "eligible_sequences_in_source_order": eligible, "excluded": base["selection_details"]["excluded"],
        "sampling_algorithm": "rng=Random(20261002); reproduce original PO sample6 then DR sample6 excluding preserved2 each; from this resulting RNG state draw full remaining PO permutation then full remaining DR permutation; keep original14 entries as literal prefix and append one PO/DR pair at a time using each dataset's next unused element",
        "sampled_six_additions": drawn, "full_continuation_permutation": continuation,
        "original14_preserved_as_literal_prefix": True, "original_four_preserved": True,
        "nested_roster_rule": "Every smaller supported even-N roster is the literal entry prefix of every larger roster",
        "selected_counts": {dataset: clips//2 for dataset in DATASETS}, "python_version": sys.version.split()[0],
        "file_bytes_and_sha256_preserved": True, "selected_file_sha256_verified": True,
        "path_mode": "project_relative_posix", "selected_validation": validation,
        "validation_implementation": "CPU NumPy official tracking coordinates; stream first64 raw depth frames and validate nearest model-grid depth/world masks; no model inference"}
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": "data/worldtrack_release", "datasets": list(DATASETS), "num_frames": FRAMES, "seed": SEED,
        "campaign_expected_clips": clips, "selection": "frozen14_prefix_plus_seeded_GT_only_balanced_continuation",
        "paper_sequence_identity_verified": False, "sequence_list": None,
        "available_sequences": {dataset: 50 for dataset in DATASETS}, "entries": entries,
        "selection_details": details, "runtime_planning_note": "Roster size is explicitly selected after separate runtime calibration; no per-objective runtime guarantee is inferred by this GT-only generator"}
    config = {**config, "campaign_expected_clips": clips}
    return manifest, config


def save_pair(root, clips, manifest, config):
    suffix = "_allframes" if manifest.get("all_frames") is True else ""
    outputs = [(Path(root) / f"docker/manifests/loss_components_{clips}clips{suffix}.json", manifest),
               (Path(root) / f"configs/loss_components_{clips}clips{suffix}.json", config)]
    for path, payload in outputs:
        if path.exists():
            previous = read_json(path)
            if "created_at_utc" in payload:
                payload["created_at_utc"] = previous.get("created_at_utc", payload["created_at_utc"])
            if payload != previous:
                raise FileExistsError(f"Existing frozen output differs; refusing overwrite: {path}")
    for path, payload in outputs:
        if not path.exists():
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)
    return [str(path.resolve()) for path, _ in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, type=int, help="Explicit even balanced roster: >=16 for historical64, >=4 for full128")
    parser.add_argument("--frames", type=int, choices=(64, 128), default=64,
                        help="Explicit128 uses every raw NPZ frame and separate _allframes outputs")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--runtime-optimization", action="store_true",
                        help="Use the validated optimized config; its only allowed change is runtime_optimization=true")
    parser.add_argument("--dry-run", action="store_true", help="Validate everything but write no output")
    args = parser.parse_args()
    if args.clips < (4 if args.frames == 128 else 16) or args.clips % 2:
        parser.error("clips must be even and >=4 for128 or >=16 for historical64")
    manifest, config = build(args.project_root, args.clips, args.runtime_optimization, args.frames)
    outputs = [] if args.dry_run else save_pair(args.project_root, args.clips, manifest, config)
    print(json.dumps({"dry_run": args.dry_run, "campaign_expected_clips": args.clips,
        "num_frames": args.frames, "all_frames": manifest.get("all_frames", False),
        "runtime_optimization": config.get("runtime_optimization", False),
        "selected_counts": manifest["selection_details"]["selected_counts"], "GT_candidates_checked": 100,
        "eligible_counts": manifest["selection_details"]["eligible_counts"],
        "excluded_ids": [f"{entry['dataset']}/{entry['sequence']}" for entry in manifest["selection_details"]["excluded"]],
        "selected_SHA256_verified": len(manifest["entries"]), "selected_depth_frames_checked": args.clips * args.frames,
        "original14_preserved_as_literal_prefix": manifest["selection_details"]["original14_preserved_as_literal_prefix"],
        "ineligible_original14_ids": manifest["selection_details"].get("ineligible_original14_ids", {}),
        "ineligible_original_four_ids": manifest["selection_details"].get("ineligible_original_four_ids", []),
        "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
