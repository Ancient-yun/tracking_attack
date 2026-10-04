"""CPU checks of the shared frozen roster, its GT eligibility, and experiment scope."""
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import struct
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docker/manifests/loss_components_14clips.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_shared_manifest_preserves_identities_and_frozen_source_hashes():
    manifest = read_json(MANIFEST_PATH)
    details = manifest["selection_details"]
    assert manifest["num_frames"] == 64
    assert manifest["seed"] == 20261002
    assert manifest["paper_sequence_identity_verified"] is False
    assert details["same_roster_for_all_objectives"] is True
    for source, expected_sha in details["source_manifests"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected_sha
    full = read_json(ROOT / "docker/manifests/synthetic_full.json")
    previous = read_json(ROOT / "docker/manifests/pgd_4clips.json")
    identities = [(e["dataset"], e["sequence"]) for e in manifest["entries"]]
    assert len(set(identities)) == len(identities) == 14
    assert Counter(ds for ds, _ in identities) == {"po_mini": 7, "ds_mini": 7}
    frozen = {(e["dataset"], e["sequence"]): e for e in full["entries"]}
    for entry in manifest["entries"]:
        identity = entry["dataset"], entry["sequence"]
        assert entry == frozen[identity]
        path = PurePosixPath(entry["path"])
        assert not path.is_absolute() and ".." not in path.parts
        assert "\\" not in entry["path"] and ":" not in entry["path"]
        assert path.as_posix() == f"data/worldtrack_release/{entry['dataset']}/{entry['sequence']}.npz"
    assert set((e["dataset"], e["sequence"]) for e in previous["entries"]).issubset(identities)


def test_roster_reproduces_the_recorded_filter_and_seeded_draw():
    manifest = read_json(MANIFEST_PATH)
    details = manifest["selection_details"]
    full = read_json(ROOT / "docker/manifests/synthetic_full.json")
    previous = read_json(ROOT / "docker/manifests/pgd_4clips.json")
    rng = random.Random(manifest["seed"])
    expected = []
    for dataset in manifest["datasets"]:
        source = [e for e in full["entries"] if e["dataset"] == dataset]
        eligible_ids = details["eligible_sequences_in_source_order"][dataset]
        excluded = details["excluded"][dataset]
        excluded_ids = {e["sequence"] for e in excluded}
        assert len(source) == manifest["available_sequences"][dataset] == 50
        assert len(eligible_ids) == details["eligible_counts"][dataset] == 48
        assert [e["sequence"] for e in source if e["sequence"] not in excluded_ids] == eligible_ids
        assert len(excluded_ids) == 2
        assert all(e["first64_gt_finite"] and e["dynamic_queries"] == 0 for e in excluded)
        preserved = [e for e in previous["entries"] if e["dataset"] == dataset]
        preserved_ids = {e["sequence"] for e in preserved}
        candidates = [e for e in source if e["sequence"] in eligible_ids and e["sequence"] not in preserved_ids]
        additions = rng.sample(candidates, 6)
        assert [e["sequence"] for e in additions] == details["sampled_six_additions"][dataset]
        expected.extend(preserved + additions[:5])
    assert manifest["entries"] == expected


def test_config_matches_all_five_objectives_and_full_frame_pgd_scope():
    config = read_json(ROOT / "configs/loss_components.json")
    assert config == {
        "checkpoint": "assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth",
        "num_frames": 64, "image_size": 512, "device": "cuda",
        "objectives": ["tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training"],
        "epsilon_255": 4, "steps": 20, "step_size_fraction": 0.25,
        "restarts": 1, "seed": 20261002, "alpha": 0.2, "reweight_scale": 5,
        "gradient_mode": "recompute", "save_inputs": True,
        "verify_saved_inputs": True, "verify_official_clean": True,
    }
    manifest = read_json(MANIFEST_PATH)
    assert manifest["num_frames"] == config["num_frames"]
    estimate = manifest["runtime_estimate"]
    for clips in (14, 16):
        expected = estimate["fixed_startup_finalization_shutdown_seconds"] + clips * estimate["per_clip_wall_seconds"]
        assert estimate[f"estimated_{clips}clips_seconds"] == pytest.approx(expected)
    assert estimate["estimated_5_objectives_seconds"] == pytest.approx(5 * estimate["estimated_14clips_seconds"])


def test_recorded_selected_gt_and_depth_support_every_objective():
    manifest = read_json(MANIFEST_PATH)
    validation = manifest["selection_details"]["selected_validation"]
    assert [(v["dataset"], v["sequence"]) for v in validation] == [
        (e["dataset"], e["sequence"]) for e in manifest["entries"]
    ]
    for record in validation:
        assert record["first64_gt_finite"] is True
        assert record["dynamic_queries"] > 0 and record["static_queries"] > 0
        assert record["num_queries"] == record["dynamic_queries"] + record["static_queries"]
        assert record["preprocessed_wh"] == [512, 288]
        depth = record["depth_first64"]
        assert depth["shape"][0] == 64 and depth["finite"] is True
        counts = depth["positive_counts_per_frame"]
        assert len(counts) == 64 and 0 < min(counts) <= max(counts) <= depth["pixels_per_frame"]
        assert depth["minimum_positive_count"] == min(counts)
        assert depth["positive_count_total"] == sum(counts)


def test_selected_available_assets_match_their_frozen_sha256_and_size():
    manifest = read_json(MANIFEST_PATH)
    records = {(v["dataset"], v["sequence"]): v for v in manifest["selection_details"]["selected_validation"]}
    paths = [ROOT / e["path"] for e in manifest["entries"]]
    if not all(path.is_file() for path in paths):
        pytest.skip("Official data assets are not available in this CPU test environment")
    for entry, path in zip(manifest["entries"], paths):
        assert path.stat().st_size == entry["bytes"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == entry["sha256"]
        depth_record = records[entry["dataset"], entry["sequence"]]["depth_first64"]
        with zipfile.ZipFile(path) as archive, archive.open("depth_map.npy") as stream:
            assert stream.read(6) == b"\x93NUMPY"
            major = stream.read(2)[0]
            length = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
            header = ast.literal_eval(stream.read(length).decode("latin1"))
            assert not header["fortran_order"] and header["shape"][0] == 128
            assert [64, *header["shape"][1:]] == depth_record["shape"]
            dtype = np.dtype(header["descr"])
            frame_pixels = int(np.prod(header["shape"][1:]))
            counts = []
            for _ in range(64):
                depth = np.frombuffer(stream.read(frame_pixels * dtype.itemsize), dtype=dtype)
                assert depth.size == frame_pixels and np.isfinite(depth).all()
                counts.append(int((depth > 0).sum()))
            assert counts == depth_record["positive_counts_per_frame"]


def test_available_gt_reproduces_all_candidate_filter_and_selected_query_counts():
    manifest = read_json(MANIFEST_PATH)
    full = read_json(ROOT / "docker/manifests/synthetic_full.json")
    if not all((ROOT / e["path"]).is_file() for e in full["entries"]):
        pytest.skip("All official candidate assets are not available in this CPU test environment")
    expected_records = {(v["dataset"], v["sequence"]): v for v in manifest["selection_details"]["selected_validation"]}
    eligible = {dataset: [] for dataset in manifest["datasets"]}
    for entry in full["entries"]:
        with np.load(ROOT / entry["path"], allow_pickle=False) as data:
            cam = data["tracks_XYZ"][:64]
            visibility = data["visibility"][:64]
            intrinsics = data["fx_fy_cx_cy"]
            extrinsics = data["extrinsics_w2c"][:64].copy()
        width, height = (960, 540) if entry["dataset"] == "po_mini" else (1280, 720)
        uv = cam[0, :, :2] / (cam[0, :, 2:3] + 1e-8) * intrinsics[:2] + intrinsics[2:]
        xy = uv * np.array([512, 288]) / np.array([width, height])
        indices = np.flatnonzero(visibility[0] & (xy[:, 0] >= 0) & (xy[:, 0] < 512)
                                 & (xy[:, 1] >= 0) & (xy[:, 1] < 288))
        first_inverse = np.linalg.inv(extrinsics[0])
        for frame in range(64):
            extrinsics[frame] = extrinsics[frame] @ first_inverse
        camera_to_world = np.linalg.inv(extrinsics)
        world = np.zeros_like(cam)
        for frame in range(64):
            world[frame] = (camera_to_world[frame, :3, :3] @ cam[frame].T).T + camera_to_world[frame, :3, 3]
        assert all(np.isfinite(array).all() for array in (cam, world, intrinsics, extrinsics))
        selected = world[:, indices]
        dynamic = np.linalg.norm(np.diff(selected, axis=0), axis=-1).sum(axis=0) > 0.01
        dynamic_count, static_count = int(dynamic.sum()), int((~dynamic).sum())
        if dynamic_count and static_count:
            eligible[entry["dataset"]].append(entry["sequence"])
        record = expected_records.get((entry["dataset"], entry["sequence"]))
        if record:
            assert record["raw_track_count"] == cam.shape[1]
            assert record["num_queries"] == len(indices)
            assert record["dynamic_queries"] == dynamic_count
            assert record["static_queries"] == static_count
    assert eligible == manifest["selection_details"]["eligible_sequences_in_source_order"]
