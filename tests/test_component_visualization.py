"""Meaningful synthetic CPU checks for current-schema visualization.

The source fixture has 128 times, a moving camera, later occlusion, fixed GT
support and independent objectives. No real campaign or GPU is accessed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import component_projection as projection
import render_component_comparisons as renderer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_run(tmp_path):
    run, project = tmp_path / "run", tmp_path / "project"
    dataset, sequence = "po_mini", "synthetic_moving_camera"
    t, h, w, q = 128, 4, 8, 4
    intrinsics = np.array([4., 4., 4., 2.], np.float64)
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (t, 1, 1))
    extrinsics[:, 0, 3] = np.linspace(0, .1, t, dtype=np.float32)
    world = np.tile(np.array([[-.6, -.2, 2.], [-.2, 0, 2.], [.2, .2, 2.], [.6, .1, 2.]], np.float64), (t, 1, 1))
    world[:, 0, 1] += np.linspace(0, .05, t)
    camera = np.einsum("tij,tnj->tni", extrinsics[:, :3, :3], world) + extrinsics[:, None, :3, 3]
    visibility = np.ones((t, q), bool)
    visibility[80:, 0] = False  # The evaluator must still retain this query.
    source = project / "data" / "worldtrack_release" / dataset / f"{sequence}.npz"
    source.parent.mkdir(parents=True)
    np.savez(source, tracks_XYZ=camera, visibility=visibility, fx_fy_cx_cy=intrinsics, extrinsics_w2c=extrinsics)
    # Mirror the adapter's actual storage rounding and inverse convention.
    inv = np.linalg.inv(extrinsics)
    recovered = np.zeros_like(camera)
    for frame in range(t):
        recovered[frame] = (inv[frame, :3, :3] @ camera[frame].T).T + inv[frame, :3, 3]
    gt = recovered.astype(np.float32)
    query_xy = projection._camera_to_xy(camera, intrinsics, np.ones(2))[0]
    dynamic = np.linalg.norm(np.diff(recovered, axis=0), axis=-1).sum(axis=0) > .01
    sequence_dir = run / "sequences" / dataset / sequence
    sequence_dir.mkdir(parents=True)
    dense_gt = np.tile(np.array([0, 0, 2.], np.float32), (t, h, w, 1))
    dense_valid = np.ones((t, h, w), bool)
    dense_valid[::2, 0, :] = False  # Nonuniform counts expose mean-of-means mistakes.
    np.save(sequence_dir / "reconstruction_gt.npy", dense_gt)
    np.save(sequence_dir / "reconstruction_valid.npy", dense_valid)
    np.savez(sequence_dir / "targets.npz", reconstruction_gt_norm=np.full(t, 2 / 3, np.float32))
    signature = "synthetic-only"
    metadata = {"sequence_name": sequence, "num_frames": t, "original_frame_count": t,
                "all_frames_used": True, "used_frame_indices": list(range(t)),
                "source_frame_counts": {"tracks_XYZ": t, "visibility": t, "depth_map": t, "extrinsics_w2c": t, "images_jpeg_bytes": t},
                "original_wh": [w, h], "preprocessed_wh": [w, h], "intrinsics": intrinsics.tolist(), "selected_point_indices": list(range(q))}
    write_json(sequence_dir / "sequence.json", metadata)
    names = ("sequence.json", "targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy")
    write_json(sequence_dir / "sequence_manifest.json", {"signature": signature, "artifact_sha256": {name: projection.sha256(sequence_dir / name) for name in names}})
    write_json(run / "run.json", {"signature": signature, "manifest": {"entries": [{"dataset": dataset, "sequence": sequence,
               "sha256": projection.sha256(source), "bytes": source.stat().st_size}]}, "config": {"num_frames": 128}})
    for index, objective in enumerate(("clean", *projection.OBJECTIVES)):
        directory = run / "conditions" / objective / dataset / sequence
        directory.mkdir(parents=True)
        pred = gt.copy()
        pred[..., 0] += index * .1
        error = np.linalg.norm(pred - gt, axis=-1)
        tracking = projection.residual_metrics(error, np.ones((t, q), bool))
        rgb = np.full((t, 3, h, w), .4, np.float32)
        if index:
            rgb[:, :, 1, 1] += index / 1000
        delta = rgb - np.full_like(rgb, .4)
        recon = dense_gt.copy()
        recon[..., 0] = index * .2 + np.arange(t, dtype=np.float32)[:, None, None] / 1000
        re = np.linalg.norm(recon - dense_gt, axis=-1)
        recon_metrics = projection.residual_metrics(re, dense_valid)
        for name, array in (("rgb_float32.npy", rgb), ("delta_float32.npy", delta), ("reconstruction_float32.npy", recon),
                            ("reconstruction_confidence_float32.npy", np.full((t, h, w), 1 + index / 10, np.float32))):
            np.save(directory / name, array)
        np.savez(directory / "tracks.npz", pred=pred, gt=gt, valid=np.ones((t, q), bool), dynamic=dynamic,
                 query_xy=query_xy, intrinsics=intrinsics)
        np.savez(directory / "components.npz", reconstruction_norm=np.full(t, .7 - index / 100, np.float32),
                 track_conf=np.ones((t, q), np.float32))
        history = [{"restart": -1, "step": 0, "loss": 1., "delta_linf": 0., "delta_l2_per_frame": [0.] * 128}]
        if index:
            for step in range(21):
                row = {"restart": 0, "step": step, "loss": 1. + step / 10,
                       "delta_linf": min(step / 1000, 4 / 255), "delta_l2_per_frame": [step / 1000] * 128}
                if step < 20:
                    row["gradient_l2_per_frame"] = [0 if frame == 0 else .1 for frame in range(128)]
                history.append(row)
            history[16]["loss"] = 4.  # selected step15 deliberately differs from final.
        write_json(directory / "history.json", history)
        record = {"dataset": dataset, "sequence": sequence, "objective": objective, "signature": signature,
                  "seed": 101, "epsilon_255": 0 if index == 0 else 4, "steps": 0 if index == 0 else 20,
                  "saved_input_verification": {"passed": True, "capture_output_errors": {}, "compact_output_errors": {},
                                               "max_reconstruction_error": 0, "max_reconstruction_confidence_error": 0},
                  "tracking_metrics": {"scale_all": 1., "epe_all_m": tracking["epe_m"], "apd3d_all": tracking["apd3d"]},
                  "reconstruction_metrics": {"scale": 1., **recon_metrics, "valid_pixels": int(dense_valid.sum())},
                  "perturbation_norms": {"linf": float(np.abs(delta).max())},
                  "selected_state": {"restart": -1 if index == 0 else 0, "step": 0 if index == 0 else 15},
                  "attack_loss": 1. if index == 0 else 4.,
                  "loss_terms": {"diagnostics": {"track_raw_conf_mean": 1.1, "track_conf_mean": 4., "reconstruction_conf_mean": 1.2,
                                                 "tracking_l21": .1 + index / 10, "reconstruction_l21": .2 + index / 10}},
                  "artifact_sha256": {path.name: projection.sha256(path) for path in directory.iterdir() if path.is_file()}}
        write_json(directory / "result.json", record)
    return run, project, dataset, sequence


@pytest.fixture
def fixture_run(tmp_path):
    return synthetic_run(tmp_path)


def test_projection_uses_moving_gt_camera_and_includes_later_occlusion(fixture_run):
    run, project, dataset, sequence = fixture_run
    context = projection.ComponentContext(run, project)
    pair = context.pair(dataset, sequence, "tracking_3d")
    assert pair["projection_checks"]["passed"]
    assert pair["projection_checks"]["gt_uv_error_max_px"] < .01
    assert pair["visibility"].shape == (128, 4)
    assert not pair["visibility"][127, 0]
    assert pair["attack_epe_per_frame_m"].shape == (128,)
    np.testing.assert_allclose(pair["attack_epe_per_frame_m"], .2, rtol=1e-5)
    assert not np.allclose(pair["gt_xy"][0], pair["gt_xy"][127])


def test_source_hash_change_is_rejected(fixture_run):
    run, project, dataset, sequence = fixture_run
    source = project / "data" / "worldtrack_release" / dataset / f"{sequence}.npz"
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Original NPZ hash/size"):
        projection.ComponentContext(run, project).pair(dataset, sequence, "tracking_3d")


def update_record(run, dataset, sequence, objective, edit):
    path = run / "conditions" / objective / dataset / sequence / "result.json"
    record = projection.read_json(path)
    edit(record)
    write_json(path, record)


@pytest.mark.parametrize("field,value,match", [("seed", 999, "seeds"), ("signature", "wrong", "signature/identity")])
def test_paired_seed_and_run_signature_required(fixture_run, field, value, match):
    run, project, dataset, sequence = fixture_run
    update_record(run, dataset, sequence, "tracking_3d", lambda record: record.update({field: value}))
    with pytest.raises(ValueError, match=match):
        projection.ComponentContext(run, project).pair(dataset, sequence, "tracking_3d")


def test_epe_record_tamper_rejected(fixture_run):
    run, project, dataset, sequence = fixture_run
    update_record(run, dataset, sequence, "tracking_3d", lambda record: record["tracking_metrics"].update(epe_all_m=100))
    with pytest.raises(ValueError, match="Tracking EPE"):
        projection.ComponentContext(run, project).pair(dataset, sequence, "tracking_3d")


def test_delta_exact_subtraction_not_only_bound(fixture_run):
    run, project, dataset, sequence = fixture_run
    directory = run / "conditions" / "tracking_3d" / dataset / sequence
    np.save(directory / "delta_float32.npy", np.zeros((128, 3, 4, 8), np.float32))
    update_record(run, dataset, sequence, "tracking_3d", lambda record: record["artifact_sha256"].update({"delta_float32.npy": projection.sha256(directory / "delta_float32.npy")}))
    with pytest.raises(ValueError, match="exact attacked minus clean"):
        projection.ComponentContext(run, project).pair(dataset, sequence, "tracking_3d")


def test_all_frame_evidence_refuses_prefix(fixture_run):
    run, project, dataset, sequence = fixture_run
    directory = run / "sequences" / dataset / sequence
    metadata = projection.read_json(directory / "sequence.json")
    metadata["used_frame_indices"] = list(range(64))
    write_json(directory / "sequence.json", metadata)
    inventory = projection.read_json(directory / "sequence_manifest.json")
    inventory["artifact_sha256"]["sequence.json"] = projection.sha256(directory / "sequence.json")
    write_json(directory / "sequence_manifest.json", inventory)
    with pytest.raises(ValueError, match="all 128"):
        projection.ComponentContext(run, project).pair(dataset, sequence, "tracking_3d")


def test_reconstruction_fixed_support_and_common_limits(fixture_run):
    run, project, dataset, sequence = fixture_run
    context = projection.ComponentContext(run, project)
    checks = context.reconstruction_checks(dataset, sequence)
    assert checks["passed"] and len(checks["conditions"]) == 6
    assert checks["vmax_m"] > 1
    assert checks["conditions"]["clean"]["valid_pixels"] == 128 * 4 * 8 - 64 * 8
    clean = checks["conditions"]["clean"]
    assert abs(clean["epe_m"] - np.mean(clean["epe_per_frame_m"])) > 1e-5


def test_pred_behind_camera_is_nan_not_clamped():
    xyz = np.array([[[100., 0., 1.], [0., 0., -1.]]], np.float32)
    xy, valid, inside, behind = projection._project(xyz, np.eye(4)[None], np.array([1, 1, 2, 2.]), np.ones(2), 8, 4)
    assert xy[0, 0, 0] > 8 and valid[0, 0] and not inside[0, 0]
    assert np.isnan(xy[0, 1]).all() and behind[0, 1] and not valid[0, 1]


def test_gt_only_query_selection_independent_of_attack(fixture_run):
    run, project, dataset, sequence = fixture_run
    context = projection.ComponentContext(run, project)
    first, metadata = renderer.select_queries(context.pair(dataset, sequence, "tracking_3d"))
    second, _ = renderer.select_queries(context.pair(dataset, sequence, "reconstruction_3d"))
    np.testing.assert_array_equal(first, second)
    assert not metadata["selection_uses_attack_results"]


def test_history_best_and_zero_gradient_rendered(fixture_run, tmp_path):
    run, project, dataset, sequence = fixture_run
    condition = projection.ComponentContext(run, project).condition(dataset, sequence, "tracking_3d")
    path = tmp_path / "history.png"
    renderer.history_plot(condition, path, "synthetic fixture")
    assert path.is_file() and Image.open(path).width > 100
    condition["record"]["selected_state"]["step"] = 20
    with pytest.raises(ValueError, match="selected history"):
        renderer.history_plot(condition, tmp_path / "invalid.png", "synthetic")


def test_probe_requires_decoded128_and_fps(monkeypatch, tmp_path):
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"synthetic-only")
    encoder = object.__new__(renderer.CPUEncoder)
    encoder.ffprobe, encoder.ffmpeg, encoder.image_id = "ffprobe", "ffmpeg", None
    class Completed:
        stdout = json.dumps({"streams": [{"nb_read_frames": "128", "avg_frame_rate": "10/1", "width": 8, "height": 4, "codec_name": "h264"}], "format": {"duration": "12.8"}})
    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **k: Completed())
    probe = encoder.probe(video, 128, 10)
    assert probe["success"] and probe["decoded_frames"] == 128
    assert probe["method"] == "ffprobe_count_frames"
    with pytest.raises(ValueError, match="decoded-frame"):
        encoder.probe(video, 64, 10)


def test_encoder_checks_frame_generator_count_before_publication(monkeypatch, tmp_path):
    encoder = object.__new__(renderer.CPUEncoder)
    encoder.ffmpeg, encoder.ffprobe, encoder.image_id = "ffmpeg", "ffprobe", None
    monkeypatch.setattr(renderer, "encode_frames", lambda command, frames: list(frames))
    with pytest.raises(ValueError, match="yielded 127"):
        encoder.encode(tmp_path / "bad.mp4", (Image.new("RGB", (8, 4)) for _ in range(127)), 8, 4, 10, 128)
    assert not (tmp_path / "bad.mp4").exists()


def test_encoder_refuses_wrong_output_resolution(monkeypatch, tmp_path):
    encoder = object.__new__(renderer.CPUEncoder)
    encoder.ffmpeg, encoder.ffprobe, encoder.image_id = "ffmpeg", "ffprobe", None
    def fake_encode(command, frames):
        list(frames)
        Path(command[-1]).write_bytes(b"synthetic-only")
    monkeypatch.setattr(renderer,"encode_frames",fake_encode)
    monkeypatch.setattr(encoder,"probe",lambda *args:{"width":10,"height":4})
    with pytest.raises(ValueError,match="dimensions"):
        encoder.encode(tmp_path/"wrong.mp4",(Image.new("RGB",(8,4)) for _ in range(128)),8,4,10,128)
    assert not (tmp_path/"wrong.mp4").exists()


def test_partial_or_preview_cannot_claim_complete():
    manifest = {"clips": [], "timelines": [], "geometry_cases": [], "errors": []}
    assert not renderer.manifest_complete(manifest, [("po_mini", "clip")])


def test_complete_manifest_requires_both_videos_and_unique_shared_frame_cases():
    identities=[("po_mini","clip")]
    manifest={"clips":[],"timelines":[{"dataset":"po_mini","sequence":"clip"}],
              "geometry_cases":[{"dataset":"po_mini","sequence":"clip","frame_index":f} for f in (0,64,127)],"errors":[]}
    for objective in projection.OBJECTIVES:
        manifest["clips"].append({"dataset":"po_mini","sequence":"clip","objective":objective,"status":"complete","frame_count":128,
          "projection_checks":{"passed":True},"reconstruction_checks":{"passed":True},
          "assets":[{"kind":kind,"probe":{"success":True,"decoded_frames":128}} for kind in ("rgb_video","tracking_video")]})
    assert renderer.manifest_complete(manifest,identities)
    manifest["geometry_cases"][2]["frame_index"]=64
    assert not renderer.manifest_complete(manifest,identities)
    manifest["geometry_cases"][2]["frame_index"]=127
    manifest["clips"].append(manifest["clips"][0])
    assert not renderer.manifest_complete(manifest,identities)


def test_output_assets_reject_path_escape(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="within output"):
        renderer.asset_record(outside, tmp_path / "assets", "rgb")


@pytest.mark.parametrize("value", ["../clip", "..", "a/b", "C:clip", "a\\b", ""])
def test_identity_path_escape_rejected(value):
    with pytest.raises(ValueError, match="Unsafe"):
        projection.safe_component(value)
