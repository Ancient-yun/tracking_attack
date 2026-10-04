"""Small saved-array fixtures exercise integrity checks without Torch/models."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("component_artifact_audit", ROOT / "scripts/audit_component_artifacts.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def artifact_digests(directory):
    return {item.name: audit_module._helpers.file_hash(item) for item in directory.iterdir()
            if item.is_file() and item.name != "result.json"}


def publish_summary(run, config, manifest, records):
    conditions = {(r["dataset"], r["sequence"], r["objective"]): {"record": r} for r in records}
    rows = audit_module.summary_rows(conditions)
    aggregate = []
    for dataset in ("po_mini", "ds_mini", "all"):
        for objective in ("clean", *config["objectives"]):
            group = [r for r in rows if r["objective"] == objective and (dataset == "all" or r["dataset"] == dataset)]
            if not group:
                continue
            item = {"dataset": dataset, "objective": objective, "completed_sequences": len(group),
                    "expected_sequences": 1, "attack_seconds_sum": sum(r["elapsed_seconds"] for r in group)}
            for key in group[0]:
                values = [row.get(key) for row in group]
                if key not in item and all(isinstance(v, (int, float)) for v in values):
                    item[key] = float(np.mean(values))
            aggregate.append(item)
    for name, data in (("sequence_metrics.csv", rows), ("aggregate_metrics.csv", aggregate)):
        fields = list(dict.fromkeys(key for row in data for key in row))
        with (run / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    write_json(run / "summary.json", {"complete": True, "completed_conditions": len(records), "expected_conditions": len(records),
        "missing_conditions": [], "failures": [], "failed_attempts": 0,
        "aggregation": "equal_weight_per_sequence; CPU fixture", "aggregate": aggregate})


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = audit_module.read_json(ROOT / "configs/loss_components.json")
    config.update(num_frames=2, steps=1, objectives=["tracking_mse"])
    entry = {"dataset": "po_mini", "sequence": "toy", "path": "data/worldtrack_release/po_mini/toy.npz",
             "bytes": 1, "sha256": "a" * 64}
    manifest = {"num_frames": 2, "entries": [entry]}
    metadata = {"config": config, "manifest": manifest, "provenance": {}, "tta": False, "frozen_weights": True}
    metadata["signature"] = audit_module.fingerprint({k: metadata[k] for k in ("config", "manifest", "provenance")})
    write_json(run / "run.json", metadata)
    sequence_dir = run / "sequences/po_mini/toy"
    sequence_dir.mkdir(parents=True)
    rgb = np.full((2, 3, 4, 4), .5, dtype=np.float32)
    gt = np.array([[[1, 1, 1], [2, 1, 2], [3, 1, 2]], [[1.1, 1, 1], [2, 1, 2], [3, 1, 2]]], dtype=np.float32)
    query = np.array([[.2, .4], [1.2, .7], [1.4, .7]], dtype=np.float64)
    dense = np.ones((2, 4, 4, 3), dtype=np.float32)
    dense[..., 2] = 2
    dense[:, 0, 0] = 0
    dense_valid = np.ones((2, 4, 4), dtype=bool)
    dense_valid[:, 0, 0] = False
    targets = {"native_tracks": gt[:, [0, 2]], "native_track_valid": np.ones((2, 2), dtype=bool),
        "training_dynamic": np.array([True, False]), "tracks": gt, "valid": np.ones((2, 3), dtype=bool),
        "reconstruction_gt_norm": np.array([np.linalg.norm(frame, axis=-1).sum() / 48 for frame in dense], dtype=np.float32),
        "reconstruction_valid_counts": np.array([15, 15], dtype=np.int64)}
    np.savez(sequence_dir / "targets.npz", **targets)
    np.save(sequence_dir / "reconstruction_gt.npy", dense)
    np.save(sequence_dir / "reconstruction_valid.npy", dense_valid)
    hashes = {name: audit_module.tensor_hash(value) for name, value in {**targets, "reconstruction_gt": dense,
        "reconstruction_valid": dense_valid, "native_query_xy": np.array([[0, 0], [1, 1]], dtype=np.int64),
        "native_point_indices": np.array([0, 2], dtype=np.int64)}.items()}
    component = {"tensor_sha256": hashes, "rgb_sha256": audit_module.tensor_hash(rgb),
        "evaluation_query_xy_sha256": audit_module.tensor_hash(query), "source_sha256": {key: "a" * 64 for key in audit_module.NATIVE_SOURCES},
        "eval_query_count": 3, "native_query_count": 2, "native_collision_count": 1}
    write_json(sequence_dir / "sequence.json", {"num_frames": 2, "num_queries": 3, "preprocessed_wh": [4, 4],
        "official_reference_batch_size": 1, "official_clean_batch_size": 1,
        "selected_point_indices": [0, 1, 2], "component_targets": component})
    write_json(sequence_dir / "official_clean_parity.json", {"passed": True, "max_abs_error": 0., "atol": 1e-5, "rtol": 1e-5})
    write_json(sequence_dir / "sequence_manifest.json", {"signature": metadata["signature"],
        "artifact_sha256": {p.name: audit_module._helpers.file_hash(p) for p in sequence_dir.iterdir() if p.is_file()}})
    seed = audit_module.stable_seed(config["seed"], "po_mini", "toy", 4)
    records = []
    clean_terms = None
    for objective in ("clean", "tracking_mse"):
        directory = run / "conditions" / objective / "po_mini/toy"
        directory.mkdir(parents=True)
        is_clean = objective == "clean"
        saved_rgb = rgb.copy() if is_clean else rgb + np.float32(.005)
        delta = saved_rgb - rgb
        recon = dense * np.float32(1.1) + np.float32(.02)
        conf = np.full((2, 4, 4), 2, dtype=np.float32)
        norm = np.array([np.linalg.norm(frame, axis=-1).sum() / 48 for frame in recon], dtype=np.float32)
        residual = np.linalg.norm(recon / norm[:, None, None, None] - dense / targets["reconstruction_gt_norm"][:, None, None, None], axis=-1)
        outputs = {"tracks": gt.copy() if is_clean else gt + np.float32(.03),
            "native_tracks": targets["native_tracks"] + np.float32(.02), "track_conf": np.full((2, 2), 2, dtype=np.float32),
            "reconstruction_norm": norm,
            "reconstruction_regression_sum": (conf * residual * dense_valid).sum((1, 2)).astype(np.float32),
            "reconstruction_log_conf_sum": (np.log(conf) * dense_valid).sum((1, 2)).astype(np.float32),
            "reconstruction_l21_sum": (residual * dense_valid).sum((1, 2)).astype(np.float32),
            "reconstruction_conf_sum": (conf * dense_valid).sum((1, 2)).astype(np.float32)}
        terms = audit_module.native_terms(outputs, targets, config)
        if is_clean:
            clean_terms = copy.deepcopy(terms)
        loss = terms["tracking_mse"]
        norms = {"epsilon_255": 0 if is_clean else 4, "linf": float(np.abs(delta).max()), "linf_per_frame": np.abs(delta).reshape(2, -1).max(1).tolist(),
                 "l2_per_frame": np.linalg.norm(delta.reshape(2, -1), axis=1).tolist(), "attacked_frames": 0 if is_clean else 2}
        np.save(directory / "rgb_float32.npy", saved_rgb)
        np.save(directory / "delta_float32.npy", delta)
        np.save(directory / "reconstruction_float32.npy", recon)
        np.save(directory / "reconstruction_confidence_float32.npy", conf)
        np.savez(directory / "components.npz", **outputs)
        np.savez(directory / "tracks.npz", pred=outputs["tracks"], gt=gt, valid=targets["valid"],
                 dynamic=np.array([True, False, False]), query_xy=query, intrinsics=np.array([2, 2, 2, 2], dtype=np.float32))
        history = [{"restart": -1, "step": 0, "objective": "tracking_mse", "loss": clean_terms["tracking_mse"], "scale": 1.,
                    "terms": clean_terms, "delta_linf": 0., "delta_linf_per_frame": [0., 0.], "delta_l2_per_frame": [0., 0.]}]
        if not is_clean:
            for step in (0, 1):
                item = {"restart": 0, "step": step, "objective": "tracking_mse", "loss": loss, "scale": 1., "terms": terms,
                        "delta_linf": norms["linf"], "delta_linf_per_frame": norms["linf_per_frame"], "delta_l2_per_frame": norms["l2_per_frame"]}
                if step == 0:
                    item.update(gradient_linf_per_frame=[.1, .1], gradient_l2_per_frame=[.2, .2])
                history.append(item)
        write_json(directory / "history.json", history)
        write_json(directory / "perturbation_norms.json", norms)
        if is_clean:
            oracle_components = {"tracking": terms["tracking_regression"] - .2 * terms["diagnostics"]["track_log_conf_mean"],
                "reconstruction": terms["reconstruction_regression"] - .2 * terms["diagnostics"]["reconstruction_log_conf_mean"],
                "joint": terms["total"], "tracking_l21": terms["diagnostics"]["tracking_l21"],
                "reconstruction_l21": terms["diagnostics"]["reconstruction_l21"]}
            write_json(directory / "native_loss_oracle.json", {"passed": True, "device": "cpu", "num_frames": 2,
                "head_gradient_max_abs_error": {name: 0. for name in ("head1_xyz", "head1_conf", "head2_xyz", "head2_conf")},
                "loss_abs_error": 0., "atol": 2e-5, "rtol": 2e-5, "components": oracle_components,
                "oracle": {"mode": "unchanged official active-loss AST + actual official geometry import", "full_training_loss_module_imported": False,
                           "losses_source_sha256": "a" * 64, "geometry_source_sha256": "a" * 64, "misc_source_sha256": "a" * 64}})
        record = {"dataset": "po_mini", "sequence": "toy", "objective": objective, "epsilon_255": 0 if is_clean else 4,
            "steps": 0 if is_clean else 1, "seed": seed, "signature": metadata["signature"], "attack_loss": loss, "loss_terms": terms,
            "selected_state": {"restart": -1 if is_clean else 0, "step": 0 if is_clean else 1}, "elapsed_seconds": 1., "condition_wall_seconds": 2.,
            "tracking_metrics": {"num_frames": 2, "num_queries": 3, "num_dynamic_queries": 1, "apd3d_all": 90., "epe_all_m": .1},
            "reconstruction_metrics": {"num_frames": 2, "valid_pixels": 30, "apd3d": 90., "epe_m": .1}, "perturbation_norms": norms,
            "saved_input_verification": {"passed": True, "capture_output_errors": {name: 0. for name in audit_module.OUTPUT_FIELDS},
                "compact_output_errors": {name: 0. for name in audit_module.OUTPUT_FIELDS}, "max_reconstruction_error": 0., "max_reconstruction_confidence_error": 0.}}
        if not is_clean:
            record["attack_config"] = {"method": "pgd", "eps": 4/255, "steps": 1, "restarts": 1, "seed": seed,
                "step_size": 1/255, "gradient_mode": "recompute", "check_replay": True, "record_per_frame": True, "random_start": True}
        record["artifact_sha256"] = artifact_digests(directory)
        write_json(directory / "result.json", record)
        records.append(record)
    publish_summary(run, config, manifest, records)
    return run


def audit(run):
    return audit_module.audit_run(run, ROOT, expected_clips=1, expected_frames=2, expected_steps=1, verify_sources=False)


def edit_artifact(run, objective, name, edit):
    directory = run / "conditions" / objective / "po_mini/toy"
    edit(directory / name)
    record = audit_module.read_json(directory / "result.json")
    record["artifact_sha256"] = artifact_digests(directory)
    write_json(directory / "result.json", record)


def test_complete_fixture_passes_without_claiming_full_campaign(saved_run):
    report = audit(saved_run)
    assert report["passed"], report["errors"]
    assert report["expected_conditions"] == report["completed_conditions_checked"] == 2
    assert not report["full_campaign_completed"]


def test_wrong_saved_delta_fails_even_when_file_digest_is_updated(saved_run):
    def change(path):
        delta = np.load(path)
        delta[1, 0, 0, 0] += np.float32(.001)
        np.save(path, delta)
    edit_artifact(saved_run, "tracking_mse", "delta_float32.npy", change)
    report = audit(saved_run)
    assert not report["passed"]
    assert any("bit-exactly" in error["check"] for error in report["errors"])


def test_confidence_below_native_floor_fails_with_updated_digest(saved_run):
    def change(path):
        confidence = np.load(path)
        confidence[0, 0, 0] = .5
        np.save(path, confidence)
    edit_artifact(saved_run, "tracking_mse", "reconstruction_confidence_float32.npy", change)
    report = audit(saved_run)
    assert any("confidence >= one" in error["check"] for error in report["errors"])


def test_nonfinite_array_produces_a_writable_failed_report(saved_run):
    def change(path):
        delta = np.load(path)
        delta[0, 0, 0, 0] = np.nan
        np.save(path, delta)
    edit_artifact(saved_run, "tracking_mse", "delta_float32.npy", change)
    report = audit(saved_run)
    assert not report["passed"] and report["status"] == "failed"
    assert any(error["check"] == "saved delta finite" for error in report["errors"])
    json.dumps(report, allow_nan=False)


def test_nonfinite_gradient_fails_and_zero_gradient_is_reported(saved_run):
    def zeros(path):
        history = audit_module.read_json(path)
        history[1]["gradient_linf_per_frame"][1] = 0.
        history[1]["gradient_l2_per_frame"][1] = 0.
        write_json(path, history)
    edit_artifact(saved_run, "tracking_mse", "history.json", zeros)
    report = audit(saved_run)
    assert report["passed"], report["errors"]
    attack = next(condition for condition in report["conditions"] if condition["objective"] == "tracking_mse")
    assert attack["zero_gradient_frame_step_count"] == 1
    assert not attack["all_recorded_gradient_norms_nonzero"]
    def invalid(path):
        history = audit_module.read_json(path)
        history[1]["gradient_linf_per_frame"][1] = -1.
        write_json(path, history)
    edit_artifact(saved_run, "tracking_mse", "history.json", invalid)
    assert not audit(saved_run)["passed"]


def test_missing_artifact_is_incomplete_and_never_passes(saved_run):
    (saved_run / "conditions/tracking_mse/po_mini/toy/components.npz").unlink()
    report = audit(saved_run)
    assert report["status"] == "incomplete" and not report["passed"]


def test_changed_gt_inventory_or_aggregate_cannot_pass(saved_run):
    summary = audit_module.read_json(saved_run / "summary.json")
    summary["aggregate"][0]["tracking_apd3d_all"] = 0.
    write_json(saved_run / "summary.json", summary)
    report = audit(saved_run)
    assert any("equal-weight" in error["check"] for error in report["errors"])
    inventory = saved_run / "sequences/po_mini/toy/sequence_manifest.json"
    value = audit_module.read_json(inventory)
    value["artifact_sha256"]["targets.npz"] = "0" * 64
    write_json(inventory, value)
    report = audit(saved_run)
    assert any(error["check"] == "SHA256 matches recorded digest" for error in report["errors"])


def test_explicit_source_snapshot_requires_exact_set_and_raw_hash_without_fallback(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "core.py"
    source.write_bytes(b"original source\n")
    expected = audit_module._helpers.file_hash(source)
    metadata = {"provenance": {"experiment_source_sha256": {"core.py": expected}}}
    monkeypatch.setattr(audit_module._helpers, "audit_source_files", lambda *args: None)
    check = audit_module.Audit(tmp_path)
    audit_module.audit_sources(check, metadata, tmp_path, snapshot)
    assert not check.errors and not check.missing
    # A matching unrelated host copy must not rescue a changed explicit copy.
    host = tmp_path / "src/st4rtrack_pgd"
    host.mkdir(parents=True)
    (host / "core.py").write_bytes(b"original source\n")
    source.write_bytes(b"changed source\n")
    check = audit_module.Audit(tmp_path)
    audit_module.audit_sources(check, metadata, tmp_path, snapshot)
    assert any(error["check"] == "SHA256 matches recorded digest" for error in check.errors)
    (snapshot / "unrecorded.py").write_text("extra")
    check = audit_module.Audit(tmp_path)
    audit_module.audit_sources(check, metadata, tmp_path, snapshot)
    assert any("exact recorded Python file set" in error["check"] for error in check.errors)


@pytest.mark.parametrize("declared", [18, 20, 22, 24, 26, 28])
def test_campaign_size_uses_matching_positive_even_declarations(declared):
    metadata = {"config": {"campaign_expected_clips": declared}, "manifest": {"campaign_expected_clips": declared}}
    assert audit_module.campaign_size(metadata) == declared
    assert audit_module.campaign_size({"config": {}, "manifest": {}}) == 14
    metadata["manifest"]["campaign_expected_clips"] = declared + 2
    with pytest.raises(ValueError):
        audit_module.campaign_size(metadata)


def test_all_frame_campaign_requires_authoritative_declarations_and_full_raw_sequence_metadata():
    metadata = {"config": {"num_frames": 64}, "manifest": {"num_frames": 64}}
    assert audit_module.campaign_frames(metadata) == (64, False)
    for value in metadata.values():
        value.update({"num_frames": 128, "campaign_expected_frames": 128, "all_frames": True})
    assert audit_module.campaign_frames(metadata) == (128, True)
    sequence = {"num_frames": 128, "requested_num_frames": 128, "original_frame_count": 128}
    assert audit_module.all_frame_sequence_metadata(sequence)
    for field in sequence:
        altered = {**sequence, field: 64}
        assert not audit_module.all_frame_sequence_metadata(altered)
    metadata["manifest"]["all_frames"] = False
    with pytest.raises(ValueError):
        audit_module.campaign_frames(metadata)


@pytest.mark.parametrize("bad", [True, 64, 127, "128", None])
def test_all_frame_campaign_cannot_silently_claim_truncated_or_unadvertised_scope(bad):
    common = {"num_frames": 128, "campaign_expected_frames": 128, "all_frames": True}
    metadata = {"config": {**common}, "manifest": {**common, "campaign_expected_frames": bad}}
    with pytest.raises(ValueError):
        audit_module.campaign_frames(metadata)
    with pytest.raises(ValueError):
        audit_module.campaign_frames({"config": {"num_frames": 128}, "manifest": {"num_frames": 128}})


def test_raw_all_frame_header_count_is_independent_of_declared_saved_metadata(tmp_path):
    path = tmp_path / "source.npz"
    values = {"images_jpeg_bytes": np.zeros(128, dtype="S1"), "tracks_XYZ": np.zeros((128, 2, 3)),
              "visibility": np.ones((128, 2), dtype=bool), "depth_map": np.ones((128, 2, 2), dtype=np.float32),
              "extrinsics_w2c": np.zeros((128, 4, 4))}
    np.savez_compressed(path, **values)
    shapes = audit_module.raw_all_frame_headers(path)
    assert audit_module.all_frame_raw_shapes(shapes)
    values["images_jpeg_bytes"] = values["images_jpeg_bytes"][:64]
    np.savez_compressed(path, **values)
    assert not audit_module.all_frame_raw_shapes(audit_module.raw_all_frame_headers(path))
