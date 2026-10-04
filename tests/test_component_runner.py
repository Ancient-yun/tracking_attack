"""CPU integration of component PGD, official metrics, artifacts and resume."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from st4rtrack_pgd import run_component_study as runner
from st4rtrack_pgd.common import VENDOR_ROOT, sha256
from st4rtrack_pgd.component_attack import OBJECTIVES, OUTPUT_KEYS, component_loss_terms
from st4rtrack_pgd.component_forward import NativeComponentForward as ActualNativeForward
from st4rtrack_pgd.evaluate_attack import evaluate_tracks
from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward as ActualBaseForward
from st4rtrack_pgd.worldtrack_adapter import WorldTrackSequence

from test_component_forward import TwoHeadToy, toy_targets


class SyntheticBase(ActualBaseForward):
    parity_calls = 0
    parity_failures = 0
    checkpoint_loads = 0

    @classmethod
    def from_checkpoint(cls, checkpoint, query_xy, *, device="cpu", vendor_root=None, **kwargs):
        cls.checkpoint_loads += 1
        assert torch.device(device).type == "cpu"
        return cls(TwoHeadToy().to(device), query_xy, vendor_root=vendor_root, checkpoint=str(checkpoint))

    def check_clean_parity(self, sequence, **kwargs):
        type(self).parity_calls += 1
        if type(self).parity_failures:
            type(self).parity_failures -= 1
            raise AssertionError("synthetic official clean parity failed")
        assert self.vendor_root == Path(sequence.metadata["vendor_root"]).resolve()
        return {"passed": True, "batch_size": 1, "max_abs_error": 0.0}


def synthetic_sequence(source, vendor_root):
    targets = toy_targets()
    second_clip = Path(source).stem == "clip_b"
    offset = 0.04 if second_clip else 0
    rgb = torch.linspace(0.52 - offset, 0.84 - offset, 2 * 3 * 4 * 6).reshape(2, 3, 4, 6)
    if second_clip:
        targets.native_tracks = targets.native_tracks + torch.tensor([0.03, -0.01, 0.04])
        targets.tracks = targets.native_tracks.clone()
    targets.metadata = {"fixture": "CPU two-head geometry", "clip": Path(source).stem,
                        "tensor_sha256": targets.tensor_hashes()}
    sequence = WorldTrackSequence(
        rgb=rgb, gt_tracks=targets.tracks, valid=targets.valid,
        dynamic=torch.tensor([True, False, True]), query_xy=targets.native_query_xy.double() + 0.2,
        metadata={"vendor_root": str(Path(vendor_root).resolve()), "intrinsics": [10., 10., 3., 2.],
                  "num_frames": 2, "num_queries": 3, "sequence_name": Path(source).stem,
                  "coordinate_frame": "first-camera world"},
    )
    return sequence, targets


@pytest.fixture
def study(tmp_path, monkeypatch):
    SyntheticBase.parity_calls = SyntheticBase.parity_failures = SyntheticBase.checkpoint_loads = 0
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic frozen model")
    entries = []
    for dataset, clip in [("po_mini", "clip_a"), ("ds_mini", "clip_b")]:
        source = tmp_path / f"{clip}.npz"
        source.write_bytes(f"synthetic source {clip}".encode())
        entries.append({"dataset": dataset, "sequence": clip, "path": str(source), "sha256": sha256(source)})
    config = {"checkpoint": str(checkpoint), "num_frames": 2, "image_size": 512, "device": "cpu",
              "seed": 241, "objectives": list(OBJECTIVES), "epsilon_255": 4., "steps": 2,
              "restarts": 1, "step_size_fraction": 0.25, "gradient_mode": "recompute",
              "alpha": 0.2, "reweight_scale": 5., "save_inputs": True,
              "verify_saved_inputs": True, "verify_official_clean": True}
    manifest = {"schema_version": 1, "num_frames": 2, "entries": entries}
    loaded, forwards = [], []

    def load(source, *, num_frames, image_size, vendor_root):
        loaded.append(Path(source))
        assert num_frames == 2 and image_size == 512
        return synthetic_sequence(source, vendor_root)

    def native(base, targets):
        actual = ActualNativeForward(base, targets)
        forwards.append(actual)
        return actual

    monkeypatch.setattr(runner, "provenance", lambda *args: {"test_source": "fixed", "official_metric_source_sha256": sha256(VENDOR_ROOT / "dust3r/track_eval_util.py")})
    monkeypatch.setattr(runner, "St4RTrackForward", SyntheticBase)
    monkeypatch.setattr(runner, "NativeComponentForward", native)
    monkeypatch.setattr(runner, "load_component_sequence", load)
    return SimpleNamespace(config=config, manifest=manifest, run_dir=tmp_path / "run", loaded=loaded,
                           forwards=forwards, monkeypatch=monkeypatch)


def run(study, *, resume=False, fail_fast=True):
    return runner.run_study(study.config, study.manifest, study.run_dir, resume=resume, fail_fast=fail_fast)


def records(study):
    return {path: json.loads(path.read_text()) for path in study.run_dir.glob("conditions/*/*/*/result.json")}


def condition_path(study, objective="tracking_3d", dataset="po_mini", clip="clip_a"):
    return study.run_dir / "conditions" / objective / dataset / clip


def test_end_to_end_all_five_losses_artifacts_official_metrics_and_resume(study):
    summary = run(study)
    assert summary["complete"]
    assert summary["completed_conditions"] == summary["expected_conditions"] == 12
    assert summary["missing_conditions"] == summary["failures"] == []
    assert summary["failed_attempts"] == 0
    assert SyntheticBase.checkpoint_loads == 1
    assert SyntheticBase.parity_calls == 2
    required = {"rgb_float32.npy", "delta_float32.npy", "reconstruction_float32.npy",
                "reconstruction_confidence_float32.npy", "components.npz", "tracks.npz",
                "history.json", "perturbation_norms.json"}
    snapshots = {path: path.read_bytes() for path in records(study)}
    for path, record in records(study).items():
        assert record["saved_input_verification"]["passed"]
        assert required <= set(record["artifact_sha256"])
        for name, digest in record["artifact_sha256"].items():
            assert sha256(path.parent / name) == digest
        saved = np.load(path.parent / "rgb_float32.npy", allow_pickle=False)
        delta = np.load(path.parent / "delta_float32.npy", allow_pickle=False)
        clean_sequence, context = synthetic_sequence(study.manifest["entries"][int(record["sequence"] == "clip_b")]["path"], VENDOR_ROOT)
        np.testing.assert_array_equal(saved - clean_sequence.rgb.numpy(), delta)
        assert saved.dtype == delta.dtype == np.float32
        assert np.max(np.abs(delta)) <= record["epsilon_255"] / 255 + 1e-7
        assert np.logical_and(saved >= 0, saved <= 1).all()
        assert record["condition_wall_seconds"] >= record["elapsed_seconds"] > 0
        with np.load(path.parent / "tracks.npz", allow_pickle=False) as pack:
            tracking = evaluate_tracks(pack["pred"], pack["gt"], pack["valid"], pack["dynamic"], pack["intrinsics"])
        assert tracking == record["tracking_metrics"]
        recon = np.load(path.parent / "reconstruction_float32.npy", allow_pickle=False)
        expected_recon = runner.reconstruction_metrics(recon, context.reconstruction_gt, context.reconstruction_valid,
                                                       clean_sequence.metadata["intrinsics"])
        assert expected_recon == record["reconstruction_metrics"]
        with np.load(path.parent / "components.npz", allow_pickle=False) as pack:
            outputs = {key: torch.from_numpy(pack[key]) for key in pack.files}
        terms = runner.scalar_tree(component_loss_terms(outputs, context))
        assert terms == record["loss_terms"]
        key = "tracking_mse" if record["objective"] == "clean" else OBJECTIVES[record["objective"]]
        assert record["attack_loss"] == terms[key]
        if record["objective"] != "clean":
            assert record["peak_vram_bytes"] == 0
            history = json.loads((path.parent / "history.json").read_text())
            assert record["attack_loss"] == max(row["loss"] for row in history)
            assert len(history) == 4
            assert all(len(row["delta_linf_per_frame"]) == 2 for row in history)
    all_rows = [row for row in summary["aggregate"] if row["dataset"] == "all"]
    assert len(all_rows) == 6
    for aggregate in all_rows:
        children = [row for row in records(study).values() if row["objective"] == aggregate["objective"]]
        assert aggregate["completed_sequences"] == aggregate["expected_sequences"] == 2
        assert aggregate["tracking_apd3d_all"] == pytest.approx(np.mean([r["tracking_metrics"]["apd3d_all"] for r in children]))
        assert aggregate["reconstruction_epe_m"] == pytest.approx(np.mean([r["reconstruction_metrics"]["epe_m"] for r in children]))
        assert aggregate["attack_seconds_sum"] == pytest.approx(sum(r["elapsed_seconds"] for r in children))
    with (study.run_dir / "sequence_metrics.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 12
    assert all("tracking_apd_drop_pp" in row and "reconstruction_epe_increase_m" in row for row in csv_rows)
    before_load_count = len(study.loaded)
    study.monkeypatch.setattr(runner, "component_attack_video", lambda *args, **kwargs: pytest.fail("completed resume reran an attack"))
    resumed = run(study, resume=True)
    assert resumed["complete"]
    assert len(study.loaded) == before_load_count
    assert snapshots == {path: path.read_bytes() for path in snapshots}


def test_initial_random_rgb_and_seed_are_matched_across_losses(study):
    actual = runner.component_attack_video
    starts = {}

    def capture(pair_fn, rgb, targets, objective, config=None, **kwargs):
        visits = []

        def spy(video, t):
            if t == 0:
                visits.append(video.detach().clone())
            return pair_fn(video, t)

        result = actual(spy, rgb, targets, objective, config, **kwargs)
        if config.method == "pgd":
            clip = pair_fn.__self__.targets.metadata["clip"]
            starts[(clip, objective)] = visits[1]
        return result

    study.monkeypatch.setattr(runner, "component_attack_video", capture)
    run(study)
    for entry in study.manifest["entries"]:
        values = [starts[(entry["sequence"], objective)] for objective in study.config["objectives"]]
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], rtol=0, atol=0)
        child_records = [r for r in records(study).values() if r["sequence"] == entry["sequence"] and r["objective"] != "clean"]
        assert len({r["seed"] for r in child_records}) == 1
        assert all(r["seed"] == runner.stable_seed(study.config["seed"], entry["dataset"], entry["sequence"], "pgd", 4) for r in child_records)


def test_resume_refuses_changed_config_source_or_artifact(study):
    run(study)
    study.config["steps"] += 1
    with pytest.raises(ValueError, match="identical"):
        run(study, resume=True)
    study.config["steps"] -= 1
    artifact = condition_path(study) / "rgb_float32.npy"
    with artifact.open("ab") as output:
        output.write(b"changed bytes")
    with pytest.raises((ValueError, RuntimeError), match="artifact|failures"):
        run(study, resume=True)


def test_changed_dataset_hash_fails_without_silent_exclusion(study):
    study.config["objectives"] = ["tracking_mse"]
    source = Path(study.manifest["entries"][0]["path"])
    source.write_bytes(b"changed canonical source")
    with pytest.raises(RuntimeError, match="failures"):
        run(study, fail_fast=False)
    summary = json.loads((study.run_dir / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["completed_conditions"] == 2
    assert summary["expected_conditions"] == 4
    assert summary["failed_attempts"] == 1
    assert "Dataset hash mismatch" in summary["failures"][0]["error"]


def test_saved_rgb_replay_failure_does_not_publish_result(study):
    study.config["objectives"] = ["tracking_3d"]
    study.manifest["entries"] = study.manifest["entries"][:1]
    actual = ActualNativeForward.predict_with_maps
    calls = 0

    def changing_capture(self, rgb):
        nonlocal calls
        calls += 1
        compact, recon, conf = actual(self, rgb)
        if calls == 4:  # two clean captures, then PGD capture and attacked RGB replay
            compact["tracks"] = compact["tracks"] + 0.01
        return compact, recon, conf

    study.monkeypatch.setattr(ActualNativeForward, "predict_with_maps", changing_capture)
    with pytest.raises(AssertionError):
        run(study)
    assert (condition_path(study, "clean") / "result.json").is_file()
    assert not (condition_path(study) / "result.json").exists()
    summary = json.loads((study.run_dir / "summary.json").read_text())
    assert summary["completed_conditions"] == 1
    assert not summary["complete"]


def test_resume_repairs_missing_attack_without_replacing_completed_conditions(study):
    run(study)
    missing_path = condition_path(study, "confidence") / "result.json"
    missing_path.unlink()
    before = {path: path.read_bytes() for path in records(study)}
    actual = runner.component_attack_video
    invocations = []

    def capture(*args, **kwargs):
        invocations.append(args[3])
        return actual(*args, **kwargs)

    study.monkeypatch.setattr(runner, "component_attack_video", capture)
    resumed = run(study, resume=True)
    assert resumed["complete"]
    assert invocations == ["confidence"]
    assert before == {path: path.read_bytes() for path in before}


def test_resume_repairs_missing_clean_without_rerunning_attacks(study):
    run(study)
    missing = condition_path(study, "clean") / "result.json"
    missing.unlink()
    attacks = {path: path.read_bytes() for path in records(study)}
    actual = runner.component_attack_video
    invoked = []

    def capture(*args, **kwargs):
        invoked.append((args[3], args[4].method))
        return actual(*args, **kwargs)

    study.monkeypatch.setattr(runner, "component_attack_video", capture)
    resumed = run(study, resume=True)
    assert resumed["complete"]
    assert invoked == [("tracking_mse", "clean")]
    assert attacks == {path: path.read_bytes() for path in attacks}


def test_failed_parity_is_rechecked_on_resume(study):
    study.config["objectives"] = ["tracking_3d"]
    study.manifest["entries"] = study.manifest["entries"][:1]
    SyntheticBase.parity_failures = 1
    with pytest.raises(AssertionError, match="parity failed"):
        run(study)
    assert SyntheticBase.parity_calls == 1
    summary = run(study, resume=True)
    assert summary["complete"]
    assert SyntheticBase.parity_calls == 2
    path = study.run_dir / "sequences/po_mini/clip_a/official_clean_parity.json"
    assert json.loads(path.read_text())["passed"]


@pytest.mark.parametrize("geometry", ["targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy", "sequence.json"])
def test_canonical_geometry_tampering_is_detected_by_audit(study, geometry):
    study.config["objectives"] = ["tracking_mse"]
    study.manifest["entries"] = study.manifest["entries"][:1]
    run(study)
    artifact = study.run_dir / "sequences/po_mini/clip_a" / geometry
    with artifact.open("ab") as output:
        output.write(b"modified canonical geometry")
    with pytest.raises((ValueError, RuntimeError)):
        runner.summarize_component_study(study.run_dir)


def test_required_artifact_manifest_cannot_be_removed_to_bypass_hash_audit(study):
    study.config["objectives"] = ["tracking_mse"]
    study.manifest["entries"] = study.manifest["entries"][:1]
    run(study)
    path = condition_path(study, "tracking_mse") / "result.json"
    record = json.loads(path.read_text())
    record["artifact_sha256"] = {}
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, RuntimeError)):
        runner.summarize_component_study(study.run_dir)


def test_condition_location_must_match_its_canonical_identity(study):
    study.config["objectives"] = ["tracking_mse"]
    study.manifest["entries"] = study.manifest["entries"][:1]
    run(study)
    source = condition_path(study, "tracking_mse")
    destination = condition_path(study, "foreign_folder")
    source.resolve().relative_to(study.run_dir.resolve())
    destination.resolve().relative_to(study.run_dir.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    with pytest.raises((ValueError, RuntimeError)):
        runner.summarize_component_study(study.run_dir)
