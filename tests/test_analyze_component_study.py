"""CPU postprocessing tests; independent-audit results are synthetic stubs.

These fixtures test analysis gating, not the auditor's large-array checks. They
never read an actual campaign or load a model, CUDA, or Docker.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_component_study.py"
SPEC = importlib.util.spec_from_file_location("component_analyzer_test", SCRIPT)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def scalar_run(root, *, clips=8, frames=128, steps=20, limited=False):
    config = {"objectives": list(ANALYZER.OBJECTIVES), "num_frames": frames,
              "steps": steps, "epsilon_255": 4, "restarts": 1, "seed": 20261002}
    manifest = {"num_frames": frames, "entries": [
        {"dataset": dataset, "sequence": f"clip_{index}"}
        for dataset in ANALYZER.DATASETS for index in range(clips // 2)]}
    if frames == 128:
        for source in (config, manifest):
            source.update(campaign_expected_clips=clips, campaign_expected_frames=128, all_frames=True)
    if limited:
        manifest["purpose"] = "explicit limited CPU preflight"
    metadata = {"config": config, "manifest": manifest, "provenance": {"official_commit": ANALYZER.COMMIT}}
    metadata["signature"] = ANALYZER.fingerprint(metadata)
    write_json(root / "run.json", metadata)
    records = []
    for entry in manifest["entries"]:
        if frames == 128:
            directory = root / "sequences" / entry["dataset"] / entry["sequence"]
            write_json(directory / "sequence.json", {"num_frames": 128, "requested_num_frames": 128,
                                                      "original_frame_count": 128})
            write_json(directory / "sequence_manifest.json", {"signature": metadata["signature"], "artifact_sha256": {
                "sequence.json": ANALYZER.sha_file(directory / "sequence.json")}})
        for index, objective in enumerate(("clean", *ANALYZER.OBJECTIVES)):
            clean = objective == "clean"
            loss = 1. if clean else 1. + (steps + 1) * .01
            record = {**entry, "objective": objective, "signature": metadata["signature"],
                      "saved_input_verification": {"passed": True}, "elapsed_seconds": 1. if clean else 10.,
                      "seed": 101, "steps": 0 if clean else steps, "epsilon_255": 0 if clean else 4,
                      "selected_state": {"restart": -1 if clean else 0, "step": 0 if clean else steps},
                      "attack_loss": loss,
                      "tracking_metrics": {"num_frames": frames, "apd3d_all": 70. - index * 3, "epe_all_m": .3 + index * .2},
                      "reconstruction_metrics": {"num_frames": frames, "apd3d": 80. - index * 2, "epe_m": .2 + index * .1},
                      "loss_terms": {"diagnostics": {"tracking_l21": .2 + index * .1,
                          "reconstruction_l21": .4 + index * .1, "track_conf_mean": 2. + index * .1,
                          "track_raw_conf_mean": 2. + index * .1, "reconstruction_conf_mean": 3. + index * .1}}}
            history = [{"restart": -1, "step": 0, "loss": 1.}]
            if not clean:
                history += [{"restart": 0, "step": step, "loss": 1. + (step + 1) * .01,
                             "gradient_l2_per_frame": [.1] * frames} for step in range(steps)]
                history += [{"restart": 0, "step": steps, "loss": loss}]
            directory = root / "conditions" / objective / entry["dataset"] / entry["sequence"]
            write_json(directory / "result.json", record)
            write_json(directory / "history.json", history)
            records.append(record)
    aggregate = []
    for dataset in (*ANALYZER.DATASETS, "all"):
        for objective in ("clean", *ANALYZER.OBJECTIVES):
            children = [r for r in records if r["objective"] == objective and (dataset == "all" or r["dataset"] == dataset)]
            item = {"dataset": dataset, "objective": objective, "completed_sequences": len(children),
                    "attack_seconds_sum": sum(r["elapsed_seconds"] for r in children)}
            for section, fields in (("tracking_metrics", ("apd3d_all", "epe_all_m")),
                                    ("reconstruction_metrics", ("apd3d", "epe_m"))):
                prefix = "tracking_" if section == "tracking_metrics" else "reconstruction_"
                item.update({prefix + field: sum(r[section][field] for r in children) / len(children) for field in fields})
            aggregate.append(item)
    write_json(root / "summary.json", {"completed_conditions": len(records), "expected_conditions": len(records),
        "complete": True, "aggregate": aggregate, "failed_attempts": 0})
    rows = []
    for record in records:
        row = {key: record[key] for key in ("dataset", "sequence", "objective")}
        row.update({"tracking_" + key: record["tracking_metrics"][key] for key in ("apd3d_all", "epe_all_m")})
        row.update({"reconstruction_" + key: record["reconstruction_metrics"][key] for key in ("apd3d", "epe_m")})
        rows.append(row)
    with (root / "sequence_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def audit_stub(root, metadata):
    scope = ANALYZER.scope_of(metadata)
    full = scope["kind"] == "full_campaign"
    return {"status": "passed", "passed": True, "errors": [], "missing": [],
            "full_campaign_completed": full, "all_frames_completed": full and scope["all_frames"],
            "all_frames": scope["all_frames"], "source_files_verified": True,
            "campaign_expected_clips": scope["campaign_expected_clips"],
            "campaign_expected_frames": scope["campaign_expected_frames"],
            "full_campaign_expected_conditions": scope["full_campaign_expected_conditions"],
            "expected_conditions": scope["expected_conditions"], "completed_conditions_checked": scope["expected_conditions"],
            "requested_scope": {"clips": scope["clips"], "frames": scope["num_frames"], "steps": scope["steps"]},
            "file_sha256": {str(path): ANALYZER.sha_file(path) for path in root.rglob("*") if path.is_file()}}


@pytest.fixture
def complete_run(tmp_path, monkeypatch):
    # Plotting is unrelated to audit completion; validate JSON and actual MD.
    monkeypatch.setattr(ANALYZER, "make_plots", lambda *args: [])
    root = tmp_path / "synthetic_run"
    metadata = scalar_run(root)
    audit = audit_stub(root, metadata)
    return root, metadata, audit


def analyze_fixture(root):
    return ANALYZER.analyze(root, bootstrap_samples=20)


def assert_no_completion(report):
    assert not report["all_five_campaign_objectives_completed"]
    assert not report["all_raw_frames_campaign_completed"]


def test_full128_requires_bound_independent_audit_and_records_scientific_scope(complete_run):
    root, _, audit = complete_run
    write_json(root / "artifact_validation.json", audit)
    before = {path: ANALYZER.sha_file(path) for path in root.rglob("*") if path.is_file()}
    report = analyze_fixture(root)
    assert report["status"] == "complete"
    assert report["all_five_campaign_objectives_completed"] and report["all_raw_frames_campaign_completed"]
    assert report["artifact_audit"]["validated_for_scope"]
    assert report["validation"]["completed_conditions"] == 48
    assert report["validation"]["raw_frame_verified_clips"] == report["bootstrap"]["common_clips"] == 8
    assert report["bootstrap"]["all_dataset_weights"] == {"po_mini": .5, "ds_mini": .5}
    science = report["scientific_interpretation"]
    assert science["selected_clips"] == 8 and science["dataset_counts"] == {"po_mini": 4, "ds_mini": 4}
    assert science["scene_independence_verified"] is False and science["scene_cluster_adjustment"] is False
    assert "loss-deletion retraining effects" in science["excluded_claims"]
    assert "causal importance of training weights" in science["excluded_claims"]
    assert "global worst-case attack" in science["excluded_claims"]
    markdown = (root / "analysis/study_report.md").read_text(encoding="utf-8")
    for phrase in ("고정 모델", "선택된 8개 클립", "손실 삭제 후 재훈련", "인과적 중요도", "전역 최악 공격",
                   "Scene 독립성을 검증하지 않았고 scene cluster 보정을 하지 않았습니다"):
        assert phrase in markdown
    assert all(ANALYZER.sha_file(path) == digest for path, digest in before.items())


def test_missing_full128_audit_withholds_complete_status(complete_run):
    root, _, _ = complete_run
    report = analyze_fixture(root)
    assert report["status"] == "incomplete"
    assert report["artifact_audit"]["status"] == "not_available"
    assert_no_completion(report)
    assert "완료·순위를 주장하지 않습니다" in (root / "analysis/study_report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("keys,value,expected_status", [
    (("status",), "failed", "failed"), (("status",), "incomplete", "incomplete"),
    (("status",), "unknown", "failed"), (("passed",), False, "failed"), (("passed",), 1, "failed"),
    (("full_campaign_completed",), False, "failed"), (("all_frames_completed",), False, "failed"),
    (("all_frames_completed",), 1, "failed"), (("all_frames",), False, "failed"),
    (("campaign_expected_clips",), 10, "failed"), (("campaign_expected_frames",), 64, "failed"),
    (("requested_scope", "clips"), 10, "failed"), (("requested_scope", "frames"), 64, "failed"),
    (("requested_scope", "steps"), 10, "failed"), (("expected_conditions",), 47, "failed"),
    (("completed_conditions_checked",), 47, "failed"), (("full_campaign_expected_conditions",), 84, "failed"),
    (("source_files_verified",), False, "failed"), (("errors",), ["contradictory error"], "failed"),
    (("missing",), ["missing evidence"], "failed"), (("file_sha256",), {}, "failed"),
])
def test_invalid_or_incomplete_full128_audit_never_claims_completion(complete_run, keys, value, expected_status):
    root, _, audit = complete_run
    target = audit
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    write_json(root / "artifact_validation.json", audit)
    report = analyze_fixture(root)
    assert report["status"] == expected_status
    assert_no_completion(report)


@pytest.mark.parametrize("name", ["run.json", "summary.json", "conditions/tracking_mse/po_mini/clip_0/result.json"])
def test_stale_audit_does_not_bind_current_scalar_evidence(complete_run, name):
    root, _, audit = complete_run
    audit["file_sha256"][str(root / name)] = "0" * 64
    write_json(root / "artifact_validation.json", audit)
    report = analyze_fixture(root)
    assert report["status"] == "failed"
    assert any(name in error.replace("\\", "/") and "audit SHA" in error for error in report["validation"]["errors"])
    assert_no_completion(report)


def test_container_audit_hash_paths_are_portable_to_host(complete_run):
    root, _, audit = complete_run
    audit["file_sha256"] = {"/workspace/runs/fixture/" + Path(path).relative_to(root).as_posix(): digest
                             for path, digest in audit["file_sha256"].items()}
    write_json(root / "artifact_validation.json", audit)
    assert analyze_fixture(root)["all_raw_frames_campaign_completed"]


@pytest.mark.parametrize("content", ["{bad JSON", "[]"])
def test_unreadable_audit_is_explicit_failed_analysis(complete_run, content):
    root, _, _ = complete_run
    (root / "artifact_validation.json").write_text(content, encoding="utf-8")
    report = analyze_fixture(root)
    assert report["status"] == "failed" and report["artifact_audit"]["status"] == "invalid"
    assert_no_completion(report)


@pytest.mark.parametrize("key", ["passed", "full_campaign_completed", "all_frames_completed", "campaign_expected_frames", "requested_scope"])
def test_missing_mandatory_full128_audit_evidence_withholds_completion(complete_run, key):
    root, _, audit = complete_run
    del audit[key]
    write_json(root / "artifact_validation.json", audit)
    report = analyze_fixture(root)
    assert report["status"] == "failed"
    assert_no_completion(report)


@pytest.mark.parametrize("key", ["num_frames", "requested_num_frames", "original_frame_count"])
def test_saved128_release_metadata_is_required_even_with_passed_audit(complete_run, key):
    root, _, audit = complete_run
    path = root / "sequences/po_mini/clip_0/sequence.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[key] = 64
    write_json(path, metadata)
    inventory_path = path.parent / "sequence_manifest.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["artifact_sha256"]["sequence.json"] = ANALYZER.sha_file(path)
    write_json(inventory_path, inventory)
    for changed in (path, inventory_path):
        audit["file_sha256"][str(changed)] = ANALYZER.sha_file(changed)
    write_json(root / "artifact_validation.json", audit)
    report = analyze_fixture(root)
    assert report["status"] == "failed" and report["validation"]["raw_frame_verified_clips"] == 7
    assert_no_completion(report)


def test_partial_full128_records_remain_incomplete_even_with_old_passed_audit(complete_run):
    root, _, audit = complete_run
    write_json(root / "artifact_validation.json", audit)
    (root / "conditions/tracking_mse/po_mini/clip_0/result.json").unlink()
    report = analyze_fixture(root)
    assert report["status"] == "incomplete" and report["validation"]["completed_conditions"] == 47
    assert_no_completion(report)


def test_historical64_audit_need_not_contain_later_all128_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(ANALYZER, "make_plots", lambda *args: [])
    root = tmp_path / "legacy_fixture"
    metadata = scalar_run(root, clips=14, frames=64)
    audit = audit_stub(root, metadata)
    for key in ("all_frames", "all_frames_completed", "campaign_expected_frames", "campaign_expected_clips", "full_campaign_expected_conditions"):
        del audit[key]
    write_json(root / "artifact_validation.json", audit)
    report = analyze_fixture(root)
    assert report["status"] == "complete" and report["all_five_campaign_objectives_completed"]
    assert not report["all_raw_frames_campaign_completed"] and not report["raw_frame_evidence"]


@pytest.mark.parametrize("with_audit", [False, True])
def test_explicit_limited_prefix_does_not_require_full128_evidence(tmp_path, monkeypatch, with_audit):
    monkeypatch.setattr(ANALYZER, "make_plots", lambda *args: [])
    root = tmp_path / "limited_fixture"
    metadata = scalar_run(root, clips=2, frames=2, steps=1, limited=True)
    if with_audit:
        write_json(root / "artifact_validation.json", audit_stub(root, metadata))
    report = analyze_fixture(root)
    assert report["status"] == "complete" and report["scope"]["kind"] == "explicit_limited_preflight"
    assert_no_completion(report)
    assert not report["raw_frame_evidence"]
