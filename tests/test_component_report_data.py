"""Synthetic scalar proofs only: no actual study, large arrays, or model runs.

Audit/old-bootstrap records here are deliberately labelled stubs; they test the
consumer's binding and arithmetic, not the independent auditor's array checks.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/component_report_data.py"
SPEC = importlib.util.spec_from_file_location("component_report_data_tests", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return sum(values) / len(values)


def make_fixture(root):
    entries = [{"dataset": d, "sequence": f"clip_{i}"} for i in range(4) for d in REPORT.DATASETS]
    config = {"objectives": list(REPORT.OBJECTIVES), "num_frames": 128, "steps": 20, "epsilon_255": 4,
              "restarts": 1, "seed": 20261002, "step_size_fraction": .25, "campaign_expected_clips": 8,
              "campaign_expected_frames": 128, "all_frames": True}
    manifest = {"entries": entries, "num_frames": 128, "campaign_expected_clips": 8,
                "campaign_expected_frames": 128, "all_frames": True}
    metadata = {"config": config, "manifest": manifest, "provenance": {"official_commit": REPORT.OFFICIAL_COMMIT},
                "frozen_weights": True, "tta": False}
    metadata["signature"] = hashlib.sha256(json.dumps({k: metadata[k] for k in ("config", "manifest", "provenance")},
                                                       sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write(root / "run.json", metadata)
    write(root / "campaign_execution.json", {"status": "complete", "synthetic_only": True})
    write(root / "resume_execution.json", {"status": "complete", "synthetic_only": True})
    records, pairs = [], []
    for clip in entries:
        index = int(clip["sequence"][-1])
        dataset_shift = 2 if clip["dataset"] == "ds_mini" else 0
        directory = root / "sequences" / clip["dataset"] / clip["sequence"]
        sequence = {"num_frames": 128, "requested_num_frames": 128, "original_frame_count": 128,
                    "all_frames_used": True, "used_frame_indices": list(range(128)), "source_frame_counts": {
                        field: 128 for field in ("images_jpeg_bytes", "tracks_XYZ", "visibility", "depth_map", "extrinsics_w2c")}}
        write(directory / "sequence.json", sequence)
        write(directory / "sequence_manifest.json", {"signature": metadata["signature"], "artifact_sha256": {
            "sequence.json": digest(directory / "sequence.json")}})
        baseline = None
        for oi, objective in enumerate(("clean", *REPORT.OBJECTIVES)):
            tracking_drop, recon_drop = oi + index, oi * 2 + index
            if objective == "tracking_3d":
                tracking_drop, recon_drop = 3 + 2 * index + dataset_shift, -1 + .5 * index
            if objective == "reconstruction_3d":
                tracking_drop, recon_drop = 8 - index, 6 + index + dataset_shift
            if objective == "confidence":
                tracking_drop, recon_drop = -3., -1.
            clean = objective == "clean"
            if clean:
                tracking_drop = recon_drop = 0.
            tracking_clean = 50. + index * 8 + dataset_shift
            recon_clean = 60. + index * 4
            record = {**clip, "objective": objective, "seed": 1000 + index, "signature": metadata["signature"],
                      "epsilon_255": 0 if clean else 4, "steps": 0 if clean else 20, "elapsed_seconds": 1. if clean else 20.,
                      "saved_input_verification": {"passed": True}, "selected_state": {"restart": -1 if clean else 0, "step": 0 if clean else 20},
                      "attack_loss": 1. if clean else 22., "tracking_metrics": {"num_frames": 128,
                          "apd3d_all": tracking_clean - tracking_drop, "epe_all_m": .2 + index * .02 + tracking_drop * .01},
                      "reconstruction_metrics": {"num_frames": 128, "apd3d": recon_clean - recon_drop,
                          "epe_m": .3 + index * .03 + recon_drop * .01},
                      "loss_terms": {"diagnostics": {field: .5 + oi * .1 for field in REPORT.DIAGNOSTIC_FIELDS.values()}}}
            condition = root / "conditions" / objective / clip["dataset"] / clip["sequence"]
            history = [{"restart": -1, "step": 0, "loss": 1.}]
            if not clean:
                history += [{"restart": 0, "step": step, "loss": 2. + step,
                             "gradient_l2_per_frame": [.1] * 128} for step in range(21)]
                history[-1].pop("gradient_l2_per_frame")
            write(condition / "history.json", history)
            record["artifact_sha256"] = {"history.json": digest(condition / "history.json")}
            write(condition / "result.json", record)
            records.append(record)
            if clean:
                baseline = record
                continue
            row = {key: record[key] for key in ("dataset", "sequence", "objective", "seed", "elapsed_seconds")}
            for field, (section, native, sign, _) in REPORT.CHANGE_FIELDS.items():
                cv, av = baseline[section][native], record[section][native]
                row.update({field: sign * (av - cv), field + "_clean": cv, field + "_attacked": av})
            for field, native in REPORT.DIAGNOSTIC_FIELDS.items():
                cv, av = baseline["loss_terms"]["diagnostics"][native], record["loss_terms"]["diagnostics"][native]
                row.update({field: av - cv, field + "_clean": cv, field + "_attacked": av})
            pairs.append(row)
    aggregates, summary_aggregates, cis = [], [], []
    for dataset in REPORT.GROUPS:
        for objective in ("clean", *REPORT.OBJECTIVES):
            children = [r for r in records if r["objective"] == objective and (dataset == "all" or r["dataset"] == dataset)]
            row = {"dataset": dataset, "objective": objective, "completed_sequences": len(children)}
            for _, (section, native, _, _) in REPORT.CHANGE_FIELDS.items():
                row[("tracking_" if section == "tracking_metrics" else "reconstruction_") + native] = mean([r[section][native] for r in children])
            summary_aggregates.append(row)
            if objective == "clean":
                continue
            selected = [r for r in pairs if r["objective"] == objective and (dataset == "all" or r["dataset"] == dataset)]
            row = {"dataset": dataset, "objective": objective, "paired_clips": len(selected), "expected_clips": len(selected)}
            for field in (*REPORT.CHANGE_FIELDS, *REPORT.DIAGNOSTIC_FIELDS):
                for suffix in ("", "_clean", "_attacked"):
                    row[field + suffix] = mean([r[field + suffix] for r in selected])
            aggregates.append(row)
            for metric in REPORT.METRICS:
                # Stub intervals intentionally unrelated to the fresh contrast
                # so a subtract-the-two-old-CIs bug cannot pass arithmetic QA.
                cis.append({"dataset": dataset, "objective": objective, "metric": metric, "estimate": row[metric],
                            "ci95": [row[metric] - 100., row[metric] + 100.],
                            "unit": REPORT.METRIC_UNITS[metric], "common_clips": len(selected)})
    write(root / "summary.json", {"complete": True, "expected_conditions": 48, "completed_conditions": 48,
                                 "missing_conditions": [], "failed_attempts": 0, "aggregate": summary_aggregates})
    sequence_rows = []
    for record in records:
        row = {key: record[key] for key in ("dataset", "sequence", "objective")}
        for _, (section, native, _, _) in REPORT.CHANGE_FIELDS.items():
            row[("tracking_" if section == "tracking_metrics" else "reconstruction_") + native] = record[section][native]
        sequence_rows.append(row)
    for name, rows in (("sequence_metrics.csv", sequence_rows), ("aggregate_metrics.csv", summary_aggregates)):
        with (root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = {"synthetic_only": True, "status": "passed", "passed": True, "full_campaign_completed": True,
             "all_frames_completed": True, "all_frames": True, "source_files_verified": True,
             "expected_conditions": 48, "completed_conditions_checked": 48, "full_campaign_expected_conditions": 48,
             "campaign_expected_clips": 8, "campaign_expected_frames": 128,
             "requested_scope": {"clips": 8, "frames": 128, "steps": 20}, "errors": [], "missing": [],
             "file_sha256": {"/workspace/run/" + p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}}
    write(root / "artifact_validation.json", audit)
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()
              and p.name not in ("campaign_execution.json", "resume_execution.json", "aggregate_metrics.csv")}
    analysis = {"schema_version": 1, "synthetic_only": True, "status": "complete", "scope": {"kind": "full_campaign",
                    "clips": 8, "dataset_counts": {"po_mini": 4, "ds_mini": 4}, "num_frames": 128,
                    "steps": 20, "epsilon_255": 4, "expected_conditions": 48, "all_frames": True, "objectives": list(REPORT.OBJECTIVES)},
                "all_five_campaign_objectives_completed": True, "all_raw_frames_campaign_completed": True,
                "analyzer_source_sha256": REPORT.ANALYZER_SHA256,
                "validation": {"completed_conditions": 48, "raw_frame_verified_clips": 8, "errors": [],
                               "warnings": [], "missing_conditions": [], "recorded_failed_attempts": 0},
                "artifact_audit": {"validated_for_scope": True}, "input_scalar_file_sha256": hashes,
                "paired_conditions": pairs, "paired_aggregates": aggregates, "bootstrap": {"common_clips": 8,
                    "common_dataset_counts": {"po_mini": 4, "ds_mini": 4}, "all_dataset_weights": {"po_mini": .5, "ds_mini": .5},
                    "samples": 2000, "seed": 20261004, "estimates": cis}}
    write(root / "analysis/study_analysis.json", analysis)
    return root


def freshen(root):
    """Re-bind synthetic stubs to test semantic checks beyond SHA rejection."""
    audit = read(root / "artifact_validation.json")
    for path in list(audit["file_sha256"]):
        if not path.startswith("/workspace/run/"):
            continue
        relative = path.removeprefix("/workspace/run/")
        audit["file_sha256"][path] = digest(root / relative)
    write(root / "artifact_validation.json", audit)
    analysis = read(root / "analysis/study_analysis.json")
    for path in list(analysis["input_scalar_file_sha256"]):
        analysis["input_scalar_file_sha256"][path] = digest(root / path)
    write(root / "analysis/study_analysis.json", analysis)


@pytest.fixture
def run(tmp_path):
    return make_fixture(tmp_path / "synthetic_full128_report")


def mutate_analysis(root, change):
    path = root / "analysis/study_analysis.json"
    value = read(path)
    change(value)
    write(path, value)


def test_complete_gate_is_read_only_and_keeps_manifest_order(run):
    before = {p.relative_to(run).as_posix(): digest(p) for p in run.rglob("*") if p.is_file()}
    study = REPORT.load_verified_study(run)
    assert study["complete_verified"]
    assert len(study["records"]) == 48
    assert study["identities"][:2] == [("po_mini", "clip_0"), ("ds_mini", "clip_0")]
    assert study["analysis_sha256"] == digest(run / "analysis/study_analysis.json")
    assert before == {p.relative_to(run).as_posix(): digest(p) for p in run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["campaign_execution.json", "resume_execution.json"])
def test_running_controller_never_permits_final_report(run, name):
    write(run / name, {"status": "running"})
    with pytest.raises(REPORT.ReportValidationError, match="not complete"):
        REPORT.load_verified_study(run)


@pytest.mark.parametrize("key,value", [("all_five_campaign_objectives_completed", False), ("all_raw_frames_campaign_completed", False),
                                      ("status", "incomplete"), ("status", "failed"), ("analyzer_source_sha256", "old-version")])
def test_analysis_completion_and_version_gates(run, key, value):
    mutate_analysis(run, lambda a: a.update({key: value}))
    with pytest.raises(REPORT.ReportValidationError):
        REPORT.load_verified_study(run)


@pytest.mark.parametrize("key,value", [("clips", 4), ("num_frames", 64), ("steps", 2), ("epsilon_255", 8),
                                      ("all_frames", False), ("expected_conditions", 24), ("kind", "explicit_limited_preflight")])
def test_wrong_scope_cannot_borrow_completion_flags(run, key, value):
    mutate_analysis(run, lambda a: a["scope"].update({key: value}))
    with pytest.raises(REPORT.ReportValidationError, match="scope"):
        REPORT.load_verified_study(run)


@pytest.mark.parametrize("key,value", [("source_files_verified", False), ("full_campaign_completed", False),
                                      ("all_frames_completed", False), ("completed_conditions_checked", 47),
                                      ("requested_scope", {"clips": 8, "frames": 64, "steps": 20})])
def test_audit_scope_is_independently_required(run, key, value):
    path = run / "artifact_validation.json"
    audit = read(path)
    audit[key] = value
    write(path, audit)
    freshen(run)
    with pytest.raises(REPORT.ReportValidationError, match="Audit gate"):
        REPORT.load_verified_study(run)


@pytest.mark.parametrize("relative", ["run.json", "summary.json", "artifact_validation.json", "sequence_metrics.csv", "aggregate_metrics.csv",
    "sequences/po_mini/clip_0/sequence.json", "conditions/tracking_3d/po_mini/clip_0/result.json", "conditions/tracking_3d/po_mini/clip_0/history.json"])
def test_stale_scalar_evidence_rejected_even_when_numbers_unchanged(run, relative):
    path = run / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(REPORT.ReportValidationError, match="SHA|bind|bound"):
        REPORT.load_verified_study(run)


@pytest.mark.parametrize("field", ["seed", "tracking_apd_drop_pp", "tracking_apd_drop_pp_clean", "reconstruction_epe_increase_m_attacked", "track_raw_conf_mean_change"])
def test_analysis_rows_crosschecked_with_original_results(run, field):
    mutate_analysis(run, lambda a: a["paired_conditions"][0].update({field: a["paired_conditions"][0][field] + 1}))
    with pytest.raises(REPORT.ReportValidationError, match="mismatch"):
        REPORT.load_verified_study(run)


def test_duplicate_foreign_and_unpaired_records_rejected(run):
    mutate_analysis(run, lambda a: a["paired_conditions"].append(a["paired_conditions"][0].copy()))
    with pytest.raises(REPORT.ReportValidationError, match="Duplicate"):
        REPORT.load_verified_study(run)


def test_wrong_seed_in_rebound_original_result_still_rejected(run):
    path = run / "conditions/reconstruction_3d/po_mini/clip_0/result.json"
    record = read(path)
    record["seed"] += 1
    write(path, record)
    freshen(run)
    with pytest.raises(REPORT.ReportValidationError, match="share.*seed"):
        REPORT.load_verified_study(run)


def test_prefix_metadata_rejected_after_binding_is_refreshed(run):
    path = run / "sequences/po_mini/clip_0/sequence.json"
    value = read(path)
    value["used_frame_indices"] = list(range(64))
    write(path, value)
    inventory_path = path.with_name("sequence_manifest.json")
    inventory = read(inventory_path)
    inventory["artifact_sha256"]["sequence.json"] = digest(path)
    write(inventory_path, inventory)
    freshen(run)
    with pytest.raises(REPORT.ReportValidationError, match="original frames"):
        REPORT.load_verified_study(run)


def test_hash_path_traversal_and_duplicate_json_keys_rejected(run):
    mutate_analysis(run, lambda a: a["input_scalar_file_sha256"].update({"../outside.json": "0" * 64}))
    with pytest.raises(REPORT.ReportValidationError, match="Unsafe"):
        REPORT.load_verified_study(run)
    path = run / "analysis/study_analysis.json"
    path.write_text('{"scope":{},"scope":{}}', encoding="utf-8")
    with pytest.raises(REPORT.ReportValidationError, match="Duplicate JSON"):
        REPORT.load_verified_study(run)


def test_ambiguous_audit_suffix_cannot_bind_two_files(run):
    path = run / "artifact_validation.json"
    audit = read(path)
    audit["file_sha256"]["/different/run/run.json"] = digest(run / "run.json")
    write(path, audit)
    freshen(run)
    with pytest.raises(REPORT.ReportValidationError, match="uniquely"):
        REPORT.load_verified_study(run)


def test_geometry_contrast_uses_raw_paired_differences_and_shared_indices(run):
    study = REPORT.load_verified_study(run)
    coupling = REPORT.compute_task_coupling(study, samples=2000, seed=17)
    contrast = coupling["geometry_contrasts"]
    assert contrast["direction"] == "tracking_3d - reconstruction_3d"
    assert len(contrast["clip_differences"]) == 8
    rng = np.random.default_rng(17)
    draws = {d: rng.integers(0, 4, size=(2000, 4)) for d in REPORT.DATASETS}
    pair_index = {(p["dataset"], p["sequence"], p["objective"]): p for p in study["analysis"]["paired_conditions"]}
    distributions = {}
    for dataset in REPORT.DATASETS:
        values = np.array([[pair_index[dataset, f"clip_{i}", "tracking_3d"][metric] - pair_index[dataset, f"clip_{i}", "reconstruction_3d"][metric]
                            for metric in REPORT.METRICS] for i in range(4)])
        assert contrast["bootstrap"]["resample_indices"][dataset] == draws[dataset].tolist()
        distributions[dataset] = values[draws[dataset]].mean(axis=1)
    distributions["all"] = (distributions["po_mini"] + distributions["ds_mini"]) / 2
    for row in contrast["estimates"]:
        column = REPORT.METRICS.index(row["metric"])
        assert row["ci95"] == pytest.approx(np.quantile(distributions[row["dataset"]][:, column], [.025, .975]))
        assert row["paired_clips"] == (8 if row["dataset"] == "all" else 4)
        assert row["ci95"][1] - row["ci95"][0] < 200  # old fake intervals have width200; subtraction is not CI
    assert REPORT.compute_task_coupling(study, samples=2000, seed=17) == coupling
    json.dumps(coupling, allow_nan=False)


def test_ratios_negative_improvements_and_no_mean_of_ratios_confusion(run):
    study = REPORT.load_verified_study(run)
    coupling = REPORT.compute_task_coupling(study, samples=20)
    row = next(r for r in coupling["impact_matrix"] if r["dataset"] == "all" and r["objective"] == "confidence")
    assert row["tracking_apd_relative_drop_pct"] == pytest.approx(100 * row["tracking_apd_drop_pp"] / row["tracking_apd_drop_pp_clean"])
    assert row["tracking_apd_relative_drop_pct"] < 0
    clip_ratio_mean = mean([p["x"] for p in coupling["scatter"]["points"] if p["objective"] == "confidence"])
    assert row["tracking_apd_relative_drop_pct"] != pytest.approx(clip_ratio_mean)
    assert all(p["x"] < 0 and p["y"] < 0 for p in coupling["scatter"]["points"] if p["objective"] == "confidence")
    assert REPORT.safe_ratio(-1., 2.) == -.5
    for denominator in (0., 1e-13, None, float("nan"), float("inf")):
        assert REPORT.safe_ratio(1., denominator) is None


def test_scatter_na_points_do_not_become_zero_or_create_a_connection(run):
    study = REPORT.load_verified_study(run)
    for pair in study["analysis"]["paired_conditions"]:
        if (pair["dataset"], pair["sequence"]) == ("po_mini", "clip_0"):
            pair["tracking_apd_drop_pp_clean"] = 0.
    coupling = REPORT.compute_task_coupling(study, samples=20)
    assert len(coupling["scatter"]["excluded"]) == 5
    assert len(coupling["scatter"]["points"]) == 35
    assert len(coupling["scatter"]["connections"]) == 7
    assert all(p["x"] is None for p in coupling["scatter"]["excluded"])
    json.dumps(coupling, allow_nan=False)


def test_partial_diagnostic_permission_cannot_authorize_final_analysis(run):
    write(run / "resume_execution.json", {"status": "running"})
    study = REPORT.load_verified_study(run, require_complete=False)
    assert not study["complete_verified"]
    with pytest.raises(REPORT.ReportValidationError, match="strict completion"):
        REPORT.compute_task_coupling(study)


def test_all_nine_standalone_plots_are_written_without_changing_evidence(run, tmp_path):
    study = REPORT.load_verified_study(run)
    before = digest(run / "analysis/study_analysis.json")
    coupling = REPORT.compute_task_coupling(study, samples=20)
    plots = REPORT.make_coupling_plots(coupling, tmp_path / "plots")
    assert len(plots) == len(set(plots)) == 9
    for filename in plots:
        path = tmp_path / "plots" / filename
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000
    assert digest(run / "analysis/study_analysis.json") == before
