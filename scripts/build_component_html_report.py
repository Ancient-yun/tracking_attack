"""Build a portable Korean report from a completed, verified component study.

This postprocessor never loads a model or changes experiment artifacts.
The final HTML is published only after the full study and rendered assets pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys

from component_report_data import (OBJECTIVES, ReportValidationError,
                                   compute_task_coupling, load_verified_study,
                                   make_coupling_plots, sha_file)
from component_report_interpretation import build_interpretation
from render_component_comparisons import summarize_normalization, summarize_tracking_timeline


METRICS = ("tracking_apd_drop_pp", "reconstruction_apd_drop_pp",
           "tracking_epe_increase_m", "reconstruction_epe_increase_m")
ORIGINAL_PLOTS = ("component_changes_heatmap.png", "paired_bootstrap_ci.png",
                  "native_l21_changes.png")
KST = timezone(timedelta(hours=9))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def script_json(value):
    """Embed data without permitting a string to terminate the JSON script."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace(
        "\u2028", "\\u2028").replace("\u2029", "\\u2029")


def write_atomic(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(contents.encode("utf-8"))
    os.replace(temporary, path)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(numerator, denominator, multiplier=1):
    if not finite(numerator) or not finite(denominator) or denominator <= 1e-12:
        return None
    result = multiplier * numerator / denominator
    return result if finite(result) else None


def checked_asset(root, relative):
    """An asset must be a real file inside the supplied asset directory."""
    root = Path(root).resolve()
    relative = str(relative).replace("\\", "/")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ReportValidationError(f"Missing or unsafe report asset: {relative}")
    return path


def validate_visualizations(assets_dir, study):
    assets_dir = Path(assets_dir).resolve()
    identities = study["identities"]
    run_dir = Path(study["run_dir"])
    manifest_path = assets_dir / "visualization_manifest.json"
    if not manifest_path.is_file():
        raise ReportValidationError("Verified visualization_manifest.json is required")
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "complete" or manifest.get("errors")
            or manifest.get("cpu_only") is not True or manifest.get("new_model_inference") is not False):
        raise ReportValidationError("Qualitative rendering is incomplete or failed")
    if (manifest.get("run_id") != run_dir.name
            or manifest.get("run_signature") != study["metadata"]["signature"]
            or manifest.get("scope") != study["analysis"]["scope"]
            or manifest.get("frame_count") != 128
            or manifest.get("analysis_json_sha256") != sha_file(Path(study["analysis_dir"]) / "study_analysis.json")
            or manifest.get("artifact_validation_sha256") != sha_file(run_dir / "artifact_validation.json")
            or manifest.get("input_scalar_file_sha256") != study["input_sha256"]):
        raise ReportValidationError("Visualization evidence belongs to a different or changed study")
    renderer_sources = ("render_component_comparisons.py", "component_projection.py", "component_report_data.py",
                        "render_pgd_comparisons.py", "render_tracking_comparisons.py", "tracking_projection.py")
    for name in renderer_sources:
        source = Path(__file__).with_name(name)
        if manifest.get("renderer_source_sha256", {}).get(name) != sha_file(source):
            raise ReportValidationError(f"Visualization renderer source differs: {name}")

    def condition_sources(evidence, dataset, sequence, objective):
        record = study["records"][(dataset, sequence, objective)]
        directory = run_dir / "conditions" / objective / dataset / sequence
        expected = {name: record["artifact_sha256"][name] for name in (
            "tracks.npz", "rgb_float32.npy", "delta_float32.npy", "reconstruction_float32.npy",
            "reconstruction_confidence_float32.npy", "components.npz", "history.json")}
        expected["result.json"] = sha_file(directory / "result.json")
        for name, digest in expected.items():
            if evidence.get(name, {}).get("sha256") != digest:
                raise ReportValidationError(f"Visualization source SHA differs: {dataset}/{sequence}/{objective}/{name}")

    def sequence_sources(evidence, dataset, sequence):
        directory = run_dir / "sequences" / dataset / sequence
        inventory = read_json(directory / "sequence_manifest.json")["artifact_sha256"]
        entry = next(e for e in study["metadata"]["manifest"]["entries"]
                     if (e["dataset"], e["sequence"]) == (dataset, sequence))
        expected = {name: inventory[name] for name in (
            "targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy")}
        expected["sequence.json"] = sha_file(directory / "sequence.json")
        expected["source_npz"] = entry["sha256"]
        for name, digest in expected.items():
            if evidence.get(name, {}).get("sha256") != digest:
                raise ReportValidationError(f"Visualization sequence source SHA differs: {dataset}/{sequence}/{name}")

    def assets_inventory(assets):
        inventory = {}
        for asset in assets:
            path = checked_asset(assets_dir, asset["path"])
            actual = sha_file(path)
            if actual != asset.get("sha256") or path.stat().st_size != asset.get("bytes"):
                raise ReportValidationError(f"Rendered asset hash or size differs: {path.name}")
            if path.name in inventory:
                raise ReportValidationError(f"Duplicate asset name: {path.name}")
            inventory[path.name] = {**asset, "url": "assets/" + path.relative_to(assets_dir).as_posix()}
        return inventory
    expected = {(d, s, o) for d, s in identities for o in OBJECTIVES}
    indexed = {}
    clean_baselines = {}
    media = []
    for clip in manifest.get("clips", []):
        key = (clip["dataset"], clip["sequence"], clip["objective"])
        if key in indexed or key not in expected or clip.get("frame_count") != 128:
            raise ReportValidationError(f"Unexpected/duplicate rendered condition: {key}")
        indexed[key] = clip
        sources = clip.get("sources", {})
        sequence_sources(sources.get("sequence", {}), *key[:2])
        condition_sources(sources.get("conditions", {}).get("clean", {}), *key[:2], "clean")
        condition_sources(sources.get("conditions", {}).get("attack", {}), *key)
        if (clip.get("status") != "complete" or clip.get("source_frame_indices") != list(range(128))
                or clip.get("preview_frame_indices") != [0, 64, 127]
                or clip.get("selected_state") != study["records"][key]["selected_state"]
                or clip.get("projection_checks", {}).get("passed") is not True
                or clip.get("reconstruction_checks", {}).get("passed") is not True):
            raise ReportValidationError(f"Rendered condition checks incomplete: {key}")
        normalization = clip.get("normalization_summary", {})
        values = normalization.get("per_frame", {})
        try:
            recomputed_norm = summarize_normalization(values.get("clean"), values.get("attack"), values.get("fixed_gt"))
        except (ValueError, TypeError) as exc:
            raise ReportValidationError(f"Invalid full-frame normalization summary: {key}: {exc}") from exc
        if normalization != recomputed_norm:
            raise ReportValidationError(f"Normalization descriptive statistics changed: {key}")
        clean_record, attack_record = study["records"][(*key[:2], "clean")], study["records"][key]
        timeline = clip.get("tracking_timeline_summary", {})
        if not (timeline.get("query_count") == clean_record["tracking_metrics"].get("num_queries")
                == attack_record["tracking_metrics"].get("num_queries")):
            raise ReportValidationError(f"Temporal summary does not cover all saved evaluation queries: {key}")
        try:
            recomputed_timeline = summarize_tracking_timeline(
                clip.get("clean_epe_per_frame_m"), clip.get("attack_epe_per_frame_m"),
                clean_record["tracking_metrics"]["epe_all_m"], attack_record["tracking_metrics"]["epe_all_m"],
                timeline.get("query_count"))
        except (ValueError, TypeError) as exc:
            raise ReportValidationError(f"Invalid full-frame temporal summary: {key}: {exc}") from exc
        if timeline != recomputed_timeline:
            raise ReportValidationError(f"Tracking temporal descriptive statistics changed: {key}")
        baseline = (values["clean"], values["fixed_gt"], clip["clean_epe_per_frame_m"], timeline["query_count"])
        previous = clean_baselines.setdefault(key[:2], baseline)
        if previous != baseline:
            raise ReportValidationError(f"Shared clean/GT temporal baseline differs across attack objectives: {key}")
        recon = clip["reconstruction_checks"].get("conditions", {})
        if set(recon) != {"clean", *OBJECTIVES} or any(recon[o].get("passed") is not True for o in recon):
            raise ReportValidationError(f"Reconstruction checks must cover clean and all five attacks: {key}")
        for objective, reproduced in recon.items():
            recorded = study["records"][(*key[:2], objective)]["reconstruction_metrics"]
            if any(not finite(reproduced.get(metric)) or not math.isclose(reproduced[metric], recorded[metric],
                       rel_tol=1e-6, abs_tol=1e-6) for metric in ("epe_m", "apd3d")):
                raise ReportValidationError(f"Reconstructed metrics differ from saved result: {key}/{objective}")
        inventory = assets_inventory(clip.get("assets", []))
        for frame in (0, 64, 127):
            for prefix in ("rgb", "tracking", "reconstruction"):
                if f"{prefix}_frame_{frame:03d}.png" not in inventory:
                    raise ReportValidationError(f"Missing {prefix} still {frame} for {key}")
        for filename in ("pgd_history.png", "normalization.png", "rgb_comparison.mp4", "tracking_comparison.mp4"):
            if filename not in inventory:
                raise ReportValidationError(f"Missing {filename} for {key}")
        for filename, asset in inventory.items():
            if asset["path"].replace("\\", "/") != f"comparisons/{key[2]}/{key[0]}/{key[1]}/{filename}":
                raise ReportValidationError(f"Rendered asset condition path differs: {key}/{filename}")
        for filename in ("rgb_comparison.mp4", "tracking_comparison.mp4"):
            asset = inventory[filename]
            probe = asset.get("probe", {})
            if (probe.get("decoded_frames") != 128 or probe.get("success") is not True
                    or probe.get("method") != "ffprobe_count_frames"
                    or probe.get("source_sha256") != asset["sha256"]):
                raise ReportValidationError(f"MP4 lacks hash-bound decoded 128-frame proof: {key}/{filename}")
            if not finite(probe.get("fps")) or probe["fps"] <= 0 or not finite(probe.get("duration")):
                raise ReportValidationError(f"Invalid preview FPS/duration: {key}/{filename}")
            if not all(isinstance(probe.get(d), int) and probe[d] > 0 for d in ("width", "height")):
                raise ReportValidationError(f"Invalid preview dimensions: {key}/{filename}")
            if not math.isclose(probe["duration"], 128 / probe["fps"], rel_tol=.01, abs_tol=.05):
                raise ReportValidationError(f"Preview duration does not match all 128 frames: {key}/{filename}")
        media.append({"dataset": key[0], "sequence": key[1], "objective": key[2],
                      "selection": clip.get("selection", {}), "checks": {
                          "projection": clip["projection_checks"], "reconstruction": clip["reconstruction_checks"]},
                      "normalization_summary": normalization, "tracking_timeline_summary": timeline,
                      "assets": inventory})
    if set(indexed) != expected:
        raise ReportValidationError(f"Missing rendered conditions: {sorted(expected - set(indexed))}")
    timelines, geometry = {}, {}
    for row in manifest.get("timelines", []):
        key = row["dataset"], row["sequence"]
        if key in timelines or key not in identities or row.get("frame_count") != 128:
            raise ReportValidationError(f"Unexpected/duplicate timeline: {key}")
        for objective in ("clean", *OBJECTIVES):
            condition_sources(row.get("sources", {}).get(objective, {}), *key, objective)
        timelines[key] = assets_inventory(row.get("assets", []))
    for row in manifest.get("geometry_cases", []):
        key = row["dataset"], row["sequence"], row["frame_index"]
        if key in geometry or key[:2] not in identities or key[2] not in (0, 64, 127):
            raise ReportValidationError(f"Unexpected/duplicate geometry case: {key}")
        for objective in ("clean", "tracking_3d", "reconstruction_3d"):
            condition_sources(row.get("sources", {}).get(objective, {}), *key[:2], objective)
        geometry[key] = assets_inventory(row.get("assets", []))
    if len(timelines) != 8 or len(geometry) != 24:
        raise ReportValidationError("Eight timelines and 24 shared-frame geometry cases are required")
    common = []
    for d, s in identities:
        timeline = timelines[(d, s)].get("tracking_epe.png")
        triptychs = [geometry[(d, s, f)].get(f"geometry_attack_triptych_frame_{f:03d}.png") for f in (0, 64, 127)]
        if not timeline or any(t is None for t in triptychs):
            raise ReportValidationError(f"Missing common assets for {d}/{s}")
        if timeline["path"].replace("\\", "/") != f"timelines/{d}/{s}/tracking_epe.png" or any(
                p["path"].replace("\\", "/") != f"geometry_cases/{d}/{s}/geometry_attack_triptych_frame_{f:03d}.png"
                for f, p in zip((0, 64, 127), triptychs)):
            raise ReportValidationError(f"Common asset clip path differs: {d}/{s}")
        common.append({"dataset": d, "sequence": s,
                       "timeline": timeline["url"],
                       "triptychs": [{"frame": f, "url": p["url"]}
                                      for f, p in zip((0, 64, 127), triptychs)]})
    return manifest, media, common


def conditions_payload(study):
    rows = []
    for row in study["analysis"]["paired_conditions"]:
        d, s, o = row["dataset"], row["sequence"], row["objective"]
        original = study["records"][(d, s, o)]
        fields = {k: row[k] for metric in METRICS for k in (metric, metric + "_clean", metric + "_attacked")}
        rows.append({"dataset": d, "sequence": s, "objective": o, **fields,
                     "tracking_apd_relative_drop_pct": ratio(fields[METRICS[0]], fields[METRICS[0] + "_clean"], 100),
                     "reconstruction_apd_relative_drop_pct": ratio(fields[METRICS[1]], fields[METRICS[1] + "_clean"], 100),
                     "tracking_epe_ratio": ratio(fields[METRICS[2] + "_attacked"], fields[METRICS[2] + "_clean"]),
                     "reconstruction_epe_ratio": ratio(fields[METRICS[3] + "_attacked"], fields[METRICS[3] + "_clean"]),
                     "selected_state": original["selected_state"], "seed": original["seed"],
                     "elapsed_seconds": original["elapsed_seconds"],
                     "condition_wall_seconds": original.get("condition_wall_seconds"),
                     "loss_terms": original["loss_terms"],
                     "clean_diagnostics": study["records"][(d, s, "clean")]["loss_terms"]["diagnostics"],
                     "perturbation_linf": original["perturbation_norms"]["linf"],
                     "source_result": f"conditions/{o}/{d}/{s}/result.json"})
    if len(rows) != 40:
        raise ReportValidationError("Detailed report must contain the 40 paired attacks")
    return rows


def summarize(study, coupling):
    all_rows = [row for row in coupling["impact_matrix"] if row["dataset"] == "all"]
    first = []
    for metric, family in zip(METRICS[:2], ("tracking", "reconstruction")):
        greatest = max(all_rows, key=lambda row: row[metric])
        action = "감소" if greatest[metric] >= 0 else "증가"
        rank = "감소량이 가장 큰" if greatest[metric] >= 0 else "증가폭이 가장 작은"
        first.append(f"선정 8클립에서 {greatest['objective']} 공격의 {family} APD는 "
                     f"{greatest[metric + '_clean']:.2f}%에서 {greatest[metric + '_attacked']:.2f}%로 "
                     f"{abs(greatest[metric]):.2f}%p {action}했습니다. 다섯 목적 중 {family} APD {rank} 결과입니다.")
    contrasts = [x for x in coupling["geometry_contrasts"]["estimates"] if x["dataset"] == "all"]
    second = []
    for x in contrasts:
        family = "Tracking" if x["metric"].startswith("tracking") else "Reconstruction"
        if "apd" not in x["metric"]:
            continue
        lo, hi = x["ci95"]
        unit = "%p"
        uncertain = "; CI가 0을 포함하므로 상대 차이는 불확실합니다" if lo <= 0 <= hi else ""
        second.append(f"{family} APD 손상의 tracking_3d−reconstruction_3d 차이는 "
                      f"{x['estimate']:+.2f}{unit} (paired 95% CI {lo:+.2f}~{hi:+.2f}{unit}){uncertain}.")
    return [*first, *second,
            "모든 요약은 클립 동일 가중 평균이며, 전체는 PO4/DR4 각 50%입니다. "
            "직접 차이는 두 공격 사이의 비교이고 실제 손상·개선은 clean 대비 값을 함께 읽어야 합니다."]


def select_cases(study, coupling):
    """Show the full roster, two fixed examples, and a labelled exploratory contrast."""
    contrasts = coupling["geometry_contrasts"]["clip_differences"]
    indexed = {(row["dataset"], row["sequence"]): row for row in contrasts}
    selected = []
    for dataset, label in (("po_mini", "PO 고정 사례"), ("ds_mini", "DR 고정 사례")):
        identity = next(key for key in study["identities"] if key[0] == dataset)
        selected.append({**indexed[identity], "label": label,
                         "criterion": "결과 수치를 사용하지 않고 원래 manifest에서 해당 dataset의 첫 클립을 선택했습니다. 대표성이나 전체 dataset 일반화를 주장하지 않습니다."})
    greatest = max(contrasts, key=lambda row: abs(row[METRICS[0]]))
    selected.append({**greatest, "label": "Tracking APD 차이가 큰 사례",
                     "criterion": "동일 클립의 tracking_3d−reconstruction_3d tracking APD 손상 차이 절댓값이 최대인 사례입니다. 결과 기반 탐색 사례이며, 동률은 manifest 순서입니다. 나머지 8클립 결과도 함께 제공합니다."})
    return selected


def source_configuration_evidence(study, project_root):
    """Bind the original launch's byte hashes to the current source files."""
    run_dir, project_root = Path(study["run_dir"]), Path(project_root).resolve()
    launch_path = run_dir / "launch.json"
    launch = read_json(launch_path)
    if (launch.get("RunName") != run_dir.name or launch.get("Clips") != 8
            or launch.get("ExpectedConditions") != 48 or launch.get("FramesPerClip") != 128
            or launch.get("AllFrames") is not True):
        raise ReportValidationError("Original launch scope differs from the completed study")
    files = {}
    for relative, field, signed_key in (("configs/loss_components_8clips_allframes.json", "ConfigSHA256", "config"),
                                       ("docker/manifests/loss_components_8clips_allframes.json", "ManifestSHA256", "manifest")):
        path = (project_root / relative).resolve()
        if not path.is_relative_to(project_root) or not path.is_file() or sha_file(path) != launch.get(field):
            raise ReportValidationError(f"Original launch source file SHA differs: {relative}")
        if read_json(path) != study["metadata"][signed_key]:
            raise ReportValidationError(f"Original launch source content differs from signed run: {relative}")
        files[relative] = {"path": str(path), "sha256": launch[field], "bytes": path.stat().st_size,
                           "matched_original_launch": True, "hash_kind": "literal file bytes"}
    return {"launch_json_sha256": sha_file(launch_path), "launch_metadata": launch,
            "configuration_files": files}


def cpu_timing_snapshot(run_dir, output_dir, builder_started):
    """Display finished CPU renderer times without claiming future QA duration."""
    path = Path(run_dir) / "html_postprocessing.json"
    rows, attempt_id = [], None
    if path.is_file():
        state = read_json(path)
        if Path(state.get("staging_dir") or "").resolve() == Path(output_dir).resolve():
            attempt_id = state.get("attempt_id")
            completed = [row for row in state.get("stages", []) if row.get("status") == "complete"]
            expected = ["render:" + objective for objective in OBJECTIVES]
            if [row.get("stage") for row in completed] != expected[:len(completed)] or len(completed) > len(expected):
                raise ReportValidationError("CPU timing snapshot has unexpected completed renderer stages")
            for row in completed:
                elapsed = row.get("elapsed_seconds")
                if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                        or not math.isfinite(elapsed) or elapsed < 0 or row.get("cpu_only") is not True):
                    raise ReportValidationError("CPU timing snapshot requires finite nonnegative CPU durations")
            rows = [{key: row.get(key) for key in ("stage", "status", "started_at_utc", "completed_at_utc", "elapsed_seconds", "cpu_only")}
                    for row in completed]
    return {"schema_version": 1, "status": "snapshot_before_browser_qa", "attempt_id": attempt_id,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "stages": rows,
            "builder_elapsed_until_html_content_snapshot_seconds": (datetime.now(timezone.utc) - builder_started).total_seconds(),
            "builder_snapshot_scope": "Elapsed builder CPU work up to HTML content assembly; excludes renderer and browser QA.",
            "final_cpu_execution_evidence_url": "data/postprocessing_execution.json",
            "not_gpu_experiment_time": True,
            "scope": "Completed CPU stages only; final QA and time until publication are recorded after QA in the linked JSON. GPU experiment, existing artifact audit/analysis and pause intervals are excluded."}


def build_report(run_dir, *, analysis_dir=None, assets_dir=None, output_dir=None,
                 language="ko", require_complete=True, command=None):
    if language != "ko" or require_complete is not True:
        raise ReportValidationError("This final report requires Korean and full completion")
    started = datetime.now(timezone.utc)
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir or run_dir / "html_report").resolve()
    if output_dir == run_dir or not output_dir.is_relative_to(run_dir):
        raise ReportValidationError("Report output must be a distinct directory inside the run")
    if any(output_dir.is_relative_to(run_dir / name) for name in ("conditions", "sequences", "analysis")):
        raise ReportValidationError("Report output must not overwrite experiment evidence")
    if (output_dir / "index.html").exists():
        raise ReportValidationError("Existing report must be preserved; build in a fresh staging directory")
    study = load_verified_study(run_dir, analysis_dir=analysis_dir, require_complete=True)
    source_assets = Path(assets_dir or output_dir / "assets").resolve()
    visual, media, common = validate_visualizations(source_assets, study)
    configuration = source_configuration_evidence(study, visual.get("project_root", Path(__file__).resolve().parents[1]))
    visual_sha = sha_file(source_assets / "visualization_manifest.json")
    target_assets = output_dir / "assets"
    if source_assets != target_assets.resolve():
        if target_assets.resolve().is_relative_to(source_assets):
            raise ReportValidationError("Asset copy target must not be inside its source")
        shutil.copytree(source_assets, target_assets, dirs_exist_ok=True)
    else:
        target_assets.mkdir(parents=True, exist_ok=True)
    coupling = compute_task_coupling(study, samples=2000, seed=20261004)
    interpretation = build_interpretation(study, coupling, visual)
    coupling_plots = make_coupling_plots(coupling, target_assets)
    analysis_root = Path(study["analysis_dir"])
    original_plot_hashes = {}
    for filename in ORIGINAL_PLOTS:
        original = analysis_root / filename
        if not original.is_file():
            raise ReportValidationError(f"Required original analysis plot missing: {filename}")
        original_plot_hashes[filename] = sha_file(original)
        shutil.copy2(original, target_assets / filename)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("sequence_metrics.csv", "aggregate_metrics.csv"):
        shutil.copy2(run_dir / filename, data_dir / filename)
    write_atomic(data_dir / "task_coupling_analysis.json", json_text(coupling))
    record_rows = conditions_payload(study)
    cpu_timing = cpu_timing_snapshot(run_dir, output_dir, started)
    write_atomic(data_dir / "postprocessing_execution.json", json_text(cpu_timing))
    shutil.copy2(run_dir / "launch.json", data_dir / "launch.json")
    runtime = {}
    runtime_path = run_dir / "postprocessing_revision.json"
    if runtime_path.is_file():
        runtime = read_json(runtime_path).get("final_report_text_updates", {})
    warnings = list(study["analysis"]["validation"].get("warnings", []))
    if runtime.get("runtime_note_markdown"):
        warnings.append(runtime["runtime_note_markdown"])
    for filename in ("runtime_observations.json", "postprocessing_revision.json"):
        if (run_dir / filename).is_file():
            shutil.copy2(run_dir / filename, data_dir / filename)
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1, "run_id": run_dir.name, "run_signature": study["metadata"]["signature"],
        "synthetic_only": study["analysis"].get("synthetic_only", False) is True,
        "generated_at_utc": now.isoformat(), "generated_at_kst": now.astimezone(KST).isoformat(),
        "status": study["analysis"]["status"], "scope": study["analysis"]["scope"],
        "objectives": list(OBJECTIVES), "clips": [{"dataset": d, "sequence": s} for d, s in study["identities"]],
        "impact": coupling["impact_matrix"], "scatter": coupling["scatter"],
        "contrasts": coupling["geometry_contrasts"], "conditions": record_rows,
        "interpretation": interpretation, "source_configuration": configuration,
        "cpu_postprocessing": cpu_timing,
        "case_selections": select_cases(study, coupling),
        "media": media, "common_media": common, "visualization": {
            "status": visual["status"], "cpu_only": True, "new_model_inference": False,
            "manifest_url": "assets/visualization_manifest.json"},
        "timing": study["analysis"]["timing"], "summary": summarize(study, coupling),
        "validation": study["analysis"]["validation"], "warnings": warnings,
        "audit": {k: study["audit"].get(k) for k in ("status", "passed", "full_campaign_completed", "all_frames_completed",
                    "source_files_verified", "completed_conditions_checked", "errors", "missing")},
        "provenance": study["metadata"].get("provenance", {}),
        "config": study["metadata"]["config"],
        "dataset_manifest": study["metadata"]["manifest"],
        "input_sha256": study["input_sha256"], "coupling_plots": coupling_plots,
        "runtime_evidence": [name for name in ("runtime_observations.json", "postprocessing_revision.json")
                             if (data_dir / name).is_file()],
        "scientific_interpretation": study["analysis"].get("scientific_interpretation"),
    }
    write_atomic(data_dir / "report_data.json", json_text(payload))
    template_path = Path(__file__).resolve().parents[1] / "templates" / "component_report.html"
    template = template_path.read_text(encoding="utf-8")
    rendered = template.replace("__TITLE__", html.escape(f"St4RTrack PGD · {run_dir.name}"))
    rendered = rendered.replace("__PAYLOAD_JSON__", script_json(payload))
    if "__PAYLOAD_JSON__" in rendered or "__TITLE__" in rendered:
        raise ReportValidationError("Unresolved HTML template marker")
    # Verify again after potentially lengthy CPU plot generation.
    refreshed = load_verified_study(run_dir, analysis_dir=analysis_dir, require_complete=True)
    if refreshed["input_sha256"] != study["input_sha256"]:
        raise ReportValidationError("Study evidence changed during report generation")
    if source_configuration_evidence(refreshed, visual.get("project_root", Path(__file__).resolve().parents[1])) != configuration:
        raise ReportValidationError("Launch or source configuration changed during report generation")
    validate_visualizations(target_assets, refreshed)
    if sha_file(target_assets / "visualization_manifest.json") != visual_sha:
        raise ReportValidationError("Visualization manifest changed during report generation")
    for filename, digest in original_plot_hashes.items():
        if sha_file(analysis_root / filename) != digest or sha_file(target_assets / filename) != digest:
            raise ReportValidationError(f"Original analysis plot changed during report generation: {filename}")
    output_files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in ("index.html", "report_manifest.json") and not path.name.endswith(".tmp"):
            output_files[path.relative_to(output_dir).as_posix()] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    code_paths = [Path(__file__), template_path]
    code_paths.extend(Path(__file__).with_name(name) for name in (
        "component_report_data.py", "component_projection.py", "render_component_comparisons.py",
        "component_report_interpretation.py",
        "render_pgd_comparisons.py", "render_tracking_comparisons.py", "tracking_projection.py",
        "verify_component_html_report.cjs", "run_component_report_pipeline.py"))
    report_manifest = {
        "schema_version": 1, "status": "built_pending_browser_qa", "started_at_utc": started.isoformat(),
        "generated_at_utc": now.isoformat(), "generated_at_kst": now.astimezone(KST).isoformat(),
        "run_id": run_dir.name, "run_signature": payload["run_signature"], "run_dir": str(run_dir),
        "project_root": visual.get("project_root", str(Path(__file__).resolve().parents[1])),
        "provenance": payload["provenance"], "config": payload["config"], "dataset_manifest": payload["dataset_manifest"],
        "execution_evidence": {"campaign": study["campaign"], "resume": study["resume"]},
        "source_configuration": configuration,
        "scope": payload["scope"], "analysis_status": payload["status"], "artifact_audit": payload["audit"],
        "synthetic_only": payload["synthetic_only"],
        "analysis_json_sha256": sha_file(analysis_root / "study_analysis.json"),
        "artifact_validation_sha256": sha_file(run_dir / "artifact_validation.json"),
        "analysis_plot_source_sha256": original_plot_hashes,
        "scalar_input_sha256": study["input_sha256"],
        "analyzer_source_sha256": study["analysis"].get("analyzer_source_sha256"),
        "source_code_sha256": {p.name: sha_file(p) for p in code_paths if p.is_file()},
        "guide_sha256": sha_file(Path(__file__).resolve().parents[1] / "docs" / "HTML_REPORT_GUIDE.md"),
        "visualization_source_proofs": {key: visual[key] for key in ("clips", "timelines", "geometry_cases")},
        "visualization_manifest_sha256": sha_file(target_assets / "visualization_manifest.json"),
        "objectives": list(OBJECTIVES), "clips": payload["clips"],
        "bootstrap": payload["contrasts"]["bootstrap"], "scatter_definition": payload["scatter"]["definition"],
        "preview_frame_indices": [0, 64, 127], "contrast_direction": payload["contrasts"]["direction"],
        "command": command or sys.argv, "outputs": output_files, "missing": [], "errors": [],
        "browser_qa": {"status": "pending"}, "builder_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "builder_time_scope": "CPU builder only: statistics/plots/assets copy/hash/HTML assembly; renderer and browser QA are separate stages.",
    }
    report_manifest["outputs"]["index.html"] = {"sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                                                       "bytes": len(rendered.encode("utf-8"))}
    write_atomic(output_dir / "report_manifest.json", json_text(report_manifest))
    write_atomic(output_dir / "index.html", rendered)
    return output_dir / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--language", choices=("ko",), default="ko")
    parser.add_argument("--require-complete", action="store_true", default=True)
    args = parser.parse_args()
    try:
        result = build_report(**vars(args))
    except (ReportValidationError, OSError, KeyError, ValueError) as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"HTML built; browser QA still required: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
