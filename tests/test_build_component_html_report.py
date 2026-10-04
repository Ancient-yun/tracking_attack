"""Synthetic CPU consumer tests, not evidence from the actual experiment.

NPY/NPZ/MP4 fixture files below are labelled dummy bytes. Their SHA and video
probe records are unit-test stubs, not independent array audits, decoded-video
checks, browser playback checks, or actual full-campaign completion evidence.
``make_builder_fixture`` is public so browser tests can reuse the scalar proof
and replace the dummy MP4s with independently encoded synthetic test videos.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw
import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import build_component_html_report as BUILDER

from test_component_report_data import digest, freshen, make_fixture, read, write

ARRAY_NAMES = ("tracks.npz", "rgb_float32.npy", "delta_float32.npy",
               "reconstruction_float32.npy", "reconstruction_confidence_float32.npy", "components.npz")
FRAMES = (0, 64, 127)
RENDERER_SOURCES = ("render_component_comparisons.py", "component_projection.py", "component_report_data.py",
                    "render_pgd_comparisons.py", "render_tracking_comparisons.py", "tracking_projection.py")


def _png(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 80), "#edf3f9")
    ImageDraw.Draw(image).text((8, 12), "SYNTHETIC UNIT FIXTURE", fill="#173b63")
    ImageDraw.Draw(image).text((8, 37), text[:45], fill="#173b63")
    image.save(path)


def _asset(path, assets, kind, **extra):
    return {"path": path.relative_to(assets).as_posix(), "kind": kind,
            "sha256": digest(path), "bytes": path.stat().st_size,
            "synthetic_only": True, **extra}


def _condition_sources(root, dataset, sequence, objective):
    directory = root / "conditions" / objective / dataset / sequence
    names = (*ARRAY_NAMES, "history.json", "result.json")
    return {name: {"path": str(directory / name), "sha256": digest(directory / name),
                   "bytes": (directory / name).stat().st_size, "synthetic_only": True}
            for name in names}


def _sequence_sources(root, metadata, dataset, sequence):
    directory = root / "sequences" / dataset / sequence
    names = ("sequence.json", "targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy")
    evidence = {name: {"path": str(directory / name), "sha256": digest(directory / name),
                       "bytes": (directory / name).stat().st_size, "synthetic_only": True}
                for name in names}
    entry = next(e for e in metadata["manifest"]["entries"]
                 if (e["dataset"], e["sequence"]) == (dataset, sequence))
    evidence["source_npz"] = {"path": str(root / entry["path"]), "sha256": entry["sha256"],
                              "bytes": entry["bytes"], "synthetic_only": True}
    return evidence


def make_builder_fixture(root):
    """Return a synthetic run Path with ``fixture_assets`` and strict stubs.

    This helper writes only under the supplied temporary fixture root. It never
    reads the real run, original NPZs, vendor, checkpoint, Docker, or a model.
    """
    root = Path(root).resolve()
    make_fixture(root)
    metadata = read(root / "run.json")
    metadata["synthetic_only"] = True
    for entry in metadata["manifest"]["entries"]:
        source = root / "synthetic_sources" / entry["dataset"] / (entry["sequence"] + ".npz")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"SYNTHETIC NPZ UNIT STUB: NOT A REAL DATASET\n")
        entry.update(path=source.relative_to(root).as_posix(), sha256=digest(source), bytes=source.stat().st_size)
    metadata["signature"] = hashlib.sha256(json.dumps(
        {k: metadata[k] for k in ("config", "manifest", "provenance")},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write(root / "run.json", metadata)
    for entry in metadata["manifest"]["entries"]:
        dataset, sequence = entry["dataset"], entry["sequence"]
        directory = root / "sequences" / dataset / sequence
        inventory = read(directory / "sequence_manifest.json")
        inventory["signature"] = metadata["signature"]
        for name in ("targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy"):
            path = directory / name
            path.write_bytes(f"SYNTHETIC ARRAY UNIT STUB {dataset}/{sequence}/{name}\n".encode())
            inventory["artifact_sha256"][name] = digest(path)
        inventory["synthetic_only"] = True
        write(directory / "sequence_manifest.json", inventory)
        for objective in ("clean", *BUILDER.OBJECTIVES):
            condition = root / "conditions" / objective / dataset / sequence
            record = read(condition / "result.json")
            record.update(signature=metadata["signature"], synthetic_only=True,
                          condition_wall_seconds=1. if objective == "clean" else 21.,
                          perturbation_norms={"linf": 0. if objective == "clean" else 4 / 255})
            for name in ARRAY_NAMES:
                path = condition / name
                path.write_bytes(f"SYNTHETIC ARRAY UNIT STUB {dataset}/{sequence}/{objective}/{name}\n".encode())
                record["artifact_sha256"][name] = digest(path)
            write(condition / "result.json", record)
    analysis = read(root / "analysis/study_analysis.json")
    analysis["status"] = "complete_with_recorded_failures"
    analysis["validation"]["recorded_failed_attempts"] = 2
    analysis["validation"]["warnings"] = ["Synthetic reporting stub: two recovered attempts; no actual failures are asserted."]
    summary = read(root / "summary.json")
    summary["failed_attempts"] = 2
    write(root / "summary.json", summary)
    analysis["timing"] = {
        "synthetic_only": True,
        "objectives": [{"objective": o, "attack_seconds_sum": 160., "condition_wall_seconds_sum": 168.,
                        "objective_finished_wall_seconds_sum": 170., "completed_timing_sessions": 1,
                        "unfinished_timing_sessions": 0} for o in BUILDER.OBJECTIVES],
        "objective_sessions": [], "scope": {"description": "Synthetic runtime records, not measured execution"},
        "study_finished_invocation_wall_seconds_sum": 850.,
    }
    write(root / "analysis/study_analysis.json", analysis)
    freshen(root)
    for filename in BUILDER.ORIGINAL_PLOTS:
        _png(root / "analysis" / filename, "Stub original plot: " + filename)
    write(root / "postprocessing_revision.json", {"synthetic_only": True,
        "final_report_text_updates": {"runtime_note_markdown": "Synthetic runtime caveat: GPU was shared; no benchmark is asserted."}})
    assets = root / "fixture_assets"
    study = BUILDER.load_verified_study(root)
    manifest = {"schema_version": 1, "synthetic_only": True, "status": "complete", "errors": [],
                "cpu_only": True, "new_model_inference": False, "run_id": root.name,
                "run_signature": metadata["signature"], "scope": analysis["scope"], "frame_count": 128,
                "analysis_json_sha256": study["analysis_sha256"],
                "artifact_validation_sha256": digest(root / "artifact_validation.json"),
                "input_scalar_file_sha256": study["input_sha256"],
                "renderer_source_sha256": {name: digest(SCRIPTS / name) for name in RENDERER_SOURCES},
                "clips": [], "timelines": [], "geometry_cases": []}
    selection = {"synthetic_only": True, "selection_uses_model_errors": False,
                 "selected_count": 2, "display_number_to_saved_query_index": {"01": 0, "02": 1}}
    for dataset, sequence in study["identities"]:
        reconstruction_checks = {"passed": True, "synthetic_only": True,
            "conditions": {o: {"passed": True, "synthetic_only": True,
                **{metric: study["records"][(dataset, sequence, o)]["reconstruction_metrics"][metric]
                   for metric in ("apd3d", "epe_m")}} for o in ("clean", *BUILDER.OBJECTIVES)}}
        for objective in BUILDER.OBJECTIVES:
            directory = assets / "comparisons" / objective / dataset / sequence
            items = []
            for frame in FRAMES:
                for prefix in ("rgb", "tracking", "reconstruction"):
                    path = directory / f"{prefix}_frame_{frame:03d}.png"
                    _png(path, f"{objective}: {prefix} frame {frame}")
                    items.append(_asset(path, assets, prefix, frame_index=frame))
            for name in ("pgd_history.png", "normalization.png"):
                path = directory / name
                _png(path, "Stub: " + name)
                items.append(_asset(path, assets, name.removesuffix(".png")))
            for name in ("rgb_comparison.mp4", "tracking_comparison.mp4"):
                path = directory / name
                path.write_bytes(f"SYNTHETIC MP4 UNIT STUB: NOT DECODABLE {dataset}/{sequence}/{objective}/{name}\n".encode())
                probe = {"synthetic_only": True, "success": True, "method": "ffprobe_count_frames",
                         "decoded_frames": 128, "fps": 10., "duration": 12.8, "width": 320, "height": 80,
                         "source_sha256": digest(path), "test_scope": "Dummy probe for consumer unit gate, not actual ffprobe"}
                items.append(_asset(path, assets, "rgb_video" if name.startswith("rgb") else "tracking_video", probe=probe))
            manifest["clips"].append({"dataset": dataset, "sequence": sequence, "objective": objective,
                "synthetic_only": True, "status": "complete", "frame_count": 128,
                "source_frame_indices": list(range(128)), "preview_frame_indices": list(FRAMES),
                "preview_fps": 10., "selected_state": study["records"][(dataset, sequence, objective)]["selected_state"],
                "selection": deepcopy(selection), "projection_checks": {"passed": True, "synthetic_only": True},
                "reconstruction_checks": deepcopy(reconstruction_checks),
                "sources": {"sequence": _sequence_sources(root, metadata, dataset, sequence), "conditions": {
                    "clean": _condition_sources(root, dataset, sequence, "clean"),
                    "attack": _condition_sources(root, dataset, sequence, objective)}}, "assets": items})
        path = assets / "timelines" / dataset / sequence / "tracking_epe.png"
        _png(path, f"Synthetic all-query timeline {dataset}/{sequence}")
        manifest["timelines"].append({"dataset": dataset, "sequence": sequence, "frame_count": 128,
            "synthetic_only": True, "sources": {o: _condition_sources(root, dataset, sequence, o)
                for o in ("clean", *BUILDER.OBJECTIVES)}, "assets": [_asset(path, assets, "tracking_timeline")]})
        for frame in FRAMES:
            path = assets / "geometry_cases" / dataset / sequence / f"geometry_attack_triptych_frame_{frame:03d}.png"
            _png(path, f"Synthetic geometry comparison frame {frame}")
            manifest["geometry_cases"].append({"dataset": dataset, "sequence": sequence, "frame_index": frame,
                "synthetic_only": True, "selection": deepcopy(selection),
                "sources": {o: _condition_sources(root, dataset, sequence, o)
                    for o in ("clean", "tracking_3d", "reconstruction_3d")},
                "assets": [_asset(path, assets, "geometry_triptych", frame_index=frame)]})
    write(assets / "visualization_manifest.json", manifest)
    write(root / "synthetic_fixture_description.json", {"synthetic_only": True,
        "description": "Strict consumer unit fixture. Array and video files/probes are stubs; no actual experiment or independent audit.",
        "assets_dir": "fixture_assets"})
    return root


@pytest.fixture
def builder_run(tmp_path):
    return make_builder_fixture(tmp_path / "SYNTHETIC_FULL128_BUILDER_STUB")


def test_full_builder_integration_keeps_input_proofs_and_publishes_bound_outputs(builder_run):
    originals = {p: digest(p) for p in builder_run.rglob("*") if p.is_file()}
    output = builder_run / "html_staging"
    index = BUILDER.build_report(builder_run, assets_dir=builder_run / "fixture_assets", output_dir=output,
                                 command=["synthetic-consumer-unit-test"])
    assert index == output / "index.html"
    report = read(output / "data/report_data.json")
    manifest = read(output / "report_manifest.json")
    assert len(report["conditions"]) == 40 and len(report["impact"]) == 15
    assert len(report["coupling_plots"]) == 9 and len(report["common_media"]) == 8
    assert all(row["clean_diagnostics"]["tracking_l21"] == .5 for row in report["conditions"])
    confidence = [r for r in report["conditions"] if r["objective"] == "confidence"]
    assert all(r["tracking_apd_drop_pp"] == -3 for r in confidence)
    assert all(r["tracking_apd_relative_drop_pct"] < 0 for r in confidence)
    assert report["warnings"][-1].startswith("Synthetic runtime caveat")
    assert report["validation"]["recorded_failed_attempts"] == 2
    assert manifest["status"] == "built_pending_browser_qa"
    assert manifest["browser_qa"]["status"] == "pending"
    assert manifest["outputs"]["index.html"]["sha256"] == digest(index)
    assert manifest["outputs"]["index.html"]["bytes"] == index.stat().st_size
    assert manifest["visualization_source_proofs"]["clips"][0]["synthetic_only"] is True
    assert manifest.get("synthetic_only") is True and report.get("synthetic_only") is True
    for relative, proof in manifest["outputs"].items():
        assert digest(output / relative) == proof["sha256"]
    embedded = re.search(r'<script type="application/json" id="report-payload">(.*?)</script>',
                         index.read_text(encoding="utf-8"), flags=re.S)
    assert embedded and json.loads(embedded.group(1)) == report
    assert "__PAYLOAD_JSON__" not in index.read_text(encoding="utf-8")
    assert all(digest(path) == before for path, before in originals.items())


def test_json_script_escape_retains_strings_without_html_termination():
    value = {"label": '</script><script>alert("x")</script>&\u2028\u2029'}
    escaped = BUILDER.script_json(value)
    assert "</script>" not in escaped and "<" not in escaped and "&" not in escaped
    assert json.loads(escaped) == value


@pytest.mark.parametrize("numerator,denominator,multiplier,expected", [
    (-3., 50., 100., -6.), (1., 0., 1., None), (1., 1e-13, 1., None),
    (1., -2., 1., None), (None, 2., 1., None), (float("inf"), 2., 1., None),
    (1e308, 1e-11, 1., None), (.8, .2, 1., 4.),
])
def test_ratios_preserve_improvement_and_return_na_for_invalid_results(numerator, denominator, multiplier, expected):
    actual = BUILDER.ratio(numerator, denominator, multiplier)
    assert actual == expected


@pytest.mark.parametrize("field", ["run_signature", "analysis_json_sha256", "artifact_validation_sha256"])
def test_visualizations_cannot_borrow_proofs_from_another_study(builder_run, field):
    path = builder_run / "fixture_assets/visualization_manifest.json"
    value = read(path)
    value[field] = "0" * 64
    write(path, value)
    with pytest.raises(BUILDER.ReportValidationError, match="different or changed"):
        BUILDER.validate_visualizations(path.parent, BUILDER.load_verified_study(builder_run))


def test_condition_array_sha_proof_must_match_the_signed_result(builder_run):
    path = builder_run / "fixture_assets/visualization_manifest.json"
    value = read(path)
    value["clips"][0]["sources"]["conditions"]["attack"]["rgb_float32.npy"]["sha256"] = "0" * 64
    write(path, value)
    with pytest.raises(BUILDER.ReportValidationError, match="source SHA differs"):
        BUILDER.validate_visualizations(path.parent, BUILDER.load_verified_study(builder_run))


@pytest.mark.parametrize("shared_group", ["timelines", "geometry_cases"])
def test_shared_png_tampering_is_rejected(builder_run, shared_group):
    assets = builder_run / "fixture_assets"
    manifest = read(assets / "visualization_manifest.json")
    path = assets / manifest[shared_group][0]["assets"][0]["path"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(BUILDER.ReportValidationError, match="hash or size differs"):
        BUILDER.validate_visualizations(assets, BUILDER.load_verified_study(builder_run))


def test_127_frame_unit_probe_cannot_claim_full_video(builder_run):
    path = builder_run / "fixture_assets/visualization_manifest.json"
    value = read(path)
    video = next(a for a in value["clips"][0]["assets"] if a["kind"] == "rgb_video")
    video["probe"]["decoded_frames"] = 127
    write(path, value)
    with pytest.raises(BUILDER.ReportValidationError, match="decoded 128-frame"):
        BUILDER.validate_visualizations(path.parent, BUILDER.load_verified_study(builder_run))


def test_scalar_change_blocks_publication_before_html_is_written(builder_run):
    path = builder_run / "conditions/tracking_3d/po_mini/clip_0/result.json"
    path.write_bytes(path.read_bytes() + b"\n")
    output = builder_run / "html_staging"
    with pytest.raises(BUILDER.ReportValidationError, match="Stale analysis SHA"):
        BUILDER.build_report(builder_run, assets_dir=builder_run / "fixture_assets", output_dir=output)
    assert not (output / "index.html").exists()


@pytest.mark.parametrize("relative", [".", "conditions/report", "sequences/report", "analysis/report"])
def test_protected_output_does_not_modify_input_evidence(builder_run, relative):
    before = {p: digest(p) for p in builder_run.rglob("*") if p.is_file()}
    with pytest.raises(BUILDER.ReportValidationError, match="distinct|evidence"):
        BUILDER.build_report(builder_run, assets_dir=builder_run / "fixture_assets", output_dir=builder_run / relative)
    assert all(digest(path) == expected for path, expected in before.items())


def test_existing_final_report_is_preserved(builder_run):
    output = builder_run / "html_report"
    output.mkdir()
    previous = output / "index.html"
    previous.write_bytes(b"Previously reviewed report: must stay untouched")
    with pytest.raises(BUILDER.ReportValidationError, match="Existing report"):
        BUILDER.build_report(builder_run, assets_dir=builder_run / "fixture_assets", output_dir=output)
    assert previous.read_bytes() == b"Previously reviewed report: must stay untouched"


def test_clean_baseline_and_recorded_failure_count_are_preserved(builder_run):
    study = BUILDER.load_verified_study(builder_run)
    rows = BUILDER.conditions_payload(study)
    assert rows[0]["clean_diagnostics"] == study["records"][("po_mini", "clip_0", "clean")]["loss_terms"]["diagnostics"]
    assert study["analysis"]["validation"]["recorded_failed_attempts"] == 2
    assert study["analysis"]["status"] == "complete_with_recorded_failures"
