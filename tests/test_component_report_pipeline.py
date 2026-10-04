"""Synthetic orchestration tests: Docker, encoders and browsers are all stubs.

No actual experiment, GPU, renderer, browser, container or background worker is
started. Fake QA records model the successful protocol, not actual browser QA.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("component_report_pipeline_test", SCRIPTS / "run_component_report_pipeline.py")
PIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPE)
FIXTURE_SPEC = importlib.util.spec_from_file_location("component_pipeline_scalar_fixtures", Path(__file__).with_name("test_component_report_data.py"))
FIXTURES = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURES)
CONTAINER_ID = "a" * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspector(status="exited", code=0, oom=False, running=False, identity=CONTAINER_ID):
    return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps([{"Id": identity, "Name": "/synthetic", "State": {
        "Status": status, "Running": running, "ExitCode": code, "OOMKilled": oom}}]))


def fake_bundle(directory, study, *, passed=False, synthetic=False):
    """Simulated builder/QA byte contract, explicitly tagged as a test stub."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.html").write_text('<!doctype html><html lang=ko>synthetic fixture<a href="data/postprocessing_execution.json">CPU time</a></html>', encoding="utf-8")
    write(directory / "data/report_data.json", {"fixture_only": True})
    write(directory / "assets/visualization_manifest.json", {"fixture_only": True})
    if not (directory / PIPE.CPU_EXECUTION_FILE).is_file():
        write(directory / PIPE.CPU_EXECUTION_FILE, {"fixture_only": True, "status": "builder_snapshot_before_qa"})
    manifest = {"fixture_only": True, "status": "complete" if passed else "built_pending_browser_qa",
                "run_id": Path(study["run_dir"]).name, "run_signature": study["metadata"]["signature"],
                "scope": study["analysis"]["scope"], "analysis_json_sha256": study["analysis_sha256"],
                "artifact_validation_sha256": study["input_sha256"]["artifact_validation.json"],
                "scalar_input_sha256": study["input_sha256"], "errors": [], "missing": [],
                "browser_qa": {"status": "passed" if passed else "pending", "synthetic_only": synthetic}}
    if passed:
        write(directory / "qa/browser_qa.json", {"fixture_qa_stub": True, "status": "passed", "synthetic_only": synthetic,
              "index_sha256": PIPE.sha_file(directory / "index.html"), "report_data_sha256": PIPE.sha_file(directory / "data/report_data.json"),
              "run_signature": study["metadata"]["signature"], "errors": [], "cpu_only": True, "new_model_inference": False,
              "original_folder": {"passed": True}, "portable_folder": {"passed": True}})
        manifest["browser_qa"].update(index_sha256=PIPE.sha_file(directory / "index.html"),
              report_data_sha256=PIPE.sha_file(directory / "data/report_data.json"), report="qa/browser_qa.json",
              sha256=PIPE.sha_file(directory / "qa/browser_qa.json"), actual_video_playback=True,
              portable_copy_verified=True, external_network_requests=0)
    manifest["outputs"] = {p.relative_to(directory).as_posix(): {"sha256": PIPE.sha_file(p), "bytes": p.stat().st_size}
                           for p in directory.rglob("*") if p.is_file() and p.name != "report_manifest.json"}
    write(directory / "report_manifest.json", manifest)


def completed_state(study):
    """Invented durations for metadata contract tests, never measured runtimes."""
    labels = ["render:" + objective for objective in PIPE.OBJECTIVES] + ["build", "browser_qa"]
    stages = [{"stage": label, "status": "complete", "returncode": 0, "elapsed_seconds": float(index + 1),
               "cpu_only": True, "new_model_inference": False, "cuda_visible_devices": "",
               "started_at_utc": "2026-01-01T00:00:00+00:00", "completed_at_utc": "2026-01-01T00:00:07+00:00",
               "argv": ["synthetic-only-stub"], "log": "synthetic-never-executed.log"}
              for index, label in enumerate(labels)]
    return {"fixture_only": True, "attempt_id": "synthetic-time-contract", "started_at_utc": "2026-01-01T00:00:00+00:00",
            "readiness": {"fixture_only": True, "run_signature": study["metadata"]["signature"]},
            "source_code_sha256": {}, "stages": stages}


def bundle_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def run(tmp_path):
    return FIXTURES.make_fixture(tmp_path / "synthetic_scalar_campaign")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    # All child source/runtime paths are local non-executable fixture files.
    scripts = tmp_path / "synthetic_code/scripts"
    templates = scripts.parent / "templates"
    docs = scripts.parent / "docs"
    scripts.mkdir(parents=True)
    templates.mkdir()
    docs.mkdir()
    for filename in ("run_component_report_pipeline.py", "component_report_data.py", "component_projection.py",
                     "component_report_interpretation.py", "render_component_comparisons.py", "build_component_html_report.py",
                     "render_pgd_comparisons.py", "render_tracking_comparisons.py", "tracking_projection.py",
                     "render_official_visualizations.py", "verify_component_html_report.cjs"):
        (scripts / filename).write_text("synthetic child source, never executed", encoding="utf-8")
    (templates / "component_report.html").write_text("synthetic template", encoding="utf-8")
    (docs / "HTML_REPORT_GUIDE.md").write_text("synthetic guide", encoding="utf-8")
    monkeypatch.setattr(PIPE, "__file__", str(scripts / "run_component_report_pipeline.py"))
    node, package, browser = tmp_path / "node", tmp_path / "playwright", tmp_path / "browser"
    node.write_text("fixture", encoding="utf-8")
    browser.write_text("fixture", encoding="utf-8")
    package.mkdir()
    monkeypatch.setattr(PIPE, "default_browser_runtime", lambda: (node, package, browser))
    return scripts


def fake_executor(run, monkeypatch, *, failure=None, synthetic_qa=False, mutate_source=None, mutate_evidence=False):
    calls = []
    def execute(argv, **kwargs):
        argv = [str(a) for a in argv]
        if len(argv) == 3 and argv[1] == "inspect":
            return inspector()
        assert kwargs.get("env", {}).get("CUDA_VISIBLE_DEVICES") == ""
        assert "--gpus" not in argv and "--synthetic" not in argv and "--frames-only" not in argv
        script = next(a for a in argv if a.endswith((".py", ".cjs")))
        name = Path(script).name
        if name == "render_component_comparisons.py":
            label = "render:" + argv[argv.index("--objective") + 1]
            directory = Path(argv[argv.index("--output-dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / (label.replace(":", "_") + ".fixture")).write_text(label, encoding="utf-8")
        elif name == "build_component_html_report.py":
            label = "build"
            directory = Path(argv[argv.index("--output-dir") + 1])
            study = PIPE.load_verified_study(run)
            fake_bundle(directory, study)
        else:
            label = "browser_qa"
            directory = Path(argv[argv.index("--report-dir") + 1])
            study = PIPE.load_verified_study(run)
            fake_bundle(directory, study, passed=True, synthetic=synthetic_qa)
            if mutate_source:
                mutate_source.write_text("source changed during mocked QA", encoding="utf-8")
            if mutate_evidence:
                path = run / "conditions/tracking_3d/po_mini/clip_0/result.json"
                path.write_bytes(path.read_bytes() + b"\n")
        calls.append(label)
        kwargs["stdout"].write("Synthetic subprocess output\n")
        return SimpleNamespace(returncode=7 if label == failure else 0)
    monkeypatch.setattr(PIPE.subprocess, "run", execute)
    return calls


@pytest.mark.parametrize("status,code,oom,running", [("running", 0, False, True), ("paused", 0, False, False),
    ("exited", 1, False, False), ("exited", 0, True, False), ("exited", 0, False, True)])
def test_readiness_refuses_unfinished_failed_or_oom_container_without_writes(run, monkeypatch, status, code, oom, running):
    monkeypatch.setattr(PIPE.subprocess, "run", lambda *args, **kwargs: inspector(status, code, oom, running))
    before = {p.relative_to(run).as_posix(): PIPE.sha_file(p) for p in run.rglob("*") if p.is_file()}
    result = PIPE.check_readiness(run, CONTAINER_ID)
    assert not result["ready"] and result["status"] == "not_ready"
    assert before == {p.relative_to(run).as_posix(): PIPE.sha_file(p) for p in run.rglob("*") if p.is_file()}


def test_exact_container_id_is_required_and_pinned(run, monkeypatch):
    monkeypatch.setattr(PIPE.subprocess, "run", lambda *args, **kwargs: inspector(identity="b" * 64))
    assert not PIPE.check_readiness(run, CONTAINER_ID)["ready"]
    assert not PIPE.check_readiness(run, "short-id")["ready"]
    assert not (run / "html_postprocessing.json").exists()


def test_check_only_is_read_only_when_fully_ready(run, environment, monkeypatch):
    monkeypatch.setattr(PIPE.subprocess, "run", lambda *args, **kwargs: inspector())
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID, check_only=True)
    assert result["ready"] and result["status"] == "ready"
    assert not (run / "html_postprocessing.json").exists()
    assert not (run / "html_postprocessing.lock").exists()
    assert not list(run.glob("html_report*"))


@pytest.mark.parametrize("name", ["campaign_execution.json", "resume_execution.json"])
def test_container_exit_does_not_override_incomplete_controller(run, environment, monkeypatch, name):
    monkeypatch.setattr(PIPE.subprocess, "run", lambda *args, **kwargs: inspector())
    write(run / name, {"status": "running"})
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "not_ready"
    assert not (run / "html_postprocessing.json").exists()


def test_success_runs_sequential_cpu_stages_and_atomically_publishes(run, environment, monkeypatch):
    originals = {name: (run / name).read_bytes() for name in ("run.json", "campaign_execution.json", "resume_execution.json", "summary.json", "artifact_validation.json")}
    calls = fake_executor(run, monkeypatch)
    replacements = []
    original_replace = PIPE.os.replace
    def replace(source, target):
        if Path(source).is_dir():
            replacements.append((Path(source), Path(target)))
        return original_replace(source, target)
    monkeypatch.setattr(PIPE.os, "replace", replace)
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "complete" and result["published_atomic"] and not result["reused"]
    assert calls == ["render:" + o for o in PIPE.OBJECTIVES] + ["build", "browser_qa"]
    assert len(replacements) == 1 and replacements[0][0].parent == run and replacements[0][1] == run / "html_report"
    assert (run / "html_report/index.html").is_file()
    assert not (run / "html_postprocessing.lock").exists()
    assert not list(run.glob("html_report.staging-*"))
    assert len(result["stages"]) == 7 and all(s["status"] == "complete" and s["cpu_only"] for s in result["stages"])
    assert all((run / s["log"]).is_file() and s["elapsed_seconds"] >= 0 for s in result["stages"])
    for name, content in originals.items():
        assert (run / name).read_bytes() == content
    recorded = read(run / "html_report" / PIPE.CPU_EXECUTION_FILE)
    manifest = read(run / "html_report/report_manifest.json")
    assert recorded["status"] == "cpu_stages_complete" and recorded["snapshot_phase"] == "after_browser_qa_before_atomic_publication"
    assert recorded["stages"] == result["stages"] and recorded["stage_elapsed_seconds_sum"] == sum(s["elapsed_seconds"] for s in result["stages"])
    assert recorded["elapsed_until_publication_seconds"] >= recorded["stage_elapsed_seconds_sum"]
    assert recorded["elapsed_until_publication_seconds"] <= result["elapsed_seconds"]
    assert manifest["cpu_postprocessing_evidence"]["only_updated_output"] == PIPE.CPU_EXECUTION_FILE


@pytest.mark.parametrize("failed", ["render:reconstruction_3d", "build", "browser_qa"])
def test_failed_stage_preserves_staging_and_never_publishes_or_changes_controllers(run, environment, monkeypatch, failed):
    controllers = {name: (run / name).read_bytes() for name in ("campaign_execution.json", "resume_execution.json")}
    calls = fake_executor(run, monkeypatch, failure=failed)
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "failed" and result["staging_preserved"]
    assert failed == calls[-1] and result["stages"][-1]["returncode"] == 7
    assert Path(result["staging_dir"]).is_dir() and not (run / "html_report").exists()
    assert not (run / "html_postprocessing.lock").exists()
    for name, content in controllers.items():
        assert (run / name).read_bytes() == content


def test_synthetic_browser_proof_cannot_publish_official_report(run, environment, monkeypatch):
    fake_executor(run, monkeypatch, synthetic_qa=True)
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "failed" and "synthetic" in result["error"]
    assert not (run / "html_report").exists()


def test_complete_existing_report_is_verified_and_reused_without_rerender(run, monkeypatch):
    study = PIPE.load_verified_study(run)
    fake_bundle(run / "html_report", study, passed=True)
    PIPE.finalize_cpu_postprocessing_evidence(run / "html_report", study, completed_state(study), 35.)
    before = {p.relative_to(run / "html_report").as_posix(): PIPE.sha_file(p) for p in (run / "html_report").rglob("*") if p.is_file()}
    def only_inspection(argv, **kwargs):
        assert argv[1] == "inspect"
        return inspector()
    monkeypatch.setattr(PIPE.subprocess, "run", only_inspection)
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "complete" and result["reused"] and result["stages"] == []
    assert before == {p.relative_to(run / "html_report").as_posix(): PIPE.sha_file(p) for p in (run / "html_report").rglob("*") if p.is_file()}


def test_invalid_existing_report_is_preserved_without_overwrite(run, monkeypatch):
    study = PIPE.load_verified_study(run)
    fake_bundle(run / "html_report", study, passed=True)
    path = run / "html_report/index.html"
    path.write_text("tampered report", encoding="utf-8")
    monkeypatch.setattr(PIPE.subprocess, "run", lambda *args, **kwargs: inspector())
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "failed" and "current HTML" in result["error"]
    assert path.read_text(encoding="utf-8") == "tampered report"
    assert not list(run.glob("html_report.staging-*"))


def test_existing_lock_prevents_duplicate_worker(run, monkeypatch):
    write(run / "html_postprocessing.lock", {"attempt_id": "other", "pid": 1})
    monkeypatch.setattr(PIPE.subprocess, "run", lambda *args, **kwargs: inspector())
    with pytest.raises(PIPE.PipelineError, match="lock"):
        PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert read(run / "html_postprocessing.lock")["attempt_id"] == "other"
    assert not (run / "html_postprocessing.json").exists()


def test_source_or_evidence_change_during_execution_blocks_publication(run, environment, monkeypatch):
    fake_executor(run, monkeypatch, mutate_source=environment / "component_projection.py")
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "failed" and "source changed" in result["error"]
    assert not (run / "html_report").exists()


def test_experiment_writer_during_qa_blocks_publication(run, environment, monkeypatch):
    fake_executor(run, monkeypatch, mutate_evidence=True)
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "failed" and "evidence changed" in result["error"]
    assert not (run / "html_report").exists()


def test_report_output_path_cannot_escape_to_evidence_or_parent(run, tmp_path):
    with pytest.raises(PIPE.PipelineError, match="Unsafe"):
        PIPE.inside(run, "../outside.json")
    with pytest.raises(PIPE.PipelineError, match="Unsafe"):
        PIPE.inside(run, "C:/outside.json")
    other = tmp_path / "outside_output"
    other.mkdir()
    marker = other / "marker"
    marker.write_text("preserve", encoding="utf-8")
    try:
        (run / "html_report").symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this host")
    with pytest.raises(PIPE.PipelineError, match="escapes"):
        PIPE.validate_run_paths(run)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_missing_runtime_records_failure_without_starting_children(run, environment, monkeypatch):
    monkeypatch.setattr(PIPE, "default_browser_runtime", lambda: (None, None, None))
    monkeypatch.setattr(PIPE.subprocess, "run", lambda *args, **kwargs: inspector())
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "failed" and "Node executable" in result["error"]
    assert result["stages"] == [] and not (run / "html_report").exists()


def test_cpu_metadata_update_changes_only_the_registered_time_file_and_manifest(run, tmp_path):
    study = PIPE.load_verified_study(run)
    directory = tmp_path / "synthetic_metadata_bundle"
    fake_bundle(directory, study, passed=True)
    before = bundle_bytes(directory)
    before_manifest = read(directory / "report_manifest.json")
    original_inputs = bundle_bytes(run)
    clock_reads = []
    def final_clock():
        clock_reads.append(bundle_bytes(directory))
        return 35.
    proof = PIPE.finalize_cpu_postprocessing_evidence(directory, study, completed_state(study), final_clock)
    after = bundle_bytes(directory)
    assert set(before) == set(after)
    assert clock_reads == [before]  # Clock is sampled after QA checks but before the metadata write.
    assert {name for name in before if before[name] != after[name]} == {PIPE.CPU_EXECUTION_FILE, "report_manifest.json"}
    assert bundle_bytes(run) == original_inputs
    manifest = read(directory / "report_manifest.json")
    assert {k: v for k, v in before_manifest.items() if k != "outputs"} == {k: v for k, v in manifest.items() if k not in ("outputs", "cpu_postprocessing_evidence")}
    assert {k: v for k, v in before_manifest["outputs"].items() if k != PIPE.CPU_EXECUTION_FILE} == {k: v for k, v in manifest["outputs"].items() if k != PIPE.CPU_EXECUTION_FILE}
    timing = read(directory / PIPE.CPU_EXECUTION_FILE)
    assert timing["stage_elapsed_seconds_sum"] == 28. and timing["elapsed_until_publication_seconds"] == 35.
    assert timing["pre_browser_qa_snapshot_sha256"] == before_manifest["outputs"][PIPE.CPU_EXECUTION_FILE]["sha256"]
    assert timing["scope"] == PIPE.CPU_TIMING_SCOPE
    assert "previous independent artifact audit and analysis" in timing["scope"]["excluded"]
    assert "do not add" in timing["scope"]["stage_elapsed_seconds_sum"]
    assert timing["unchanged_output_sha256"]["qa/browser_qa.json"] == proof["qa_proof_sha256"]
    assert timing["browser_qa_bindings"]["index_sha256"] == proof["index_sha256"]


@pytest.mark.parametrize("missing", ["registered_output", "html_link"])
def test_time_update_requires_the_linked_preexisting_browser_checked_snapshot(run, tmp_path, missing):
    study = PIPE.load_verified_study(run)
    directory = tmp_path / "synthetic_missing_snapshot"
    fake_bundle(directory, study, passed=True)
    if missing == "registered_output":
        (directory / PIPE.CPU_EXECUTION_FILE).unlink()
        manifest = read(directory / "report_manifest.json")
        manifest["outputs"].pop(PIPE.CPU_EXECUTION_FILE)
        write(directory / "report_manifest.json", manifest)
    else:
        (directory / "index.html").write_text("synthetic HTML without a timing link", encoding="utf-8")
        qa = read(directory / "qa/browser_qa.json")
        qa["index_sha256"] = PIPE.sha_file(directory / "index.html")
        write(directory / "qa/browser_qa.json", qa)
        manifest = read(directory / "report_manifest.json")
        manifest["browser_qa"].update(index_sha256=qa["index_sha256"], sha256=PIPE.sha_file(directory / "qa/browser_qa.json"))
        for name in ("index.html", "qa/browser_qa.json"):
            manifest["outputs"][name] = {"sha256": PIPE.sha_file(directory / name), "bytes": (directory / name).stat().st_size}
        write(directory / "report_manifest.json", manifest)
    before = bundle_bytes(directory)
    with pytest.raises(PIPE.PipelineError, match="snapshot|already link"):
        PIPE.finalize_cpu_postprocessing_evidence(directory, study, completed_state(study), 35.)
    assert bundle_bytes(directory) == before


@pytest.mark.parametrize("invalid", ["missing_stage", "extra_stage", "failed", "negative", "nonfinite", "cuda", "naive_time", "backward_time", "short_wall"])
def test_invalid_stage_measurements_fail_before_the_qa_checked_bundle_changes(run, tmp_path, invalid):
    study = PIPE.load_verified_study(run)
    directory = tmp_path / "synthetic_invalid_times"
    fake_bundle(directory, study, passed=True)
    state, elapsed = completed_state(study), 35.
    if invalid == "missing_stage":
        state["stages"].pop(0)
    elif invalid == "extra_stage":
        state["stages"].append(None)
    elif invalid == "failed":
        state["stages"][-1]["returncode"] = 1
    elif invalid == "negative":
        state["stages"][0]["elapsed_seconds"] = -1.
    elif invalid == "nonfinite":
        state["stages"][0]["elapsed_seconds"] = float("inf")
    elif invalid == "cuda":
        state["stages"][0]["cuda_visible_devices"] = "0"
    elif invalid == "naive_time":
        state["stages"][0]["started_at_utc"] = "2026-01-01T00:00:00"
    elif invalid == "backward_time":
        state["stages"][0]["completed_at_utc"] = "2025-01-01T00:00:00+00:00"
    else:
        elapsed = 27.
    before = bundle_bytes(directory)
    with pytest.raises(PIPE.PipelineError, match="timing|completion|elapsed"):
        PIPE.finalize_cpu_postprocessing_evidence(directory, study, state, elapsed)
    assert bundle_bytes(directory) == before


@pytest.mark.parametrize("filename", ["index.html", "data/report_data.json", "qa/browser_qa.json"])
def test_core_byte_change_during_metadata_update_blocks_atomic_publication_and_preserves_stage(run, environment, monkeypatch, filename):
    fake_executor(run, monkeypatch)
    original_write = PIPE.write_json
    changed = []
    def write_and_tamper(path, value):
        original_write(path, value)
        if Path(path).as_posix().endswith(PIPE.CPU_EXECUTION_FILE):
            stage = Path(path).parents[1]
            target = stage / filename
            target.write_bytes(target.read_bytes() + b"\nsynthetic-tampering-stub")
            changed.append(target)
    monkeypatch.setattr(PIPE, "write_json", write_and_tamper)
    original_inputs = {name: (run / name).read_bytes() for name in ("run.json", "campaign_execution.json", "resume_execution.json", "artifact_validation.json")}
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert changed and result["status"] == "failed" and "output bytes changed" in result["error"]
    assert result["staging_preserved"] and Path(result["staging_dir"]).is_dir() and not (run / "html_report").exists()
    assert all((run / name).read_bytes() == value for name, value in original_inputs.items())


@pytest.mark.parametrize("relative", ["component_report_interpretation.py", "render_pgd_comparisons.py", "render_tracking_comparisons.py",
                                     "tracking_projection.py", "render_official_visualizations.py", "../docs/HTML_REPORT_GUIDE.md"])
def test_all_scientific_render_helpers_interpretation_and_guide_are_frozen(run, environment, monkeypatch, relative):
    fake_executor(run, monkeypatch, mutate_source=(environment / relative).resolve())
    result = PIPE.run_pipeline(run, run.parent, CONTAINER_ID)
    assert result["status"] == "failed" and "source changed" in result["error"]
    assert result["staging_preserved"] and not (run / "html_report").exists()


@pytest.mark.parametrize("changed", ["snapshot_phase", "elapsed_until_publication_seconds", "scope"])
def test_reuse_rejects_manifest_timing_scope_that_does_not_match_its_bound_metadata(run, tmp_path, changed):
    study = PIPE.load_verified_study(run)
    directory = tmp_path / "synthetic_scope_tampering"
    fake_bundle(directory, study, passed=True)
    PIPE.finalize_cpu_postprocessing_evidence(directory, study, completed_state(study), 35.)
    manifest = read(directory / "report_manifest.json")
    manifest["cpu_postprocessing_evidence"][changed] = "invented scope cannot be accepted"
    write(directory / "report_manifest.json", manifest)
    with pytest.raises(PIPE.PipelineError, match="metadata update|scopes"):
        PIPE.verify_complete_report(directory, study, require_cpu_evidence=True)
