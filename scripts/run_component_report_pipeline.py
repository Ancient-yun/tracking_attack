"""Foreground CPU postprocessing of an already completed Docker experiment.

This program never waits for the experiment, resumes it, schedules work, loads
a model, or changes campaign evidence. Use --check-only for read-only readiness.
Reports are published by a same-directory atomic rename only after browser QA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from component_report_data import (OBJECTIVES, ReportValidationError,
                                   load_verified_study, read_json, sha_file)


class PipelineError(ReportValidationError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise PipelineError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inside(root, name):
    root = Path(root).resolve()
    text = str(name).replace("\\", "/")
    require(text and ":" not in text and not text.startswith("/") and all(part not in ("", ".", "..") for part in text.split("/")),
            f"Unsafe report path: {name}")
    path = (root / text).resolve()
    require(path.is_relative_to(root), f"Report path escapes output: {name}")
    return path


def validate_run_paths(run_dir):
    root = Path(run_dir).resolve()
    require(root.is_dir(), f"Run directory does not exist: {root}")
    for name in ("html_report", "html_postprocessing.json", "html_postprocessing.json.tmp", "html_postprocessing.lock", "html_postprocessing_logs"):
        require((root / name).resolve().is_relative_to(root), f"Postprocessing target escapes run: {name}")
    return root


def inspect_container(container_id, docker="docker"):
    require(isinstance(container_id, str) and len(container_id) == 64 and all(c in "0123456789abcdef" for c in container_id),
            "An exact full Docker container ID is required")
    result = subprocess.run([str(docker), "inspect", container_id], capture_output=True, text=True, check=False,
                            timeout=30, encoding="utf-8", errors="replace")
    require(result.returncode == 0, f"Docker inspection failed: {result.stderr.strip()}")
    try:
        values = json.loads(result.stdout)
    except ValueError as exc:
        raise PipelineError("Docker inspection did not return JSON") from exc
    require(isinstance(values, list) and len(values) == 1 and values[0].get("Id") == container_id,
            "Docker inspection did not identify the pinned container")
    state = values[0].get("State", {})
    require(state.get("Status") == "exited" and state.get("Running") is False and state.get("ExitCode") == 0
            and state.get("OOMKilled") is False, "Pinned experiment container has not exited successfully without OOM")
    return {"id": container_id, "name": values[0].get("Name"), "status": state["Status"], "exit_code": state["ExitCode"],
            "oom_killed": state["OOMKilled"], "started_at": state.get("StartedAt"), "finished_at": state.get("FinishedAt")}


def check_readiness(run_dir, container_id, analysis_dir=None, docker="docker"):
    """Only Docker inspection and scalar reads; no status/output is written."""
    proof = {"schema_version": 1, "checked_at_utc": now(), "ready": False, "status": "not_ready", "container_id": container_id}
    try:
        root = validate_run_paths(run_dir)
        proof["container"] = inspect_container(container_id, docker)
        study = load_verified_study(root, analysis_dir=analysis_dir, require_complete=True)
        proof.update(ready=True, status="ready", run_signature=study["metadata"]["signature"],
                     analysis_json_sha256=study["analysis_sha256"], scalar_input_sha256=study["input_sha256"],
                     scope=study["analysis"]["scope"])
    except (ReportValidationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        proof["reason"] = str(exc)
    return proof


def default_browser_runtime():
    bundle = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    nodes = [bundle / "node/bin/node.exe", bundle / "node/bin/node"]
    node = next((p for p in nodes if p.is_file()), None)
    if node is None and shutil.which("node"):
        node = Path(shutil.which("node"))
    packages = [bundle / "node/node_modules/playwright", bundle / "node/node_modules/playwright-core"]
    playwright = next((p for p in packages if p.is_dir()), None)
    browser_candidates = []
    for env_name in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        folder = os.environ.get(env_name)
        if folder:
            browser_candidates += [Path(folder) / "Microsoft/Edge/Application/msedge.exe",
                                   Path(folder) / "Google/Chrome/Application/chrome.exe"]
    for cache in (Path.home() / ".cache/ms-playwright", Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ms-playwright"):
        if cache.is_dir():
            browser_candidates += sorted(cache.glob("chromium*/chrome-win*/chrome.exe"), reverse=True)
            browser_candidates += sorted(cache.glob("chromium*/chrome-linux*/chrome"), reverse=True)
    browser = next((p for p in browser_candidates if p.is_file()), None)
    if browser is None:
        located = shutil.which("chromium") or shutil.which("google-chrome")
        browser = Path(located) if located else None
    return node, playwright, browser


def verify_complete_report(report_dir, study):
    """Validate current QA/manifest/output bytes, including a relocated bundle."""
    root = Path(report_dir).resolve()
    require(root.is_dir(), f"Report directory missing: {root}")
    manifest = read_json(root / "report_manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("run_signature") == study["metadata"]["signature"],
            "Report is not complete for this frozen run")
    require(manifest.get("scope") == study["analysis"]["scope"] and manifest.get("run_id") == Path(study["run_dir"]).name,
            "Report scope/identity differs from the completed experiment")
    require(manifest.get("analysis_json_sha256") == study["analysis_sha256"]
            and manifest.get("artifact_validation_sha256") == study["input_sha256"]["artifact_validation.json"]
            and manifest.get("scalar_input_sha256") == study["input_sha256"], "Report is bound to stale experiment evidence")
    require(manifest.get("errors") == [] and manifest.get("missing") == [], "Report manifest has failures or missing assets")
    browser = manifest.get("browser_qa", {})
    require(browser.get("status") == "passed" and browser.get("synthetic_only") is False,
            "Real browser QA did not pass (synthetic proofs cannot publish a report)")
    qa = read_json(root / "qa/browser_qa.json")
    require(qa.get("status") == "passed" and qa.get("synthetic_only") is False and qa.get("errors") == [], "QA proof is incomplete or synthetic")
    require(qa.get("index_sha256") == sha_file(root / "index.html"), "Browser QA does not bind the current HTML")
    require(qa.get("run_signature") == study["metadata"]["signature"]
            and qa.get("report_data_sha256") == sha_file(root / "data/report_data.json"), "Browser QA data/run binding differs")
    require(qa.get("cpu_only") is True and qa.get("new_model_inference") is False
            and qa.get("original_folder", {}).get("passed") is True and qa.get("portable_folder", {}).get("passed") is True,
            "Both original-folder and portable-copy browser QA must pass without new inference")
    require(browser.get("index_sha256") == qa["index_sha256"] and browser.get("report_data_sha256") == qa["report_data_sha256"]
            and browser.get("report") == "qa/browser_qa.json" and browser.get("sha256") == sha_file(root / "qa/browser_qa.json")
            and browser.get("actual_video_playback") is True and browser.get("portable_copy_verified") is True
            and browser.get("external_network_requests") == 0, "Manifest does not bind actual offline playback/portable QA proof")
    outputs = manifest.get("outputs", {})
    require(isinstance(outputs, dict) and {"index.html", "data/report_data.json", "assets/visualization_manifest.json", "qa/browser_qa.json"} <= set(outputs),
            "Report output inventory is incomplete")
    verified = {}
    for relative, reference in outputs.items():
        path = inside(root, relative)
        require(path.is_file() and sha_file(path) == reference.get("sha256") and path.stat().st_size == reference.get("bytes"),
                f"Report output SHA/size differs: {relative}")
        verified[relative] = reference["sha256"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "report_manifest.json"}
    require(actual == set(outputs), "Report output inventory does not cover its exact file set")
    return {"status": "complete", "index_sha256": verified["index.html"], "outputs_sha256": verified,
            "report_manifest_sha256": sha_file(root / "report_manifest.json"), "qa_proof_sha256": verified["qa/browser_qa.json"]}


def run_pipeline(run_dir, project_root, container_id, analysis_dir=None, *, check_only=False, node=None,
                 playwright_module=None, browser_executable=None, fps=10., docker="docker", python=None):
    readiness = check_readiness(run_dir, container_id, analysis_dir, docker)
    if check_only or not readiness["ready"]:
        return readiness
    root = validate_run_paths(run_dir)
    project = Path(project_root).resolve()
    require(project.is_dir(), "Project root must exist")
    require(isinstance(fps, (int, float)) and not isinstance(fps, bool) and math.isfinite(fps) and fps > 0, "Preview FPS must be finite and positive")
    final = root / "html_report"
    target = root / "html_postprocessing.json"
    lock = root / "html_postprocessing.lock"
    attempt = uuid.uuid4().hex
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"attempt_id": attempt, "pid": os.getpid(), "started_at_utc": now()}, stream)
    except FileExistsError as exc:
        raise PipelineError("Another postprocessing lock already exists; no duplicate pipeline was started") from exc
    started = time.perf_counter()
    state = {"schema_version": 1, "attempt_id": attempt, "status": "running", "started_at_utc": now(),
             "cpu_only": True, "new_model_inference": False, "numerical_experiment_modified": False,
             "readiness": readiness, "stages": [], "staging_dir": None, "output_dir": str(final)}
    stage_directory, published = None, False
    try:
        write_json(target, state)
        study = load_verified_study(root, analysis_dir=analysis_dir, require_complete=True)
        if final.exists():
            proof = verify_complete_report(final, study)
            state.update(status="complete", reused=True, verification=proof, completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started)
            write_json(target, state)
            return state
        defaults = default_browser_runtime()
        node = Path(node or defaults[0]).resolve() if node or defaults[0] else None
        playwright = Path(playwright_module or defaults[1]).resolve() if playwright_module or defaults[1] else None
        browser = Path(browser_executable or defaults[2]).resolve() if browser_executable or defaults[2] else None
        require(node is not None and node.is_file(), "Node executable missing; pass --node")
        require(playwright is not None and playwright.exists(), "Playwright module missing; pass --playwright-module")
        require(browser is not None and browser.is_file(), "Browser executable missing; pass --browser-executable")
        scripts = Path(__file__).resolve().parent
        source_paths = [Path(__file__), scripts / "component_report_data.py", scripts / "component_projection.py",
                        scripts / "render_component_comparisons.py", scripts / "build_component_html_report.py",
                        scripts / "verify_component_html_report.cjs", scripts.parent / "templates/component_report.html"]
        require(all(path.is_file() for path in source_paths), "Pipeline source, browser QA script, or template is missing")
        source_sha = {str(p): sha_file(p) for p in source_paths}
        state["source_code_sha256"] = source_sha
        state["browser_runtime"] = {"node": str(node), "playwright_module": str(playwright), "browser_executable": str(browser)}
        stage_directory = root / ("html_report.staging-" + attempt)
        require(stage_directory.resolve().parent == root and not stage_directory.exists(), "Unsafe/existing staging output")
        stage_directory.mkdir()
        log_directory = root / "html_postprocessing_logs" / attempt
        log_directory.mkdir(parents=True)
        state["staging_dir"] = str(stage_directory)
        analysis_root = Path(study["analysis_dir"])
        interpreter = str(python or sys.executable)

        def execute(label, argv):
            stage = {"stage": label, "argv": [str(a) for a in argv], "status": "running", "started_at_utc": now(),
                     "cpu_only": True, "new_model_inference": False, "cuda_visible_devices": "",
                     "log": str((log_directory / (label.replace(":", "_") + ".log")).relative_to(root))}
            state["stages"].append(stage)
            write_json(target, state)
            stage_started = time.perf_counter()
            child_env = dict(os.environ)
            child_env["CUDA_VISIBLE_DEVICES"] = ""
            with (root / stage["log"]).open("w", encoding="utf-8") as output:
                result = subprocess.run(stage["argv"], cwd=str(project), env=child_env, stdout=output, stderr=subprocess.STDOUT, check=False)
            stage.update(returncode=result.returncode, status="complete" if result.returncode == 0 else "failed",
                         completed_at_utc=now(), elapsed_seconds=time.perf_counter() - stage_started)
            write_json(target, state)
            require(result.returncode == 0, f"Postprocessing stage failed: {label}; see {stage['log']}")

        for objective in OBJECTIVES:
            execute("render:" + objective, [interpreter, "-u", scripts / "render_component_comparisons.py", "--run-dir", root,
                    "--project-root", project, "--analysis-dir", analysis_root, "--output-dir", stage_directory / "assets",
                    "--objective", objective, "--fps", str(fps), "--preview-frames", "0", "64", "127"])
        execute("build", [interpreter, "-u", scripts / "build_component_html_report.py", "--run-dir", root,
                "--analysis-dir", analysis_root, "--assets-dir", stage_directory / "assets", "--output-dir", stage_directory,
                "--language", "ko", "--require-complete"])
        execute("browser_qa", [node, scripts / "verify_component_html_report.cjs", "--report-dir", stage_directory,
                "--playwright-module", playwright, "--browser-executable", browser])
        proof = verify_complete_report(stage_directory, study)
        refreshed = check_readiness(root, container_id, analysis_root, docker)
        require(refreshed["ready"] and refreshed["analysis_json_sha256"] == readiness["analysis_json_sha256"]
                and refreshed["scalar_input_sha256"] == readiness["scalar_input_sha256"], "Experiment evidence changed during postprocessing")
        require(all(sha_file(Path(p)) == digest for p, digest in source_sha.items()), "Postprocessing source changed during execution")
        require(not final.exists() and final.resolve().parent == root, "Final report appeared or escaped the run; refusing to replace it")
        os.replace(stage_directory, final)
        published = True
        final_proof = verify_complete_report(final, study)
        require(final_proof == proof, "Report bytes changed during atomic publication")
        state.update(status="complete", reused=False, published_atomic=True, verification=final_proof,
                     completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started, staging_preserved=False,
                     readiness_at_publication=refreshed)
        write_json(target, state)
        return state
    except Exception as exc:
        if published and stage_directory is not None and final.is_dir() and not stage_directory.exists():
            os.replace(final, stage_directory)  # Preserve this attempt; do not leave failed final HTML.
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at_utc=now(),
                     elapsed_seconds=time.perf_counter() - started,
                     staging_preserved=bool(stage_directory is not None and stage_directory.is_dir()))
        write_json(target, state)
        return state
    finally:
        if lock.is_file() and read_json(lock).get("attempt_id") == attempt:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--container-id", required=True, help="Exact full64 id of the finished experiment container")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--node", type=Path)
    parser.add_argument("--playwright-module", type=Path)
    parser.add_argument("--browser-executable", type=Path)
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--docker", default="docker", help="Docker executable, used only for readonly inspection")
    parser.add_argument("--python", type=Path, help="Optional child Python; default is this process's interpreter")
    args = parser.parse_args()
    try:
        result = run_pipeline(**vars(args))
    except (ReportValidationError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "ready", "reason", "error", "output_dir", "reused") if key in result}, ensure_ascii=False))
    return 0 if result["status"] in ("ready", "complete") else 2 if result["status"] == "not_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
