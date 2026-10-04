"""Resume the unchanged numerical campaign and finalize its verified report."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path("runs/docker/lossstudy_allframes_20261004_183750")
ANALYZER_SHA256 = "c867a873aaffa8086d3ca8c7eeb99cacc2ccbb6f522507a55012c0c6d3c4b9c4"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(run):
    campaign = read(run / "campaign_execution.json")
    audit = read(run / "artifact_validation.json")
    analysis = read(run / "analysis/study_analysis.json")
    if (campaign.get("status") != "complete" or not audit.get("full_campaign_completed")
            or not analysis.get("all_raw_frames_campaign_completed")
            or analysis["validation"]["completed_conditions"] != 48
            or analysis.get("analyzer_source_sha256") != ANALYZER_SHA256):
        raise RuntimeError("Full48 completion or the exact strengthened analyzer was not verified")

    report_path = run / "analysis/study_report.md"
    before = digest(report_path)
    report = report_path.read_text(encoding="utf-8")
    document = analysis["loss_structure_document"]
    target = os.path.relpath(Path("docs/loss_structure.md").resolve(), report_path.parent.resolve()).replace(os.sep, "/")
    report = report.replace(f"]({document})", f"]({target})")
    plan_path = run / "postprocessing_revision.json"
    plan = read(plan_path)
    runtime_note = plan["final_report_text_updates"]["runtime_note_markdown"]
    if runtime_note not in report:
        report = report.rstrip() + "\n\n" + runtime_note + "\n"
    report_path.write_text(report, encoding="utf-8")
    plan.update(status="complete", completed_at_utc=now(), final_report_sha256=digest(report_path),
                final_report_before_text_updates_sha256=before)
    plan["final_report_text_updates"].update(status="complete", applied_at_utc=now())
    write(plan_path, plan)
    return {"completed_conditions": 48, "full_campaign_completed": True,
            "artifact_audit_passed": True, "final_analyzer_sha256": ANALYZER_SHA256,
            "report_relative_path": "analysis/study_report.md",
            "report_sha256": digest(report_path), "report_text_updates_only": True}


def main():
    target = RUN / "resume_execution.json"
    state = {"status": "running", "started_at_utc": now(), "user_requested_resume": True,
             "historical_pause_record": "requested_pause.json", "pause_superseded_by_user": True,
             "completed_before_resume": 24, "expected_conditions": 48,
             "remaining_objectives": ["reconstruction_3d", "confidence", "joint_training"],
             "frames_per_clip": 128, "clips_per_objective": 8, "steps": 20,
             "analyzer_sha256": ANALYZER_SHA256}
    write(target, state)
    try:
        if digest(Path("scripts/analyze_component_study.py")) != ANALYZER_SHA256:
            raise RuntimeError("Mounted analyzer changed before launch")
        command = [sys.executable, "-u", "scripts/run_loss_campaign.py", "--config",
                   "configs/loss_components_8clips_allframes.json", "--manifest",
                   "docker/manifests/loss_components_8clips_allframes.json", "--run-dir", str(RUN), "--resume"]
        code = subprocess.run(command, check=False).returncode
        if code:
            raise RuntimeError(f"Resumed campaign returned exit code {code}")
        state.update(finalize(RUN), status="complete", completed_at_utc=now())
        write(target, state)
        print("RESUMED CAMPAIGN COMPLETE: all48 conditions audited and analyzed.", flush=True)
        return 0
    except Exception as exc:
        state.update(status="failed", failed_at_utc=now(), error=str(exc), full_campaign_completed=False)
        write(target, state)
        print(f"RESUMED CAMPAIGN FAILED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
