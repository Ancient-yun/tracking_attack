"""Resume the frozen study and exit at the user-requested loss boundary.

This external control wrapper does not edit the signed numerical source,
configuration, manifest, or any completed condition. The original campaign
driver is deliberately not invoked, since its next stages require all losses.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


STOP_OBJECTIVE = "tracking_3d"
ALLOWED_OBJECTIVES = {"tracking_mse", STOP_OBJECTIVE}
RUN_DIR = Path("runs/docker/lossstudy_allframes_20261004_183750")


def install_boundary_stop(study, run_dir=RUN_DIR):
    original_event = study.event

    def boundary_event(directory, kind, **values):
        if kind == "objective_started" and values.get("objective") not in ALLOWED_OBJECTIVES:
            raise SystemExit("User-requested boundary blocks the next loss objective")
        original_event(directory, kind, **values)
        if kind != "objective_completed" or values.get("objective") != STOP_OBJECTIVE:
            return

        summary = study.summarize_component_study(directory, verify=False)
        meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        entries = meta["manifest"]["entries"]
        if len(entries) != 8 or meta["config"]["num_frames"] != 128:
            raise SystemExit("Requested stop roster/frame count differs from the frozen study")
        for objective in ("clean", "tracking_mse", STOP_OBJECTIVE):
            for entry in entries:
                base = directory / "conditions" / objective / entry["dataset"] / entry["sequence"]
                record = json.loads((base / "result.json").read_text(encoding="utf-8"))
                if (record["signature"] != meta["signature"]
                        or record["tracking_metrics"]["num_frames"] != 128
                        or record["reconstruction_metrics"]["num_frames"] != 128
                        or not record["saved_input_verification"]["passed"]
                        or (objective != "clean" and record["steps"] != 20)):
                    raise SystemExit("Completed condition failed the requested-stop metadata checks")
        future = set(meta["config"]["objectives"]) - ALLOWED_OBJECTIVES
        if any((directory / "conditions" / objective).exists() for objective in future):
            raise SystemExit("Unexpected future-loss artifacts exist")
        if summary["completed_conditions"] != 24 or summary["complete"]:
            raise SystemExit("Requested stop must retain an honest partial 24/48 summary")

        request = json.loads((directory / "pause_request.json").read_text(encoding="utf-8"))
        request.update(status="paused", paused_at_utc=study.utc_now(),
                       completed_conditions=24, original_expected_conditions=48,
                       completed_attack_objectives=["tracking_mse", STOP_OBJECTIVE],
                       not_started_objectives=sorted(future),
                       frames_per_clip=128, clips_per_objective=8,
                       full_campaign_completed=False,
                       boundary_verified=True,
                       verification_scope="signed result metadata and saved-input replay records; not the full 48-condition independent audit")
        study.write_json(directory / "requested_pause.json", request)
        original_event(directory, "user_requested_pause", objective=STOP_OBJECTIVE,
                       completed_conditions=24, expected_conditions=48,
                       next_objectives_started=False)
        print("USER REQUESTED PAUSE: tracking_3d finished for all 8 clips; later losses were not started.", flush=True)
        raise SystemExit(0)

    study.event = boundary_event
    return boundary_event


def main():
    import st4rtrack_pgd.run_component_study as study

    # Refuse to reinterpret a run in which a later loss has already started.
    for line in (RUN_DIR / "events.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["event"] == "objective_started" and item["objective"] not in ALLOWED_OBJECTIVES:
            raise SystemExit("A later loss already started; cannot claim this requested stop boundary")
    install_boundary_stop(study)
    sys.argv = [__file__, "--config", "configs/loss_components_8clips_allframes.json",
                "--manifest", "docker/manifests/loss_components_8clips_allframes.json",
                "--run-dir", str(RUN_DIR), "--resume", "--fail-fast"]
    study.main()
    raise SystemExit("Boundary exit was not reached; refusing normal campaign completion")


if __name__ == "__main__":
    main()
