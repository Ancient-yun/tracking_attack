#!/usr/bin/env python3
"""Read Docker timestamps and saved results without starting any experiment.

Writes only RUN_DIR/container.log and RUN_DIR/timing.json. Attack durations are
the synchronized attack_video intervals recorded by run_attack.py; Docker wall
time also includes imports, hashes, loading, parity, save/replay, and aggregation.
All phase intervals come from observed stdout boundaries, not new core timers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any


TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\S+)\s(.*)$")
LOADING = re.compile(r"^\[(\d+)/(\d+)\]\s+([^/\s]+)/([^\s]+): loading$")
LABEL = r"(?:clean|(?:noise|fgsm|pgd)_eps[\d.eE+\-]+)"
START = re.compile(rf"^\s*({LABEL}): starting\b")
DONE = re.compile(rf"^\s*({LABEL}): APD=")
SKIP = re.compile(rf"^\s*({LABEL}): already complete\b")
PROGRESS = re.compile(r"^\s*restart (\d+), step (\d+)/(\d+), loss=([^\s]+)")


def _date(value: str | None) -> datetime | None:
    if not value or value.startswith("0001-"):
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _condition_key(dataset: str, sequence: str, label: str) -> str:
    return f"{dataset}/{sequence}/{label}"


def _record_label(record: dict) -> str:
    return "clean" if record["method"] == "clean" else f"{record['method']}_eps{float(record['epsilon_255']):g}"


def _parse_events(raw_log: str) -> tuple[list[dict], dict[str, dict], int]:
    timestamped = []
    for line_number, line in enumerate(raw_log.splitlines()):
        match = TIMESTAMP.match(line)
        if match:
            stamp = _date(match[1])
            if stamp is not None:
                timestamped.append((stamp, line_number, match[2]))
    timestamped.sort(key=lambda entry: (entry[0], entry[1]))
    events, conditions = [], {}
    clip, active = None, None
    for stamp, _number, message in timestamped:
        loading = LOADING.match(message)
        if loading:
            clip = (loading[3], loading[4])
            active = None
            events.append({"time": stamp, "kind": "loading", "clip": "/".join(clip)})
            continue
        if clip is None:
            continue
        if "checking official clean parity" in message:
            events.append({"time": stamp, "kind": "parity", "clip": "/".join(clip)})
            continue
        start, done, skip = START.match(message), DONE.match(message), SKIP.match(message)
        if start or done or skip:
            match = start or done or skip
            key = _condition_key(*clip, match[1])
            condition = conditions.setdefault(key, {"dataset": clip[0], "sequence": clip[1], "label": match[1]})
            if start:
                # Fresh runs contain one start. A resume can revisit a failed
                # condition; retain every interval instead of replacing history.
                condition.setdefault("starts", []).append(stamp)
                condition["start"] = stamp
                active = key
                kind = "condition_start"
            elif done:
                condition["completion"] = stamp
                active = None
                kind = "condition_complete"
            else:
                condition["skip"] = stamp
                active = None
                kind = "condition_skipped"
            events.append({"time": stamp, "kind": kind, "clip": "/".join(clip), "condition": key})
            continue
        progress = PROGRESS.match(message)
        if progress and active:
            loss = float(progress[4])
            conditions[active]["last_progress"] = {
                "at_utc": _iso(stamp), "restart": int(progress[1]),
                "step": int(progress[2]), "total_steps": int(progress[3]),
                "loss": loss if math.isfinite(loss) else None,
            }
        if message.startswith("Results:"):
            events.append({"time": stamp, "kind": "results", "clip": "/".join(clip)})
        elif message.startswith("FAILED "):
            events.append({"time": stamp, "kind": "failure", "clip": "/".join(clip), "condition": active})
            active = None
    return events, conditions, len(timestamped)


def _phase_kind(previous: dict, following: dict) -> str:
    kind = previous["kind"]
    if kind == "container_start":
        return "startup"
    if kind == "loading":
        return "clip_loading_model_transfer"
    if kind == "parity":
        return "official_clean_parity_and_sequence_metadata"
    if kind == "condition_start":
        return "condition_wall"
    if kind in ("condition_complete", "condition_skipped"):
        if following["kind"] == "loading":
            return "between_clips"
        if following["kind"] == "condition_start":
            return "between_conditions"
        return "finalization"
    if kind == "results":
        return "shutdown"
    return "failure_or_other_finalization"


def build_report(run_dir: Path, launch: dict, inspected: dict, raw_log: str,
                 observed_at: datetime | None = None) -> dict[str, Any]:
    """Pure snapshot analysis, also usable with CPU fixture data."""
    observed_at = observed_at or datetime.now(timezone.utc)
    state = inspected["State"]
    start = _date(state.get("StartedAt"))
    finish = _date(state.get("FinishedAt"))
    if start is None:
        raise ValueError("Docker inspect does not contain a real StartedAt timestamp")
    running = bool(state.get("Running"))
    end = observed_at if running or finish is None else finish
    wall = max(0.0, (end - start).total_seconds())
    events, conditions, timestamped_count = _parse_events(raw_log)
    events = [event for event in events if start <= event["time"] <= end]
    boundaries = [{"time": start, "kind": "container_start"}, *events,
                  {"time": end, "kind": "snapshot" if running else "container_finish"}]
    segments, totals = [], defaultdict(float)
    for previous, following in zip(boundaries, boundaries[1:]):
        seconds = max(0.0, (following["time"] - previous["time"]).total_seconds())
        category = _phase_kind(previous, following)
        totals[category] += seconds
        segments.append({"category": category, "start_at_utc": _iso(previous["time"]),
                         "end_at_utc": _iso(following["time"]), "seconds": seconds,
                         "clip": previous.get("clip"), "condition": previous.get("condition"),
                         "ongoing_at_snapshot": following["kind"] == "snapshot"})

    result_records = {}
    for path in sorted((run_dir / "sequences").glob("*/*/*/result.json")):
        record = _read_json(path)
        attack = float(record["elapsed_seconds"])
        if not math.isfinite(attack) or attack < 0:
            raise ValueError(f"Invalid saved attack duration: {path}")
        key = _condition_key(record["dataset"], record["sequence"], _record_label(record))
        if key in result_records:
            raise ValueError(f"Duplicate saved timing identity: {key}")
        result_records[key] = record
        conditions.setdefault(key, {"dataset": record["dataset"], "sequence": record["sequence"],
                                    "label": _record_label(record)})["result_path"] = str(path)

    rendered_conditions = []
    for key, condition in conditions.items():
        record = result_records.get(key)
        began, ended = condition.get("start"), condition.get("completion")
        executed = began is not None
        observed_wall = (ended - began).total_seconds() if began and ended else None
        ongoing_wall = (end - began).total_seconds() if began and ended is None else None
        attack = float(record["elapsed_seconds"]) if record else None
        rendered_conditions.append({
            "identity": key, "dataset": condition["dataset"], "sequence": condition["sequence"],
            "label": condition["label"], "saved_result_present": record is not None,
            "executed_in_this_container": executed,
            "skipped_existing_result": "skip" in condition,
            "start_at_utc": _iso(began), "completion_at_utc": _iso(ended),
            "observed_condition_wall_seconds": observed_wall,
            "ongoing_observed_condition_seconds": ongoing_wall,
            "saved_attack_seconds": attack,
            "saved_attack_method": record["method"] if record else None,
            "observed_condition_wall_minus_attack_seconds": observed_wall - attack if observed_wall is not None and attack is not None else None,
            "result_path": condition.get("result_path"),
            "last_progress": condition.get("last_progress"),
        })
    rendered_conditions.sort(key=lambda value: value["start_at_utc"] or value["identity"])
    measured = [value for value in rendered_conditions if value["executed_in_this_container"] and value["saved_attack_seconds"] is not None]
    attack_sum = sum(value["saved_attack_seconds"] for value in measured)
    pgd_sum = sum(value["saved_attack_seconds"] for value in measured if value["saved_attack_method"] == "pgd")
    clean_sum = sum(value["saved_attack_seconds"] for value in measured if value["saved_attack_method"] == "clean")
    summary = _read_json(run_dir / "summary.json", {})
    metadata = _read_json(run_dir / "run.json", {})
    expected = summary.get("expected_conditions")
    if expected is None and metadata:
        config = metadata["config"]
        count_per_clip = sum(1 if method == "clean" else len(config["epsilons_255"]) for method in config["methods"])
        expected = len(metadata["manifest"]["entries"]) * count_per_clip
    complete = (not running and state.get("Status") == "exited" and state.get("ExitCode") == 0
                and summary.get("complete") is True and expected == len(result_records))
    for category in ("startup", "clip_loading_model_transfer", "official_clean_parity_and_sequence_metadata",
                     "condition_wall", "between_conditions", "between_clips", "finalization", "shutdown",
                     "failure_or_other_finalization"):
        totals.setdefault(category, 0.0)
    setup_sum = totals["clip_loading_model_transfer"] + totals["official_clean_parity_and_sequence_metadata"]
    phases = {name: {"seconds": seconds, "percent_of_container_wall": seconds / wall * 100 if wall else 0.0}
              for name, seconds in sorted(totals.items())}
    requested = _date(launch.get("RequestedAtUtc") or launch.get("requested_at_utc"))
    clip_names = list(dict.fromkeys(event["clip"] for event in events if "clip" in event))
    clips = []
    for clip in clip_names:
        clip_conditions = [condition for condition in rendered_conditions
                           if f"{condition['dataset']}/{condition['sequence']}" == clip]
        clip_segments = [segment for segment in segments if segment.get("clip") == clip]
        clips.append({"identity": clip,
                      "setup_observed_seconds": sum(segment["seconds"] for segment in clip_segments
                                                   if segment["category"] in ("clip_loading_model_transfer", "official_clean_parity_and_sequence_metadata")),
                      "completed_condition_wall_seconds": sum(condition["observed_condition_wall_seconds"] or 0 for condition in clip_conditions),
                      "completed_attack_seconds": sum(condition["saved_attack_seconds"] or 0 for condition in clip_conditions if condition["executed_in_this_container"]),
                      "saved_conditions": sum(condition["saved_result_present"] for condition in clip_conditions)})
    current = next((condition for condition in reversed(rendered_conditions)
                    if condition["start_at_utc"] and condition["completion_at_utc"] is None), None)
    return {
        "schema_version": 1, "complete": complete,
        "status": "running" if running else ("complete" if complete else "incomplete_or_failed"),
        "run_dir": str(run_dir), "container_name": inspected.get("Name", "").lstrip("/"),
        "container_id": inspected.get("Id"), "container_state": state,
        "run_signature": metadata.get("signature"),
        "execution_scope": {"num_clips": len(metadata.get("manifest", {}).get("entries", [])),
                            **{key: metadata.get("config", {}).get(key) for key in
                               ("num_frames", "methods", "epsilons_255", "steps", "restarts")}},
        "snapshot_observed_at_utc": _iso(observed_at),
        "container_started_at_utc": state.get("StartedAt"),
        "container_finished_at_utc": state.get("FinishedAt") if finish else None,
        "container_wall_seconds": wall,
        "container_wall_scope": "Docker StartedAt to FinishedAt; if running, StartedAt to this snapshot",
        "launch_requested_to_container_start_seconds": (start - requested).total_seconds() if requested else None,
        "launch_requested_to_finish_or_snapshot_seconds": (end - requested).total_seconds() if requested else None,
        "completed_conditions": len(result_records), "expected_conditions": expected,
        "completed_attack_seconds_in_this_container": attack_sum,
        "completed_pgd_attack_seconds_in_this_container": pgd_sum,
        "completed_clean_attack_seconds_in_this_container": clean_sum,
        "wall_minus_completed_attack_seconds": wall - attack_sum,
        "wall_minus_completed_attack_interpretation": "overhead outside synchronized attack intervals" if complete else "includes unfinished work and unmeasured attack time; this is not overhead",
        "setup_observed_seconds": setup_sum,
        "setup_percent_of_container_wall": setup_sum / wall * 100 if wall else 0.0,
        "phase_totals": phases, "phase_segments": segments,
        "phase_partition_sum_seconds": sum(segment["seconds"] for segment in segments),
        "phase_partition_minus_wall_seconds": sum(segment["seconds"] for segment in segments) - wall,
        "clips": clips, "conditions": rendered_conditions, "current_condition": current,
        "timestamped_log_lines": timestamped_count,
        "summary_complete": summary.get("complete"), "summary_failed_attempts": summary.get("failed_attempts"),
        "interpretation": [
            "Saved attack seconds include the attack's initial clean forward, PGD recomputation/update work, final candidate forward, and progress logging, with CUDA synchronization.",
            "Observed condition wall also includes the pre-attack synchronization, official metrics, artifact saves, saved-RGB replay and metrics, hashes, and result publication.",
            "Clip loading-to-parity boundaries include decoding/preprocessing, model loading on the first clip, and device transfer; these cannot be separated further with existing logs.",
            "Dataset SHA checks occur before the loading marker, so they fall into startup or between-clips intervals.",
            "Finalization includes cleanup and summarize_run rehashing/re-evaluating all completed artifacts; shutdown follows the final Results log.",
            "Docker daemon log timestamps are observed phase boundaries, not separately instrumented GPU or file-I/O timers.",
            "Image building, prior downloads, and CPU QA run after container exit are outside container_wall_seconds.",
        ],
    }


def _docker(docker: str, arguments: list[str], timeout: int) -> str:
    completed = subprocess.run([docker, *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"Docker {' '.join(arguments[:2])} failed ({completed.returncode}): {completed.stdout.strip()}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--docker", default="docker", help="Docker CLI path")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")
    output = run_dir / "timing.json"
    try:
        launch = _read_json(run_dir / "launch.json")
        if not isinstance(launch, dict):
            raise FileNotFoundError(f"Missing launch.json in {run_dir}")
        container = launch.get("ContainerId") or launch.get("container_id") or launch.get("ContainerName") or launch.get("container_name")
        if not isinstance(container, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container):
            raise ValueError("launch.json contains no valid Docker container identity")
        first = json.loads(_docker(args.docker, ["inspect", container], 15))[0]
        raw_log = _docker(args.docker, ["logs", "--timestamps", container], 30)
        inspected = json.loads(_docker(args.docker, ["inspect", container], 15))[0]
        if first["State"].get("Running") and not inspected["State"].get("Running"):
            raw_log = _docker(args.docker, ["logs", "--timestamps", container], 30)
        if launch.get("ContainerId") and inspected.get("Id") != launch["ContainerId"]:
            raise ValueError("Inspected container differs from launch.json ContainerId")
        (run_dir / "container.log").write_text(raw_log, encoding="utf-8")
        report = build_report(run_dir, launch, inspected, raw_log)
        _write_json(output, report)
        print(json.dumps({"status": report["status"], "complete": report["complete"],
                          "container_wall_seconds": report["container_wall_seconds"],
                          "completed_conditions": report["completed_conditions"],
                          "expected_conditions": report["expected_conditions"],
                          "completed_pgd_attack_seconds": report["completed_pgd_attack_seconds_in_this_container"],
                          "current_condition": report["current_condition"]["identity"] if report["current_condition"] else None,
                          "timing_json": str(output)}, ensure_ascii=False))
        return 0 if report["status"] in ("running", "complete") else 1
    except Exception as error:
        _write_json(output, {"complete": False, "status": "query_failed", "run_dir": str(run_dir),
                             "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                             "error": repr(error), "traceback": traceback.format_exc()})
        print(json.dumps({"status": "query_failed", "error": repr(error), "timing_json": str(output)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
