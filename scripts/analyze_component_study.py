"""Read-only CPU analysis of matched St4RTrack loss-component experiments.

Only the output directory is written. No model, CUDA, Docker, saved predictions
or source data are loaded or modified. Metric records are checked against the
CSV/summary; the separate artifact auditor owns large-array/source validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

OBJECTIVES = ("tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training")
DATASETS = ("po_mini", "ds_mini")
COMMIT = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"
CHANGE_FIELDS = {
    "tracking_apd_drop_pp": ("tracking_metrics", "apd3d_all", -1, "percentage_points"),
    "tracking_epe_increase_m": ("tracking_metrics", "epe_all_m", 1, "metres"),
    "reconstruction_apd_drop_pp": ("reconstruction_metrics", "apd3d", -1, "percentage_points"),
    "reconstruction_epe_increase_m": ("reconstruction_metrics", "epe_m", 1, "metres"),
    "tracking_native_l21_change": ("diagnostics", "tracking_l21", 1, "dimensionless"),
    "reconstruction_native_l21_change": ("diagnostics", "reconstruction_l21", 1, "dimensionless"),
    "track_conf_mean_change": ("diagnostics", "track_conf_mean", 1, "confidence_score"),
    "track_raw_conf_mean_change": ("diagnostics", "track_raw_conf_mean", 1, "confidence_score"),
    "reconstruction_conf_mean_change": ("diagnostics", "reconstruction_conf_mean", 1, "confidence_score"),
}
CI_FIELDS = tuple(CHANGE_FIELDS)[:4]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    return float(np.mean(values)) if values else None


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("event timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("event timestamp must include its UTC offset")
    return parsed.astimezone(timezone.utc)


def same_number(first, second):
    return finite(first) and finite(second) and bool(np.isclose(first, second, rtol=1e-7, atol=1e-8))


def scope_of(metadata):
    manifest, config = metadata["manifest"], metadata["config"]
    ids = [(row["dataset"], row["sequence"]) for row in manifest["entries"]]
    counts = Counter(dataset for dataset, _ in ids)
    markers = " ".join(str(value) for value in (
        manifest.get("purpose", ""), manifest.get("scope", ""),
        config.get("scope", ""), metadata.get("scope", ""), metadata.get("analysis_scope", ""),
    )).lower()
    limited = any(word in markers for word in ("preflight", "limited", "synthetic", "smoke", "pilot"))
    declaration_key = "campaign_expected_clips"
    declarations = {"config": config.get(declaration_key), "manifest": manifest.get(declaration_key)}
    present = [declaration_key in source for source in (config, manifest)]
    declaration_errors = []
    if not any(present):
        campaign_clips = 14
        declaration_mode = "legacy_default_14"
    else:
        declaration_mode = "explicit"
        if not all(present):
            declaration_errors.append("campaign_expected_clips must be explicitly declared in both config and manifest")
        for source, value in declarations.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 2:
                declaration_errors.append(f"{source} campaign_expected_clips must be a positive even integer")
        if all(present) and declarations["config"] != declarations["manifest"]:
            declaration_errors.append("config and manifest campaign_expected_clips declarations differ")
        campaign_clips = declarations["config"] if not declaration_errors else None
    frame_keys = ("campaign_expected_frames", "all_frames")
    frame_declarations = {source: {key: values.get(key) for key in frame_keys}
                          for source, values in (("config", config), ("manifest", manifest))}
    if not any(key in source for source in (config, manifest) for key in frame_keys):
        campaign_frames, all_frames, frame_policy = 64, False, "legacy_frame_prefix"
    else:
        campaign_frames, frame_policy = 128, "all_raw_128_frames"
        all_frames = all(source.get("all_frames") is True for source in (config, manifest))
        for source, values in (("config", config), ("manifest", manifest)):
            if not all(key in values for key in frame_keys):
                declaration_errors.append(f"{source} must declare both campaign_expected_frames and all_frames")
            value = values.get("campaign_expected_frames")
            if not isinstance(value, int) or isinstance(value, bool) or value != 128:
                declaration_errors.append(f"{source} campaign_expected_frames must be the raw release length 128")
            if values.get("all_frames") is not True:
                declaration_errors.append(f"{source} all_frames must be explicitly true for the 128-frame campaign")
            actual_frames = values.get("num_frames")
            if not isinstance(actual_frames, int) or isinstance(actual_frames, bool) or actual_frames != 128:
                declaration_errors.append(f"{source} num_frames must equal all 128 raw frames")
        if frame_declarations["config"] != frame_declarations["manifest"]:
            declaration_errors.append("config and manifest full-frame declarations differ")
    if manifest.get("num_frames", config["num_frames"]) != config["num_frames"]:
        declaration_errors.append("config and manifest num_frames differ")
    full = (not declaration_errors and campaign_clips is not None and len(ids) == campaign_clips and len(set(ids)) == len(ids)
            and all(counts[d] == campaign_clips // 2 for d in DATASETS)
            and len(config["objectives"]) == len(OBJECTIVES) and set(config["objectives"]) == set(OBJECTIVES)
            and config["num_frames"] == campaign_frames and config["steps"] == 20
            and config["epsilon_255"] == 4 and not limited)
    return {"kind": "full_campaign" if full else "explicit_limited_preflight" if limited else "unrecognized",
            "clips": len(ids), "dataset_counts": dict(counts), "num_frames": config["num_frames"],
            "steps": config["steps"], "epsilon_255": config["epsilon_255"],
            "objectives": list(config["objectives"]), "declared_scope": markers,
            "campaign_expected_clips": campaign_clips, "campaign_declaration_mode": declaration_mode,
            "campaign_declarations": declarations, "declaration_errors": declaration_errors,
            "campaign_expected_frames": campaign_frames, "all_frames": all_frames,
            "frame_policy": frame_policy, "frame_declarations": frame_declarations,
            "expected_conditions": len(ids) * (1 + len(config["objectives"])),
            "full_campaign_expected_conditions": campaign_clips * (1 + len(OBJECTIVES)) if campaign_clips is not None else None}


def raw_frame_evidence(run_dir, ids, scope, signature, hashes, errors):
    """Validate scalar release-length evidence without loading model arrays.

    Historical prefix experiments did not record the raw release length. They
    remain analyzable, but cannot establish completion of an all-frame study.
    """
    if scope["frame_policy"] != "all_raw_128_frames":
        return []
    evidence = []
    for dataset, sequence in sorted(ids):
        directory = run_dir / "sequences" / dataset / sequence
        metadata_path, inventory_path = directory / "sequence.json", directory / "sequence_manifest.json"
        row = {"dataset": dataset, "sequence": sequence, "passed": False}
        try:
            metadata, inventory = read_json(metadata_path), read_json(inventory_path)
            for path in (metadata_path, inventory_path):
                hashes[str(path.relative_to(run_dir))] = sha_file(path)
            row.update({key: metadata.get(key) for key in ("num_frames", "requested_num_frames", "original_frame_count")})
            if any(not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] != 128
                   for key in ("num_frames", "requested_num_frames", "original_frame_count")):
                raise ValueError("num_frames/requested_num_frames/original_frame_count must all equal raw release length 128")
            if inventory.get("signature") != signature:
                raise ValueError("sequence inventory signature does not match frozen run")
            if inventory.get("artifact_sha256", {}).get("sequence.json") != hashes[str(metadata_path.relative_to(run_dir))]:
                raise ValueError("sequence original-frame metadata SHA differs from its inventory")
            row["passed"] = True
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"{metadata_path.relative_to(run_dir)}: {exc}")
        evidence.append(row)
    return evidence


def audit_completion_evidence(audit, scope, hashes):
    """Bind an independent audit to this scope and the consumed saved evidence.

    Hash keys can be absolute container or host paths. Match their logical run
    suffixes so a valid audit remains usable after moving between those hosts.
    Historical audits need not contain the later all-frame declaration fields.
    """
    problems = []
    if audit.get("status") != "passed" or audit.get("passed") is not True:
        return False, ["independent artifact audit is not explicitly passed"]
    if audit.get("errors") != [] or audit.get("missing") != []:
        problems.append("passed artifact audit contains errors or missing evidence")
    full = scope["kind"] == "full_campaign"
    if full:
        if audit.get("full_campaign_completed") is not True:
            problems.append("artifact audit does not establish full campaign completion")
        if audit.get("source_files_verified") is not True:
            problems.append("full campaign audit did not verify original/source files")
        if scope["all_frames"]:
            if audit.get("all_frames_completed") is not True or audit.get("all_frames") is not True:
                problems.append("artifact audit does not establish all raw frames completion")
        elif audit.get("all_frames_completed", False) is not False or audit.get("all_frames", False) is not False:
            problems.append("historical prefix audit incorrectly claims all raw frames")
        expected_clips = scope["campaign_expected_clips"]
        clip_declaration = audit.get("campaign_expected_clips", 14 if scope["campaign_declaration_mode"] == "legacy_default_14" else None)
        if type(clip_declaration) is not int or clip_declaration != expected_clips:
            problems.append("artifact audit campaign clip declaration differs from frozen run")
        frame_declaration = audit.get("campaign_expected_frames", 64 if not scope["all_frames"] else None)
        if type(frame_declaration) is not int or frame_declaration != scope["campaign_expected_frames"]:
            problems.append("artifact audit campaign frame declaration differs from frozen run")
        full_count = audit.get("full_campaign_expected_conditions", scope["full_campaign_expected_conditions"] if not scope["all_frames"] else None)
        if type(full_count) is not int or full_count != scope["full_campaign_expected_conditions"]:
            problems.append("artifact audit full campaign condition count differs from frozen run")
    requested = audit.get("requested_scope")
    expected_scope = {"clips": scope["clips"], "frames": scope["num_frames"], "steps": scope["steps"]}
    if full or requested is not None:
        if not isinstance(requested, dict) or any(type(requested.get(key)) is not int or requested.get(key) != value
                                                  for key, value in expected_scope.items()):
            problems.append("artifact audit requested scope differs from actual run")
    for key in ("expected_conditions", "completed_conditions_checked"):
        if full or key in audit:
            if type(audit.get(key)) is not int or audit[key] != scope["expected_conditions"]:
                problems.append(f"artifact audit {key} differs from actual run")
    if full:
        audited_hashes = audit.get("file_sha256", {})
        if not isinstance(audited_hashes, dict):
            audited_hashes = {}
        for relative, current_hash in hashes.items():
            if relative in ("artifact_validation.json", "events.jsonl"):
                continue  # Events are recorded timing evidence, not audited arrays.
            suffix = relative.replace("\\", "/")
            matches = [value for key, value in audited_hashes.items()
                       if str(key).replace("\\", "/") == suffix or str(key).replace("\\", "/").endswith("/" + suffix)]
            if len(matches) != 1 or matches[0] != current_hash:
                problems.append(f"artifact audit SHA does not bind current saved evidence: {relative}")
    return not problems, problems


def history_diagnostics(history, record, config):
    """Descriptive stalls; gradient magnitudes have objective-specific units."""
    losses = [row.get("loss") for row in history]
    if not history or not all(finite(value) for value in losses):
        raise ValueError("history is empty or has a nonfinite objective")
    selected = record["selected_state"]
    matching = [row for row in history if all(row.get(k) == v for k, v in selected.items())]
    if len(matching) != 1 or not same_number(matching[0]["loss"], record["attack_loss"]):
        raise ValueError("selected state does not match the recorded attack objective")
    if record["objective"] != "clean" and not same_number(max(losses), record["attack_loss"]):
        raise ValueError("saved PGD state is not the clean-inclusive best objective")
    derivatives = [row for row in history if "gradient_l2_per_frame" in row]
    expected_steps = config["steps"] * config["restarts"] if record["objective"] != "clean" else 0
    if len(derivatives) != expected_steps:
        raise ValueError("history does not contain the requested gradient iterations")
    gradient_norms = []
    for row in derivatives:
        values = row["gradient_l2_per_frame"]
        if len(values) != config["num_frames"] or not all(finite(v) and v >= 0 for v in values):
            raise ValueError("history gradient norms have invalid frame support")
        gradient_norms.extend(values)
    within_restart = defaultdict(list)
    for row in history:
        if row["restart"] >= 0:
            within_restart[row["restart"]].append(row)
    transitions = []
    for rows in within_restart.values():
        rows.sort(key=lambda row: row["step"])
        for first, second in zip(rows, rows[1:]):
            transitions.append(abs(second["loss"] - first["loss"]) <= 1e-8 * max(1, abs(first["loss"])))
    derivative_restarts = defaultdict(list)
    for row in derivatives:
        derivative_restarts[row["restart"]].append(row)
    gradient_ratios = []
    for rows in derivative_restarts.values():
        rows.sort(key=lambda row: row["step"])
        first_norm = float(np.median(rows[0]["gradient_l2_per_frame"]))
        last_norm = float(np.median(rows[-1]["gradient_l2_per_frame"]))
        if first_norm > 0:
            gradient_ratios.append(last_norm / first_norm)
    selected_kind = ("clean" if selected["restart"] == -1 else "initialization" if selected["step"] == 0
                     else "final" if selected["step"] == config["steps"] else "earlier_iterate")
    return {"selected_state": selected, "selected_kind": selected_kind,
            "recorded_gradient_steps": len(derivatives),
            "zero_gradient_frame_fraction": mean([int(v == 0) for v in gradient_norms]),
            "tiny_gradient_frame_fraction": mean([int(v <= 1e-12) for v in gradient_norms]),
            "gradient_l2_per_frame_median": float(np.median(gradient_norms)) if gradient_norms else None,
            "last_over_initial_median_gradient_l2": mean(gradient_ratios),
            "relative_gradient_collapse_definition": "per restart median(last recorded frame L2)/median(initial frame L2); final saved state may be an earlier iterate",
            "near_flat_transition_fraction": mean([int(v) for v in transitions]),
            "near_flat_definition": "abs(next_loss-loss) <= 1e-8*max(1,abs(loss)); within each restart",
            "native_stop_gradient_warning": "detached dynamic weights are recomputed between steps; loss need not rise monotonically"}


def record_value(record, section, field):
    container = record["loss_terms"]["diagnostics"] if section == "diagnostics" else record[section]
    value = container[field]
    if not finite(value):
        raise ValueError(f"nonfinite metric: {section}.{field}")
    return float(value)


def paired_row(record, clean, history):
    row = {key: record[key] for key in ("dataset", "sequence", "objective", "seed", "elapsed_seconds")}
    row["condition_wall_seconds"] = record.get("condition_wall_seconds")
    row["history"] = history
    for name, (section, field, sign, unit) in CHANGE_FIELDS.items():
        clean_value = record_value(clean, section, field)
        attacked_value = record_value(record, section, field)
        row[name] = sign * (attacked_value - clean_value)
        row[name + "_clean"] = clean_value
        row[name + "_attacked"] = attacked_value
    return row


def bootstrap_intervals(rows, objectives, common_ids, *, samples=2000, seed=20261004):
    """Paired, stratified percentile bootstrap, reusing indices for all losses.

    Each stratum retains its original number of common clips. When both PO and
    DR are represented, all uses 50/50 dataset weight, never a pooled bootstrap.
    """
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    strata = {dataset: sorted(pair for pair in common_ids if pair[0] == dataset) for dataset in DATASETS}
    strata = {dataset: pairs for dataset, pairs in strata.items() if pairs}
    if not strata:
        return {"samples": samples, "seed": seed, "common_clips": 0, "estimates": [],
                "reason": "no clips completed for every requested objective plus clean"}
    resamples = {dataset: rng.integers(0, len(pairs), size=(samples, len(pairs)))
                 for dataset, pairs in strata.items()}
    indexed = {(row["dataset"], row["sequence"], row["objective"]): row for row in rows}
    estimates = []
    for objective in objectives:
        distributions, points = {}, {}
        for dataset, pairs in strata.items():
            values = np.array([[indexed[(d, clip, objective)][field] for field in CI_FIELDS] for d, clip in pairs])
            distributions[dataset] = values[resamples[dataset]].mean(axis=1)
            points[dataset] = values.mean(axis=0)
        distributions["all"] = np.mean(list(distributions.values()), axis=0)
        points["all"] = np.mean(list(points.values()), axis=0)
        for dataset in (*strata, "all"):
            for i, field in enumerate(CI_FIELDS):
                lower, upper = np.quantile(distributions[dataset][:, i], [.025, .975])
                estimates.append({"dataset": dataset, "objective": objective, "metric": field,
                                  "estimate": float(points[dataset][i]), "ci95": [float(lower), float(upper)],
                                  "unit": CHANGE_FIELDS[field][3], "common_clips": len(common_ids) if dataset == "all" else len(strata[dataset])})
    return {"samples": samples, "seed": seed, "common_clips": len(common_ids),
            "common_dataset_counts": {d: len(v) for d, v in strata.items()},
            "method": "paired stratified percentile bootstrap; identical sampled clip indices across objectives/metrics",
            "all_dataset_weights": {d: 1 / len(strata) for d in strata}, "estimates": estimates,
            "scope": "conditional variability over these development clips; excludes attack-seed/model/GT uncertainty",
            "small_sample_warning": "one clip in a stratum gives a degenerate CI, not evidence of population certainty"}


def aggregate_pairs(rows, objectives, expected_counts):
    aggregate = []
    for objective in objectives:
        all_rows = [row for row in rows if row["objective"] == objective]
        for dataset in (*DATASETS, "all"):
            group = [row for row in all_rows if dataset == "all" or row["dataset"] == dataset]
            if not group:
                continue
            item = {"dataset": dataset, "objective": objective, "paired_clips": len(group),
                    "expected_clips": sum(expected_counts.values()) if dataset == "all" else expected_counts.get(dataset, 0),
                    "attack_seconds_sum": sum(row["elapsed_seconds"] for row in group),
                    "selected_state_counts": dict(Counter(row["history"]["selected_kind"] for row in group))}
            for field in CHANGE_FIELDS:
                item[field] = mean([row[field] for row in group])
                item[field + "_clean"] = mean([row[field + "_clean"] for row in group])
                item[field + "_attacked"] = mean([row[field + "_attacked"] for row in group])
                if dataset == "all":
                    dataset_means = [mean([row[field] for row in group if row["dataset"] == d]) for d in DATASETS]
                    item[field + "_balanced_datasets"] = mean([value for value in dataset_means if value is not None])
            for field in ("zero_gradient_frame_fraction", "tiny_gradient_frame_fraction", "near_flat_transition_fraction", "last_over_initial_median_gradient_l2"):
                item[field] = mean([row["history"][field] for row in group if row["history"][field] is not None])
            item["mean_scope"] = "equal weight per paired clip; all balanced-dataset deltas are separately named"
            aggregate.append(item)
    return aggregate


def event_timing(events, records, objectives, snapshot_time, errors):
    active, sessions, pending_load = {}, [], {}
    clean = defaultdict(lambda: {"captures": 0, "wall_seconds": 0.0})
    loads = Counter()
    preparation = defaultdict(list)
    active_objective = None
    sessions_study, study_start = [], None
    for event in events:
        try:
            when = timestamp(event["time_utc"])
            kind, objective = event["event"], event.get("objective")
            if kind == "study_started":
                study_start = when
            elif kind == "study_completed" and study_start is not None:
                sessions_study.append((when - study_start).total_seconds())
                study_start = None
            if kind == "objective_started":
                if objective in active:
                    sessions.append({"objective": objective, "started_at_utc": active[objective].isoformat(),
                                     "finished_at_utc": None, "complete": False, "interrupted_before_next_start": True})
                active[objective], active_objective = when, objective
            elif kind == "objective_completed":
                start = active.pop(objective, None)
                if start is None:
                    errors.append(f"objective completion has no start: {objective}")
                else:
                    utc_elapsed = (when - start).total_seconds()
                    monotonic_elapsed = event.get("wall_seconds")
                    if utc_elapsed < 0 or not finite(monotonic_elapsed) or monotonic_elapsed < 0:
                        errors.append(f"invalid objective timing: {objective}")
                    else:
                        sessions.append({"objective": objective, "started_at_utc": start.isoformat(),
                            "finished_at_utc": when.isoformat(), "wall_seconds": monotonic_elapsed,
                            "utc_wall_seconds": utc_elapsed, "complete": True})
                active_objective = None
            elif kind == "clip_loading":
                key = (objective, event["dataset"], event["sequence"])
                pending_load[key] = when
                loads[(objective, event["dataset"])] += 1
            elif kind == "clean_completed":
                owner = objective or active_objective or "unknown"
                clean[owner]["captures"] += 1
                value = event.get("condition_wall_seconds")
                if finite(value) and value >= 0:
                    clean[owner]["wall_seconds"] += value
            elif kind == "condition_started":
                key = (objective, event["dataset"], event["sequence"])
                start = pending_load.pop(key, None)
                if start:
                    elapsed = (when - start).total_seconds()
                    if elapsed >= 0:
                        preparation[objective].append(elapsed)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid timing event: {exc}")
    for objective, start in active.items():
        sessions.append({"objective": objective, "started_at_utc": start.isoformat(), "finished_at_utc": None,
                         "complete": False, "observed_wall_to_snapshot_seconds": max(0, (snapshot_time - start).total_seconds())})
    timing = []
    for objective in objectives:
        group = [record for record in records if record["objective"] == objective]
        finished = [session for session in sessions if session["objective"] == objective and session["complete"]]
        wall = sum(session["wall_seconds"] for session in finished) if finished else None
        timing.append({"objective": objective, "attack_seconds_sum": sum(record["elapsed_seconds"] for record in group),
                       "condition_wall_seconds_sum": sum(record.get("condition_wall_seconds", 0) for record in group),
                       "objective_finished_wall_seconds_sum": wall,
                       "objective_wall_target_seconds": 7200, "objective_wall_vs_target_ratio": wall / 7200 if wall is not None else None,
                       "completed_timing_sessions": len(finished), "unfinished_timing_sessions": sum(
                           session["objective"] == objective and not session["complete"] for session in sessions),
                       "clip_load_counts": {dataset: loads[(objective, dataset)] for dataset in DATASETS},
                       "clean_baseline_captures": clean[objective]["captures"],
                       "clean_baseline_wall_seconds": clean[objective]["wall_seconds"],
                       "pre_attack_preparation_wall_seconds_sum": sum(preparation[objective]),
                       "pre_attack_preparation_observations": len(preparation[objective])})
    return {"objectives": timing, "objective_sessions": sessions,
            "study_finished_invocation_wall_seconds_sum": sum(sessions_study) if sessions_study else None,
            "clean_saved_baseline_attack_seconds_sum": sum(record["elapsed_seconds"] for record in records if record["objective"] == "clean"),
            "scope": {"attack_seconds": "result elapsed: synchronized attack engine including its internal clean forward; excludes save/eval/replay",
                      "condition_wall_seconds": "attack plus capture/metrics/save/replay; excludes preceding GT/model load and separately saved clean baseline",
                      "objective_wall_seconds": "completed objective events, including its loads/clean/save/parity/skip overhead; excludes between-invocation downtime",
                      "pre_attack_preparation": "UTC clip_loading to condition_started; includes GT/model/parity/serialization and any clean baseline, not isolated dataset IO",
                      "clean_amortization": "clean baseline is saved once per clip; internal clean objective forwards still occur in every attack"}}


def make_plots(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    paths = []
    groups = {(row["objective"], row["dataset"]): row for row in report["paired_aggregates"]}
    scope, validation = report["scope"], report["validation"]
    planned = scope["clips"]
    paired_counts = [groups.get((objective, "all"), {}).get("paired_clips", 0) for objective in scope["objectives"]]
    paired_range = (str(min(paired_counts)) if min(paired_counts) == max(paired_counts)
                    else f"{min(paired_counts)}-{max(paired_counts)}") if paired_counts else "0"
    protocol = "all raw" if scope["all_frames"] else "prefix"
    figure_scope = (
        f"Scope: {scope['kind']} | status: {report['status']} | T={scope['num_frames']} ({protocol}) | PGD={scope['steps']} steps\n"
        f"Planned clips: {planned} (PO={scope['dataset_counts'].get('po_mini', 0)}, DR={scope['dataset_counts'].get('ds_mini', 0)}) | "
        f"Completed conditions (incl. clean): {validation['completed_conditions']}/{scope['expected_conditions']}\n"
        f"Paired clips per requested objective: {paired_range}/{planned} | All-requested bootstrap common clips: {report['bootstrap']['common_clips']}/{planned}"
    )
    columns = ("tracking_apd_drop_pp", "reconstruction_apd_drop_pp", "tracking_epe_increase_m",
               "reconstruction_epe_increase_m", "track_raw_conf_mean_change", "reconstruction_conf_mean_change")
    labels = ("Tracking APD\ndrop (pp)", "Recon APD\ndrop (pp)", "Tracking EPE\nincrease (m)",
              "Recon EPE\nincrease (m)", "Head1 raw conf\nchange (score)", "Head2 conf\nchange (score)")
    values = np.array([[groups.get((objective, "all"), {}).get(field, np.nan) for field in columns] for objective in OBJECTIVES], float)
    denominators = np.array([max(1e-12, float(np.max(np.abs(column[np.isfinite(column)]))))
                             if np.isfinite(column).any() else 1 for column in values.T])
    colors = np.ma.masked_invalid(values / denominators)
    figure, axis = plt.subplots(figsize=(12.4, 6.2), constrained_layout=True)
    figure.suptitle(f"Paired clean-to-attack changes\n{figure_scope}", fontsize=10.5)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    panel = axis.imshow(colors, cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    axis.set_xticks(np.arange(len(columns)), labels)
    axis.set_yticks(np.arange(5), OBJECTIVES)
    for r in range(5):
        for c in range(len(columns)):
            axis.text(c, r, f"{values[r,c]:.3g}" if np.isfinite(values[r,c]) else "missing", ha="center", va="center",
                      color="white" if np.isfinite(values[r,c]) and abs(colors[r,c]) > .72 else "#222222", fontsize=10)
    axis.set_title("Actual units in labels; color normalized separately within each column", fontsize=10)
    figure.colorbar(panel, ax=axis, label="Signed change / column maximum absolute change", shrink=.8)
    path = output_dir / "component_changes_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.name)
    estimates = {(row["objective"], row["metric"]): row for row in report["bootstrap"]["estimates"] if row["dataset"] == "all"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 6.1), constrained_layout=True)
    figure.suptitle(f"Matched clips: paired 95% bootstrap intervals\n{figure_scope}", fontsize=10.5)
    for axis, family, unit in zip(axes, ("apd", "epe"), ("APD drop (percentage points)", "EPE increase (metres)")):
        for head, offset, color in (("tracking", -.13, "#2864ad"), ("reconstruction", .13, "#d57521")):
            metric = f"{head}_{'apd_drop_pp' if family == 'apd' else 'epe_increase_m'}"
            labeled = False
            for i, objective in enumerate(OBJECTIVES):
                entry = estimates.get((objective, metric))
                if entry:
                    value = entry["estimate"]
                    axis.errorbar(value, i + offset, xerr=[[max(0, value - entry["ci95"][0])], [max(0, entry["ci95"][1] - value)]],
                                  fmt="o", color=color, capsize=3, label=head if not labeled else None)
                    labeled = True
        axis.axvline(0, color="#777777", linewidth=.8)
        axis.set_yticks(np.arange(5), OBJECTIVES)
        axis.invert_yaxis()
        axis.set_xlabel(unit)
        axis.grid(axis="x", alpha=.2)
    axes[0].set_title("Tracking and reconstruction APD", fontsize=10)
    axes[1].set_title("EPE; equal available-dataset weights", fontsize=10)
    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, names, loc="best")
    path = output_dir / "paired_bootstrap_ci.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.name)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.7), constrained_layout=True)
    figure.suptitle(f"Unweighted native geometry changes\n{figure_scope}", fontsize=10.5)
    for axis, field, title in zip(axes, ("tracking_native_l21_change", "reconstruction_native_l21_change"),
                                  ("Head1 unweighted native L21", "Head2 unweighted native L21")):
        changes = [groups.get((objective, "all"), {}).get(field, np.nan) for objective in OBJECTIVES]
        axis.barh(np.arange(5), changes, color="#628da8")
        axis.set_yticks(np.arange(5), OBJECTIVES)
        axis.invert_yaxis()
        axis.axvline(0, color="#777777", linewidth=.8)
        axis.set_title(title)
        axis.set_xlabel("Paired change (dimensionless; native normalization)")
    path = output_dir / "native_l21_changes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.name)
    return paths


def format_value(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def make_report(report, metadata):
    scope, validation = report["scope"], report["validation"]
    counts = scope["dataset_counts"]
    campaign_clips = scope["campaign_expected_clips"]
    campaign_label = f"5×{campaign_clips}" if campaign_clips is not None else "선언된 전체"
    lines = [f"# St4RTrack 손실별 PGD 분석 — {report['status']}", "",
             f"기록된 범위: {scope['clips']}클립(PO {counts.get('po_mini',0)}, DR {counts.get('ds_mini',0)}) × "
             f"{len(scope['objectives'])}개 공격 목적함수 + clean {scope['clips']}개. "
             f"완료 조건 {validation['completed_conditions']}/{scope['expected_conditions']}, "
             f"기록된 실패 시도 {validation['recorded_failed_attempts']}, saved-input replay 통과 {validation['saved_replay_passed_conditions']}개.", ""]
    if scope["frame_policy"] == "all_raw_128_frames":
        lines += [f"클립당 원본 128프레임 전체 사용 선언을 검사했습니다. 저장된 release-length 메타데이터 검증은 "
                  f"{validation['raw_frame_verified_clips']}/{scope['clips']}개 통과했습니다. "
                  "실제 배열의 프레임 길이와 원본 데이터 SHA는 별도 artifact auditor 검증 범위입니다.", ""]
    else:
        lines += [f"기존 prefix 프로토콜로 클립당 {scope['num_frames']}프레임을 평가한 기록입니다. "
                  "이 결과는 새 원본 128프레임 전체 실험의 완료 근거가 아닙니다.", ""]
    if report["status"] in ("incomplete", "failed") or scope["kind"] != "full_campaign":
        lines += [f"**{campaign_label} 실험의 완료·순위를 주장하지 않습니다.** 아래 값은 현재 저장된 clean 짝이 있는 조건의 기술 통계입니다. "
                  "목적함수별 완료 표본이 다르면 직접 순위를 비교하면 안 됩니다.", ""]
    else:
        lines += [f"{campaign_clips}개 개발 클립의 5개 공격과 공통 clean 기록이 모두 있습니다. 이는 전체 WorldTrack 벤치마크 재현이 아닙니다.", ""]
    lines += ["## clean 대비 변화", "",
              "APD 감소는 clean−attack(%p), EPE 증가는 attack−clean(m)입니다. 양수이면 정확도가 낮아졌습니다. "
              "아래 all 평균은 클립마다 동일 가중치이며 PO·DR 클립 수가 같은 전체 캠페인에서는 각 50%와 같습니다.", "",
              "| 목적함수 | paired/expected | Tracking APD 감소 (%p) | Recon APD 감소 (%p) | Tracking EPE 증가 (m) | Recon EPE 증가 (m) |",
              "|---|---:|---:|---:|---:|---:|"]
    groups = {(row["objective"], row["dataset"]): row for row in report["paired_aggregates"]}
    for objective in OBJECTIVES:
        row = groups.get((objective, "all"), {})
        table_fields = ("tracking_apd_drop_pp", "reconstruction_apd_drop_pp", "tracking_epe_increase_m", "reconstruction_epe_increase_m")
        lines.append(f"| {objective} | {row.get('paired_clips',0)}/{scope['clips']} | " + " | ".join(format_value(row.get(field)) for field in table_fields) + " |")
    lines += ["", "PO·DR별 수치와 native 비가중 L21, confidence 변화, 선택된 iterate·gradient stall 통계는 study_analysis.json에 있습니다. "
              "Native L21은 정규화된 **무차원** 거리이며 EPE의 metre와 비교할 수 없습니다. "
              "Confidence는 1 이상인 모델 점수이며 확률이 아닙니다. confidence만 변해도 weighted loss가 변하므로 기하 오차와 함께 읽어야 합니다.", "",
              "Stall 통계는 exact-zero gradient·고정 threshold·연속 objective 변화와 마지막/초기 gradient 비율의 기술 값입니다. "
              "서로 다른 단위의 gradient 절대 크기는 직접 비교하지 않으며, stall을 자동 수렴·공격 실패 판정으로 사용하지 않습니다.", "",
              "![손실별 변화](component_changes_heatmap.png)", "",
              "![클립 bootstrap 신뢰구간](paired_bootstrap_ci.png)", "",
              "![비가중 native 기하 오차](native_l21_changes.png)", "",
              "## 신뢰구간과 해석 범위", "",
              f"모든 요청 목적함수와 clean이 함께 완료된 공통 {report['bootstrap']['common_clips']}개 클립에만 "
              f"seed {report['bootstrap']['seed']}, {report['bootstrap']['samples']}회 paired stratified percentile bootstrap을 적용했습니다. "
              "PO·DR 안에서 클립 수를 유지해 복원 추출하며, 목적함수와 지표 전부에 같은 추출 인덱스를 사용합니다. "
              "둘 다 존재하면 all CI는 PO/DR 각 50%입니다. 불균형 제한 실행의 일반 all 클립 평균과 다를 수 있습니다.", "",
              "95% CI는 선택된 개발 클립에 조건부인 표본 변동만 나타냅니다. 단일 attack seed, 단일 checkpoint, "
              "GT·클립 선택·전체 데이터셋의 불확실성은 포함하지 않습니다. 한 클립뿐인 strata의 CI는 퇴화합니다. "
              "다섯 손실 비교에 대한 다중 비교 보정이나 모집단 유의성 검정을 수행하지 않았습니다.", "",
              f"이는 고정 모델에 대한 입력 공격 민감성 비교입니다. 선택된 {scope['clips']}개 클립에서 하나의 base attack seed를 사용하고, "
              "클립별 파생 restart seed를 목적함수 사이에 공유합니다. 손실 삭제 후 재훈련, training weight의 인과적 중요도, "
              "전역 최악 공격을 측정한 결과가 아닙니다. Bootstrap의 표본 단위는 클립이며 프레임을 독립 표본으로 세지 않습니다. "
              "Scene 독립성을 검증하지 않았고 scene cluster 보정을 하지 않았습니다.", "",
              "## 실제 시간과 2시간 목표", "",
              "| 목적함수 | 저장된 attack 합계 (분) | 완료 objective wall 합계 (분) | 120분 대비 | load 횟수 | shared clean capture |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in report["timing"]["objectives"]:
        lines.append(f"| {row['objective']} | {row['attack_seconds_sum']/60:.2f} | "
                     f"{format_value(row['objective_finished_wall_seconds_sum']/60 if row['objective_finished_wall_seconds_sum'] is not None else None,2)} | "
                     f"{format_value(row['objective_wall_vs_target_ratio'],2)}× | {sum(row['clip_load_counts'].values())} | {row['clean_baseline_captures']} |")
    lines += ["", "Attack 시간은 동기화된 엔진 내부 계산과 반복별 internal clean forward를 포함하고 저장·평가·replay를 제외합니다. "
              "Condition wall은 이후 capture·저장·평가·replay를 포함합니다. Objective wall은 앞선 GT/model load·parity·공유 clean 저장과 resume skip도 포함합니다. "
              "종료 event가 없는 실행은 실제 완료 wall로 집계하지 않습니다. 재실행 사이 중단 시간과 container 시작 시간은 포함하지 않으며, "
              "dataset I/O만의 시간은 기록되지 않아 분리 추정하지 않았습니다.", "",
              "## 목적함수와 검증 근거", "",
              "```text", "s[t] = max(sum_ALL_pixels ||P2[t]|| / (3HW), 1e-8)",
              "e1 = ||P1/s_pred − G1/s_gt|| ; e2 = ||P2/s_pred − G2/s_gt||",
              "tracking_3d = mean(c1_eff * e1)", "reconstruction_3d = mean_valid_pixels_times(C2 * e2)",
              "confidence = −0.2 * (mean(log(max(c1_eff,1))) + mean_valid(log(C2)))",
              "joint_training = tracking_3d + reconstruction_3d + confidence",
              "c1_dynamic = 5 * max_static_confidence.detach()",
              "tracking_mse = mean ||median_norm_scale * evaluation_tracks − GT||²", "```", "",
              f"공식 commit `{metadata['provenance'].get('official_commit', COMMIT)}`의 "
              f"[훈련 설정](https://github.com/HavenFeng/St4RTrack/blob/{COMMIT}/scripts_run/train_seq_reweight.sh#L26), "
              f"[활성 손실](https://github.com/HavenFeng/St4RTrack/blob/{COMMIT}/dust3r/losses.py#L376), "
              f"[평가 정렬](https://github.com/HavenFeng/St4RTrack/blob/{COMMIT}/dust3r/track_eval_util.py), "
              f"[구현 근거 문서]({report['loss_structure_document']})를 따릅니다.", "",
              "모델과 GT/query/mask는 고정되고 모든 RGB 프레임에 같은 L∞ 예산을 적용합니다. "
              "같은 클립은 목적함수마다 같은 restart seed를 사용합니다. 저장 입력은 각 목적함수의 clean·초기화·중간·마지막 iterate 중 최고값이며, "
              "최저 APD를 직접 선택한 결과가 아닙니다. Native query의 round/clamp/collision 처리와 benchmark truncate query는 별도입니다. "
              "Native /3 정규화와 dynamic weight stop-gradient는 공식 소스 동작을 보존했습니다.", "",
              "공식 native objective의 활성 head-1 tracking·head-2 reconstruction 두 branch를 geometry 두 항과 "
              "두 branch의 log-confidence 항으로 분해했습니다. Confidence는 별도 세 번째 head가 아닙니다. "
              "Tracking objective도 head-2 normalization을 통해 연결되므로 독립 head 개입으로 해석할 수 없습니다.", "",
              "Reconstruction 지표는 resized model grid의 고정 실제 depth GT에 공식 median/threshold 수식을 적용한 값입니다. "
              "논문의 original-resolution reconstruction benchmark나 Sim(3) 점수와 동일한 수치가 아닙니다. "
              "훈련 augmentation·원래 mesh-vertex query population·TTA를 재현하지 않았습니다.", "",
              f"별도 대용량 artifact audit 상태: **{report['artifact_audit']['status']}**. "
              "본 분석은 저장된 scalar/CSV/history/events의 일관성을 검사하며, 원본 데이터·예측 배열의 전체 SHA 검사는 별도 auditor 결과로 구분합니다."]
    if report["validation"]["errors"]:
        lines += ["", "## 검증 오류", "", *[f"- {error}" for error in report["validation"]["errors"]]]
    if report["validation"]["missing_conditions"]:
        lines += ["", f"누락 조건 {len(report['validation']['missing_conditions'])}개는 JSON의 missing_conditions에 기록했습니다."]
    return "\n".join(lines) + "\n"


def analyze(run_dir, output_dir=None, *, bootstrap_samples=2000, seed=20261004):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else run_dir / "analysis"
    # Outputs have fixed new names. Do not allow them to replace run evidence.
    if output_dir == run_dir or any(output_dir.is_relative_to(run_dir / name) for name in ("conditions", "sequences")):
        raise ValueError("analysis output must have its own directory")
    metadata = read_json(run_dir / "run.json")
    config, manifest = metadata["config"], metadata["manifest"]
    scope = scope_of(metadata)
    errors, warnings = [], []
    errors.extend(scope["declaration_errors"])
    ids = {(row["dataset"], row["sequence"]) for row in manifest["entries"]}
    if len(ids) != len(manifest["entries"]) or any(d not in DATASETS for d, _ in ids):
        errors.append("manifest has duplicate clips or unexpected datasets")
    if scope["kind"] == "unrecognized":
        campaign_label = str(scope["campaign_expected_clips"]) if scope["campaign_expected_clips"] is not None else "validly declared"
        errors.append(f"run is neither the balanced {campaign_label}-clip campaign nor an explicitly marked limited/preflight run")
    if not config["objectives"] or set(config["objectives"]) - set(OBJECTIVES) or len(config["objectives"]) != len(set(config["objectives"])):
        errors.append("requested objectives are invalid or duplicated")
    signature = fingerprint({"config": config, "manifest": manifest, "provenance": metadata["provenance"]})
    if signature != metadata.get("signature"):
        errors.append("run signature does not match config/manifest/provenance")
    expected = {(d, clip, objective) for d, clip in ids for objective in ("clean", *config["objectives"])}
    indexed, histories = {}, {}
    hashes = {"run.json": sha_file(run_dir / "run.json")}
    frame_evidence = raw_frame_evidence(run_dir, ids, scope, signature, hashes, errors)
    for path in sorted((run_dir / "conditions").glob("*/*/*/result.json")):
        try:
            record = read_json(path)
            key = (record["dataset"], record["sequence"], record["objective"])
            canonical = run_dir / "conditions" / key[2] / key[0] / key[1] / "result.json"
            if key not in expected or key in indexed or path != canonical or record.get("signature") != signature:
                raise ValueError("foreign/duplicate/stale/misplaced condition")
            if not record.get("saved_input_verification", {}).get("passed"):
                raise ValueError("saved input replay did not pass")
            if not finite(record["elapsed_seconds"]) or record["elapsed_seconds"] < 0:
                raise ValueError("invalid attack elapsed time")
            if record["objective"] != "clean" and (record["steps"] != config["steps"] or record["epsilon_255"] != config["epsilon_255"]):
                raise ValueError("attack budget differs from config")
            if scope["frame_policy"] == "all_raw_128_frames" and any(
                    record.get(section, {}).get("num_frames") != 128
                    for section in ("tracking_metrics", "reconstruction_metrics")):
                raise ValueError("tracking and reconstruction metrics must both cover all 128 raw frames")
            history_path = path.parent / "history.json"
            histories[key] = history_diagnostics(read_json(history_path), record, config)
            indexed[key] = record
            hashes[str(path.relative_to(run_dir))] = sha_file(path)
            hashes[str(history_path.relative_to(run_dir))] = sha_file(history_path)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"{path.relative_to(run_dir)}: {exc}")
    missing = sorted(expected - set(indexed))
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    if summary is None:
        warnings.append("summary.json is not available; run may still be active")
    else:
        hashes["summary.json"] = sha_file(summary_path)
        if summary.get("completed_conditions") != len(indexed) or summary.get("expected_conditions") != len(expected):
            errors.append("summary condition counts disagree with saved records")
        if summary.get("complete") != (not missing):
            errors.append("summary completion flag disagrees with saved records")
        summary_groups = {(row["dataset"], row["objective"]): row for row in summary.get("aggregate", [])}
        if len(summary_groups) != len(summary.get("aggregate", [])):
            errors.append("summary contains duplicate aggregate groups")
        for (dataset, objective), aggregate in summary_groups.items():
            children = [r for r in indexed.values() if r["objective"] == objective and (dataset == "all" or r["dataset"] == dataset)]
            checks = {"attack_seconds_sum": sum(r["elapsed_seconds"] for r in children)}
            for section, fields in (("tracking_metrics", ("apd3d_all", "epe_all_m")), ("reconstruction_metrics", ("apd3d", "epe_m"))):
                prefix = "tracking_" if section == "tracking_metrics" else "reconstruction_"
                for field in fields:
                    checks[prefix + field] = mean([r[section][field] for r in children])
            if aggregate.get("completed_sequences") != len(children):
                errors.append(f"summary aggregate support differs: {dataset}/{objective}")
            for field, value in checks.items():
                if value is not None and not same_number(value, aggregate.get(field)):
                    errors.append(f"summary aggregate differs: {dataset}/{objective}/{field}")
    csv_path = run_dir / "sequence_metrics.csv"
    if csv_path.is_file():
        hashes["sequence_metrics.csv"] = sha_file(csv_path)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        csv_index = {(row["dataset"], row["sequence"], row["objective"]): row for row in csv_rows}
        if len(csv_index) != len(csv_rows) or set(csv_index) != set(indexed):
            errors.append("sequence CSV identities disagree with saved records")
        for key in set(csv_index) & set(indexed):
            record = indexed[key]
            for section, fields in (("tracking_metrics", ("apd3d_all", "epe_all_m")), ("reconstruction_metrics", ("apd3d", "epe_m"))):
                prefix = "tracking_" if section == "tracking_metrics" else "reconstruction_"
                for field in fields:
                    try:
                        if not same_number(float(csv_index[key][prefix + field]), record[section][field]):
                            raise ValueError("value differs")
                    except (ValueError, KeyError, TypeError):
                        errors.append(f"sequence CSV metric differs: {key}, {prefix+field}")
    else:
        warnings.append("sequence_metrics.csv is not available")
    paired = []
    for key, record in sorted(indexed.items()):
        if key[2] == "clean":
            continue
        clean = indexed.get((key[0], key[1], "clean"))
        if clean is None:
            continue
        same_clip_records = [indexed.get((key[0], key[1], objective)) for objective in config["objectives"]]
        seeds = {child["seed"] for child in same_clip_records if child is not None}
        if len(seeds) > 1 or seeds and clean["seed"] not in seeds:
            errors.append(f"objectives do not share the restart seed: {key[:2]}")
        try:
            paired.append(paired_row(record, clean, histories[key]))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid paired metrics {key}: {exc}")
    common = {pair for pair in ids if all((*pair, objective) in indexed for objective in ("clean", *config["objectives"]))}
    paired_ids = {(row["dataset"], row["sequence"], row["objective"]) for row in paired}
    common = {pair for pair in common if all((*pair, objective) in paired_ids for objective in config["objectives"])}
    events_path = run_dir / "events.jsonl"
    events = []
    if events_path.is_file():
        hashes["events.jsonl"] = sha_file(events_path)
        for i, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    warnings.append(f"events line {i} is incomplete/invalid")
    failure_path = run_dir / "failures.jsonl"
    failed_attempts = len([line for line in failure_path.read_text(encoding="utf-8").splitlines() if line.strip()]) if failure_path.is_file() else 0
    if summary and summary.get("failed_attempts", 0) != failed_attempts:
        errors.append("failure count differs between summary and failure log")
    snapshot = datetime.now(timezone.utc)
    timing = event_timing(events, list(indexed.values()), config["objectives"], snapshot, errors)
    audit_path = run_dir / "artifact_validation.json"
    audit_qualified = False
    audit_problems = []
    if audit_path.is_file():
        hashes["artifact_validation.json"] = sha_file(audit_path)
        try:
            audit = read_json(audit_path)
            if not isinstance(audit, dict):
                raise ValueError("artifact audit must be a JSON object")
            artifact_audit = {"status": audit.get("status", "unknown"), "passed": audit.get("passed", False),
                              "errors": len(audit.get("errors", [])), "missing": len(audit.get("missing", [])),
                              "recorded_evidence_only": audit.get("recorded_evidence_only", []), "path": str(audit_path),
                              **{key: audit.get(key) for key in ("full_campaign_completed", "all_frames_completed", "all_frames",
                                  "campaign_expected_clips", "campaign_expected_frames", "requested_scope", "source_files_verified")}}
            if artifact_audit["status"] == "incomplete":
                warnings.append("independent artifact audit is incomplete; completion is withheld")
            elif artifact_audit["status"] != "passed":
                errors.append("the separate artifact audit is not passed")
            else:
                audit_qualified, audit_problems = audit_completion_evidence(audit, scope, hashes)
                errors.extend(audit_problems)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            artifact_audit = {"status": "invalid", "passed": False, "path": str(audit_path)}
            errors.append(f"independent artifact audit is unreadable/invalid: {exc}")
    else:
        artifact_audit = {"status": "not_available", "passed": False, "path": str(audit_path)}
        warnings.append("independent artifact audit is not available; no full campaign completion claim")
    artifact_audit["validated_for_scope"] = audit_qualified
    artifact_audit["completion_evidence_errors"] = audit_problems
    full_audit_missing = scope["kind"] == "full_campaign" and artifact_audit["status"] == "not_available"
    status = ("incomplete" if missing or summary is None or artifact_audit["status"] == "incomplete" or full_audit_missing else
              "failed" if errors else "complete_with_recorded_failures" if failed_attempts else "complete")
    full_completed = scope["kind"] == "full_campaign" and audit_qualified and status.startswith("complete")
    scientific_interpretation = {
        "comparison": "fixed-model input-attack sensitivity on the selected development clips",
        "selected_clips": scope["clips"], "dataset_counts": scope["dataset_counts"],
        "attack_randomness": "one base attack seed; one derived restart seed per clip shared across objectives",
        "base_attack_seed": config.get("seed"),
        "bootstrap_unit": "clip; paired stratified resampling retains all frames within each sampled clip",
        "scene_independence_verified": False, "scene_cluster_adjustment": False,
        "excluded_claims": ["loss-deletion retraining effects", "causal importance of training weights",
                            "global worst-case attack", "population significance or external-scene generalization"],
        "native_structure": "two active tracking/reconstruction branches, algebraically split into two weighted geometry terms and both branches' log-confidence term",
    }
    report = {"schema_version": 1, "status": status, "snapshot_at_utc": snapshot.isoformat(), "run_dir": str(run_dir),
              "scope": scope, "all_five_campaign_objectives_completed": full_completed,
              "all_raw_frames_campaign_completed": full_completed and scope["all_frames"],
              "validation": {"errors": errors, "warnings": warnings, "completed_conditions": len(indexed),
                             "missing_conditions": missing, "recorded_failed_attempts": failed_attempts,
                             "saved_replay_passed_conditions": len(indexed),
                             "raw_frame_verified_clips": sum(row["passed"] for row in frame_evidence)},
              "raw_frame_evidence": frame_evidence,
              "units": {field: spec[3] for field, spec in CHANGE_FIELDS.items()},
              "paired_aggregates": aggregate_pairs(paired, config["objectives"], scope["dataset_counts"]),
              "paired_conditions": paired, "bootstrap": bootstrap_intervals(paired, config["objectives"], common, samples=bootstrap_samples, seed=seed),
              "timing": timing, "artifact_audit": artifact_audit, "input_scalar_file_sha256": hashes,
              "loss_structure_document": (Path(__file__).resolve().parents[1] / "docs/loss_structure.md").as_posix(),
              "analyzer_source_sha256": sha_file(Path(__file__).resolve()),
              "scientific_interpretation": scientific_interpretation,
              "interpretation": "fixed-model input-attack sensitivity; one base attack seed and selected development clips; clip-unit paired bootstrap without verified scene independence or scene-cluster adjustment; no loss-deletion retraining, causal training-weight importance, global worst-case attack, or population significance claim"}
    output_dir.mkdir(parents=True, exist_ok=True)
    report["plots"] = make_plots(report, output_dir)
    (output_dir / "study_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "study_report.md").write_text(make_report(report, metadata), encoding="utf-8")
    return report


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.output_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    print(json.dumps({"status": report["status"], "completed_conditions": report["validation"]["completed_conditions"],
                      "expected_conditions": report["scope"]["expected_conditions"], "common_paired_clips": report["bootstrap"]["common_clips"],
                      "output_dir": str(args.output_dir or args.run_dir / "analysis")}, ensure_ascii=False))
    return 1 if report["status"] == "failed" else 2 if report["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
