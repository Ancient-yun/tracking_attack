"""Strict, read-only CPU data layer for the completed 8-clip HTML report.

No experiment package, model, torch, CUDA, or Docker is imported. Large arrays
remain the renderer's responsibility: this module binds the independent audit
to fresh scalar evidence and checks all scalar computations against results.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

OBJECTIVES = ("tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training")
DATASETS = ("po_mini", "ds_mini")
GROUPS = ("all", *DATASETS)
ANALYZER_SHA256 = "c867a873aaffa8086d3ca8c7eeb99cacc2ccbb6f522507a55012c0c6d3c4b9c4"
OFFICIAL_COMMIT = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"
CHANGE_FIELDS = {
    "tracking_apd_drop_pp": ("tracking_metrics", "apd3d_all", -1, "percentage_points"),
    "reconstruction_apd_drop_pp": ("reconstruction_metrics", "apd3d", -1, "percentage_points"),
    "tracking_epe_increase_m": ("tracking_metrics", "epe_all_m", 1, "metres"),
    "reconstruction_epe_increase_m": ("reconstruction_metrics", "epe_m", 1, "metres"),
}
DIAGNOSTIC_FIELDS = {
    "tracking_native_l21_change": "tracking_l21",
    "reconstruction_native_l21_change": "reconstruction_l21",
    "track_conf_mean_change": "track_conf_mean",
    "track_raw_conf_mean_change": "track_raw_conf_mean",
    "reconstruction_conf_mean_change": "reconstruction_conf_mean",
}
METRICS = tuple(CHANGE_FIELDS)
METRIC_UNITS = {name: item[3] for name, item in CHANGE_FIELDS.items()}
SCIENTIFIC_LIMITS = [
    "Fixed-model input-attack sensitivity on eight selected development clips; no training-loss deletion or causal weight-importance claim.",
    "One shared derived attack seed per clip, one restart; no global worst-case guarantee.",
    "Clip-unit, PO/DR-stratified bootstrap; scene independence is unverified and no scene-cluster adjustment is made.",
    "Different tasks have different evaluation populations and alignment; equal relative APD drops are not equal physical damage.",
    "Native L21 is dimensionless and confidence is a score, not a probability; native objective magnitudes do not rank attack strength.",
    "The saved iterate maximizes its attack objective, not benchmark EPE or minimum APD.",
    "Contrast signs compare attacks; actual damage or improvement versus clean must also be checked.",
]


class ReportValidationError(ValueError):
    """Saved report inputs do not establish the requested verified scope."""


def _require(condition, message):
    if not condition:
        raise ReportValidationError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ReportValidationError(f"Non-finite JSON constant: {value}")


def _read_json(path):
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_duplicates, parse_constant=_bad_constant)
        return value, hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportValidationError(f"Cannot read {path}: {exc}") from exc


def read_json(path):
    return _read_json(path)[0]


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value, label):
    _require(_finite(value), f"Non-finite or nonnumeric {label}")
    return float(value)


def _same(first, second, label):
    _require(_finite(first) and _finite(second) and bool(np.isclose(first, second, rtol=1e-7, atol=1e-8)),
             f"Scalar mismatch: {label}")


def safe_ratio(numerator, denominator):
    """None for missing/near-zero denominators; preserve negative numerators."""
    if not _finite(numerator) or not _finite(denominator) or denominator <= 1e-12:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def _ratios(row):
    tracking = safe_ratio(row["tracking_apd_drop_pp"], row["tracking_apd_drop_pp_clean"])
    reconstruction = safe_ratio(row["reconstruction_apd_drop_pp"], row["reconstruction_apd_drop_pp_clean"])
    return {"tracking_apd_relative_drop_pct": None if tracking is None else 100 * tracking,
            "reconstruction_apd_relative_drop_pct": None if reconstruction is None else 100 * reconstruction,
            "tracking_epe_ratio": safe_ratio(row["tracking_epe_increase_m_attacked"], row["tracking_epe_increase_m_clean"]),
            "reconstruction_epe_ratio": safe_ratio(row["reconstruction_epe_increase_m_attacked"], row["reconstruction_epe_increase_m_clean"])}


def _relative_path(root, name):
    _require(isinstance(name, str), "Hash path must be a string")
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized)
    _require(not parts.is_absolute() and ":" not in normalized and all(p not in ("", ".", "..") for p in normalized.split("/")),
             f"Unsafe relative evidence path: {name}")
    path = (root / Path(*parts.parts)).resolve()
    _require(path.is_relative_to(root), f"Evidence path escapes run: {name}")
    return path


def _audit_digest(audit, relative):
    mapping = audit.get("file_sha256", {})
    _require(isinstance(mapping, dict), "Audit SHA map is missing")
    suffix = relative.replace("\\", "/")
    matches = [value for name, value in mapping.items() if str(name).replace("\\", "/") == suffix
               or str(name).replace("\\", "/").endswith("/" + suffix)]
    _require(len(matches) == 1, f"Audit does not uniquely bind {relative}")
    return matches[0]


def _checked_key(value, label):
    _require(isinstance(value, str) and value not in ("", ".", "..") and "/" not in value and "\\" not in value and ":" not in value,
             f"Unsafe {label}: {value}")
    return value


def _index(rows, keys, label):
    _require(isinstance(rows, list), f"{label} is not a list")
    result = {}
    for row in rows:
        _require(isinstance(row, dict) and all(key in row for key in keys), f"Malformed {label} row")
        identity = tuple(row[key] for key in keys)
        _require(identity not in result, f"Duplicate {label} identity: {identity}")
        result[identity] = row
    return result


def _crosscheck_aggregate(row, children, dataset, objective, label):
    _require(row.get("paired_clips", row.get("completed_sequences")) == len(children), f"Wrong {label} support: {dataset}/{objective}")
    for field, (section, native, _, _) in CHANGE_FIELDS.items():
        if "paired_clips" in row:
            for suffix in ("", "_clean", "_attacked"):
                _same(row.get(field + suffix), float(np.mean([child[field + suffix] for child in children])),
                      f"{label}/{dataset}/{objective}/{field + suffix}")
        else:
            prefix = "tracking_" if section == "tracking_metrics" else "reconstruction_"
            _same(row.get(prefix + native), float(np.mean([child[section][native] for child in children])),
                  f"{label}/{dataset}/{objective}/{prefix + native}")


def _load_verified_study(run_dir, analysis_dir=None, require_complete=True):
    """Load fresh signed scalar evidence. No files are created or modified.

    ``require_complete=False`` permits diagnostic snapshots only, while still
    refusing integrity errors. It never grants a completion flag and cannot
    make ``compute_task_coupling`` accept an incomplete campaign.
    """
    root = Path(run_dir).resolve()
    analysis_root = Path(analysis_dir).resolve() if analysis_dir is not None else root / "analysis"
    _require(root.is_dir(), f"Missing run directory: {root}")
    analysis, analysis_sha = _read_json(analysis_root / "study_analysis.json")
    audit, audit_sha = _read_json(root / "artifact_validation.json")
    _require(isinstance(analysis, dict) and isinstance(audit, dict), "Analysis/audit must be JSON objects")
    scalar_hashes = analysis.get("input_scalar_file_sha256", {})
    _require(isinstance(scalar_hashes, dict) and scalar_hashes, "Analysis has no bound scalar SHA evidence")
    current_hashes = {}
    for relative, expected in scalar_hashes.items():
        _require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                 f"Malformed scalar SHA: {relative}")
        path = _relative_path(root, relative)
        actual = sha_file(path) if path.is_file() else None
        _require(actual == expected, f"Stale analysis SHA: {relative}")
        current_hashes[relative] = actual
        if relative not in ("artifact_validation.json", "events.jsonl") and audit.get("status") == "passed":
            _require(_audit_digest(audit, relative) == actual, f"Stale audit SHA: {relative}")
    _require(scalar_hashes.get("artifact_validation.json") == audit_sha, "Analysis does not bind the current artifact audit")

    def consume(relative, *, bind_analysis=True, bind_audit=True):
        path = _relative_path(root, relative)
        value, digest = _read_json(path)
        if bind_analysis:
            _require(scalar_hashes.get(relative) == digest, f"Missing/stale analysis binding: {relative}")
        if bind_audit and audit.get("status") == "passed":
            _require(_audit_digest(audit, relative) == digest, f"Missing/stale audit binding: {relative}")
        current_hashes[relative] = digest
        return value

    metadata = consume("run.json")
    config, manifest = metadata["config"], metadata["manifest"]
    _require(metadata.get("signature") == _fingerprint({"config": config, "manifest": manifest, "provenance": metadata["provenance"]}),
             "Run signature differs from frozen config/manifest/provenance")
    _require(config.get("objectives") == list(OBJECTIVES), "Expected all five objectives in frozen order")
    for source, label in ((config, "config"), (manifest, "manifest")):
        for key, value in (("num_frames", 128), ("campaign_expected_frames", 128), ("campaign_expected_clips", 8)):
            _require(type(source.get(key)) is int and source[key] == value, f"Wrong {label}.{key}")
        _require(source.get("all_frames") is True, f"{label} does not declare all raw frames")
    _require(type(config.get("steps")) is int and config["steps"] == 20 and config.get("epsilon_255") == 4
             and type(config.get("restarts")) is int and config["restarts"] == 1
             and config.get("step_size_fraction") == .25 and config.get("seed") == 20261002,
             "Expected PGD20, epsilon4/255, step1/255, one restart and base seed20261002")
    _require(metadata["provenance"].get("official_commit") == OFFICIAL_COMMIT, "Unexpected official source commit")
    identities = [(_checked_key(row["dataset"], "dataset"), _checked_key(row["sequence"], "sequence")) for row in manifest["entries"]]
    _require(len(identities) == len(set(identities)) == 8 and Counter(d for d, _ in identities) == {"po_mini": 4, "ds_mini": 4},
             "Expected unique balanced PO4/DR4 manifest")
    _require(metadata.get("frozen_weights") is True and metadata.get("tta") is False, "Frozen model/no-TTA contract is missing")
    scope = analysis.get("scope", {})
    for key, value in {"kind": "full_campaign", "clips": 8, "dataset_counts": {"po_mini": 4, "ds_mini": 4},
                       "num_frames": 128, "steps": 20, "epsilon_255": 4, "expected_conditions": 48,
                       "all_frames": True, "objectives": list(OBJECTIVES)}.items():
        _require(scope.get(key) == value and (not isinstance(value, bool) or scope.get(key) is value), f"Wrong analysis scope.{key}")
    _require(analysis.get("analyzer_source_sha256") == ANALYZER_SHA256, "Unexpected analyzer version; preserve/version the approved analysis separately")
    validation = analysis.get("validation", {})
    _require(validation.get("errors") == [], "Analysis records integrity errors")
    _require(audit.get("errors") == [], "Artifact audit records integrity errors")
    campaign = consume("campaign_execution.json", bind_analysis=False, bind_audit=False)
    resume = consume("resume_execution.json", bind_analysis=False, bind_audit=False)
    if require_complete:
        _require(campaign.get("status") == resume.get("status") == "complete", "Campaign or resume is not complete")
        _require(analysis.get("status") in ("complete", "complete_with_recorded_failures"), "Analysis is incomplete or failed")
        _require(analysis.get("all_five_campaign_objectives_completed") is True and analysis.get("all_raw_frames_campaign_completed") is True,
                 "Analysis does not establish all-five all-frame completion")
        _require(validation.get("completed_conditions") == 48 and validation.get("raw_frame_verified_clips") == 8
                 and validation.get("missing_conditions") == [], "Analysis scope has missing conditions or frames")
        _require(analysis.get("artifact_audit", {}).get("validated_for_scope") is True, "Analysis audit binding did not pass")
        expected_audit = {"status": "passed", "passed": True, "full_campaign_completed": True, "all_frames_completed": True,
                          "all_frames": True, "source_files_verified": True, "expected_conditions": 48,
                          "completed_conditions_checked": 48, "full_campaign_expected_conditions": 48,
                          "campaign_expected_clips": 8, "campaign_expected_frames": 128,
                          "requested_scope": {"clips": 8, "frames": 128, "steps": 20}, "missing": []}
        for key, value in expected_audit.items():
            _require(audit.get(key) == value and (not isinstance(value, bool) or audit.get(key) is value), f"Audit gate failed: {key}")
        bootstrap = analysis.get("bootstrap", {})
        _require(bootstrap.get("common_clips") == 8 and bootstrap.get("common_dataset_counts") == {"po_mini": 4, "ds_mini": 4}
                 and bootstrap.get("all_dataset_weights") == {"po_mini": .5, "ds_mini": .5}
                 and bootstrap.get("samples") == 2000 and bootstrap.get("seed") == 20261004,
                 "Wrong paired bootstrap common population/weights/settings")

    sequences, records, histories = {}, {}, {}
    expected = {(d, clip, objective) for d, clip in identities for objective in ("clean", *OBJECTIVES)}
    actual_paths = {(path.parents[1].name, path.parent.name, path.parents[2].name) for path in (root / "conditions").glob("*/*/*/result.json")}
    _require(actual_paths <= expected, "Unexpected conditions in run")
    if require_complete:
        _require(actual_paths == expected, "Run does not have exactly 48 condition results")
    for dataset, sequence in identities:
        prefix = f"sequences/{dataset}/{sequence}"
        sequence_metadata = consume(prefix + "/sequence.json")
        inventory = consume(prefix + "/sequence_manifest.json")
        _require(inventory.get("signature") == metadata["signature"] and inventory.get("artifact_sha256", {}).get("sequence.json") == current_hashes[prefix + "/sequence.json"],
                 f"Sequence inventory binding failed: {dataset}/{sequence}")
        _require(all(type(sequence_metadata.get(k)) is int and sequence_metadata[k] == 128 for k in
                     ("num_frames", "requested_num_frames", "original_frame_count")) and sequence_metadata.get("all_frames_used") is True
                 and sequence_metadata.get("used_frame_indices") == list(range(128)), f"Not all original frames: {dataset}/{sequence}")
        raw_counts = sequence_metadata.get("source_frame_counts", {})
        _require(set(raw_counts) == {"images_jpeg_bytes", "tracks_XYZ", "visibility", "depth_map", "extrinsics_w2c"}
                 and all(type(v) is int and v == 128 for v in raw_counts.values()), f"Raw temporal field lengths differ: {dataset}/{sequence}")
        sequences[dataset, sequence] = sequence_metadata
        for objective in ("clean", *OBJECTIVES):
            identity = dataset, sequence, objective
            if identity not in actual_paths:
                continue
            prefix = f"conditions/{objective}/{dataset}/{sequence}"
            record = consume(prefix + "/result.json")
            _require(tuple(record.get(key) for key in ("dataset", "sequence", "objective")) == identity
                     and record.get("signature") == metadata["signature"], f"Foreign/misplaced result: {identity}")
            _require(record.get("saved_input_verification", {}).get("passed") is True, f"Saved replay not passed: {identity}")
            _require(type(record.get("seed")) is int and 0 <= record["seed"] < 2 ** 32, f"Invalid attack seed: {identity}")
            _require(record.get("steps") == (0 if objective == "clean" else 20)
                     and record.get("epsilon_255") == (0 if objective == "clean" else 4), f"Wrong attack budget: {identity}")
            _require(_number(record.get("elapsed_seconds"), str(identity)) >= 0, f"Negative attack runtime: {identity}")
            for section in ("tracking_metrics", "reconstruction_metrics"):
                _require(type(record.get(section, {}).get("num_frames")) is int and record[section]["num_frames"] == 128,
                         f"Metrics omit raw frames: {identity}/{section}")
            for _, (section, field, _, _) in CHANGE_FIELDS.items():
                value = _number(record[section].get(field), f"{identity}/{field}")
                _require(value >= 0 and ("apd" not in field or value <= 100), f"Metric outside range: {identity}/{field}")
            history = consume(prefix + "/history.json")
            _require(record.get("artifact_sha256", {}).get("history.json") == current_hashes[prefix + "/history.json"],
                     f"Condition history SHA differs: {identity}")
            selected = record.get("selected_state", {})
            selected_rows = [row for row in history if all(row.get(k) == v for k, v in selected.items())]
            _require(set(selected) == {"restart", "step"} and len(selected_rows) == 1, f"Invalid selected state: {identity}")
            _same(record.get("attack_loss"), selected_rows[0].get("loss"), f"Selected objective/{identity}")
            if objective != "clean":
                _same(record.get("attack_loss"), max(_number(row.get("loss"), str(identity)) for row in history), f"Best objective/{identity}")
            records[identity], histories[identity] = record, history
        clip_records = [r for (d, clip, _), r in records.items() if (d, clip) == (dataset, sequence)]
        _require(len({r["seed"] for r in clip_records}) <= 1, f"Objectives do not share the clean/attack seed: {dataset}/{sequence}")

    paired = _index(analysis.get("paired_conditions"), ("dataset", "sequence", "objective"), "paired_conditions")
    _require(set(paired) == {key for key in records if key[2] != "clean"}, "Paired analysis does not match actual attack identities")
    for identity, row in paired.items():
        record, baseline = records[identity], records.get((*identity[:2], "clean"))
        _require(baseline is not None and row.get("seed") == record["seed"] == baseline["seed"], f"Paired seed/clean mismatch: {identity}")
        _same(row.get("elapsed_seconds"), record["elapsed_seconds"], f"Paired elapsed/{identity}")
        for field, (section, native, sign, _) in CHANGE_FIELDS.items():
            clean_value, attacked_value = baseline[section][native], record[section][native]
            for key, expected_value in ((field + "_clean", clean_value), (field + "_attacked", attacked_value),
                                        (field, sign * (attacked_value - clean_value))):
                _same(row.get(key), expected_value, f"Paired metrics/{identity}/{key}")
        for field, native in DIAGNOSTIC_FIELDS.items():
            for suffix, child in (("_clean", baseline), ("_attacked", record)):
                _same(row.get(field + suffix), child["loss_terms"]["diagnostics"].get(native), f"Native diagnostic/{identity}/{field + suffix}")
            _same(row.get(field), row[field + "_attacked"] - row[field + "_clean"], f"Native diagnostic delta/{identity}/{field}")

    summary = consume("summary.json")
    _require(summary.get("completed_conditions") == len(records) and summary.get("expected_conditions") == 48,
             "Summary counts differ from saved results")
    _require(validation.get("completed_conditions") == len(records), "Analysis count differs from saved results")
    missing = sorted(expected - set(records))
    _require(summary.get("complete") == (not missing) and summary.get("missing_conditions") == [list(k) for k in missing],
             "Summary completion/missing conditions differ")
    aggregates = _index(analysis.get("paired_aggregates"), ("dataset", "objective"), "paired_aggregates")
    summary_aggregates = _index(summary.get("aggregate"), ("dataset", "objective"), "summary aggregate")
    for dataset in GROUPS:
        for objective in ("clean", *OBJECTIVES):
            children = [r for (d, _, o), r in records.items() if o == objective and (dataset == "all" or d == dataset)]
            if not children:
                continue
            _require((dataset, objective) in summary_aggregates, "Missing summary aggregate")
            _crosscheck_aggregate(summary_aggregates[dataset, objective], children, dataset, objective, "summary")
            if objective != "clean":
                pairs = [row for (d, _, o), row in paired.items() if o == objective and (dataset == "all" or d == dataset)]
                _require((dataset, objective) in aggregates, "Missing paired aggregate")
                _crosscheck_aggregate(aggregates[dataset, objective], pairs, dataset, objective, "paired analysis")
    _require(set(aggregates) == {(d, o) for d in GROUPS for o in OBJECTIVES if any(key[2] == o and (d == "all" or key[0] == d) for key in paired)},
             "Foreign/missing paired aggregate groups")
    for filename, keys, references in (("sequence_metrics.csv", ("dataset", "sequence", "objective"), records),
                                       ("aggregate_metrics.csv", ("dataset", "objective"), summary_aggregates)):
        path = root / filename
        digest = sha_file(path) if path.is_file() else None
        _require(digest is not None and digest == _audit_digest(audit, filename), f"CSV is not bound by current audit: {filename}")
        if filename == "sequence_metrics.csv":
            _require(scalar_hashes.get(filename) == digest, "Sequence CSV is not bound by analysis")
        current_hashes[filename] = digest
        with path.open(encoding="utf-8-sig", newline="") as stream:
            found = _index(list(csv.DictReader(stream)), keys, filename)
        _require(set(found) == set(references), f"CSV identities differ: {filename}")
        for identity, reference in references.items():
            for _, (section, native, _, _) in CHANGE_FIELDS.items():
                field = ("tracking_" if section == "tracking_metrics" else "reconstruction_") + native
                expected_value = reference[section][native] if filename == "sequence_metrics.csv" else reference[field]
                try:
                    value = float(found[identity][field])
                except (ValueError, KeyError, TypeError) as exc:
                    raise ReportValidationError(f"Malformed CSV metric: {filename}/{identity}/{field}") from exc
                _same(value, expected_value, f"CSV/{filename}/{identity}/{field}")
    current_hashes["artifact_validation.json"] = audit_sha
    # Recheck to detect an input writer changing scalar evidence during loading.
    for relative, digest in current_hashes.items():
        _require(sha_file(_relative_path(root, relative)) == digest, f"Evidence changed while loading: {relative}")
    _require(sha_file(analysis_root / "study_analysis.json") == analysis_sha, "Analysis changed while loading")
    return {"run_dir": str(root), "analysis_dir": str(analysis_root), "metadata": metadata, "analysis": analysis,
            "audit": audit, "campaign": campaign, "resume": resume, "summary": summary, "records": records,
            "histories": histories, "sequences": sequences, "identities": identities,
            "input_sha256": current_hashes, "analysis_sha256": analysis_sha, "complete_verified": bool(require_complete)}


def load_verified_study(run_dir, analysis_dir=None, require_complete=True):
    """Read-only strict gate; malformed/missing evidence is a validation error."""
    try:
        return _load_verified_study(run_dir, analysis_dir, require_complete)
    except ReportValidationError:
        raise
    except (OSError, KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ReportValidationError(f"Malformed or missing report evidence: {exc}") from exc


def compute_task_coupling(study, samples=2000, seed=20261004):
    """Fresh contrasts reuse paired clip indices; original analysis is untouched."""
    _require(study.get("complete_verified") is True, "Cross-task final analysis requires the strict completion gate")
    _require(type(samples) is int and samples > 0 and type(seed) is int and seed >= 0, "Invalid bootstrap samples/seed")
    analysis, identities = study["analysis"], study["identities"]
    paired = _index(analysis["paired_conditions"], ("dataset", "sequence", "objective"), "paired_conditions")
    _require(len(paired) == 40 and len(identities) == len(set(identities)) == 8, "Incomplete/duplicate paired population")
    cis = _index(analysis["bootstrap"]["estimates"], ("dataset", "objective", "metric"), "bootstrap estimates")
    impact = []
    for group in GROUPS:
        for objective in OBJECTIVES:
            source = next(row for row in analysis["paired_aggregates"] if (row["dataset"], row["objective"]) == (group, objective))
            row = {**source, **_ratios(source), "ci95": {}}
            for metric in METRICS:
                ci = cis.get((group, objective, metric))
                _require(ci is not None and ci.get("unit") == METRIC_UNITS[metric] and ci.get("common_clips") == source["paired_clips"],
                         f"Missing/wrong original CI: {group}/{objective}/{metric}")
                _same(ci.get("estimate"), source[metric], f"Original CI estimate/{group}/{objective}/{metric}")
                limits = ci.get("ci95")
                _require(isinstance(limits, list) and len(limits) == 2 and all(_finite(v) for v in limits) and limits[0] <= limits[1], "Invalid original CI endpoints")
                row["ci95"][metric] = list(limits)
            impact.append(row)
    points, excluded, point_index, connections, differences = [], [], {}, [], []
    for dataset, sequence in identities:
        for objective in OBJECTIVES:
            row = paired[dataset, sequence, objective]
            ratios = _ratios(row)
            point = {"dataset": dataset, "sequence": sequence, "objective": objective,
                     "seed": row["seed"], "x": ratios["tracking_apd_relative_drop_pct"], "y": ratios["reconstruction_apd_relative_drop_pct"]}
            if point["x"] is None or point["y"] is None:
                excluded.append({**point, "reason": "Missing/non-finite/near-zero clean APD denominator; never replaced by zero"})
            else:
                points.append(point)
                point_index[dataset, sequence, objective] = point
        a, b = paired[dataset, sequence, "tracking_3d"], paired[dataset, sequence, "reconstruction_3d"]
        _require(a["seed"] == b["seed"], "Geometry contrast does not share a seed")
        difference = {"dataset": dataset, "sequence": sequence, "seed": a["seed"]}
        for metric in METRICS:
            _same(a[metric + "_clean"], b[metric + "_clean"], "Geometry contrast does not share clean")
            difference[metric] = a[metric] - b[metric]
        differences.append(difference)
        first, second = point_index.get((dataset, sequence, "tracking_3d")), point_index.get((dataset, sequence, "reconstruction_3d"))
        if first is not None and second is not None:
            connections.append({"dataset": dataset, "sequence": sequence, "from": [first["x"], first["y"]],
                                "to": [second["x"], second["y"]], "from_objective": "tracking_3d", "to_objective": "reconstruction_3d"})
    rng = np.random.default_rng(seed)
    ordered = {d: sorted([row for row in differences if row["dataset"] == d], key=lambda row: row["sequence"]) for d in DATASETS}
    _require(all(len(rows) == 4 for rows in ordered.values()), "Contrast needs common PO4/DR4")
    indices = {d: rng.integers(0, 4, size=(samples, 4)) for d in DATASETS}
    distributions, estimates = {}, {}
    for dataset in DATASETS:
        values = np.asarray([[row[metric] for metric in METRICS] for row in ordered[dataset]], dtype=np.float64)
        distributions[dataset] = values[indices[dataset]].mean(axis=1)
        estimates[dataset] = values.mean(axis=0)
    distributions["all"] = .5 * distributions["po_mini"] + .5 * distributions["ds_mini"]
    estimates["all"] = .5 * estimates["po_mini"] + .5 * estimates["ds_mini"]
    entries = []
    for dataset in GROUPS:
        for index, metric in enumerate(METRICS):
            lower, upper = np.quantile(distributions[dataset][:, index], [.025, .975])
            count = 8 if dataset == "all" else 4
            entries.append({"dataset": dataset, "metric": metric, "estimate": float(estimates[dataset][index]),
                            "ci95": [float(lower), float(upper)], "unit": METRIC_UNITS[metric],
                            "paired_clips": count, "common_clips": count})
    return {"schema_version": 1, "run_id": Path(study["run_dir"]).name, "run_signature": study["metadata"]["signature"],
            "analysis_sha256": study["analysis_sha256"], "scope": analysis["scope"], "objectives": list(OBJECTIVES),
            "input_scalar_sha256": study["input_sha256"], "impact_matrix": impact,
            "ratio_definition": "Aggregate ratios use mean clean and mean attacked metrics; scatter uses per-clip ratios. Negative changes are preserved.",
            "scatter": {"points": points, "excluded": excluded, "connections": connections,
                        "definition": "x/y = 100 * tracking/reconstruction APD drop / that clip's clean APD; denominator >1e-12",
                        "mean_marker": "none", "connection_direction": "tracking_3d -> reconstruction_3d"},
            "geometry_contrasts": {"direction": "tracking_3d - reconstruction_3d", "clip_differences": differences,
                                   "estimates": entries, "bootstrap": {"seed": seed, "samples": samples,
                                       "method": "paired stratified percentile bootstrap of per-clip differences",
                                       "metric_order": list(METRICS), "unit": "clip", "dataset_weights": {"po_mini": .5, "ds_mini": .5},
                                       "ordered_clips": {d: [r["sequence"] for r in ordered[d]] for d in DATASETS},
                                       "resample_indices": {d: indices[d].tolist() for d in DATASETS},
                                       "indices_shared_across_metrics": True}},
            "interpretation": SCIENTIFIC_LIMITS}


def make_coupling_plots(coupling, output_dir):
    """Write nine standalone PNGs, sharing axes/scales across PO/DR panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    impact = {(row["dataset"], row["objective"]): row for row in coupling["impact_matrix"]}
    contrasts = {(row["dataset"], row["metric"]): row for row in coupling["geometry_contrasts"]["estimates"]}
    points = coupling["scatter"]["points"]
    limits = {}
    for family, metrics in (("apd", METRICS[:2]), ("epe", METRICS[2:])):
        limits[family] = max(1e-12, max(abs(row[metric]) for row in coupling["impact_matrix"] for metric in metrics))
    colors = dict(zip(OBJECTIVES, ("#4178b4", "#c44937", "#3b956b", "#9a69ae", "#a37b34")))
    xy = [0.] + [point[k] for point in points for k in ("x", "y")]
    low, high = min(xy), max(xy)
    pad = max(5., (high - low) * .08)
    xy_limits = low - pad, high + pad
    contrast_limits = {}
    for family, metrics in (("apd", METRICS[:2]), ("epe", METRICS[2:])):
        bound = max(1e-9, max(abs(value) for row in coupling["geometry_contrasts"]["estimates"] if row["metric"] in metrics for value in row["ci95"] + [row["estimate"]]))
        contrast_limits[family] = -bound * 1.15, bound * 1.15
    caption = "Verified full campaign: 48/48 conditions | all raw T=128 | PGD20, epsilon=4/255"
    for dataset in GROUPS:
        suffix = "" if dataset == "all" else "_" + dataset
        support = 8 if dataset == "all" else 4
        figure, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
        for axis, family, metrics, unit in ((axes[0], "apd", METRICS[:2], "APD drop (pp)"), (axes[1], "epe", METRICS[2:], "EPE increase (m)")):
            values = np.asarray([[impact[dataset, objective][metric] for metric in metrics] for objective in OBJECTIVES])
            image = axis.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -limits[family], limits[family]))
            axis.set_xticks((0, 1), ("Tracking", "Reconstruction"))
            axis.set_yticks(range(5), OBJECTIVES)
            axis.set_title(unit + "; shared scale across tasks and datasets")
            for index, objective in enumerate(OBJECTIVES):
                for column, metric in enumerate(metrics):
                    value = impact[dataset, objective][metric]
                    lo, hi = impact[dataset, objective]["ci95"][metric]
                    axis.text(column, index, f"{value:.3g}\n[{lo:.3g}, {hi:.3g}]", ha="center", va="center", fontsize=9,
                              color="white" if abs(value) / limits[family] > .7 else "#111111")
            figure.colorbar(image, ax=axis, shrink=.75, label=unit)
        figure.suptitle(f"Attack objective x task damage | {dataset}, paired clips={support}\n{caption}\nCells: change and recorded paired 95% CI; positive=damage, negative=improvement", fontsize=10)
        target = output / f"task_impact_matrix{suffix}.png"
        figure.savefig(target, dpi=180)
        plt.close(figure)
        paths.append(target.name)

        figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
        for link in coupling["scatter"]["connections"]:
            if dataset != "all" and link["dataset"] != dataset:
                continue
            axis.annotate("", xy=link["to"], xytext=link["from"], arrowprops={"arrowstyle": "->", "color": "#888888", "alpha": .55})
            axis.annotate(link["sequence"], xy=link["to"], xytext=(4, 4), textcoords="offset points", fontsize=7, color="#555555")
        for objective in OBJECTIVES:
            for source in DATASETS:
                if dataset != "all" and source != dataset:
                    continue
                group = [p for p in points if p["objective"] == objective and p["dataset"] == source]
                if group:
                    axis.scatter([p["x"] for p in group], [p["y"] for p in group], marker="o" if source == "po_mini" else "^",
                                 color=colors[objective], s=48, alpha=.85, edgecolor="white", linewidth=.5)
        axis.axhline(0, color="#555555", lw=.8)
        axis.axvline(0, color="#555555", lw=.8)
        axis.plot(xy_limits, xy_limits, linestyle="--", color="#bbbbbb", lw=.7)
        axis.set_xlim(xy_limits)
        axis.set_ylim(xy_limits)
        axis.set_xlabel("Tracking relative APD drop (%)")
        axis.set_ylabel("Reconstruction relative APD drop (%)")
        axis.grid(alpha=.15)
        excluded = sum(dataset == "all" or p["dataset"] == dataset for p in coupling["scatter"]["excluded"])
        count = sum(dataset == "all" or p["dataset"] == dataset for p in points)
        axis.set_title(f"Same-clip task damage | {dataset}, {count} points, {excluded} excluded\n{caption}\nArrow: tracking_3d -> reconstruction_3d; diagonal is equal relative drop, not equal physical damage", fontsize=10)
        handles = [Line2D([], [], marker="o", linestyle="none", color=color, label=objective) for objective, color in colors.items()]
        handles += [Line2D([], [], marker="o" if d == "po_mini" else "^", linestyle="none", color="#555555", label=d) for d in DATASETS]
        axis.legend(handles=handles, loc="best", fontsize=8)
        target = output / f"cross_task_apd_scatter{suffix}.png"
        figure.savefig(target, dpi=180)
        plt.close(figure)
        paths.append(target.name)

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for axis, family, metrics, unit in ((axes[0], "apd", METRICS[:2], "Difference in APD drop (pp)"), (axes[1], "epe", METRICS[2:], "Difference in EPE increase (m)")):
            for index, metric in enumerate(metrics):
                row = contrasts[dataset, metric]
                lo, hi = row["ci95"]
                value = row["estimate"]
                # Draw exact percentile endpoints; do not clip intervals around
                # a point estimate that can lie outside a percentile interval.
                axis.plot((lo, hi), (index, index), color="#426d96", lw=2)
                axis.scatter([value], [index], color="#426d96", s=40, zorder=3)
                axis.annotate(f"{value:.3g} [{lo:.3g}, {hi:.3g}]", (value, index), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
            axis.axvline(0, color="#777777", lw=.8)
            axis.set_xlim(contrast_limits[family])
            axis.set_ylim(-.6, 1.6)
            axis.set_yticks((0, 1), ("Tracking", "Reconstruction"))
            axis.invert_yaxis()
            axis.set_xlabel(unit)
            axis.grid(axis="x", alpha=.15)
        figure.suptitle(f"Geometry attack contrast: tracking_3d - reconstruction_3d | {dataset}, paired clips={support}\n{caption}\nFresh CI of paired differences; positive=larger relative damage under tracking_3d (check clean deltas separately)", fontsize=10)
        target = output / f"geometry_attack_contrasts{suffix}.png"
        figure.savefig(target, dpi=180)
        plt.close(figure)
        paths.append(target.name)
    return paths
