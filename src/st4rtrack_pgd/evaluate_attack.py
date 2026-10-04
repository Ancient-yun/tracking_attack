"""Re-evaluate saved query trajectories through the pinned official evaluator."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .common import VENDOR_ROOT, load_official_metrics, resolve_path, sha256, write_json


def evaluate_tracks(pred, gt, valid, dynamic, intrinsics, vendor_root=VENDOR_ROOT) -> dict:
    pred, gt = np.asarray(pred), np.asarray(gt)
    valid, dynamic = np.asarray(valid, dtype=bool), np.asarray(dynamic, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Invalid trajectory shapes: {pred.shape}, {gt.shape}")
    if valid.shape != pred.shape[:2] or dynamic.shape != (pred.shape[1],):
        raise ValueError("Evaluation masks have incompatible shapes")
    if not valid.all():
        raise ValueError("Official WorldTrack evaluates all times of first-frame queries; expected all-true valid mask")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise FloatingPointError("Non-finite trajectory: refused official evaluator's implicit finite-distance exclusion")
    if pred.shape[1] == 0:
        raise ValueError("No queries to evaluate")
    module = load_official_metrics(vendor_root)
    results = {"num_frames": pred.shape[0], "num_queries": pred.shape[1],
               "num_dynamic_queries": int(dynamic.sum()),
               "metric_alignment": "official_global_median_norm_ratio",
               "dynamic_alignment": "independent_global_scale_on_dynamic_subset"}
    for name, subset in (("all", np.ones_like(dynamic)), ("dynamic", dynamic)):
        if not subset.any():
            results.update({f"apd3d_{name}": None, f"epe_{name}_m": None, f"scale_{name}": None})
            for threshold in (0.1, 0.3, 0.5, 1.0):
                results[f"within_{threshold:g}m_{name}"] = None
            continue
        apd, aligned, fractions, transform, epe = module.compute_average_pts_within_thresh(
            gt[:, subset], pred[:, subset], scaling="global", intrinsics_params=np.asarray(intrinsics),
            use_fixed_metric_threshold=True, compute_epe=True)
        if not np.isfinite(aligned).all() or not np.isfinite([apd, epe, transform[0]]).all():
            raise FloatingPointError("Official metric computation produced a non-finite result")
        results[f"apd3d_{name}"] = float(apd * 100.0)
        results[f"epe_{name}_m"] = float(epe)
        results[f"scale_{name}"] = float(transform[0])
        for key, threshold in module.PIXEL_TO_FIXED_METRIC_THRESH.items():
            results[f"within_{threshold:g}m_{name}"] = float(fractions[key] * 100.0)
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(run_dir: Path, vendor_root=VENDOR_ROOT) -> dict:
    run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    recorded_metric_hash = run_metadata["provenance"].get("official_metric_source_sha256")
    metric_path = resolve_path(vendor_root) / "dust3r" / "track_eval_util.py"
    if recorded_metric_hash and sha256(metric_path) != recorded_metric_hash:
        raise ValueError("Official evaluator source differs from the source recorded for this run")
    expected = {(entry["dataset"], entry["sequence"], method, float(epsilon))
                for entry in run_metadata["manifest"]["entries"]
                for method in run_metadata["config"]["methods"]
                for epsilon in ([0] if method == "clean" else run_metadata["config"]["epsilons_255"])}
    completed = set()
    rows, groups, clean = [], defaultdict(list), {}
    for artifact in sorted((run_dir / "sequences").glob("*/*/*/tracks.npz")):
        condition = artifact.parent
        if not (condition / "result.json").exists():
            continue
        record = json.loads((condition / "result.json").read_text(encoding="utf-8"))
        identity = (record["dataset"], record["sequence"], record["method"], float(record["epsilon_255"]))
        if record.get("signature") != run_metadata["signature"] or identity not in expected:
            raise ValueError(f"Foreign/stale result refused: {condition}")
        if identity in completed:
            raise ValueError(f"Duplicate condition result: {condition}")
        for name, digest in record.get("artifact_sha256", {}).items():
            if Path(name).name != name or not (condition / name).is_file() or sha256(condition / name) != digest:
                raise ValueError(f"Saved artifact is missing or changed: {condition / name}")
        completed.add(identity)
        with np.load(artifact, allow_pickle=False) as saved:
            metrics = evaluate_tracks(saved["pred"], saved["gt"], saved["valid"], saved["dynamic"], saved["intrinsics"], vendor_root)
        row = {key: record[key] for key in ("dataset", "sequence", "method", "epsilon_255", "steps", "seed", "elapsed_seconds")}
        row.update(metrics)
        rows.append(row)
        groups[(row["dataset"], row["method"], row["epsilon_255"])].append(row)
        if row["method"] == "clean":
            clean[(row["dataset"], row["sequence"])] = row
    for row in rows:
        baseline = clean.get((row["dataset"], row["sequence"]))
        if baseline is not None:
            for subset in ("all", "dynamic"):
                apd, epe = f"apd3d_{subset}", f"epe_{subset}_m"
                row[f"apd_drop_{subset}_pp"] = baseline[apd] - row[apd] if row[apd] is not None else None
                row[f"epe_increase_{subset}_m"] = row[epe] - baseline[epe] if row[epe] is not None else None
    aggregate = []
    for (dataset, method, epsilon), members in sorted(groups.items()):
        summary = {"dataset": dataset, "method": method, "epsilon_255": epsilon, "completed_sequences": len(members),
                   "expected_sequences": sum(entry["dataset"] == dataset for entry in run_metadata["manifest"]["entries"])}
        for key in ("apd3d_all", "apd3d_dynamic", "epe_all_m", "epe_dynamic_m",
                    "apd_drop_all_pp", "apd_drop_dynamic_pp", "epe_increase_all_m", "epe_increase_dynamic_m", "elapsed_seconds"):
            values = [member[key] for member in members if member.get(key) is not None]
            summary[key] = float(np.mean(values)) if values else None
            if "dynamic" in key:
                summary["sequences_with_dynamic_queries"] = len(values)
        aggregate.append(summary)
    failures_path = run_dir / "failures.jsonl"
    failures = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines()] if failures_path.exists() else []
    write_csv(run_dir / "sequence_metrics.csv", rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate)
    summary = {"aggregation": "unweighted_sequence_mean_matches_official", "completed_conditions": len(rows),
               "expected_conditions": len(expected), "complete": completed == expected,
               "missing_conditions": [list(identity) for identity in sorted(expected - completed)],
               "failed_attempts": len(failures), "failures": failures, "aggregate": aggregate}
    write_json(run_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vendor-root", default=str(VENDOR_ROOT))
    args = parser.parse_args()
    summary = summarize_run(resolve_path(args.run_dir), resolve_path(args.vendor_root))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
