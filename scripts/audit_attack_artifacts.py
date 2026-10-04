"""Audit saved clean/PGD artifacts with NumPy and the standard library only.

This never imports the experiment, PyTorch, an evaluator, or a model. It checks
the recorded official parity/reload evidence, not a new model replay. Missing
conditions or final summaries produce ``incomplete`` (exit 2), never a pass.
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
import subprocess

import numpy as np

ARTIFACTS = {
    "tracks.npz", "history.json", "perturbation_norms.json",
    "rgb_float32.npy", "delta_float32.npy",
}
TRACK_FIELDS = {"pred", "gt", "valid", "dynamic", "query_xy", "intrinsics"}
PINNED_OFFICIAL_COMMIT = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"
METRIC_KEYS = ("apd3d_all", "apd3d_dynamic", "epe_all_m", "epe_dynamic_m")
MEAN_KEYS = METRIC_KEYS + (
    "apd_drop_all_pp", "apd_drop_dynamic_pp", "epe_increase_all_m",
    "epe_increase_dynamic_m", "elapsed_seconds",
)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def close(left, right, *, atol=1e-5, rtol=1e-5):
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), abs_tol=atol, rel_tol=rtol)
    except (ValueError, TypeError):
        return False


def bit_equal(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(
        left.view(np.uint8).reshape(-1), right.view(np.uint8).reshape(-1))


class Audit:
    def __init__(self, run_dir):
        self.root = Path(run_dir).resolve()
        self.checked = 0
        self.errors = []
        self.missing = []
        self.hashes = {}
        self.canonical_comparisons = []

    def check(self, name, passed, path=None, detail=None):
        self.checked += 1
        if not bool(passed):
            item = {"check": name}
            if path is not None:
                item["path"] = str(path)
            if detail is not None:
                item["detail"] = detail
            self.errors.append(item)

    def require(self, path):
        path = Path(path)
        if not path.is_file():
            self.missing.append(str(path))
            return False
        return True

    def hash(self, path, expected=None):
        path = Path(path)
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = file_hash(path)
        actual = self.hashes[key]
        if expected is not None:
            self.check("SHA256 matches recorded digest", actual == expected, path)
        return actual


def audit_history(audit, path, frames, steps, epsilon, method):
    history = read_json(path)
    audit.check("history is a nonempty list", isinstance(history, list) and bool(history), path)
    if not isinstance(history, list) or not history:
        return {}
    gradient_entries = []
    for state in history:
        audit.check("history loss/scale finite", np.isfinite([state["loss"], state["scale"]]).all(), path)
        audit.check("history scale positive", state["scale"] > 0, path)
        delta = np.asarray(state.get("delta_linf_per_frame", []), dtype=np.float64)
        l2 = np.asarray(state.get("delta_l2_per_frame", []), dtype=np.float64)
        audit.check("history delta norms cover every frame", delta.shape == l2.shape == (frames,), path)
        audit.check("history delta norms finite/nonnegative", np.isfinite(delta).all()
                    and np.isfinite(l2).all() and (delta >= 0).all() and (l2 >= 0).all(), path)
        audit.check("history Linf within epsilon", delta.size > 0 and delta.max() <= epsilon + 1e-7, path)
        audit.check("history global/per-frame Linf agree", delta.size > 0
                    and close(state["delta_linf"], float(delta.max()), atol=1e-7, rtol=0), path)
        if "gradient_linf_per_frame" in state or "gradient_l2_per_frame" in state:
            gradients = np.asarray(state.get("gradient_linf_per_frame", []), dtype=np.float64)
            gradients_l2 = np.asarray(state.get("gradient_l2_per_frame", []), dtype=np.float64)
            audit.check("gradient norms cover all frames", gradients.shape == gradients_l2.shape == (frames,), path)
            audit.check("all-frame gradients finite and nonzero", np.isfinite(gradients).all()
                        and np.isfinite(gradients_l2).all() and (gradients > 0).all()
                        and (gradients_l2 > 0).all(), path)
            gradient_entries.append((state["restart"], state["step"]))
    if method == "pgd":
        audit.check("exactly 20 PGD gradient steps in restart zero", gradient_entries == [(0, x) for x in range(steps)], path)
        audit.check("PGD history includes clean plus final step", len(history) == steps + 2
                    and history[0]["restart"] == -1 and history[0]["step"] == 0
                    and history[-1]["restart"] == 0 and history[-1]["step"] == steps, path)
    else:
        audit.check("clean history contains only the baseline", len(history) == 1 and not gradient_entries
                    and history[0]["restart"] == -1 and history[0]["step"] == 0, path)
    return {"history_entries": len(history), "gradient_steps": len(gradient_entries),
            "gradient_frames_per_step": frames if gradient_entries else 0,
            "maximum_recorded_loss": max(state["loss"] for state in history)}


def audit_condition(audit, directory, expected, signature, frames, steps, epsilon):
    result_path = directory / "result.json"
    if not audit.require(result_path):
        return None
    record = read_json(result_path)
    audit.hash(result_path)
    identity = (record["dataset"], record["sequence"], record["method"], float(record["epsilon_255"]))
    audit.check("condition identity matches expected", identity == expected, result_path)
    audit.check("condition signature matches run", record.get("signature") == signature, result_path)
    audit.check("condition step count", record["steps"] == (steps if expected[2] == "pgd" else 0), result_path)
    audit.check("condition elapsed/loss/scale finite", np.isfinite(
        [record["elapsed_seconds"], record["loss"], record["scale"]]).all(), result_path)
    audit.check("condition elapsed nonnegative and scale positive", record["elapsed_seconds"] >= 0
                and record["scale"] > 0, result_path)
    attack = record["attack_config"]
    audit.check("attack configuration matches PGD setting", attack["method"] == expected[2]
                and close(attack["eps"], epsilon, atol=1e-12, rtol=0)
                and attack["steps"] == steps and attack["restarts"] == 1
                and attack["gradient_mode"] == "recompute" and attack["check_replay"] is True
                and attack["record_per_frame"] is True, result_path)
    digests = record.get("artifact_sha256", {})
    audit.check("all five saved artifacts have SHA256", set(digests) == ARTIFACTS, result_path)
    required_present = True
    for name in sorted(ARTIFACTS):
        path = directory / name
        if not audit.require(path):
            required_present = False
            continue
        audit.hash(path, digests.get(name, "missing"))
    if not required_present:
        return None
    with np.load(directory / "tracks.npz", allow_pickle=False) as archive:
        audit.check("trajectory fields complete", set(archive.files) == TRACK_FIELDS, directory)
        arrays = {name: archive[name] for name in TRACK_FIELDS}
    gt, pred, valid, dynamic, query = (arrays[key] for key in ("gt", "pred", "valid", "dynamic", "query_xy"))
    n = gt.shape[1] if gt.ndim == 3 else 0
    audit.check("trajectory shapes cover 64 frames", n > 0 and gt.shape == pred.shape == (frames, n, 3), directory)
    audit.check("trajectory/query/intrinsics finite", all(np.isfinite(arrays[key]).all()
                for key in ("gt", "pred", "query_xy", "intrinsics")), directory)
    audit.check("fixed mask/query shapes and bool types", valid.shape == (frames, n)
                and valid.dtype == np.bool_ and valid.all() and dynamic.shape == (n,)
                and dynamic.dtype == np.bool_ and query.shape == (n, 2), directory)
    metrics = record["metrics"]
    audit.check("metric counts match trajectories", metrics["num_frames"] == frames
                and metrics["num_queries"] == n and metrics["num_dynamic_queries"] == int(dynamic.sum()), result_path)
    for key in METRIC_KEYS:
        audit.check("metric finite or absent dynamic subset", (metrics[key] is None and "dynamic" in key
                    and not dynamic.any()) or (metrics[key] is not None and math.isfinite(float(metrics[key]))), result_path)
    replay = record.get("saved_input_verification")
    audit.check("saved-input reload verification passed", isinstance(replay, dict) and replay.get("passed") is True, result_path)
    if isinstance(replay, dict):
        audit.check("reload track error finite/nonnegative", close(replay.get("max_track_error"), replay.get("max_track_error"))
                    and replay["max_track_error"] >= 0, result_path)
        for key in METRIC_KEYS:
            audit.check("saved-input replay metric agrees", close(metrics[key], replay.get("metrics", {}).get(key)), result_path, key)
    history_info = audit_history(audit, directory / "history.json", frames, steps, epsilon, expected[2])
    if history_info:
        audit.check("saved result selects the strongest recorded objective", close(
            record["loss"], history_info["maximum_recorded_loss"], atol=1e-6, rtol=1e-6), result_path)
    rgb = np.load(directory / "rgb_float32.npy", mmap_mode="r", allow_pickle=False)
    delta = np.load(directory / "delta_float32.npy", mmap_mode="r", allow_pickle=False)
    audit.check("RGB/delta float32 video shapes", rgb.dtype == delta.dtype == np.float32
                and rgb.ndim == 4 and rgb.shape[0:2] == (frames, 3) and rgb.shape == delta.shape, directory)
    audit.check("RGB finite in [0,1]", np.isfinite(rgb).all() and (rgb >= 0).all() and (rgb <= 1).all(), directory)
    if rgb.ndim == 4 and query.ndim == 2 and query.shape[-1] == 2:
        audit.check("saved queries inside RGB map", (query[:, 0] >= 0).all()
                    and (query[:, 0] < rgb.shape[-1]).all() and (query[:, 1] >= 0).all()
                    and (query[:, 1] < rgb.shape[-2]).all(), directory)
    audit.check("delta finite", np.isfinite(delta).all(), directory)
    linf = np.max(np.abs(delta).reshape(frames, -1), axis=1)
    audit.check("saved delta Linf within epsilon", float(linf.max()) <= epsilon + 1e-7, directory)
    if expected[2] == "clean":
        audit.check("clean delta is exactly zero", np.count_nonzero(delta) == 0, directory)
    norms = read_json(directory / "perturbation_norms.json")
    audit.check("norms cover all frame IDs in order", len(norms) == frames
                and [item["frame"] for item in norms] == list(range(frames)), directory)
    for t, item in enumerate(norms[:frames]):
        audit.check("recorded frame Linf agrees with saved delta", close(item["linf"], float(linf[t]), atol=1e-7, rtol=0), directory)
        actual_l2 = float(np.linalg.norm(np.asarray(delta[t], dtype=np.float64).reshape(-1)))
        audit.check("recorded frame L2 agrees with saved delta", close(item["l2"], actual_l2, atol=1e-6, rtol=1e-5), directory)
        audit.check("recorded epsilon agrees", close(item["epsilon"], epsilon, atol=1e-12, rtol=0), directory)
    row = {key: record[key] for key in ("dataset", "sequence", "method", "epsilon_255", "steps", "seed", "elapsed_seconds")}
    row.update(metrics)
    return {"record": record, "arrays": arrays, "rgb": rgb, "delta": delta, "row": row,
            "details": {"dataset": expected[0], "sequence": expected[1], "method": expected[2],
                        "rgb_shape": list(rgb.shape), "num_queries": n,
                        "delta_linf": float(linf.max()), "nonzero_delta_frames": int((linf > 0).sum()), **history_info}}


def audit_aggregates(audit, completed, metadata):
    rows = [condition["row"] for condition in completed.values()]
    clean = {(row["dataset"], row["sequence"]): row for row in rows if row["method"] == "clean"}
    for row in rows:
        baseline = clean.get((row["dataset"], row["sequence"]))
        if baseline is None:
            continue
        for subset in ("all", "dynamic"):
            apd, epe = f"apd3d_{subset}", f"epe_{subset}_m"
            row[f"apd_drop_{subset}_pp"] = baseline[apd] - row[apd] if row[apd] is not None else None
            row[f"epe_increase_{subset}_m"] = row[epe] - baseline[epe] if row[epe] is not None else None
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], float(row["epsilon_255"]))].append(row)
    expected = {}
    for key, members in groups.items():
        item = {name: float(np.mean([member[name] for member in members if member.get(name) is not None]))
                if any(member.get(name) is not None for member in members) else None for name in MEAN_KEYS}
        item["completed_sequences"] = len(members)
        item["expected_sequences"] = sum(entry["dataset"] == key[0] for entry in metadata["manifest"]["entries"])
        expected[key] = item
    summary_path = audit.root / "summary.json"
    if audit.require(summary_path):
        summary = read_json(summary_path)
        audit.hash(summary_path)
        audit.check("summary reports all 8 conditions complete", summary["complete"] is True
                    and summary["completed_conditions"] == summary["expected_conditions"] == 8
                    and not summary["missing_conditions"], summary_path)
        audit.check("summary reports zero failures", summary.get("failed_attempts") == 0
                    and summary.get("failed_this_invocation", 0) == 0 and not summary.get("failures"), summary_path)
        actual = {(x["dataset"], x["method"], float(x["epsilon_255"])): x for x in summary["aggregate"]}
        audit.check("summary aggregate groups complete and unique", set(actual) == set(expected)
                    and len(actual) == len(summary["aggregate"]), summary_path)
        for key in set(actual) & set(expected):
            for field, value in expected[key].items():
                audit.check("summary unweighted mean matches condition records", close(value, actual[key].get(field)), summary_path,
                            {"group": list(key), "field": field})
    for filename, grouping, reference, fields in (
        ("sequence_metrics.csv", ("dataset", "sequence", "method", "epsilon_255"),
         {(row["dataset"], row["sequence"], row["method"], float(row["epsilon_255"])): row for row in rows}, MEAN_KEYS),
        ("aggregate_metrics.csv", ("dataset", "method", "epsilon_255"), expected,
         MEAN_KEYS + ("completed_sequences", "expected_sequences")),
    ):
        path = audit.root / filename
        if not audit.require(path):
            continue
        audit.hash(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            actual_rows = list(csv.DictReader(stream))
        actual = {tuple(float(row[field]) if field == "epsilon_255" else row[field] for field in grouping): row for row in actual_rows}
        audit.check("CSV identities complete and unique", set(actual) == set(reference) and len(actual) == len(actual_rows), path)
        for key in set(actual) & set(reference):
            for field in fields:
                recorded = actual[key].get(field)
                recorded = None if recorded in (None, "") else recorded
                audit.check("CSV metric/mean matches condition records", close(reference[key].get(field), recorded), path,
                            {"identity": list(key), "field": field})
    return [{"dataset": key[0], "method": key[1], "epsilon_255": key[2], **value}
            for key, value in sorted(expected.items())]


def audit_source_files(audit, metadata, project_root):
    project = Path(project_root).resolve()
    sources = []
    for entry in metadata["manifest"]["entries"]:
        sources.append((project / "data" / "worldtrack_release" / entry["dataset"] / f"{entry['sequence']}.npz",
                        entry["sha256"], entry["bytes"]))
    provenance = metadata["provenance"]
    for name, evidence in provenance["checkpoint_files"].items():
        sources.append((project / "assets" / "checkpoints" / name, evidence["sha256"], evidence["bytes"]))
    # Docker checks out LF from the pinned Git blob, whereas the original
    # Windows checkout can have CRLF. Verify this single official file against
    # Git's exact bytes and accept only the explicit CRLF -> LF conversion.
    # Experiment core sources below still require their exact raw SHA256.
    vendor = project / "vendor" / "St4RTrack"
    metric_path = vendor / "dust3r" / "track_eval_util.py"
    if audit.require(metric_path):
        raw_hash = audit.hash(metric_path)
        commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
        tracked_status = subprocess.check_output(
            ["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=no"], text=True)
        audit.check("official Git HEAD is the fixed pinned commit", commit == PINNED_OFFICIAL_COMMIT
                    and provenance["official_commit"] == PINNED_OFFICIAL_COMMIT, vendor)
        audit.check("official Git tracked status clean", not tracked_status.strip(), vendor)
        blob = subprocess.check_output(["git", "-C", str(vendor), "show", "HEAD:dust3r/track_eval_util.py"])
        blob_hash = hashlib.sha256(blob).hexdigest()
        raw = metric_path.read_bytes()
        canonical = raw.replace(b"\r\n", b"\n")
        canonical_hash = hashlib.sha256(canonical).hexdigest()
        audit.check("official metric host CRLF-to-LF bytes equal pinned Git blob", canonical == blob, metric_path)
        audit.check("official metric Git blob SHA256 equals recorded Docker source", blob_hash
                    == provenance["official_metric_source_sha256"], metric_path)
        audit.canonical_comparisons.append({
            "path": str(metric_path), "scope": "official metric source only",
            "raw_host_sha256": raw_hash, "git_head": commit,
            "git_blob_sha256": blob_hash, "canonical_host_sha256": canonical_hash,
            "recorded_docker_sha256": provenance["official_metric_source_sha256"],
            "canonicalization": "Exact bytes.replace(b'\\r\\n', b'\\n'); no other whitespace changes",
            "canonical_bytes_equal_git_blob": canonical == blob,
            "tracked_status_clean": not tracked_status.strip(),
        })
    for name, digest in provenance["experiment_source_sha256"].items():
        sources.append((project / "src" / "st4rtrack_pgd" / name, digest, None))
    for path, digest, size in sources:
        if audit.require(path):
            if size is not None:
                audit.check("source file advertised size matches", path.stat().st_size == size, path)
            audit.hash(path, digest)


def audit_run(run_dir, project_root, *, verify_sources=True):
    audit = Audit(run_dir)
    details, aggregate, completed = [], [], {}
    metadata = None
    try:
        if audit.require(audit.root / "run.json"):
            metadata = read_json(audit.root / "run.json")
            audit.hash(audit.root / "run.json")
            config, manifest = metadata["config"], metadata["manifest"]
            signature = metadata["signature"]
            audit.check("run signature matches canonical fingerprint", signature == fingerprint(
                {"config": config, "manifest": manifest, "provenance": metadata["provenance"]}), audit.root / "run.json")
            audit.check("run has requested 4x64 clean/PGD-20 epsilon4 restart1 settings", len(manifest["entries"]) == 4
                        and Counter(x["dataset"] for x in manifest["entries"]) == {"po_mini": 2, "ds_mini": 2}
                        and config["methods"] == ["clean", "pgd"] and config["epsilons_255"] == [4]
                        and config["num_frames"] == manifest["num_frames"] == 64
                        and config["steps"] == 20 and config["restarts"] == 1
                        and config.get("verify_saved_inputs") is True and config.get("verify_official_clean") is True, audit.root / "run.json")
            frames, steps = config["num_frames"], config["steps"]
            expected_paths = set()
            for entry in manifest["entries"]:
                dataset, sequence = entry["dataset"], entry["sequence"]
                audit.check("safe sequence path components", all(Path(x).name == x and x not in (".", "..")
                            and "/" not in x and "\\" not in x for x in (dataset, sequence)), audit.root / "run.json")
                sequence_dir = audit.root / "sequences" / dataset / sequence
                parity_path, sequence_path = sequence_dir / "official_clean_parity.json", sequence_dir / "sequence.json"
                if audit.require(parity_path):
                    parity = read_json(parity_path)
                    audit.hash(parity_path)
                    audit.check("recorded official clean parity passed", parity.get("passed") is True
                                and parity.get("batch_size") == 1 and close(parity.get("max_abs_error"), parity.get("max_abs_error"))
                                and parity["max_abs_error"] >= 0, parity_path)
                if audit.require(sequence_path):
                    sequence_metadata = read_json(sequence_path)
                    audit.hash(sequence_path)
                    audit.check("sequence metadata contains 64 frames", sequence_metadata["num_frames"] == frames, sequence_path)
                for method, epsilon255 in (("clean", 0.0), ("pgd", 4.0)):
                    label = "clean" if method == "clean" else "pgd_eps4"
                    directory = sequence_dir / label
                    expected_paths.add((directory / "result.json").resolve())
                    identity = (dataset, sequence, method, epsilon255)
                    try:
                        condition = audit_condition(audit, directory, identity, signature, frames, steps, epsilon255 / 255)
                        if condition is not None:
                            audit.check("no duplicate condition identity", identity not in completed, directory)
                            completed[identity] = condition
                            details.append(condition["details"])
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                        audit.check("condition readable and schema valid", False, directory, repr(error))
                left, right = completed.get((dataset, sequence, "clean", 0.0)), completed.get((dataset, sequence, "pgd", 4.0))
                if left is not None and right is not None:
                    for name in ("gt", "valid", "dynamic", "query_xy", "intrinsics"):
                        audit.check("clean/PGD share bit-identical evaluation arrays", bit_equal(left["arrays"][name], right["arrays"][name]), sequence_dir, name)
                    audit.check("clean/PGD RGB shapes agree", left["rgb"].shape == right["rgb"].shape, sequence_dir)
                    if left["rgb"].shape == right["rgb"].shape:
                        for t in range(frames):
                            difference = np.subtract(right["rgb"][t], left["rgb"][t], dtype=np.float32)
                            audit.check("saved PGD RGB minus clean equals saved delta bit-exactly", bit_equal(difference, right["delta"][t]), sequence_dir, {"frame": t})
            actual_paths = {path.resolve() for path in (audit.root / "sequences").glob("*/*/*/result.json")}
            audit.check("no foreign/unexpected condition results", actual_paths <= expected_paths, audit.root)
            if verify_sources:
                audit_source_files(audit, metadata, project_root)
            aggregate = audit_aggregates(audit, completed, metadata)
            failure_path = audit.root / "failures.jsonl"
            if failure_path.exists():
                audit.hash(failure_path)
                audit.check("no recorded failed attempts", not failure_path.read_text(encoding="utf-8").strip(), failure_path)
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
        audit.check("run metadata and summary readable/schema valid", False, audit.root, repr(error))
    expected_conditions = 8
    # Even a stale summary may still say complete while the run is being copied.
    # Missing artifacts always prevent a pass and are reported as incomplete;
    # integrity errors remain visible in the report for the next complete audit.
    status = ("incomplete" if audit.missing else "failed" if audit.errors else
              "incomplete" if len(completed) != expected_conditions else "passed")
    return {"schema_version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(audit.root), "status": status, "passed": status == "passed",
            "scope": "CPU-only saved-artifact integrity and recorded replay/parity evidence; no new model inference or official metric execution",
            "source_files_verified": verify_sources,
            "expected_conditions": expected_conditions, "completed_conditions_checked": len(completed),
            "checked_assertions": audit.checked, "errors": audit.errors,
            "missing": sorted(set(audit.missing)), "conditions": details,
            "recomputed_unweighted_aggregate_from_condition_records": aggregate,
            "file_sha256": audit.hashes,
            "canonical_source_comparisons": audit.canonical_comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = audit_run(args.run_dir, args.project_root)
    target = args.run_dir / "artifact_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "completed_conditions_checked": report["completed_conditions_checked"],
                      "expected_conditions": report["expected_conditions"], "errors": len(report["errors"]),
                      "missing": len(report["missing"]), "report": str(target.resolve())}, indent=2))
    return 0 if report["passed"] else 2 if report["status"] == "incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
