"""Matched all-frame PGD attacks on the checkpoint's active loss components.

The five conditions share videos, evaluation queries, GT, RGB perturbation
budget, iterations, and initial random noise. Training-form objectives use
their own documented raster support, separate from benchmark query sampling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .common import VENDOR_ROOT, json_hash, load_official_metrics, provenance, resolve_path, sha256, write_json
from .component_attack import component_attack_video, component_loss_terms
from .component_forward import NativeComponentForward, check_official_component_oracle, load_component_sequence
from .evaluate_attack import evaluate_tracks, write_csv
from .pgd_video import AttackConfig
from .run_attack import stable_seed
from .st4rtrack_forward import St4RTrackForward

OBJECTIVES = ("tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training")
REQUIRED_ARTIFACTS = {"rgb_float32.npy", "delta_float32.npy", "reconstruction_float32.npy",
                      "reconstruction_confidence_float32.npy", "components.npz", "tracks.npz",
                      "history.json", "perturbation_norms.json"}
SEQUENCE_ARTIFACTS = {"sequence.json", "targets.npz", "reconstruction_gt.npy", "reconstruction_valid.npy",
                      "official_clean_parity.json"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def scalar_tree(value):
    if isinstance(value, dict):
        return {key: scalar_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return value


def event(run_dir: Path, kind: str, **values) -> None:
    record = {"time_utc": utc_now(), "event": kind, **values}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")


def synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def reconstruction_metrics(pred, gt, valid, intrinsics, vendor_root=VENDOR_ROOT) -> dict:
    """Official threshold/median-scale formula on fixed model-grid GT pixels.

    This is explicitly a model-grid spatial-correspondence reconstruction
    metric, not the paper's original-resolution/Sim(3) reconstruction score.
    """
    pred, gt, valid = array(pred), array(gt), array(valid).astype(bool)
    if pred.shape != gt.shape or pred.shape[:-1] != valid.shape or pred.shape[-1] != 3:
        raise ValueError("Dense reconstruction prediction/GT/mask shapes disagree")
    if not valid.any() or not np.isfinite(pred).all() or not np.isfinite(gt[valid]).all():
        raise FloatingPointError("Invalid dense reconstruction values/support")
    module = load_official_metrics(vendor_root)
    # Flatten all valid frame/pixel positions, as the official recon evaluator.
    reference = np.ascontiguousarray(gt[valid])[None]
    predicted = np.ascontiguousarray(pred[valid])[None]
    apd, aligned, fractions, transform, epe = module.compute_average_pts_within_thresh(
        reference, predicted, scaling="global", intrinsics_params=np.asarray(intrinsics),
        use_fixed_metric_threshold=True, compute_epe=True)
    if not np.isfinite([apd, epe, transform[0]]).all() or not np.isfinite(aligned).all():
        raise FloatingPointError("Reconstruction metrics are non-finite")
    result = {"apd3d": float(apd * 100), "epe_m": float(epe), "scale": float(transform[0]),
              "valid_pixels": int(valid.sum()), "num_frames": int(pred.shape[0]),
              "evaluation_grid": "official_RGB_preprocessed_model_grid",
              "alignment": "official_global_median_norm_ratio",
              "population": "fixed_finite_positive_depth_GT_pixels"}
    for key, threshold in module.PIXEL_TO_FIXED_METRIC_THRESH.items():
        result[f"within_{threshold:g}m"] = float(fractions[key] * 100)
    return result


def checked_config(config: dict, manifest: dict) -> None:
    if not isinstance(config.get("runtime_optimization", False), bool):
        raise ValueError("runtime_optimization must be a boolean")
    if "all_frames" in config or "all_frames" in manifest:
        if config.get("all_frames") is not True or manifest.get("all_frames") is not True:
            raise ValueError("Config and manifest must both explicitly declare all_frames=true")
        for item in (config, manifest):
            if type(item.get("campaign_expected_frames")) is not int or item["campaign_expected_frames"] != 128 or item["num_frames"] != 128:
                raise ValueError("All-frame WorldTrack study requires all 128 released frames")
    if not manifest.get("entries"):
        raise ValueError("Manifest has no entries")
    ids = [(row["dataset"], row["sequence"]) for row in manifest["entries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clips in manifest")
    if "campaign_expected_clips" in config or "campaign_expected_clips" in manifest:
        n = config.get("campaign_expected_clips")
        if (type(n) is not int or n <= 0 or n % 2 or
                type(manifest.get("campaign_expected_clips")) is not int or manifest["campaign_expected_clips"] != n):
            raise ValueError("Config and manifest require the same positive even campaign_expected_clips")
        limited = "explicitly limited" in str(manifest.get("purpose", "")).lower()
        if not limited and (len(ids) != n or any(sum(dataset == name for dataset, _ in ids) != n // 2
                                                   for name in ("po_mini", "ds_mini"))):
            raise ValueError("Full campaign requires the declared balanced PO/DR roster")
    for dataset, sequence in ids:
        if any(Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name
               for name in (dataset, sequence)):
            raise ValueError("Unsafe dataset/sequence path component")
    objectives = config["objectives"]
    if not objectives or len(objectives) != len(set(objectives)) or set(objectives) - set(OBJECTIVES):
        raise ValueError("Invalid/duplicate loss objectives")
    if config["num_frames"] != manifest["num_frames"] or config["image_size"] != 512:
        raise ValueError("Frame count/official RGB resolution mismatch")
    if not config.get("save_inputs", True) or not config.get("verify_saved_inputs", True):
        raise ValueError("Component study requires float RGB saves and replay verification")
    if config["alpha"] != .2 or config["reweight_scale"] != 5:
        raise ValueError("This checkpoint's active confidence loss uses alpha=.2 and max*5")
    AttackConfig(eps=float(config["epsilon_255"]) / 255, steps=config["steps"],
                 restarts=config["restarts"], gradient_mode=config["gradient_mode"])


def tensors_to_npz(path: Path, values: dict) -> None:
    np.savez(path, **{key: array(value) for key, value in values.items() if isinstance(value, torch.Tensor)})


def verify_files(directory: Path, record: dict, signature: str) -> None:
    if record.get("signature") != signature:
        raise ValueError(f"Foreign/stale condition: {directory}")
    if not record.get("saved_input_verification", {}).get("passed"):
        raise ValueError(f"Condition lacks successful RGB/model replay: {directory}")
    if not REQUIRED_ARTIFACTS <= set(record.get("artifact_sha256", {})):
        raise ValueError(f"Condition is missing required artifact digests: {directory}")
    if record.get("objective") == "clean" and "native_loss_oracle.json" not in record.get("artifact_sha256", {}):
        raise ValueError(f"Clean condition lacks native loss oracle: {directory}")
    for name, digest in record["artifact_sha256"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError(f"Changed/missing artifact: {directory / name}")


def verify_sequence(directory: Path, signature: str) -> None:
    inventory = json.loads((directory / "sequence_manifest.json").read_text(encoding="utf-8"))
    if inventory.get("signature") != signature or not SEQUENCE_ARTIFACTS <= set(inventory.get("artifact_sha256", {})):
        raise ValueError(f"Incomplete/foreign sequence provenance: {directory}")
    for name, digest in inventory["artifact_sha256"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError(f"Changed/missing sequence target: {directory / name}")
    if not json.loads((directory / "official_clean_parity.json").read_text(encoding="utf-8")).get("passed"):
        raise ValueError("Official clean parity did not pass")


def check_pack(actual, expected) -> dict:
    if set(actual) != set(expected):
        raise ValueError("Prediction fields differ at replay")
    errors = {}
    for name in actual:
        first, second = array(actual[name]), array(expected[name])
        if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise FloatingPointError(f"Invalid replay output: {name}")
        errors[name] = float(np.max(np.abs(first - second)))
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6, err_msg=name)
    return errors


def save_condition(directory, result, forward, sequence, targets, config, identity,
                   signature, *, vendor_root=VENDOR_ROOT) -> dict:
    """Publish result.json only after raw floats and both heads replay correctly."""
    directory.mkdir(parents=True, exist_ok=True)
    capture, reconstruction, confidence_map = forward.predict_with_maps(result.rgb)
    capture_error = check_pack(capture, result.outputs)
    reconstruction, confidence_map = array(reconstruction), array(confidence_map)
    np.save(directory / "rgb_float32.npy", array(result.rgb).astype(np.float32))
    np.save(directory / "delta_float32.npy", array(result.delta).astype(np.float32))
    np.save(directory / "reconstruction_float32.npy", reconstruction.astype(np.float32))
    np.save(directory / "reconstruction_confidence_float32.npy", confidence_map.astype(np.float32))
    tensors_to_npz(directory / "components.npz", capture)
    np.savez(directory / "tracks.npz", pred=array(capture["tracks"]), gt=array(sequence.gt_tracks),
             valid=array(sequence.valid), dynamic=array(sequence.dynamic), query_xy=array(sequence.query_xy),
             intrinsics=np.asarray(sequence.metadata["intrinsics"]))
    write_json(directory / "history.json", result.history)
    perturbation = array(result.delta).reshape(len(sequence.rgb), -1)
    norms = {"epsilon_255": identity["epsilon_255"], "linf": float(np.abs(perturbation).max()),
             "linf_per_frame": np.abs(perturbation).max(axis=1).tolist(),
             "l2_per_frame": np.linalg.norm(perturbation, axis=1).tolist(),
             "attacked_frames": int(np.any(perturbation != 0, axis=1).sum())}
    if norms["linf"] > float(identity["epsilon_255"]) / 255 + 1e-7:
        raise ValueError("Saved RGB attack exceeds the requested pixel budget")
    write_json(directory / "perturbation_norms.json", norms)
    cpu_targets = {key: value.detach().cpu() for key, value in targets.items()}
    recomputed_terms = scalar_tree(component_loss_terms(capture, cpu_targets,
                                      alpha=config["alpha"], reweight_scale=config["reweight_scale"]))
    for key in ("tracking_mse", "tracking_regression", "reconstruction_regression", "confidence", "total"):
        np.testing.assert_allclose(recomputed_terms[key], result.terms[key], rtol=2e-5, atol=2e-5, err_msg=key)
    if identity["objective"] == "clean":
        oracle = check_official_component_oracle(capture, torch.from_numpy(reconstruction),
            torch.from_numpy(confidence_map), forward.targets, vendor_root=vendor_root)
        np.testing.assert_allclose(recomputed_terms["total"], oracle["components"]["joint"], rtol=2e-5, atol=2e-5)
        write_json(directory / "native_loss_oracle.json", oracle)
    tracking = evaluate_tracks(array(capture["tracks"]), array(sequence.gt_tracks), array(sequence.valid),
                               array(sequence.dynamic), sequence.metadata["intrinsics"], vendor_root)
    dense_gt, dense_valid = forward.targets.reconstruction_gt, forward.targets.reconstruction_valid
    recon = reconstruction_metrics(reconstruction, dense_gt, dense_valid,
                                   sequence.metadata["intrinsics"], vendor_root)
    # Float reload preserves precisely the attacked input, with no image encoding.
    saved_rgb = np.load(directory / "rgb_float32.npy", allow_pickle=False)
    if saved_rgb.dtype != np.float32 or not np.isfinite(saved_rgb).all() or saved_rgb.min() < 0 or saved_rgb.max() > 1:
        raise FloatingPointError("Saved RGB is invalid")
    replay, replay_recon, replay_conf = forward.predict_with_maps(
        torch.from_numpy(saved_rgb).to(result.rgb.device))
    replay_errors = check_pack(replay, capture)
    recon_error = float(np.max(np.abs(array(replay_recon) - reconstruction)))
    conf_error = float(np.max(np.abs(array(replay_conf) - confidence_map)))
    np.testing.assert_allclose(array(replay_recon), reconstruction, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(array(replay_conf), confidence_map, rtol=1e-5, atol=1e-6)
    # A changed prediction or source cannot silently retain the recorded metric.
    replay_tracking = evaluate_tracks(array(replay["tracks"]), array(sequence.gt_tracks), array(sequence.valid),
                                      array(sequence.dynamic), sequence.metadata["intrinsics"], vendor_root)
    if replay_tracking != tracking:
        raise ValueError("Saved-input replay changed official tracking metrics")
    record = {**identity, "signature": signature, "attack_loss": result.loss,
              "loss_terms": result.terms, "selected_state": result.selected_state,
              "elapsed_seconds": result.elapsed_seconds,
              "attack_time_scope": "synchronized attack computation; excludes save/metrics/replay",
              "tracking_metrics": tracking, "reconstruction_metrics": recon,
              "saved_input_verification": {"passed": True, "capture_output_errors": capture_error,
                   "compact_output_errors": replay_errors, "max_reconstruction_error": recon_error,
                   "max_reconstruction_confidence_error": conf_error},
              "completed_at_utc": utc_now(), "perturbation_norms": norms,
              "artifact_sha256": {item.name: sha256(item) for item in directory.iterdir()
                                  if item.is_file() and item.name != "result.json"}}
    write_json(directory / "result.json", record)
    return record


def summarize_component_study(run_dir: Path, *, verify=True) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    ids = [(row["dataset"], row["sequence"]) for row in meta["manifest"]["entries"]]
    expected = {(dataset, sequence, objective) for dataset, sequence in ids
                for objective in ("clean", *meta["config"]["objectives"])}
    records, seen = [], set()
    for path in sorted((run_dir / "conditions").glob("*/*/*/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["dataset"], record["sequence"], record["objective"])
        canonical = run_dir / "conditions" / record["objective"] / record["dataset"] / record["sequence"] / "result.json"
        if path != canonical:
            raise ValueError(f"Condition directory differs from its recorded identity: {path}")
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected/duplicate condition: {path}")
        if verify:
            verify_files(path.parent, record, meta["signature"])
            verify_sequence(run_dir / "sequences" / record["dataset"] / record["sequence"], meta["signature"])
        seen.add(key)
        records.append(record)
    baseline = {(row["dataset"], row["sequence"]): row for row in records if row["objective"] == "clean"}
    rows = []
    for record in records:
        row = {key: record[key] for key in ("dataset", "sequence", "objective", "epsilon_255", "steps", "seed", "elapsed_seconds")}
        row["condition_wall_seconds"] = record.get("condition_wall_seconds")
        row.update({"tracking_" + key: value for key, value in record["tracking_metrics"].items()
                    if isinstance(value, (int, float))})
        row.update({"reconstruction_" + key: value for key, value in record["reconstruction_metrics"].items()
                    if isinstance(value, (int, float))})
        terms = record["loss_terms"]
        row.update({"loss_" + key: value for key, value in terms.items() if isinstance(value, (int, float))})
        row.update({"diagnostic_" + key: value for key, value in terms.get("diagnostics", {}).items()
                    if isinstance(value, (int, float))})
        clean = baseline.get((record["dataset"], record["sequence"]))
        if clean:
            row["tracking_apd_drop_pp"] = clean["tracking_metrics"]["apd3d_all"] - record["tracking_metrics"]["apd3d_all"]
            row["tracking_epe_increase_m"] = record["tracking_metrics"]["epe_all_m"] - clean["tracking_metrics"]["epe_all_m"]
            row["reconstruction_apd_drop_pp"] = clean["reconstruction_metrics"]["apd3d"] - record["reconstruction_metrics"]["apd3d"]
            row["reconstruction_epe_increase_m"] = record["reconstruction_metrics"]["epe_m"] - clean["reconstruction_metrics"]["epe_m"]
        rows.append(row)
    aggregate = []
    for dataset in ("po_mini", "ds_mini", "all"):
        for objective in ("clean", *meta["config"]["objectives"]):
            group = [row for row in rows if row["objective"] == objective
                     and (dataset == "all" or row["dataset"] == dataset)]
            if not group:
                continue
            item = {"dataset": dataset, "objective": objective, "completed_sequences": len(group),
                    "expected_sequences": sum(dataset == "all" or d == dataset for d, _ in ids),
                    "attack_seconds_sum": sum(row["elapsed_seconds"] for row in group)}
            for field in group[0]:
                values = [row.get(field) for row in group]
                if field not in item and all(isinstance(value, (int, float)) for value in values):
                    item[field] = float(np.mean(values))
            aggregate.append(item)
    write_csv(run_dir / "sequence_metrics.csv", rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate)
    failure_path = run_dir / "failures.jsonl"
    failures = [json.loads(line) for line in failure_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if failure_path.exists() else []
    summary = {"complete": seen == expected, "completed_conditions": len(seen),
               "expected_conditions": len(expected), "missing_conditions": sorted(expected - seen),
               "failures": failures, "failed_attempts": len(failures),
               "aggregation": f"equal_weight_per_sequence; all is the mean of the {len(ids)} matched clips",
               "tracking_protocol": "unchanged official WorldTrack retained queries and all times",
               "reconstruction_protocol": "fixed dense GT at preprocessed model grid, official median/threshold formula",
               "aggregate": aggregate}
    write_json(run_dir / "summary.json", summary)
    return summary


def run_study(config, manifest, run_dir: Path, *, resume=False, fail_fast=False, vendor_root=VENDOR_ROOT):
    checked_config(config, manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"] % 2**32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint = resolve_path(config["checkpoint"])
    version = provenance(checkpoint, vendor_root)
    signature = json_hash({"config": config, "manifest": manifest, "provenance": version})
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not resume or previous["signature"] != signature:
            raise ValueError("Existing run requires --resume and exactly identical config/manifest/source/runtime")
    else:
        write_json(metadata_path, {"schema_version": 1, "signature": signature, "config": config,
                   "manifest": manifest, "provenance": version, "created_at_utc": utc_now(),
                   "tta": False, "frozen_weights": True,
                   "shared_initial_noise": "stable_seed(base,dataset,sequence,pgd,epsilon); independent of objective",
                   "loss_scope": "checkpoint active native-form terms on WorldTrack GT; original training augmentation is not repeated"})
    event(run_dir, "study_started", signature=signature)
    model = None
    failed = 0
    for objective_index, objective in enumerate(config["objectives"]):
        objective_start = time.perf_counter()
        event(run_dir, "objective_started", objective=objective)
        print(f"LOSS [{objective_index+1}/{len(config['objectives'])}] {objective}: starting", flush=True)
        for entry_index, entry in enumerate(manifest["entries"]):
            dataset, sequence_name = entry["dataset"], entry["sequence"]
            condition_dir = run_dir / "conditions" / objective / dataset / sequence_name
            clean_dir = run_dir / "conditions" / "clean" / dataset / sequence_name
            sequence_dir = run_dir / "sequences" / dataset / sequence_name
            try:
                record_path = condition_dir / "result.json"
                if resume and record_path.exists() and clean_dir.joinpath("result.json").exists():
                    verify_files(condition_dir, json.loads(record_path.read_text(encoding="utf-8")), signature)
                    verify_files(clean_dir, json.loads((clean_dir / "result.json").read_text(encoding="utf-8")), signature)
                    verify_sequence(sequence_dir, signature)
                    print(f"[{entry_index+1}/{len(manifest['entries'])}] {dataset}/{sequence_name}/{objective}: verified existing", flush=True)
                    event(run_dir, "condition_skipped", objective=objective, dataset=dataset, sequence=sequence_name)
                    continue
                source = resolve_path(entry["path"])
                if entry.get("sha256") and sha256(source) != entry["sha256"]:
                    raise ValueError(f"Dataset hash mismatch: {source}")
                event(run_dir, "clip_loading", objective=objective, dataset=dataset, sequence=sequence_name)
                print(f"[{entry_index+1}/{len(manifest['entries'])}] {dataset}/{sequence_name}/{objective}: loading", flush=True)
                sequence, context = load_component_sequence(source, num_frames=config["num_frames"],
                                                            image_size=config["image_size"], vendor_root=vendor_root)
                if len(sequence.rgb) != config["num_frames"]:
                    raise ValueError("Actual frame count does not match frozen study")
                if config.get("all_frames", False):
                    if (sequence.metadata.get("original_frame_count") != 128 or len(sequence.rgb) != 128
                            or sequence.metadata.get("all_frames_used") is not True
                            or sequence.metadata.get("used_frame_indices") != list(range(128))):
                        raise ValueError("All-frame run did not load every released RGB/GT frame in source order")
                sequence = sequence.to(device)
                if model is None:
                    base = St4RTrackForward.from_checkpoint(checkpoint, sequence.query_xy, device=device, vendor_root=vendor_root)
                    model = base.model
                else:
                    base = St4RTrackForward(model, sequence.query_xy, vendor_root=vendor_root, checkpoint=str(checkpoint))
                optimized = config.get("runtime_optimization", False)
                forward = (NativeComponentForward(base, context, optimize_runtime=True)
                           if optimized else NativeComponentForward(base, context))
                runtime_kwargs = {"pair_frames_fn": forward.pair_frames_fn} if optimized else {}
                target_keys = ("native_tracks", "native_track_valid", "training_dynamic", "tracks", "valid",
                               "reconstruction_gt_norm", "reconstruction_valid_counts")
                targets = {key: context[key].to(device) for key in target_keys}
                sequence_dir.mkdir(parents=True, exist_ok=True)
                if not (sequence_dir / "sequence_manifest.json").exists():
                    write_json(sequence_dir / "sequence.json", {**sequence.metadata, "component_targets": context.metadata,
                        "runtime_optimization": forward.metadata.get("runtime_optimization", {"enabled": False})})
                    tensors_to_npz(sequence_dir / "targets.npz", targets)
                    np.save(sequence_dir / "reconstruction_gt.npy", array(forward.targets.reconstruction_gt))
                    np.save(sequence_dir / "reconstruction_valid.npy", array(forward.targets.reconstruction_valid))
                    write_json(sequence_dir / "official_clean_parity.json", base.check_clean_parity(sequence))
                    write_json(sequence_dir / "sequence_manifest.json", {"signature": signature,
                        "artifact_sha256": {name: sha256(sequence_dir / name) for name in sorted(SEQUENCE_ARTIFACTS)}})
                else:
                    verify_sequence(sequence_dir, signature)
                seed = stable_seed(config["seed"], dataset, sequence_name, "pgd", config["epsilon_255"])
                identity = {"dataset": dataset, "sequence": sequence_name, "objective": objective,
                            "epsilon_255": float(config["epsilon_255"]), "steps": config["steps"], "seed": seed}
                if clean_dir.joinpath("result.json").exists():
                    verify_files(clean_dir, json.loads((clean_dir / "result.json").read_text(encoding="utf-8")), signature)
                else:
                    print("  clean: capturing both heads and native loss terms", flush=True)
                    clean_start = time.perf_counter()
                    clean_result = component_attack_video(forward.pair_fn, sequence.rgb, targets, "tracking_mse",
                        AttackConfig(method="clean", eps=0, seed=seed), alpha=config["alpha"], reweight_scale=config["reweight_scale"])
                    clean_record = save_condition(clean_dir, clean_result, forward, sequence, targets, config,
                        {**identity, "objective": "clean", "epsilon_255": 0., "steps": 0}, signature, vendor_root=vendor_root)
                    clean_record["condition_wall_seconds"] = time.perf_counter() - clean_start
                    write_json(clean_dir / "result.json", clean_record)
                    event(run_dir, "clean_completed", dataset=dataset, sequence=sequence_name,
                          condition_wall_seconds=clean_record["condition_wall_seconds"])
                    del clean_result
                if resume and record_path.exists():
                    verify_files(condition_dir, json.loads(record_path.read_text(encoding="utf-8")), signature)
                    event(run_dir, "condition_skipped", objective=objective, dataset=dataset, sequence=sequence_name)
                    continue
                synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                condition_start = time.perf_counter()
                event(run_dir, "condition_started", **identity)
                print(f"  {objective}: PGD {config['steps']} steps eps={config['epsilon_255']}/255", flush=True)

                def progress(item):
                    event(run_dir, "iteration", **identity, restart=item["restart"], step=item["step"], loss=item["loss"])
                    if item["restart"] >= 0 and (item["step"] % 5 == 0 or item["step"] == config["steps"]):
                        print(f"    restart {item['restart']+1}, step {item['step']}/{config['steps']}, loss={item['loss']:.6f}", flush=True)
                        event(run_dir, "progress", **identity, step=item["step"], loss=item["loss"])

                attack_config = AttackConfig(method="pgd", eps=config["epsilon_255"] / 255, steps=config["steps"],
                    step_size=config["epsilon_255"] / 255 * config["step_size_fraction"], restarts=config["restarts"],
                    seed=seed, gradient_mode=config["gradient_mode"], check_replay=True, record_per_frame=True)
                result = component_attack_video(forward.pair_fn, sequence.rgb, targets, objective, attack_config,
                    alpha=config["alpha"], reweight_scale=config["reweight_scale"], progress_callback=progress,
                    **runtime_kwargs)
                synchronize(device)
                peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                record = save_condition(condition_dir, result, forward, sequence, targets, config, identity,
                                        signature, vendor_root=vendor_root)
                record.update({"peak_vram_bytes": peak, "attack_config": asdict(attack_config),
                               "runtime_optimization": forward.metadata.get("runtime_optimization", {"enabled": False}),
                               "condition_wall_seconds": time.perf_counter() - condition_start})
                write_json(condition_dir / "result.json", record)
                event(run_dir, "condition_completed", **identity, elapsed_seconds=record["elapsed_seconds"],
                      condition_wall_seconds=record["condition_wall_seconds"])
                print(f"  completed {objective}: tracking APD={record['tracking_metrics']['apd3d_all']:.4f}%, "
                      f"reconstruction APD={record['reconstruction_metrics']['apd3d']:.4f}%, "
                      f"attack={record['elapsed_seconds']:.1f}s", flush=True)
                del result, forward, context, targets, sequence
            except Exception as exc:
                failed += 1
                failure = {"dataset": dataset, "sequence": sequence_name,
                           "objective": objective, "time_utc": utc_now(), "error": str(exc), "traceback": traceback.format_exc()}
                with (run_dir / "failures.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(failure, allow_nan=False) + "\n")
                event(run_dir, "condition_failed", dataset=dataset, sequence=sequence_name, objective=objective, error=str(exc))
                print(f"FAILED {dataset}/{sequence_name}/{objective}: {exc}", flush=True)
                if fail_fast:
                    summarize_component_study(run_dir, verify=False)
                    raise
        objective_wall = time.perf_counter() - objective_start
        event(run_dir, "objective_completed", objective=objective, wall_seconds=objective_wall)
        print(f"LOSS {objective}: wall={objective_wall:.1f}s", flush=True)
        summarize_component_study(run_dir, verify=False)
    summary = summarize_component_study(run_dir)
    event(run_dir, "study_completed", complete=summary["complete"], completed_conditions=summary["completed_conditions"],
          expected_conditions=summary["expected_conditions"], failed_this_invocation=failed)
    print(f"Results: {run_dir} ({summary['completed_conditions']}/{summary['expected_conditions']})", flush=True)
    if failed or not summary["complete"]:
        raise RuntimeError("Component study contains failures or missing conditions")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--objectives", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-frames", type=int)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(resolve_path(args.manifest).read_text(encoding="utf-8"))
    if args.objectives is not None:
        if args.objectives != config["objectives"]:
            manifest["purpose"] = "explicitly limited objective selection; not the full matched campaign"
        config["objectives"] = args.objectives
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        manifest["entries"] = manifest["entries"][:args.limit]
        manifest["purpose"] = "explicitly limited preflight; not the full matched campaign"
    if args.steps is not None:
        if args.steps != config["steps"]:
            manifest["purpose"] = "explicitly limited step calibration; not the full matched campaign"
        config["steps"] = args.steps
    if args.num_frames is not None:
        if args.num_frames != config["num_frames"]:
            manifest["purpose"] = "explicitly limited frame calibration; not the full matched campaign"
        config["num_frames"] = manifest["num_frames"] = args.num_frames
    run_study(config, manifest, resolve_path(args.run_dir), resume=args.resume, fail_fast=args.fail_fast)


if __name__ == "__main__":
    main()
