"""Run fixed-query clean, uniform noise, FGSM, and all-frame PGD experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .common import ROOT, VENDOR_ROOT, json_hash, provenance, resolve_path, sha256, write_json
from .evaluate_attack import evaluate_tracks, summarize_run
from .pgd_video import AttackConfig, attack_video, predict_tracks
from .st4rtrack_forward import St4RTrackForward
from .worldtrack_adapter import load_sequence


def stable_seed(base: int, dataset: str, sequence: str, method: str, epsilon: float) -> int:
    key = f"{base}:{dataset}:{sequence}:{method}:{epsilon:g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _append_failure(run_dir: Path, record: dict) -> None:
    with (run_dir / "failures.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _conditions(config):
    for method in config["methods"]:
        if method == "clean":
            yield method, 0.0
        else:
            for epsilon in config["epsilons_255"]:
                yield method, float(epsilon)


def run_experiment(config: dict, manifest: dict, run_dir: Path, *, resume=False, fail_fast=False,
                   vendor_root=VENDOR_ROOT) -> dict:
    if not manifest.get("entries"):
        raise ValueError("Manifest is empty")
    identities = [(entry["dataset"], entry["sequence"]) for entry in manifest["entries"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Manifest contains duplicate dataset/sequence identities")
    if len(config["methods"]) != len(set(config["methods"])) or len(config["epsilons_255"]) != len(set(config["epsilons_255"])):
        raise ValueError("Duplicate methods or epsilon values")
    if set(config["methods"]) - {"clean", "noise", "fgsm", "pgd"} or "clean" not in config["methods"]:
        raise ValueError("Methods must include clean and contain only clean/noise/fgsm/pgd")
    if int(config["num_frames"]) != manifest["num_frames"]:
        raise ValueError("Config frame count differs from frozen manifest")
    if config.get("image_size", 512) != 512:
        raise ValueError("Official evaluation requires image_size=512")
    if not config.get("save_inputs", True) and config.get("verify_saved_inputs", True):
        raise ValueError("verify_saved_inputs requires save_inputs")
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a compatible torch build or explicitly use --device cpu")
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
        if not resume:
            raise FileExistsError(f"Run exists: {run_dir}; use --resume with identical inputs or a new directory")
        if previous["signature"] != signature:
            raise ValueError("Resume refused: config, manifest, source, checkpoint, or runtime changed")
    else:
        write_json(metadata_path, {"schema_version": 1, "signature": signature,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                   "manifest": manifest, "provenance": version, "tta": False,
                   "perturbation_space": "post_official_resize_pre_normalization_rgb_0_1",
                   "official_clean_batch_size": 1, "confidence_filtering": False})
    forward = None
    failed = 0
    for entry_index, entry in enumerate(manifest["entries"]):
        dataset, sequence_name = entry["dataset"], entry["sequence"]
        if any(Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name for name in (dataset, sequence_name)):
            raise ValueError("Manifest dataset/sequence names must be single safe path components")
        sequence_dir = run_dir / "sequences" / dataset / sequence_name
        try:
            source = resolve_path(entry["path"])
            if entry.get("sha256") and sha256(source) != entry["sha256"]:
                raise ValueError(f"Dataset hash mismatch: {source}")
            print(f"[{entry_index + 1}/{len(manifest['entries'])}] {dataset}/{sequence_name}: loading", flush=True)
            sequence = load_sequence(source, num_frames=config["num_frames"], image_size=512, vendor_root=vendor_root)
            if len(sequence.rgb) != config["num_frames"]:
                raise ValueError(f"Requested {config['num_frames']} frames but sequence contains {len(sequence.rgb)}")
            if forward is None:
                forward = St4RTrackForward.from_checkpoint(checkpoint, sequence.query_xy, vendor_root=vendor_root, device=str(device))
            else:
                forward = St4RTrackForward(forward.model, sequence.query_xy, vendor_root=vendor_root,
                                           checkpoint=str(checkpoint))
            clean_rgb = sequence.rgb.to(device)
            gt, valid = sequence.gt_tracks.to(device), sequence.valid.to(device)
            if config.get("verify_official_clean", True):
                print("  checking official clean parity (batch_size=1)", flush=True)
                with torch.no_grad():
                    actual = predict_tracks(forward.pair_fn, clean_rgb)
                    reference = forward.official_predict(sequence).to(device)
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
                write_json(sequence_dir / "official_clean_parity.json", {"passed": True, "batch_size": 1,
                           "max_abs_error": float((actual - reference).abs().max().item())})
            write_json(sequence_dir / "sequence.json", sequence.metadata)
        except Exception as error:
            failed += 1
            _append_failure(run_dir, {"dataset": dataset, "sequence": sequence_name, "stage": "load_or_clean_parity",
                                    "error": repr(error), "traceback": traceback.format_exc()})
            print(f"FAILED {dataset}/{sequence_name}: {error}", flush=True)
            if fail_fast:
                raise
            continue
        for method, epsilon_255 in _conditions(config):
            label = "clean" if method == "clean" else f"{method}_eps{epsilon_255:g}"
            condition_dir = sequence_dir / label
            result_path = condition_dir / "result.json"
            if resume and result_path.exists():
                old = json.loads(result_path.read_text(encoding="utf-8"))
                if old["signature"] != signature:
                    raise ValueError(f"Invalid resumed condition: {condition_dir}")
                for name, digest in old.get("artifact_sha256", {}).items():
                    if not (condition_dir / name).is_file() or sha256(condition_dir / name) != digest:
                        raise ValueError(f"Resumed artifact missing or altered: {condition_dir / name}")
                print(f"  {label}: already complete", flush=True)
                continue
            try:
                epsilon = epsilon_255 / 255.0
                attack_config = AttackConfig(method=method, eps=epsilon, steps=config["steps"],
                    step_size=epsilon * config.get("step_size_fraction", 0.25), restarts=config["restarts"],
                    seed=stable_seed(config["seed"], dataset, sequence_name, method, epsilon_255),
                    random_start=config.get("random_start", True),
                    record_per_frame=config.get("record_per_frame", True),
                    gradient_mode=config.get("gradient_mode", "recompute"),
                    check_replay=config.get("check_replay", True))
                print(f"  {label}: starting ({attack_config.steps} steps for PGD)", flush=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                def progress(state):
                    if method == "pgd" and state["restart"] >= 0 and (state["step"] % 5 == 0 or state["step"] == config["steps"]):
                        print(f"    restart {state['restart'] + 1}, step {state['step']}/{config['steps']}, loss={state['loss']:.6f}", flush=True)
                result = attack_video(forward.pair_fn, clean_rgb, gt, valid, attack_config,
                                      progress_callback=progress)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                attack_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                metrics = evaluate_tracks(_numpy(result.tracks), _numpy(gt), _numpy(valid), _numpy(sequence.dynamic),
                                          sequence.metadata["intrinsics"], vendor_root)
                condition_dir.mkdir(parents=True, exist_ok=True)
                np.savez(condition_dir / "tracks.npz", pred=_numpy(result.tracks), gt=_numpy(gt), valid=_numpy(valid),
                         dynamic=_numpy(sequence.dynamic), query_xy=_numpy(sequence.query_xy),
                         intrinsics=np.asarray(sequence.metadata["intrinsics"]))
                write_json(condition_dir / "history.json", result.history)
                delta = result.delta.detach()
                frame_norms = [{"frame": frame, "linf": float(item.abs().max()), "l2": float(item.norm()),
                                "epsilon": epsilon} for frame, item in enumerate(delta)]
                write_json(condition_dir / "perturbation_norms.json", frame_norms)
                artifacts = ["tracks.npz", "history.json", "perturbation_norms.json"]
                if config.get("save_inputs", True):
                    np.save(condition_dir / "rgb_float32.npy", _numpy(result.rgb).astype(np.float32))
                    np.save(condition_dir / "delta_float32.npy", _numpy(delta).astype(np.float32))
                    artifacts.extend(["rgb_float32.npy", "delta_float32.npy"])
                verification = None
                if config.get("verify_saved_inputs", True):
                    replay_rgb = torch.from_numpy(np.load(condition_dir / "rgb_float32.npy", allow_pickle=False)).to(device)
                    torch.testing.assert_close(replay_rgb, result.rgb, rtol=0, atol=0)
                    with torch.no_grad():
                        replay_tracks = predict_tracks(forward.pair_fn, replay_rgb)
                    torch.testing.assert_close(replay_tracks, result.tracks, rtol=1e-5, atol=1e-5)
                    replay_metrics = evaluate_tracks(_numpy(replay_tracks), _numpy(gt), _numpy(valid), _numpy(sequence.dynamic),
                                                    sequence.metadata["intrinsics"], vendor_root)
                    for key in ("apd3d_all", "epe_all_m", "apd3d_dynamic", "epe_dynamic_m"):
                        if metrics[key] is not None and not np.isclose(metrics[key], replay_metrics[key], rtol=1e-5, atol=1e-5):
                            raise AssertionError(f"Saved input metric changed: {key}")
                    verification = {"passed": True, "metrics": replay_metrics,
                                    "max_track_error": float((replay_tracks - result.tracks).abs().max())}
                record = {"dataset": dataset, "sequence": sequence_name, "method": method, "epsilon_255": epsilon_255,
                          "steps": config["steps"] if method == "pgd" else int(method == "fgsm"),
                          "seed": attack_config.seed, "elapsed_seconds": elapsed, "signature": signature,
                          "attack_config": asdict(attack_config), "loss": result.loss, "scale": result.scale,
                          "model": forward.metadata,
                          "peak_vram_bytes": attack_peak,
                          "peak_vram_measurement": "attack_phase_including_clean_objective_forward",
                          "metrics": metrics, "saved_input_verification": verification,
                          "artifact_sha256": {name: sha256(condition_dir / name) for name in artifacts}}
                write_json(result_path, record)
                print(f"  {label}: APD={metrics['apd3d_all']:.4f}%, EPE={metrics['epe_all_m']:.6f}m, {elapsed:.1f}s", flush=True)
                del result, delta
                if config.get("verify_saved_inputs", True):
                    del replay_rgb, replay_tracks
            except Exception as error:
                failed += 1
                _append_failure(run_dir, {"dataset": dataset, "sequence": sequence_name, "method": method,
                                         "epsilon_255": epsilon_255, "stage": "attack_or_evaluation",
                                         "error": repr(error), "traceback": traceback.format_exc()})
                print(f"FAILED {dataset}/{sequence_name}/{label}: {error}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if fail_fast:
                    raise
    summary = summarize_run(run_dir, vendor_root)
    summary["failed_this_invocation"] = failed
    write_json(run_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--vendor-root", default=str(VENDOR_ROOT))
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(resolve_path(args.manifest).read_text(encoding="utf-8"))
    for key in ("checkpoint", "device", "steps", "restarts"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    summary = run_experiment(config, manifest, resolve_path(args.run_dir), resume=args.resume, fail_fast=args.fail_fast,
                             vendor_root=resolve_path(args.vendor_root))
    print(f"Results: {resolve_path(args.run_dir)} ({summary['completed_conditions']}/{summary['expected_conditions']})")
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
