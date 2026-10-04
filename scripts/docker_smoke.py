#!/usr/bin/env python3
"""Bounded CUDA smoke: one PO clip and one DR clip, exactly two frames each.

This script never calls the benchmark runner. It checks pinned author assets,
official clean parity, two-pass input gradients, shared frame-zero gradient
accumulation, and one tiny FGSM projection for each of two real clips.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"
EXPECTED_CHECKPOINT_SHA256 = "cae4712e0265f7d5ecadade19a0cb2ccf7b7e786fa0f230a80603a3437bcdfd3"
FRAMES = 2
EPSILON = 4.0 / 255.0


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(vendor: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(vendor), *arguments], check=True, capture_output=True,
        text=True, timeout=15,
    ).stdout.strip()


def _select_clip(data_root: Path, dataset: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise FileNotFoundError(f"Expected one existing {dataset} .npz: {path}")
        return path
    files = sorted((data_root / dataset).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No WorldTrack files in {data_root / dataset}")
    return files[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/docker/smoke/report.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth")
    parser.add_argument("--vendor-root", type=Path, default=ROOT / "vendor/St4RTrack")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/worldtrack_release")
    parser.add_argument("--po-sequence", help="Optional single PO .npz; still uses exactly two frames")
    parser.add_argument("--dr-sequence", help="Optional single DR .npz; still uses exactly two frames")
    parser.add_argument("--device", default="cuda:0", help="CUDA device only; CPU unit tests run separately")
    parser.add_argument("--timeout-seconds", type=int, default=120,
                        help="Linux wall-clock timeout, 1..240 seconds (default: 120)")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "passed": False,
        "scope": "CUDA smoke only: one PO and one DR clip, exactly two frames, one FGSM step each",
        "benchmark_runner_called": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": args.timeout_seconds,
        "num_frames_per_clip": FRAMES,
        "current_stage": "initialization",
        "timings_seconds": {},
        "clips": {},
    }
    alarm_armed = False

    def stage(name: str, operation):
        report["current_stage"] = name
        print(f"[smoke] {name}", flush=True)
        tick = time.perf_counter()
        result = operation()
        report["timings_seconds"][name] = time.perf_counter() - tick
        _write_report(output, report)
        return result

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"Docker smoke exceeded {args.timeout_seconds}s during {report['current_stage']}")

    try:
        if not 1 <= args.timeout_seconds <= 240:
            raise ValueError("Smoke timeout must be between 1 and 240 seconds")
        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("Execute this bounded smoke inside the Linux Docker container (SIGALRM required)")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout_seconds)
        alarm_armed = True
        _write_report(output, report)

        vendor, checkpoint, data_root = (path.expanduser().resolve() for path in
                                         (args.vendor_root, args.checkpoint, args.data_root))

        def asset_checks():
            commit = _git(vendor, "rev-parse", "HEAD")
            if commit != EXPECTED_COMMIT:
                raise RuntimeError(f"Official commit mismatch: {commit}, expected {EXPECTED_COMMIT}")
            if _git(vendor, "status", "--porcelain", "--untracked-files=no"):
                raise RuntimeError("Pinned official tracked source has modifications")
            submodules = _git(vendor, "submodule", "status", "--recursive")
            if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
                raise RuntimeError(f"Official submodules are missing or changed: {submodules}")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Read-only author checkpoint is missing: {checkpoint}")
            digest = _sha256(checkpoint)
            if digest != EXPECTED_CHECKPOINT_SHA256:
                raise RuntimeError(f"Checkpoint SHA-256 mismatch: {digest}")
            report["assets"] = {
                "official_commit": commit, "tracked_source_unchanged": True,
                "submodule_status": submodules, "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": digest,
            }
        stage("pinned_assets", asset_checks)

        def runtime_imports():
            sys.path.insert(0, str(ROOT / "src"))
            import numpy as np
            import torch
            from st4rtrack_pgd.evaluate_attack import evaluate_tracks
            from st4rtrack_pgd.pgd_video import loss_and_gradient, predict_tracks, project_rgb
            from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward, normalize_rgb
            from st4rtrack_pgd.worldtrack_adapter import load_sequence
            return (np, torch, evaluate_tracks, loss_and_gradient, predict_tracks,
                    project_rgb, St4RTrackForward, normalize_rgb, load_sequence)

        (np, torch, evaluate_tracks, loss_and_gradient, predict_tracks, project_rgb,
         St4RTrackForward, normalize_rgb, load_sequence) = stage("runtime_imports", runtime_imports)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requires a working --gpus all container and CUDA-enabled PyTorch")
        torch.set_num_threads(4)
        torch.manual_seed(20261004)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report["runtime"] = {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "tf32": False, "tta": False,
        }

        sequences = {}
        for dataset, explicit in (("po_mini", args.po_sequence), ("ds_mini", args.dr_sequence)):
            path = _select_clip(data_root, dataset, explicit)
            sequence = stage(f"{dataset}_loader", lambda p=path: load_sequence(p, num_frames=FRAMES, vendor_root=vendor))
            if len(sequence.rgb) != FRAMES:
                raise AssertionError(f"{dataset}: expected exactly {FRAMES} decoded frames")
            if not torch.isfinite(sequence.rgb).all() or not torch.isfinite(sequence.gt_tracks).all():
                raise FloatingPointError(f"{dataset}: non-finite input or GT")
            if not sequence.valid.all() or sequence.valid.dtype != torch.bool:
                raise AssertionError(f"{dataset}: official fixed-query mask changed")
            if not ((sequence.rgb >= 0) & (sequence.rgb <= 1)).all():
                raise AssertionError(f"{dataset}: RGB outside [0,1]")
            if not np.allclose(sequence.metadata["first_extrinsic_w2c"], np.eye(4), atol=1e-5):
                raise AssertionError(f"{dataset}: first-camera world frame is not normalized")
            sequences[dataset] = sequence
            report["clips"][dataset] = {
                "source": str(path), "source_sha256": _sha256(path),
                "sequence": sequence.metadata["sequence_name"],
                "rgb_shape": list(sequence.rgb.shape), "gt_shape": list(sequence.gt_tracks.shape),
                "query_shape": list(sequence.query_xy.shape),
                "num_dynamic_queries": int(sequence.dynamic.sum()),
                "loader_passed": True,
            }

        first = sequences["po_mini"]
        loaded_forward = stage("strict_model_load", lambda: St4RTrackForward.from_checkpoint(
            checkpoint, first.query_xy, vendor_root=vendor, device=device))
        model = loaded_forward.model
        report["model"] = loaded_forward.metadata

        for dataset, sequence in sequences.items():
            def clip_check():
                forward = St4RTrackForward(model, sequence.query_xy, vendor_root=vendor, checkpoint=str(checkpoint))
                rgb = sequence.rgb.to(device)
                gt, valid = sequence.gt_tracks.to(device), sequence.valid.to(device)
                torch.cuda.reset_peak_memory_stats(device)
                clean = predict_tracks(forward.pair_fn, rgb)
                official = forward.official_predict(sequence)
                torch.testing.assert_close(clean, official, rtol=1e-5, atol=1e-5)
                parity_error = float((clean - official).abs().max())

                # Check actual model views and sum both normalized-view gradient
                # contributions back to their source video frames. For (0,0),
                # both branches must accumulate into the same frame-zero leaf.
                context = {"rgb": rgb, "t": 0}
                contributions = torch.zeros_like(rgb)
                branch_records = []

                def before_forward(_module, inputs):
                    view1, view2 = inputs
                    active_rgb, t = context["rgb"], context["t"]
                    for branch, frame, view in (("anchor", 0, view1), ("target", t, view2)):
                        expected = normalize_rgb(active_rgb[frame]).unsqueeze(0)
                        if not torch.equal(view["img"].detach(), expected.detach()):
                            raise AssertionError(f"{dataset}: {branch} did not use shared video frame {frame}")
                        if view["img"].requires_grad:
                            def record(gradient, source_frame=frame, source_branch=branch, pair_t=t):
                                if not torch.isfinite(gradient).all():
                                    raise FloatingPointError("Non-finite model-view gradient")
                                contributions[source_frame].add_(gradient.squeeze(0).detach(), alpha=2.0)
                                branch_records.append({"pair_t": pair_t, "branch": source_branch,
                                                       "video_frame": source_frame,
                                                       "gradient_l2": float(gradient.norm())})
                            view["img"].register_hook(record)

                def checked_pair(active_rgb, t):
                    context.update(rgb=active_rgb, t=t)
                    return forward.pair_fn(active_rgb, t)

                handle = model.register_forward_pre_hook(before_forward)
                try:
                    derivative = loss_and_gradient(checked_pair, rgb, gt, valid, mode="recompute", check_replay=True)
                finally:
                    handle.remove()
                norms = derivative.gradient.flatten(1).norm(dim=1)
                if not torch.isfinite(norms).all() or not (norms > 0).all():
                    raise AssertionError(f"{dataset}: each of the two frames needs a finite nonzero gradient")
                torch.testing.assert_close(derivative.gradient, contributions, rtol=2e-5, atol=1e-6)
                self_pair = [record for record in branch_records if record["pair_t"] == 0]
                if len(self_pair) != 2 or any(record["video_frame"] != 0 or record["gradient_l2"] <= 0 for record in self_pair):
                    raise AssertionError(f"{dataset}: both self-pair branches must backpropagate into shared frame zero")
                if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                    raise AssertionError("Model weights were not frozen")

                # Exactly one FGSM update from the already computed clean
                # derivative; no restarts or multi-step experiment is invoked.
                attacked = project_rgb(rgb + EPSILON * derivative.gradient.sign(), rgb, EPSILON).detach()
                delta = attacked - rgb
                linf = delta.abs().flatten(1).amax(dim=1)
                if not torch.isfinite(attacked).all() or not ((attacked >= 0) & (attacked <= 1)).all():
                    raise AssertionError("Tiny FGSM left the RGB box")
                if not (linf <= EPSILON + 1e-7).all() or not (linf > 0).all():
                    raise AssertionError("Tiny FGSM violated a per-frame perturbation bound")
                attacked_tracks = predict_tracks(forward.pair_fn, attacked)
                clean_metrics = evaluate_tracks(clean.cpu().numpy(), sequence.gt_tracks.numpy(), sequence.valid.numpy(),
                                                sequence.dynamic.numpy(), sequence.metadata["intrinsics"], vendor)
                attacked_metrics = evaluate_tracks(attacked_tracks.cpu().numpy(), sequence.gt_tracks.numpy(), sequence.valid.numpy(),
                                                   sequence.dynamic.numpy(), sequence.metadata["intrinsics"], vendor)
                artifact = output.parent / f"{dataset}_smoke_tracks.npz"
                np.savez(artifact, clean=clean.cpu().numpy(), fgsm=attacked_tracks.cpu().numpy(),
                         clean_rgb=rgb.cpu().numpy(), fgsm_rgb=attacked.cpu().numpy(), delta=delta.cpu().numpy(),
                         gt=sequence.gt_tracks.numpy(), valid=sequence.valid.numpy(), dynamic=sequence.dynamic.numpy(),
                         query_xy=sequence.query_xy.numpy())
                with np.load(artifact, allow_pickle=False) as saved:
                    replay_rgb = torch.from_numpy(saved["fgsm_rgb"]).to(device)
                    torch.testing.assert_close(replay_rgb, attacked, rtol=0, atol=0)
                    if not np.array_equal(saved["fgsm_rgb"] - saved["clean_rgb"], saved["delta"]):
                        raise AssertionError("Saved float RGB-clean does not equal saved delta")
                replay_tracks = predict_tracks(forward.pair_fn, replay_rgb)
                torch.testing.assert_close(replay_tracks, attacked_tracks, rtol=1e-5, atol=1e-5)
                replay_metrics = evaluate_tracks(replay_tracks.cpu().numpy(), sequence.gt_tracks.numpy(), sequence.valid.numpy(),
                                                 sequence.dynamic.numpy(), sequence.metadata["intrinsics"], vendor)
                for key, value in attacked_metrics.items():
                    if value is None:
                        if replay_metrics[key] is not None:
                            raise AssertionError(f"Saved input changed undefined metric {key}")
                    elif isinstance(value, (float, int)):
                        if not np.isclose(value, replay_metrics[key], rtol=1e-5, atol=1e-5):
                            raise AssertionError(f"Saved input changed metric {key}")
                    elif value != replay_metrics[key]:
                        raise AssertionError(f"Saved input changed metric metadata {key}")
                torch.cuda.synchronize(device)
                report["clips"][dataset].update({
                    "passed": True, "official_clean_batch_size": 1,
                    "official_clean_max_abs_error": parity_error,
                    "gradient_mode": "recompute", "gradient_l2_per_frame": norms.cpu().tolist(),
                    "gradient_loss": derivative.loss, "gradient_scale": derivative.scale,
                    "shared_anchor_inputs_match": True, "self_pair_both_branches_reach_frame_zero": True,
                    "model_view_gradient_accumulation_max_abs_error": float((derivative.gradient - contributions).abs().max()),
                    "model_view_gradient_records": branch_records,
                    "tiny_attack": {"method": "fgsm", "steps": 1, "epsilon_255": 4.0,
                                    "linf_per_frame": linf.cpu().tolist(), "projection_passed": True},
                    "clean_metrics": clean_metrics, "tiny_fgsm_metrics": attacked_metrics,
                    "saved_float_input_verification": {"passed": True,
                        "rgb_reload_bit_exact": True, "saved_delta_equals_attacked_minus_clean": True,
                        "max_track_error": float((replay_tracks - attacked_tracks).abs().max()),
                        "metrics_match": True},
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                    "tracks_artifact": str(artifact), "tracks_artifact_sha256": _sha256(artifact),
                })

            stage(f"{dataset}_cuda_smoke", clip_check)

        report["passed"] = True
        report["current_stage"] = "complete"
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(output, report)
        print(f"[smoke] PASSED in {report['elapsed_seconds']:.1f}s; report: {output}", flush=True)
        return 0
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = time.perf_counter() - started
        if alarm_armed:
            signal.alarm(0)
            alarm_armed = False
        _write_report(output, report)
        print(f"[smoke] FAILED during {report['current_stage']}: {error}; report: {output}", file=sys.stderr, flush=True)
        return 1
    finally:
        if alarm_armed:
            signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
