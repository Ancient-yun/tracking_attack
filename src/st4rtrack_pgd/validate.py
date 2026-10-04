"""Check official clean parity and exact recomputation on a real small clip."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .common import VENDOR_ROOT, resolve_path, write_json
from .evaluate_attack import evaluate_tracks
from .pgd_video import loss_and_gradient, predict_tracks
from .st4rtrack_forward import St4RTrackForward
from .tracking_loss import aligned_tracking_loss
from .worldtrack_adapter import load_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", default="assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth")
    parser.add_argument("--num-frames", type=int, default=2, help="Keep short: full mode retains all graphs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--output", default="runs/implementation_validation.json")
    parser.add_argument("--vendor-root", default=str(VENDOR_ROOT))
    args = parser.parse_args()
    # Match run_attack's FP32 numerical policy. TF32 convolution can otherwise
    # quantize tiny finite-difference perturbations and select different
    # backward arithmetic, which is unsuitable for an input-gradient check.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    vendor = resolve_path(args.vendor_root)
    sequence = load_sequence(resolve_path(args.sequence), num_frames=args.num_frames, vendor_root=vendor)
    forward = St4RTrackForward.from_checkpoint(resolve_path(args.checkpoint), sequence.query_xy,
                                             vendor_root=vendor, device=args.device)
    rgb = sequence.rgb.to(args.device)
    gt, valid = sequence.gt_tracks.to(args.device), sequence.valid.to(args.device)
    reference = forward.official_predict(sequence).to(args.device)
    actual = predict_tracks(forward.pair_fn, rgb)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    full = loss_and_gradient(forward.pair_fn, rgb, gt, valid, mode="full")
    recompute = loss_and_gradient(forward.pair_fn, rgb, gt, valid, mode="recompute")
    torch.testing.assert_close(full.tracks, recompute.tracks, rtol=1e-5, atol=1e-5)
    # The author's DPT head uses CUDA bilinear interpolation backward, which
    # PyTorch 2.7 documents as nondeterministic. Measure each implementation's
    # repeated-backward noise before judging their difference. The double-
    # precision CPU tests still enforce the exact mathematical identity.
    full_repeat = loss_and_gradient(forward.pair_fn, rgb, gt, valid, mode="full")
    recompute_repeat = loss_and_gradient(forward.pair_fn, rgb, gt, valid, mode="recompute")
    def gradient_error(left, right):
        difference = left - right
        return {"max_absolute": float(difference.abs().max()),
                "relative_l2": float(difference.norm() / left.norm().clamp_min(1e-12))}

    cross_error = gradient_error(full.gradient, recompute.gradient)
    repeat_errors = {"full": gradient_error(full.gradient, full_repeat.gradient),
                     "recompute": gradient_error(recompute.gradient, recompute_repeat.gradient)}
    repeat_max = max(item["max_absolute"] for item in repeat_errors.values())
    repeat_relative = max(item["relative_l2"] for item in repeat_errors.values())
    strict_close = bool(torch.isclose(full.gradient, recompute.gradient,
                                      rtol=2e-4, atol=2e-5).all())
    absolute_limit = max(2e-5, 2 * repeat_max)
    relative_limit = min(1e-3, max(2e-4, 2 * repeat_relative))
    gradient_passed = (cross_error["max_absolute"] <= absolute_limit
                       and cross_error["relative_l2"] <= relative_limit)
    norms = recompute.gradient.flatten(1).norm(dim=1)
    if not torch.isfinite(norms).all() or not (norms > 0).all():
        raise AssertionError("Each evaluated frame must receive a finite nonzero gradient")
    report = {"passed": gradient_passed, "num_frames": args.num_frames, "sequence": sequence.metadata,
              "allow_tf32": False,
              "official_clean_batch_size": 1,
              "official_clean_max_error": float((actual - reference).abs().max()),
              "full_vs_recompute_gradient_max_error": float((full.gradient - recompute.gradient).abs().max()),
              "full_vs_recompute_gradient_relative_l2_error": float((full.gradient - recompute.gradient).norm() / full.gradient.norm()),
              "gradient_l2_per_frame": norms.cpu().tolist(), "loss": recompute.loss,
              "gradient_validation": {"passed": gradient_passed,
                  "strict_elementwise_close": strict_close,
                  "cross_error": cross_error, "repeated_backward_errors": repeat_errors,
                  "max_absolute_limit": absolute_limit,
                  "relative_l2_limit": relative_limit,
                  "policy": "Difference must fit twice the measured repeated-backward noise, with relative L2 capped at 1e-3",
                  "nondeterminism_reference": "https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html"},
              "clean_metrics": evaluate_tracks(actual.cpu().numpy(), gt.cpu().numpy(), valid.cpu().numpy(),
                 sequence.dynamic.cpu().numpy(), sequence.metadata["intrinsics"], vendor)}
    if args.finite_difference:
        # Probe the largest gradient among pixels away from RGB box boundaries.
        steps = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
        eligible = (rgb > max(steps)) & (rgb < 1 - max(steps))
        if not eligible.any():
            raise ValueError("Finite difference requires at least one RGB pixel away from box boundaries")
        scores = recompute.gradient.abs().masked_fill(~eligible, -1)
        flat_index = int(scores.flatten().argmax())
        def median_identity(tracks):
            norms_flat = tracks.norm(dim=-1).flatten()
            order = norms_flat.argsort()
            middle = len(order) // 2
            central = order[middle - 1:middle + 1] if len(order) % 2 == 0 else order[middle:middle + 1]
            return [{"frame": int(index) // tracks.shape[1],
                     "query_index": int(index) % tracks.shape[1],
                     "pixel_xy": sequence.query_xy[int(index) % tracks.shape[1]].long().tolist(),
                     "norm": float(norms_flat[index])} for index in central]

        def probe_direction(direction, analytical):
            probes = []
            for step in steps:
                plus, minus = rgb + step * direction, rgb - step * direction
                with torch.no_grad():
                    upper_tracks = predict_tracks(forward.pair_fn, plus)
                    lower_tracks = predict_tracks(forward.pair_fn, minus)
                    upper = float(aligned_tracking_loss(upper_tracks, gt, valid))
                    lower = float(aligned_tracking_loss(lower_tracks, gt, valid))
                numerical = (upper - lower) / (2 * step)
                error = abs(numerical - analytical) / max(abs(numerical), abs(analytical), 1e-8)
                realized_direction = (plus - minus) / (2 * step)
                realized_analytical = float((recompute.gradient * realized_direction).sum())
                probes.append({"step": step, "analytical": analytical, "numerical": numerical,
                               "realized_direction_analytical": realized_analytical,
                               "realized_direction_relative_l2_error": float((realized_direction - direction).norm() / direction.norm()),
                               "relative_error": error, "loss_plus": upper, "loss_minus": lower,
                               "median_plus": median_identity(upper_tracks),
                               "median_minus": median_identity(lower_tracks)})
            return probes

        single_pixel = torch.zeros_like(rgb)
        single_pixel.flatten()[flat_index] = 1.0
        pixel_probes = probe_direction(single_pixel, float(recompute.gradient.flatten()[flat_index]))
        report["single_pixel_diagnostic"] = {"flat_index": flat_index, "probes": pixel_probes}
        # A deep FP32 forward has a finite numerical noise floor. A unit-L2
        # gradient direction changes more output signal than one RGB scalar;
        # its central difference still checks the same complete sequence loss.
        direction = recompute.gradient.masked_fill(~eligible, 0)
        direction = direction / direction.norm()
        analytical = float((recompute.gradient * direction).sum())
        probes = probe_direction(direction, analytical)
        # A single large step can cross median/ReLU boundaries. Require a
        # stable adjacent pair of step probes, not one lucky value.
        stable = [right for left, right in zip(probes, probes[1:])
                  if left["relative_error"] <= 0.15 and right["relative_error"] <= 0.15
                  and abs(left["numerical"] - right["numerical"]) <= 0.10 * max(abs(analytical), 1e-8)]
        report["finite_difference"] = {"method": "unit_l2_gradient_direction", "probes": probes,
                                        "baseline_median": median_identity(recompute.tracks),
                                        "direction_l2_per_frame": direction.flatten(1).norm(dim=1).cpu().tolist(),
                                        "stable_adjacent_pair_found": bool(stable),
                                        "selected_probe": stable[0] if stable else None}
        if not stable:
            report["passed"] = False
            write_json(resolve_path(args.output), report)
    if not gradient_passed:
        report["passed"] = False
    write_json(resolve_path(args.output), report)
    print(json.dumps({key: value for key, value in report.items() if key != "sequence"},
                     indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise AssertionError("Implementation validation failed; inspect the saved gradient/finite-difference diagnostics")


if __name__ == "__main__":
    main()
