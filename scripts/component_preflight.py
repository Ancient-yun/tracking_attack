"""Bounded real-model verification before the matched loss-component campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from st4rtrack_pgd.common import provenance, resolve_path, write_json
from st4rtrack_pgd.component_attack import component_loss_and_gradient, component_loss_terms, predict_components
from st4rtrack_pgd.component_forward import NativeComponentForward, check_official_component_oracle, load_component_sequence
from st4rtrack_pgd.run_component_study import OBJECTIVES
from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward


def relative(a, b):
    return float(torch.linalg.vector_norm((a - b).double()) /
                 torch.linalg.vector_norm(a.double()).clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/loss_components.json")
    parser.add_argument("--manifest", default="docker/manifests/loss_components_14clips.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    manifest = json.loads(resolve_path(args.manifest).read_text())
    entries = [next(row for row in manifest["entries"] if row["dataset"] == dataset)
               for dataset in ("po_mini", "ds_mini")]
    started = time.perf_counter()
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {"passed": False, "scope": "real checkpoint, PO+DR, 2 frames; no full campaign", "checks": []}
    write_json(resolve_path(args.output), report)
    model = None
    for entry in entries:
        print(f"PREFLIGHT {entry['dataset']}/{entry['sequence']}: loading 2 frames", flush=True)
        sequence, context = load_component_sequence(resolve_path(entry["path"]), num_frames=2)
        sequence = sequence.to("cuda")
        if model is None:
            base = St4RTrackForward.from_checkpoint(resolve_path(config["checkpoint"]), sequence.query_xy, device="cuda")
            model = base.model
        else:
            base = St4RTrackForward(model, sequence.query_xy, checkpoint=str(resolve_path(config["checkpoint"])))
        optimized = config.get("runtime_optimization", False)
        forward = NativeComponentForward(base, context, optimize_runtime=optimized)
        runtime_kwargs = {"pair_frames_fn": forward.pair_frames_fn} if optimized else {}
        target_keys = ("native_tracks", "native_track_valid", "training_dynamic", "tracks", "valid",
                       "reconstruction_gt_norm", "reconstruction_valid_counts")
        targets = {name: context[name].to("cuda") for name in target_keys}
        checks = {"dataset": entry["dataset"], "sequence": entry["sequence"],
                  "num_frames": 2, "num_eval_queries": len(sequence.query_xy),
                  "num_native_queries": len(context.native_query_xy),
                  "official_clean_parity": base.check_clean_parity(sequence), "objectives": {},
                  "runtime_optimization": forward.metadata.get("runtime_optimization", {"enabled": False})}
        packed, maps, conf = forward.predict_with_maps(sequence.rgb)
        checks["native_captured_output_oracle"] = check_official_component_oracle(packed, maps, conf, context)
        checks["native_rgb_oracle"] = forward.check_official_loss_parity(
            sequence.rgb, loss_fn=lambda values, gt: component_loss_terms(values, gt)["total"], repeat_noise_check=True)
        print("  official query and native loss/gradient oracles passed", flush=True)
        for objective in OBJECTIVES:
            repeats = {}
            for mode in ("full", "recompute"):
                repeats[mode] = [component_loss_and_gradient(forward.pair_fn, sequence.rgb, targets, objective,
                    mode=mode, check_replay=True, **runtime_kwargs) for _ in range(2)]
            full, recompute = repeats["full"][0], repeats["recompute"][0]
            full_noise = relative(full.gradient, repeats["full"][1].gradient)
            replay_noise = relative(recompute.gradient, repeats["recompute"][1].gradient)
            cross = relative(full.gradient, recompute.gradient)
            allowance = max(1e-5, 1.5 * max(full_noise, replay_noise))
            if max(full_noise, replay_noise) > .002 or cross > .002 or cross > allowance:
                raise AssertionError(f"{objective} gradient disagreement exceeds measured repeat noise: {cross}/{allowance}")
            np.testing.assert_allclose(full.loss, recompute.loss, rtol=2e-5, atol=2e-5)
            per_frame = recompute.gradient.abs().flatten(1).sum(1).cpu().tolist()
            if not all(np.isfinite(value) and value > 0 for value in per_frame):
                raise AssertionError(f"{objective} clean gradient does not reach all frames")
            checks["objectives"][objective] = {"passed": True, "loss": full.loss,
                "full_repeat_relative_l2": full_noise, "recompute_repeat_relative_l2": replay_noise,
                "full_recompute_relative_l2": cross, "measured_noise_allowance": allowance,
                "relative_error_cap": .002, "gradient_l1_per_frame": per_frame,
                "strict_elementwise_close": bool(torch.allclose(full.gradient, recompute.gradient, atol=2e-5, rtol=2e-5))}
            print(f"  {objective}: loss={full.loss:.6g}, gradient relative error={cross:.4g}", flush=True)
            del repeats, full, recompute
        report["checks"].append(checks)
        write_json(resolve_path(args.output), report)
        del packed, maps, conf, forward, context, targets, sequence
    report.update({"passed": True, "elapsed_seconds": time.perf_counter() - started,
                   "provenance": provenance(resolve_path(config["checkpoint"]))})
    write_json(resolve_path(args.output), report)
    print(f"Preflight passed in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
