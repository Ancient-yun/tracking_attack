"""Measure exact original and optimized component VJPs in one ABAB process."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
ALIAS = "_st4rtrack_original_interleaved"
TARGET_KEYS = (
    "native_tracks", "native_track_valid", "training_dynamic", "tracks", "valid",
    "reconstruction_gt_norm", "reconstruction_valid_counts",
)


def absolute(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path, calibration: dict) -> dict[str, str]:
    """Reject altered/missing/extra Python source before executing the alias."""
    expected = calibration["provenance"]["experiment_source_sha256"]
    actual = {path.name: file_sha256(path) for path in sorted(snapshot.glob("*.py"))}
    if len(expected) != 13 or set(actual) != set(expected):
        raise AssertionError(f"Original source set differs: expected 13 files, got {sorted(actual)}")
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise AssertionError(f"Original calibration source SHA differs: {mismatches}")
    return actual


def load_original_package(snapshot: Path):
    if ALIAS in sys.modules:
        raise RuntimeError("Original alias is already imported; use a fresh benchmark process")
    spec = importlib.util.spec_from_file_location(
        ALIAS, snapshot / "__init__.py", submodule_search_locations=[str(snapshot)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot alias-load original package: {snapshot}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[ALIAS] = package
    spec.loader.exec_module(package)
    return tuple(importlib.import_module(f"{ALIAS}.{name}") for name in (
        "component_forward", "component_attack", "st4rtrack_forward",
    ))


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def independent_gradient_diagnostics(helper, originals: list, optimized: list) -> dict:
    result = helper(optimized, originals, relative_l2_cap=1e-3)
    result.update({
        "mode": "independent_repeated_ABAB_original_vs_optimized_gradients",
        "reference": "exact original calibration source engine in the same process",
        "native_repeat_label": "independent optimized B1/B2 gradients",
        "official_repeat_label": "independent original A1/A2 gradients",
        "comparison_graphs": "four independently constructed recompute graphs",
    })
    result.pop("backward_calls", None)
    result.pop("retain_graph", None)
    return result


def run(args, report: dict, report_path: Path) -> None:
    # Imports stay here so source verification helpers can be tested without CUDA.
    import torch
    from st4rtrack_pgd.component_attack import component_loss_and_gradient
    from st4rtrack_pgd.component_forward import NativeComponentForward, gradient_repeat_diagnostics, source_frame_counts
    from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward

    snapshot = absolute(args.snapshot_dir)
    calibration_path = absolute(args.calibration_run)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    hashes = verify_snapshot(snapshot, calibration)
    old_forward_module, old_engine, old_base_module = load_original_package(snapshot)
    report.update({
        "snapshot_dir": str(snapshot), "snapshot_source_sha256": hashes,
        "calibration_run": str(calibration_path), "calibration_run_sha256": file_sha256(calibration_path),
        "original_engine_module": old_engine.__name__,
        "original_engine_file": old_engine.__file__,
    })
    if not torch.cuda.is_available():
        raise RuntimeError("Interleaved benchmark requires CUDA")
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    vendor = absolute(args.vendor_root)
    sequence_path = absolute(args.sequence)
    sequence_sha = file_sha256(sequence_path)
    manifest_match = next((entry for entry in calibration["manifest"]["entries"]
                           if Path(entry["path"]).name == sequence_path.name), None)
    if manifest_match is None or sequence_sha != manifest_match["sha256"]:
        raise AssertionError("Benchmark input differs from the original calibration sequence")
    print("INTERLEAVED: verified 13 original sources; loading shared original RGB/GT and model", flush=True)
    sequence, context = old_forward_module.load_component_sequence(
        sequence_path, num_frames=args.frames, vendor_root=vendor,
    )
    report.update({"sequence_sha256": sequence_sha, "target_sha256": context.tensor_hashes(),
                   "input_policy": "one original loader result; identical RGB/GT/query tensors for A and B"})
    raw_counts = source_frame_counts(absolute(args.sequence))
    report["source_frame_counts"] = raw_counts
    report["all_frames_used"] = all(value == len(sequence.rgb) for value in raw_counts.values())
    if len(sequence.rgb) != args.frames:
        raise ValueError("Benchmark did not load every requested frame")
    sequence = sequence.to("cuda")
    original_base = old_base_module.St4RTrackForward.from_checkpoint(
        absolute(args.checkpoint), sequence.query_xy, vendor_root=vendor, device="cuda",
    )
    original_forward = old_forward_module.NativeComponentForward(original_base, context)
    optimized_base = St4RTrackForward(
        original_base.model, sequence.query_xy, vendor_root=vendor, checkpoint=original_base.checkpoint,
    )
    targets = {key: context[key].to("cuda") for key in TARGET_KEYS}
    report.update({"model": original_forward.metadata, "gpu": torch.cuda.get_device_name(),
                   "torch": torch.__version__, "timings": [], "precision": "float32",
                   "allow_tf32": False, "autocast": False})
    save_report(report_path, report)
    results = {"original": [], "optimized": []}
    for label, repeat in (("original", 0), ("optimized", 0), ("original", 1), ("optimized", 1)):
        optimized_forward = None
        if label == "optimized":
            optimized_forward = NativeComponentForward(optimized_base, context, optimize_runtime=True)
            forward = optimized_forward
            engine = component_loss_and_gradient
            keywords = {"pair_frames_fn": forward.pair_frames_fn}
            report["optimized_model"] = forward.metadata
        else:
            if getattr(original_base.model, "_st4rtrack_runtime_handle", None) is not None:
                raise AssertionError("Optimized model hook remained enabled before original measurement")
            forward, engine, keywords = original_forward, old_engine.component_loss_and_gradient, {}
        try:
            with torch.no_grad():
                forward.pair_fn(sequence.rgb, 0)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            starting_allocated = torch.cuda.memory_allocated()
            started = time.perf_counter()
            result = engine(forward.pair_fn, sequence.rgb, targets, args.objective,
                            mode="recompute", check_replay=True, **keywords)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            measurement = {
                "label": label, "repeat": repeat, "seconds": seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "starting_allocated_bytes": starting_allocated,
                "peak_incremental_allocated_bytes": torch.cuda.max_memory_allocated() - starting_allocated,
                "loss": result.loss,
            }
            saved = {"gradient": result.gradient.detach().cpu(),
                     "outputs": {name: value.detach().cpu() for name, value in result.outputs.items()},
                     "loss": result.loss}
            del result
            results[label].append(saved)
            report["timings"].append(measurement)
            save_report(report_path, report)
            print(f"INTERLEAVED {label} {repeat + 1}: {seconds:.3f}s; loss={saved['loss']:.8g}", flush=True)
        finally:
            if optimized_forward is not None:
                optimized_forward.restore_runtime()
                del forward, optimized_forward
            torch.cuda.synchronize()
    errors = {}
    for name in results["original"][0]["outputs"]:
        errors[name] = 0.0
        for original in results["original"]:
            for optimized in results["optimized"]:
                left, right = original["outputs"][name], optimized["outputs"][name]
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                errors[name] = max(errors[name], float((left - right).abs().max()))
    loss_errors = [abs(a["loss"] - b["loss"]) for a in results["original"] for b in results["optimized"]]
    for a in results["original"]:
        for b in results["optimized"]:
            if abs(a["loss"] - b["loss"]) > 2e-5 + 2e-5 * abs(a["loss"]):
                raise AssertionError("Interleaved original/optimized loss differs")
    gradient_check = independent_gradient_diagnostics(
        gradient_repeat_diagnostics, [entry["gradient"] for entry in results["original"]],
        [entry["gradient"] for entry in results["optimized"]],
    )
    medians = {label: statistics.median(item["seconds"] for item in report["timings"] if item["label"] == label)
               for label in results}
    report.update({
        "output_max_abs_errors": errors, "loss_max_abs_error": max(loss_errors),
        "gradient_parity": gradient_check, "median_seconds": medians,
        "speedup": medians["original"] / medians["optimized"],
        "time_reduction_percent": 100 * (1 - medians["optimized"] / medians["original"]),
        "peak_allocated_bytes": {label: max(item["peak_allocated_bytes"] for item in report["timings"]
                                             if item["label"] == label) for label in results},
        "memory_interpretation": "allocated peak includes fixed GT/model/input; reserved peak is allocator history dependent",
        "timing_scope": "CUDA-synchronized full-video recompute loss/RGB gradient; one-pair warmup, load, CPU copies, validation and JSON excluded",
        "comparison_order": ["original_1", "optimized_1", "original_2", "optimized_2"],
        "background_load_limitation": "ABAB reduces time-separated background-load bias; concurrent GPU work can still affect timings",
    })
    save_report(report_path, report)
    if not gradient_check["passed"]:
        raise AssertionError("RGB gradient failed repeat-noise gate with relative L2 cap 0.001")
    report["passed"] = True
    save_report(report_path, report)
    print(f"INTERLEAVED passed: {report['speedup']:.3f}x; reduction={report['time_reduction_percent']:.1f}%", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--objective", default="joint_training")
    parser.add_argument("--snapshot-dir", default="runs/docker/runtime_optimization_20261004/baseline_source_snapshot")
    parser.add_argument("--calibration-run", default="runs/docker/loss_calibration_20261004/run.json")
    parser.add_argument("--sequence", default="data/worldtrack_release/po_mini/cab_e_3rd_13.npz")
    parser.add_argument("--checkpoint", default="assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth")
    parser.add_argument("--vendor-root", default="vendor/St4RTrack")
    args = parser.parse_args()
    report = {"passed": False, "frames": args.frames, "objective": args.objective,
              "scope": "same-process interleaved exact original source versus semantic-preserving runtime optimization"}
    report_path = absolute(args.output_dir) / "interleaved_report.json"
    save_report(report_path, report)
    try:
        run(args, report, report_path)
    except Exception:
        report.update({"passed": False, "traceback": traceback.format_exc()})
        save_report(report_path, report)
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
