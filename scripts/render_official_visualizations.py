"""Call the pinned official St4RTrack visualizer on saved clean/PGD outputs.

The official drawing, projection, traces, point colors and 15 fps encoding are
used without modification. This wrapper verifies saved inputs, adapts camera
intrinsics to the saved image size, optionally subsamples queries independently
of predictions, and joins the already rendered official videos side by side.
No model is loaded and no inference or attack is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image


PINNED_COMMIT = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"
OFFICIAL_SOURCE = "dust3r/datasets/tapvid3d.py"
OFFICIAL_FPS = 15


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_bytes(vendor, *arguments):
    return subprocess.run(["git", "-C", str(vendor), *arguments], capture_output=True, check=True).stdout


def import_official(vendor, run):
    head = git_bytes(vendor, "rev-parse", "HEAD").decode().strip()
    require(head == PINNED_COMMIT == run["provenance"]["official_commit"], "Official Git commit differs from saved run")
    status = git_bytes(vendor, "status", "--porcelain", "--untracked-files=no").decode().strip()
    require(not status, f"Official tracked checkout is modified: {status}")
    source = vendor / OFFICIAL_SOURCE
    blob = git_bytes(vendor, "show", f"HEAD:{OFFICIAL_SOURCE}")
    source_bytes = source.read_bytes()
    require(source_bytes.replace(b"\r\n", b"\n") == blob, "Official visualizer source differs from pinned Git blob")
    sys.path.insert(0, str(vendor))
    module = importlib.import_module("dust3r.datasets.tapvid3d")
    actual_source = Path(inspect.getsourcefile(module.visualize_results)).resolve()
    require(actual_source == source.resolve(), f"Imported visualizer from unexpected location: {actual_source}")
    return module, {
        "repository": run["provenance"]["official_repository"],
        "commit": head, "source_path": str(source), "tracked_checkout_clean": True,
        "source_sha256": sha256(source), "git_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "source_matches_git_blob_after_crlf_to_lf": True,
        "entry_point": "dust3r.datasets.tapvid3d.visualize_results",
        "source_was_modified": False, "functions_reimplemented": False, "monkeypatch_used": False,
        "wrapper_sha256": sha256(Path(__file__)),
    }


def checked_saved_arrays(directory, run, dataset, sequence):
    record = read_json(directory / "result.json")
    require(record["signature"] == run["signature"], f"Result signature mismatch: {directory}")
    require((record["dataset"], record["sequence"]) == (dataset, sequence), "Result identity mismatch")
    require(record["saved_input_verification"]["passed"], "Saved input model replay did not pass")
    paths = {name: directory / name for name in ("tracks.npz", "rgb_float32.npy")}
    hashes = {}
    for name, path in paths.items():
        hashes[name] = sha256(path)
        require(hashes[name] == record["artifact_sha256"][name], f"Saved artifact checksum mismatch: {path}")
    with np.load(paths["tracks.npz"], allow_pickle=False) as archive:
        tracks = {key: archive[key].copy() for key in archive.files}
    rgb = np.load(paths["rgb_float32.npy"], mmap_mode="r", allow_pickle=False)
    require(rgb.dtype == np.float32 and rgb.ndim == 4 and rgb.shape[1] == 3, "Saved RGB must be float32 [T,3,H,W]")
    require(np.isfinite(rgb).all() and float(rgb.min()) >= 0 and float(rgb.max()) <= 1, "Saved RGB is invalid")
    require(tracks["pred"].dtype == np.float32 and np.isfinite(tracks["pred"]).all(), "Saved predictions are invalid")
    sources = {name: {"path": str(paths[name]), "sha256": digest} for name, digest in hashes.items()}
    sources["result.json"] = {"path": str(directory / "result.json"), "sha256": sha256(directory / "result.json")}
    return record, tracks, rgb, sources


def selected_query_indices(count, limit, seed):
    # Query selection does not access clean/PGD predictions or their errors.
    if limit and count > limit:
        return np.random.RandomState(seed).choice(count, limit, replace=False)
    return np.arange(count, dtype=np.int64)


def official_subset_orders(gt, input_indices, seed):
    """Record the unchanged official function's own random/sorting decisions."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(gt.shape[1], 300, replace=False) if gt.shape[1] > 300 else np.arange(gt.shape[1])
    indices = indices[gt[0, indices, 1].argsort()]
    indices3d = rng.choice(len(indices), 100, replace=False) if len(indices) > 100 else np.arange(len(indices))
    indices3d.sort()
    return input_indices[indices], input_indices[indices[indices3d]]


def probe_video(path, frames):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,codec_name:format=duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)
    stream = probe["streams"][0]
    require(int(stream["nb_read_frames"]) == frames, f"Video frame count mismatch: {path}")
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    require(numerator / denominator == OFFICIAL_FPS, f"Official 15 fps changed: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "probe": probe, "frame_count": frames, "fps": OFFICIAL_FPS}


def join_official_videos(clean, attacked, output, font_path, frames):
    # Only completed MP4 files are combined. No tracks are drawn by this wrapper.
    for char in "':,;[]\\":
        require(char not in str(font_path), "Font path contains unsupported FFmpeg filter punctuation")
    background = "0x141922"
    common = f"pad=iw:ih+64:0:64:color={background}"
    labels = (("0:v", "CLEAN INPUT", "a"), ("1:v", "PGD INPUT (4/255, 20 STEPS)", "b"))
    filters = []
    for source, label, target in labels:
        filters.append(
            f"[{source}]{common},drawtext=fontfile={font_path}:text='{label}':x=14:y=8:"
            "fontsize=22:fontcolor=white,"
            f"drawtext=fontfile={font_path}:text='GT = circle   Prediction = cross':x=14:y=37:"
            f"fontsize=16:fontcolor=0xced7e4[{target}]"
        )
    filters.append("[a][b]hstack=inputs=2[v]")
    process = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(clean), "-i", str(attacked),
        "-filter_complex", ";".join(filters), "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "16",
        "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], capture_output=True, text=True)
    require(process.returncode == 0, f"FFmpeg comparison failed: {process.stderr}")
    return probe_video(output, frames)


def render_clip(module, sequence_dir, project_root, output_root, run, seed, max_points, font_path):
    dataset, sequence = sequence_dir.parent.name, sequence_dir.name
    metadata = read_json(sequence_dir / "sequence.json")
    frame_count = int(metadata["num_frames"])
    indices = np.asarray(metadata["selected_point_indices"], dtype=np.int64)
    entries = [entry for entry in run["manifest"]["entries"] if (entry["dataset"], entry["sequence"]) == (dataset, sequence)]
    require(len(entries) == 1, "Expected one matching saved manifest entry")
    entry = entries[0]
    source = project_root / "data" / "worldtrack_release" / dataset / f"{sequence}.npz"
    require(sha256(source) == entry["sha256"] and source.stat().st_size == entry["bytes"], "Original source NPZ differs from saved manifest")
    # Decode and construct first-camera GT/world-to-camera coordinates using the
    # actual original loader, without copying its normalization implementation.
    originals, _, _, intrinsics, source_gt, _, _, extrinsics = module.load_npz_data(
        str(source), num_frames=frame_count, normalize_cam=True
    )
    require(list(originals[0].size) == metadata["original_wh"], "Original image size differs from saved preprocessing")
    source_gt = source_gt[:, indices].astype(np.float32)
    output_dir = output_root / f"{dataset}__{sequence}"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_indices = selected_query_indices(len(indices), max_points, seed)
    displayed_gt = source_gt[:, query_indices]
    order2d, order3d = official_subset_orders(displayed_gt, query_indices, seed)
    resized = np.asarray(metadata["preprocessed_wh"], dtype=np.float64) / np.asarray(metadata["original_wh"], dtype=np.float64)
    resized_intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    resized_intrinsics *= np.asarray([resized[0], resized[1], resized[0], resized[1]])
    clip = {
        "dataset": dataset, "sequence": sequence, "frame_count": frame_count,
        "run_signature": run["signature"], "official_commit": PINNED_COMMIT,
        "official_source_sha256": sha256(Path(inspect.getsourcefile(module.visualize_results))),
        "source_npz": {"path": str(source), "sha256": entry["sha256"]},
        "source_frame_indices": list(range(frame_count)), "fps": OFFICIAL_FPS,
        "image_size_wh": metadata["preprocessed_wh"], "original_image_size_wh": metadata["original_wh"],
        "intrinsics_used": resized_intrinsics.tolist(),
        "extrinsics_basis": "official load_npz_data(normalize_cam=True), first-camera world",
        "query_selection": {
            "seed": seed, "limit": max_points, "saved_query_count": len(indices),
            "policy": "seeded uniform random subset of saved official queries before the unchanged official renderer; 0 passes all queries to its defaults",
            "selection_uses_model_errors": False, "clean_and_pgd_same_indices": True,
            "passed_saved_query_indices": query_indices.tolist(),
            "official_2d_saved_query_order": order2d.tolist(), "official_3d_saved_query_order": order3d.tolist(),
            "official_2d_source_point_order": indices[order2d].tolist(), "official_3d_source_point_order": indices[order3d].tolist(),
        },
        "conditions": {}, "comparisons": {},
    }
    clean_tracks = None
    clean_rgb = None
    for condition in ("clean", "pgd_eps4"):
        directory = sequence_dir / condition
        record, tracks, rgb, source_hashes = checked_saved_arrays(directory, run, dataset, sequence)
        require(np.array_equal(tracks["gt"], source_gt), "Saved GT does not match the official loader's original GT")
        require(np.array_equal(tracks["intrinsics"], intrinsics), "Saved original intrinsics disagree")
        require(rgb.shape[0] == frame_count and list(rgb.shape[-1:-3:-1]) == metadata["preprocessed_wh"], "Saved RGB size mismatch")
        if clean_tracks is None:
            clean_tracks, clean_rgb = tracks, rgb
            require(record["method"] == "clean", "Unexpected clean method")
        else:
            require(record["method"] == "pgd" and record["epsilon_255"] == 4 and record["steps"] == 20, "Unexpected PGD settings")
            for key in ("gt", "query_xy", "valid", "dynamic", "intrinsics"):
                require(np.array_equal(clean_tracks[key], tracks[key]), f"Clean/PGD query field differs: {key}")
            require(float(np.max(np.abs(rgb - clean_rgb))) <= 4 / 255 + 1e-7, "Saved PGD RGB exceeds attack bound")
        scale = float(record["metrics"]["scale_all"])
        require(np.isfinite(scale) and scale > 0, "Invalid saved official scale")
        aligned_predictions = tracks["pred"] * np.float32(scale)
        errors = np.linalg.norm(aligned_predictions - tracks["gt"], axis=-1)
        require(np.isclose(float(errors.mean()), record["metrics"]["epe_all_m"], rtol=1e-6, atol=1e-6), "Aligned predictions do not match recorded official EPE")
        imgs = [Image.fromarray(np.rint(frame.transpose(1, 2, 0) * 255).astype(np.uint8)) for frame in rgb]
        condition_output = output_dir / condition
        started = time.perf_counter()
        np.random.seed(seed)  # Equal official random decisions for clean/PGD.
        print(f"{dataset}/{sequence}/{condition}: calling official visualize_results ({frame_count} frames, {len(query_indices)} input queries)", flush=True)
        # THIS is the original upstream function, including its original 2D and
        # 3D drawing helpers, traces, sampling/sorting, and mediapy writes.
        module.visualize_results(
            imgs, displayed_gt.copy(), aligned_predictions[:, query_indices].copy(),
            extrinsics.copy(), resized_intrinsics.copy(), save_path=str(condition_output),
        )
        import matplotlib.pyplot as plt
        plt.close("all")  # Release figures retained by the official 3D helper.
        elapsed = time.perf_counter() - started
        clip["conditions"][condition] = {
            "function_called": "dust3r.datasets.tapvid3d.visualize_results", "completed": True,
            "elapsed_seconds": elapsed, "sources": source_hashes, "metrics": record["metrics"],
            "prediction_alignment": "saved per-condition official all-query scale_all; no pose fit", "scale_all": scale,
            "rgb_display_conversion": "round(float32 saved RGB * 255) to uint8; compressed preview",
            "videos": {kind: probe_video(condition_output / f"{kind}_tracks.mp4", frame_count) for kind in ("2d", "3d")},
        }
        print(f"{dataset}/{sequence}/{condition}: official videos saved in {elapsed:.1f}s", flush=True)
        write_json(output_dir / "manifest.json", clip)
    for kind in ("2d", "3d"):
        comparison = output_root / f"{dataset}__{sequence}__official_{kind}_comparison.mp4"
        clip["comparisons"][kind] = join_official_videos(
            output_dir / "clean" / f"{kind}_tracks.mp4", output_dir / "pgd_eps4" / f"{kind}_tracks.mp4",
            comparison, font_path, frame_count,
        )
    write_json(output_dir / "manifest.json", clip)
    return clip


def resume_completed_clip(sequence_dir, output_root, run, official, seed, max_points):
    """Skip only complete clips whose input/result/video checksums still match."""
    dataset, sequence = sequence_dir.parent.name, sequence_dir.name
    path = output_root / f"{dataset}__{sequence}" / "manifest.json"
    if not path.is_file():
        return None
    clip = read_json(path)
    if set(clip.get("conditions", {})) != {"clean", "pgd_eps4"} or set(clip.get("comparisons", {})) != {"2d", "3d"}:
        return None
    require(clip["run_signature"] == run["signature"], "Resume run signature mismatch")
    require(clip["official_commit"] == official["commit"] and clip["official_source_sha256"] == official["source_sha256"], "Resume official renderer differs")
    require(clip["query_selection"]["seed"] == seed and clip["query_selection"]["limit"] == max_points, "Resume sampling settings differ")
    require(clip["dataset"] == dataset and clip["sequence"] == sequence, "Resume clip identity differs")
    for condition, record in clip["conditions"].items():
        require(record["completed"] and record["function_called"] == "dust3r.datasets.tapvid3d.visualize_results", "Resume clip lacks completed official call")
        for filename, evidence in record["sources"].items():
            source = sequence_dir / condition / filename
            require(sha256(source) == evidence["sha256"], f"Resume source changed: {source}")
        result = read_json(sequence_dir / condition / "result.json")
        require(result["signature"] == run["signature"] and result["saved_input_verification"]["passed"], "Resume input replay evidence differs")
        for kind, evidence in record["videos"].items():
            video = output_root / f"{dataset}__{sequence}" / condition / f"{kind}_tracks.mp4"
            require(sha256(video) == evidence["sha256"], f"Resume official video changed: {video}")
            probe_video(video, clip["frame_count"])
    for kind, evidence in clip["comparisons"].items():
        video = output_root / f"{dataset}__{sequence}__official_{kind}_comparison.mp4"
        require(sha256(video) == evidence["sha256"], f"Resume comparison changed: {video}")
        probe_video(video, clip["frame_count"])
    clip["resume_verification_passed"] = True
    return clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--vendor-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=32, help="Queries passed to official renderer; 0 uses its original 300/100 limits")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--font-path", type=Path, default=None, help="Default: Matplotlib's bundled DejaVuSans.ttf")
    parser.add_argument("--only-sequence", default=None, help="Optional dataset/sequence for a bounded render")
    parser.add_argument("--resume", action="store_true", help="Verify and skip already completed official clips")
    args = parser.parse_args()
    require(args.max_points >= 0, "--max-points must be nonnegative")
    if args.font_path is None:
        import matplotlib
        args.font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    require(args.font_path.is_file(), f"Font missing: {args.font_path}")
    root, run_dir = args.project_root.resolve(), args.run_dir.resolve()
    vendor = (args.vendor_root or root / "vendor" / "St4RTrack").resolve()
    output = (args.output_dir or run_dir / "official_visualizations").resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = read_json(run_dir / "run.json")
    module, official = import_official(vendor, run)
    manifest = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_signature": run["signature"], "run_dir": str(run_dir), "cpu_only": True,
        "new_model_inference": False, "official_renderer": official,
        "legend": "Official per-query matching colors: GT circles/solid traces, predictions crosses/dotted 3D traces",
        "comparison_wrapper": "joins completed official MP4 files with labels only; no plotting or coordinate changes",
        "fps": OFFICIAL_FPS, "fps_is_original_capture_rate": False, "clips": [],
    }
    report_path = output / "manifest.json"
    write_json(report_path, manifest)
    started = time.perf_counter()
    sequence_dirs = sorted(path.parent.parent for path in (run_dir / "sequences").glob("*/*/pgd_eps4/result.json"))
    if args.only_sequence:
        sequence_dirs = [path for path in sequence_dirs if f"{path.parent.name}/{path.name}" == args.only_sequence]
        require(len(sequence_dirs) == 1, "--only-sequence did not identify one completed result")
    else:
        require(len(sequence_dirs) == 4, "Expected all four completed clean/PGD sequences")
    try:
        for index, directory in enumerate(sequence_dirs, 1):
            print(f"[{index}/{len(sequence_dirs)}] {directory.parent.name}/{directory.name}", flush=True)
            clip = resume_completed_clip(directory, output, run, official, args.seed, args.max_points) if args.resume else None
            if clip is None:
                clip = render_clip(module, directory, root, output, run, args.seed, args.max_points, args.font_path)
            else:
                source = Path(clip["source_npz"]["path"])
                require(sha256(source) == clip["source_npz"]["sha256"], "Resume original NPZ changed")
                print("Completed official clip verified; skipping its rendering.", flush=True)
            manifest["clips"].append(clip)
            manifest["elapsed_seconds"] = time.perf_counter() - started
            write_json(report_path, manifest)
        # Confirm that importing/calling the upstream visualizer never changed
        # any tracked source file.
        _, final_source = import_official(vendor, run)
        require(final_source["source_sha256"] == official["source_sha256"], "Official source changed while rendering")
        manifest["status"] = "completed"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["official_source_unchanged_after_rendering"] = True
        write_json(report_path, manifest)
        print(json.dumps({"status": "completed", "clips": len(manifest["clips"]), "elapsed_seconds": manifest["elapsed_seconds"], "manifest": str(report_path)}), flush=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(report_path, manifest)
        raise


if __name__ == "__main__":
    main()
