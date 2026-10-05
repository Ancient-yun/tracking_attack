"""Plain four-loss, all-eight-clip saved-result videos, using CPU only.

CPU only: no model, torch, checkpoint loading, inference, or experiment edits.
All 64 videos contain all 128 frames, without any text, charts, or heatmaps.
Dense metrics use every fixed GT-valid
pixel; stride-eight samples are used only to display the 3D point clouds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from component_projection import ComponentContext, THRESHOLDS, metric_check, require, sha256
from component_report_data import load_verified_study
from render_component_comparisons import CPUEncoder, write_json
from render_pgd_comparisons import to_rgb8
from render_tracking_comparisons import select_queries, clipped_segment, mask_for, GT, CLEAN, ATTACK, BG

OBJECTIVES = ("tracking_3d", "reconstruction_3d", "confidence", "joint_training")
FRAMES, FPS = 128, 10.0
VIDEO_SIZE = {"tracking": (1024, 288), "reconstruction": (1600, 600)}
POSTERS = (0, 64, 127)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def report_inventory(directory):
    """Bind every file of the original report without changing its contents."""
    return {p.relative_to(directory).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()}


def asset(path, output):
    return {"path": path.relative_to(output).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def frame_error(source, condition, t):
    pred = condition["arrays"]["reconstruction_float32.npy"][t]
    scale = np.float32(condition["record"]["reconstruction_metrics"]["scale"])
    require(np.isfinite(pred).all(), "Nonfinite dense saved prediction")
    return np.linalg.norm(pred * scale - source["dense_gt"][t], axis=-1)


def prepare_reconstruction(source, conditions):
    """One frame at a time: full metrics, GT-only display subset, shared bounds."""
    height, width = source["dense_valid"].shape[1:]
    rows, cols = np.meshgrid(np.arange(0, height, 8), np.arange(0, width, 8), indexing="ij")
    raster = (rows * width + cols).ravel().astype(np.int64)
    gt_min, gt_max = np.full(3, np.inf), np.full(3, -np.inf)
    all_min, all_max = np.full(3, np.inf), np.full(3, -np.inf)
    selected_counts, mask_hash = [], hashlib.sha256()
    maximum = 0.0
    metrics = {}
    for objective, condition in conditions.items():
        count, summed = 0, 0.0
        within = np.zeros(4, dtype=np.int64)
        per_frame = []
        for t in range(FRAMES):
            valid = np.asarray(source["dense_valid"][t], dtype=bool)
            error = frame_error(source, condition, t)
            values = error[valid]
            require(values.size and np.isfinite(values).all(), "Empty/nonfinite fixed GT support")
            frame_within = np.asarray([np.count_nonzero(values <= q) for q in THRESHOLDS])
            count += int(values.size)
            summed += float(values.sum(dtype=np.float64))
            within += frame_within
            per_frame.append({"source_frame_index": t, "valid_pixels": int(values.size),
                              "epe_m": float(values.mean()),
                              "apd3d": float((frame_within / values.size).mean() * 100)})
            maximum = max(maximum, float(values.max()))
            selected = raster[valid.ravel()[raster]]
            require(selected.size, "Empty GT stride-eight display sample")
            dense = condition["arrays"]["reconstruction_float32.npy"][t].reshape(-1, 3)
            points = dense[selected] * np.float32(condition["record"]["reconstruction_metrics"]["scale"])
            require(np.isfinite(points).all(), "Nonfinite displayed point cloud")
            all_min = np.minimum(all_min, points.min(axis=0))
            all_max = np.maximum(all_max, points.max(axis=0))
            if objective == "clean":
                gt = source["dense_gt"][t].reshape(-1, 3)[selected]
                require(np.isfinite(gt).all(), "Nonfinite displayed GT")
                gt_min = np.minimum(gt_min, gt.min(axis=0))
                gt_max = np.maximum(gt_max, gt.max(axis=0))
                all_min = np.minimum(all_min, gt.min(axis=0))
                all_max = np.maximum(all_max, gt.max(axis=0))
                selected_counts.append(int(selected.size))
                mask_hash.update(valid.ravel()[raster].tobytes())
        global_metrics = {"epe_m": summed / count, "apd3d": float((within / count).mean() * 100),
                          "valid_pixels": count, "per_frame": per_frame, "passed": True}
        recorded = condition["record"]["reconstruction_metrics"]
        metric_check(global_metrics["epe_m"], recorded["epe_m"], objective + " dense EPE")
        metric_check(global_metrics["apd3d"], recorded["apd3d"], objective + " dense APD")
        require(count == recorded["valid_pixels"], "Fixed GT valid-pixel count differs")
        global_metrics["recorded"] = {"epe_m": recorded["epe_m"], "apd3d": recorded["apd3d"], "valid_pixels": recorded["valid_pixels"]}
        global_metrics["absolute_difference"] = {"epe_m": abs(global_metrics["epe_m"] - recorded["epe_m"]),
                                                   "apd3d": abs(global_metrics["apd3d"] - recorded["apd3d"])}
        metrics[objective] = global_metrics
        print(f"  verified dense metrics {objective}: APD={global_metrics['apd3d']:.6f}% EPE={global_metrics['epe_m']:.6f}m", flush=True)
    center = (gt_min + gt_max) / 2
    extent = max(float(np.max(np.abs(np.concatenate((all_min - center, all_max - center))))), 1e-6) * 1.05
    detail_extent = max(float(np.max(np.abs(np.concatenate((gt_min - center, gt_max - center))))), 1e-6) * 1.05
    outside_counts = {"GT": [0] * FRAMES}
    for objective, condition in conditions.items():
        counts = []
        for t in range(FRAMES):
            valid = np.asarray(source["dense_valid"][t], dtype=bool)
            selected = raster[valid.ravel()[raster]]
            points = condition["arrays"]["reconstruction_float32.npy"][t].reshape(-1, 3)[selected] * np.float32(condition["record"]["reconstruction_metrics"]["scale"])
            counts.append(int(np.count_nonzero(np.any(np.abs(points - center) > detail_extent, axis=1))))
        outside_counts[objective] = counts
    metadata = {"raster_stride": 8, "raster_start_yx": [0, 0], "raster_flat_indices": raster.tolist(),
                "selection_uses_model_errors": False, "selection_uses_confidence": False,
                "selection_policy": "Same frame-specific fixed GT-valid raster-stride-eight pixels in all three clouds and all four attacks",
                "displayed_points_per_frame": selected_counts,
                "gt_valid_display_mask_sha256": mask_hash.hexdigest(),
                "gt_all_frames_bbox_min_m": gt_min.tolist(), "gt_all_frames_bbox_max_m": gt_max.tolist(),
                "common_center_translation_m": center.tolist(), "common_center_policy": "GT display-pixel bbox center over all 128 frames; identical subtraction for every condition",
                "all_conditions_all_frames_bbox_min_m": all_min.tolist(), "all_conditions_all_frames_bbox_max_m": all_max.tolist(),
                "axis_limits_centered_m": [-extent, extent], "bounds_margin_fraction": 0.05,
                "local_detail_axis_limits_centered_m": [-detail_extent, detail_extent],
                "local_detail_bounds_policy": "Common GT display-pixel bounding extent across all 128 frames, with 5% margin; same centered cube in every condition and objective",
                "global_inset_axis_limits_centered_m": [-extent, extent],
                "display_outside_count": outside_counts,
                "display_outside_count_population": "Frame-specific GT-valid stride-eight displayed pixels; any coordinate outside the shared GT-detail axis limits",
                "point_coordinates_clamped": False, "global_inset_clipping": False,
                "clipping": False, "all_selected_points_retained_in_global_inset": True,
                "main_detail_crops_to_GT_bounds": True,
                "main_detail_crop_policy": "Show the subset inside the GT-detail window; every outside point remains unmodified in the global inset; per-frame outside counts are preserved in manifest data for external HTML captions only",
                "bounds_policy": "Maximum absolute centered coordinate over GT, clean and all four attacks' displayed pixels across all 128 frames; no outlier hiding or clipping",
                "view": {"elevation_degrees": 18, "azimuth_degrees": -70, "projection": "orthographic", "static": True},
                "coordinate_basis": "Saved first-camera world XYZ in metres after recorded float32 metric scale; one shared GT-center display translation",
                "full_valid_residual_max_m": maximum,
                "heatmaps_in_video": False, "text_overlay": False,
                "fixed_GT_mask_for_metrics": True,
                "metric_population": "Every GT-valid pixel across all 128 frames; displayed cloud subset never changes metrics"}
    return raster, center, extent, metadata, metrics


def plain_tracking_panel(rgb_frame, projected, kind, t, selected):
    """Saved RGB with only fixed-GT query dots and eight-frame trails."""
    panel = Image.fromarray(to_rgb8(rgb_frame))
    require(panel.size == (512, 288), "Tracking input must remain the saved 512x288 raster")
    draw = ImageDraw.Draw(panel)
    gt_xy, pred_xy = np.asarray(projected["gt_xy"]), np.asarray(projected[kind + "_xy"])
    gt_valid = np.asarray(projected["projection_valid"]["gt"], dtype=bool)
    pred_valid = np.asarray(projected["projection_valid"][kind], dtype=bool)
    visibility = np.asarray(projected["visibility"], dtype=bool)
    gt_inside = mask_for(projected, "in_frame", "gt", t, selected)
    pred_inside = mask_for(projected, "in_frame", kind, t, selected)
    behind = mask_for(projected, "behind_camera", kind, t, selected)
    invalid = ~np.isfinite(pred_xy[t, selected]).all(axis=-1) & ~behind
    out = pred_valid[t, selected] & ~pred_inside
    color = CLEAN if kind == "clean" else ATTACK

    def segment(start, end, shade):
        clipped = clipped_segment(start, end, width=512, height=288)
        if clipped is not None:
            draw.line(clipped, fill=shade, width=1)

    for slot, query in enumerate(selected):
        for previous in range(max(0, t - 7), t):
            if gt_valid[previous, query] and gt_valid[previous + 1, query]:
                shade = tuple(int(c * .65) for c in GT)
                segment(gt_xy[previous, query], gt_xy[previous + 1, query], shade)
            if pred_valid[previous, query] and pred_valid[previous + 1, query]:
                shade = tuple(int(c * .75) for c in color)
                segment(pred_xy[previous, query], pred_xy[previous + 1, query], shade)
        if gt_inside[slot]:
            x, y = map(float, gt_xy[t, query])
            # GT occlusion is retained, indicated only by darker green dots.
            shade = GT if visibility[t, query] else tuple(int(c * .5) for c in GT)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=shade)
        if pred_inside[slot]:
            x, y = map(float, pred_xy[t, query])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=1)
    return panel, {"gt_visible": int(visibility[t, selected].sum()),
                   "gt_occluded": int((~visibility[t, selected]).sum()),
                   "pred_in_frame": int(pred_inside.sum()), "pred_out_of_frame": int(out.sum()),
                   "pred_behind_camera": int(behind.sum()), "pred_nonfinite": int(invalid.sum())}


def tracking_image(clean_rgb, attack_rgb, projected, selected, t):
    clean, clean_status = plain_tracking_panel(clean_rgb, projected, "clean", t, selected)
    attack, attack_status = plain_tracking_panel(attack_rgb, projected, "attack", t, selected)
    canvas = Image.new("RGB", VIDEO_SIZE["tracking"])
    canvas.paste(clean, (0, 0))
    canvas.paste(attack, (512, 0))
    return canvas, {"clean": clean_status, "attack": attack_status}


class ReconstructionCanvas:
    """Pure RGB-colored clouds; captions and numerical disclosures live in HTML."""
    def __init__(self, source, conditions, objective, raster, center, extent, metadata):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        self.plt, self.source, self.conditions, self.objective = plt, source, conditions, objective
        self.raster, self.center, self.extent, self.metadata = raster, center, extent, metadata
        self.fig = plt.figure(figsize=(16, 6), dpi=100, facecolor="#101620")
        self.clouds, self.global_clouds = [], []
        detail_extent = metadata["local_detail_axis_limits_centered_m"][1]
        for i in range(3):
            x = i / 3
            ax = self.fig.add_axes([x + .006, .008, .321, .984], projection="3d", facecolor="#101620")
            ax.set_proj_type("ortho")
            ax.view_init(elev=18, azim=-70)
            ax.set(xlim=(-detail_extent, detail_extent), ylim=(-detail_extent, detail_extent), zlim=(-detail_extent, detail_extent))
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            self.clouds.append(ax.scatter([], [], [], s=5.0, depthshade=False, marker=".", rasterized=True))
            inset_rect = [x + .228, .022, .096, .27]
            inset = self.fig.add_axes(inset_rect, projection="3d", facecolor="#101620")
            inset.set_proj_type("ortho")
            inset.view_init(elev=18, azim=-70)
            inset.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
            inset.set_box_aspect((1, 1, 1))
            inset.set_axis_off()
            self.global_clouds.append(inset.scatter([], [], [], s=1.0, depthshade=False, marker=".", rasterized=True))
            self.fig.add_artist(Rectangle((inset_rect[0], inset_rect[1]), inset_rect[2], inset_rect[3],
                                          transform=self.fig.transFigure, fill=False, edgecolor="#354050", linewidth=.7))
        for x in (1 / 3, 2 / 3):
            self.fig.add_artist(Rectangle((x - .0004, 0), .0008, 1, transform=self.fig.transFigure,
                                          facecolor="#263140", edgecolor="none"))
        # No Text artists are created explicitly, and every 3D axis is hidden.
        require(not self.fig.texts, "Plain reconstruction canvas has figure text")

    def frame(self, t):
        source, conds = self.source, self.conditions
        valid = np.asarray(source["dense_valid"][t], dtype=bool)
        selected = self.raster[valid.ravel()[self.raster]]
        rgb = conds["clean"]["arrays"]["rgb_float32.npy"][t]
        colors = rgb.transpose(1, 2, 0).reshape(-1, 3)[selected]
        clouds = [source["dense_gt"][t].reshape(-1, 3)[selected]]
        for name in ("clean", self.objective):
            c = conds[name]
            clouds.append(c["arrays"]["reconstruction_float32.npy"][t].reshape(-1, 3)[selected]
                          * np.float32(c["record"]["reconstruction_metrics"]["scale"]))
        detail_extent = self.metadata["local_detail_axis_limits_centered_m"][1]
        for collection, global_collection, points in zip(self.clouds, self.global_clouds, clouds):
            p = points - self.center
            inside = np.all(np.abs(p) <= detail_extent, axis=1)
            collection._offsets3d = (p[inside, 0], p[inside, 1], p[inside, 2])
            collection.set_facecolors(colors[inside])
            collection.set_edgecolors(colors[inside])
            global_collection._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
            global_collection.set_facecolors(colors)
            global_collection.set_edgecolors(colors)
        self.fig.canvas.draw()
        frame = Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy(), "RGB")
        require(frame.size == VIDEO_SIZE["reconstruction"], "Unexpected plain point-cloud canvas size")
        return frame

    def close(self):
        self.plt.close(self.fig)


def clip_descriptors(metadata):
    counters = {"po_mini": 0, "ds_mini": 0}
    descriptions = []
    for index, entry in enumerate(metadata["manifest"]["entries"]):
        dataset, sequence = entry["dataset"], entry["sequence"]
        require(dataset in counters, "Unexpected source dataset")
        counters[dataset] += 1
        group = "PO" if dataset == "po_mini" else "DR"
        prefix = group.lower() + f"{counters[dataset]:02d}"
        descriptions.append({"id": prefix, "prefix": prefix, "dataset": dataset, "sequence": sequence,
            "display_group": group, "clip_index": counters[dataset], "global_clip_index": index + 1,
            "source_manifest_index": index, "source_frame_indices": list(range(FRAMES)),
            "source_sequence_npz_sha256": entry["sha256"],
            "source_frame_identity": "Saved full raw frame order 0..127, with no temporal resampling"})
    require(len(descriptions) == 8 and counters == {"po_mini": 4, "ds_mini": 4}, "Expected original balanced eight-clip manifest")
    return descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-only", action="store_true", help="Frame-zero PNGs and verified metrics; never mark 64-video rendering complete")
    parser.add_argument("--clip-limit", type=int, choices=range(1, 9), help="Preview only: first N clips in original manifest order")
    args = parser.parse_args()
    require(args.preview_only or args.clip_limit is None, "--clip-limit is restricted to previews; complete output requires all eight clips")
    started = time.perf_counter()
    run, project, output = args.run_dir.resolve(), args.project_root.resolve(), args.output_dir.resolve()
    require(output.is_relative_to(run / "presentation" / ".build"), "New media must remain inside presentation/.build")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "task_visualization_manifest.json"
    require(not manifest_path.exists(), "Choose a fresh output directory; existing media is not silently overwritten")
    print("Strict original completed-study scalar gate and original-report hash inventory", flush=True)
    study = load_verified_study(run, require_complete=True)
    frozen_before = report_inventory(run / "html_report")
    source_names = (Path(__file__).name, "component_projection.py", "component_report_data.py",
                    "render_component_comparisons.py", "render_tracking_comparisons.py", "render_pgd_comparisons.py", "tracking_projection.py")
    source_sha = {n: sha256(Path(__file__).parent / n) for n in source_names}
    context = ComponentContext(run, project, metadata=study["metadata"], records=study["records"])
    clip_order = clip_descriptors(study["metadata"])
    selected_clips = clip_order[:args.clip_limit] if args.clip_limit else clip_order
    encoder = None if args.preview_only else CPUEncoder()
    manifest = {"schema_version": 2, "status": "rendering", "started_at_utc": now(), "run_id": run.name,
        "run_signature": study["metadata"]["signature"], "analysis_sha256": study["analysis_sha256"],
        "cpu_only": True, "new_model_inference": False, "GPU_used": False,
        "source_code_sha256": source_sha, "input_scalar_sha256": study["input_sha256"],
        "objectives": list(OBJECTIVES), "clip_order": clip_order, "rendered_clip_order": selected_clips,
        "plain_video": True, "text_overlay": False, "tracking_query_ids_drawn": False,
        "heatmaps_in_video": False, "charts_in_video": False, "legends_in_video": False,
        "numbers_overlay": False, "axis_labels_in_video": False, "ticklabels_in_video": False,
        "frame_count": FRAMES, "fps": FPS, "source_frame_indices": list(range(FRAMES)),
        "video_size_wh": {k: list(v) for k, v in VIDEO_SIZE.items()},
        "preview_fps_is_capture_fps": False, "rgb_difference_panels": False,
        "tracking_panel_order": ["clean_RGB_with_GT_and_clean_prediction", "attack_RGB_with_GT_and_attack_prediction"],
        "reconstruction_panel_order": ["GT", "Clean", "Attack"],
        "encoder_policy": {"cpu_only": True, "threads": 2, "codec": "libx264", "crf": 16},
        "scope": {"clips": len(selected_clips), "full_campaign_clips": 8, "attacks_per_clip": 4,
                  "tasks_per_attack": 2, "expected_full_media": 64, "preview_only": args.preview_only,
                  "cases": len(selected_clips) * 4, "comparison_groups": len(selected_clips) * 2,
                  "original_completed_conditions": 48, "focus_conditions_with_clean": 40,
                  "focus_attack_conditions": 32, "rendered_focus_conditions_with_clean": len(selected_clips) * 5,
                  "excluded_presentation_objectives": ["tracking_mse"]},
        "media": {}, "preview_media": {}, "sequence_common": {}, "errors": []}
    write_json(manifest_path, manifest)
    try:
        for clip in selected_clips:
            prefix, dataset, sequence = clip["prefix"], clip["dataset"], clip["sequence"]
            print(f"Preparing {prefix}: {dataset}/{sequence}: saved arrays and fixed inputs", flush=True)
            source = context.source(dataset, sequence)
            conditions = {o: context.condition(dataset, sequence, o) for o in ("clean", *OBJECTIVES)}
            pairs = {o: context.pair(dataset, sequence, o) for o in OBJECTIVES}
            selected, selection = select_queries(pairs[OBJECTIVES[0]])
            require(len(selected) == 18, "Requested fixed eighteen GT-selected tracking queries")
            for objective in OBJECTIVES[1:]:
                other_selected, other_selection = select_queries(pairs[objective])
                require(np.array_equal(selected, other_selected) and selection == other_selection, "GT query choices differ across attack objectives")
            raster, center, extent, common, dense_checks = prepare_reconstruction(source, conditions)
            common["comparison_objectives"] = list(OBJECTIVES)
            common["plain_video"] = True
            common["display_marker_size_points_squared"] = {"main": 5.0, "global_inset": 1.0}
            common["cloud_rgb_policy"] = "Same saved clean RGB at identical GT-valid raster pixels for all clouds and objectives"
            selection["query_ids_drawn_in_video"] = False
            tracking_metadata = {"tracking_selection": selection, "trailing_positions": 8,
                "comparison_objectives": list(OBJECTIVES),
                "projection_camera": "Fixed source GT per-frame camera; no predicted camera or new 2D inference",
                "prediction_scale": "Recorded evaluation float32 metric scale applied once before GT-camera projection",
                "point_coordinates_clamped": False, "crossing_segments_clipped_to_viewport": True,
                "panel_size_wh": [512, 288], "text_overlay": False, "query_ids_drawn": False,
                "colors_rgb": {"GT": list(GT), "clean_prediction": list(CLEAN), "attack_prediction": list(ATTACK)},
                "GT_occlusion": "All fixed GT support retained; occluded GT dots are darker green"}
            manifest["sequence_common"][prefix] = {"dataset": dataset, "sequence": sequence,
                "clip_descriptor": clip, "tracking_selection": selection,
                "tracking": tracking_metadata, "reconstruction": common,
                "reconstruction_metric_reproduction": dense_checks,
                "source_frame_identity": {"source_frame_indices": list(range(FRAMES)),
                    "sequence_metadata_sha256": source["evidence"]["sequence.json"]["sha256"],
                    "source_npz_sha256": source["evidence"]["source_npz"]["sha256"]}}
            write_json(manifest_path, manifest)
            for objective in OBJECTIVES:
                clean, attack = conditions["clean"], conditions[objective]
                for task in ("tracking", "reconstruction"):
                    media_id = f"{prefix}_{objective}_{task}"
                    directory = output / media_id
                    directory.mkdir(exist_ok=True)
                    path = directory / "comparison.mp4"
                    canvas = None if task == "tracking" else ReconstructionCanvas(source, conditions, objective, raster, center, extent, common)
                    frame_statuses, poster_records = [], {}
                    first_render_seconds = None
                    if args.preview_only:
                        tick = time.perf_counter()
                        if task == "tracking":
                            frame, _ = tracking_image(clean["arrays"]["rgb_float32.npy"][0], attack["arrays"]["rgb_float32.npy"][0], pairs[objective], selected, 0)
                        else:
                            frame = canvas.frame(0)
                        seconds = time.perf_counter() - tick
                        png = directory / "frame_000.png"
                        frame.save(png)
                        manifest["preview_media"][media_id] = {"id": media_id, "dataset": dataset,
                            "sequence": sequence, "objective": objective, "task": task, **asset(png, output),
                            "first_frame_render_seconds": seconds, "plain_video": True, "text_overlay": False,
                            "selected_state": attack["record"]["selected_state"]}
                        if canvas is not None:
                            canvas.close()
                        print(f"  PREVIEW {media_id}: {frame.width}x{frame.height}, {seconds:.3f}s/frame", flush=True)
                        write_json(manifest_path, manifest)
                        continue
                    width, height = VIDEO_SIZE[task]
                    def frames():
                        nonlocal first_render_seconds
                        for t in range(FRAMES):
                            tick = time.perf_counter()
                            if task == "tracking":
                                frame, status = tracking_image(clean["arrays"]["rgb_float32.npy"][t],
                                    attack["arrays"]["rgb_float32.npy"][t], pairs[objective], selected, t)
                                frame_statuses.append({"frame": t, **status})
                            else:
                                frame = canvas.frame(t)
                            if t == 0:
                                first_render_seconds = time.perf_counter() - tick
                                print(f"  {media_id} first-frame={first_render_seconds:.3f}s", flush=True)
                            if t in POSTERS:
                                png = directory / f"frame_{t:03d}.png"
                                frame.save(png)
                                poster_records[str(t)] = asset(png, output)
                            if t % 64 == 0:
                                print(f"  {media_id} frame {t}/127", flush=True)
                            yield frame
                    try:
                        probe = encoder.encode(path, frames(), width, height, FPS, FRAMES)
                    finally:
                        if canvas is not None:
                            canvas.close()
                    require(set(poster_records) == {str(t) for t in POSTERS}, "Missing source-frame posters")
                    source_evidence = {"sequence": source["evidence"], "clean": clean["evidence"], "attack": attack["evidence"]}
                    manifest["media"][media_id] = {"id": media_id, "dataset": dataset, "sequence": sequence,
                        "prefix": prefix, "display_group": clip["display_group"], "clip_index": clip["clip_index"],
                        "objective": objective, "task": task, **asset(path, output), "probe": probe,
                        "posters": poster_records, "source_frame_indices": list(range(FRAMES)),
                        "source_frame_identity": manifest["sequence_common"][prefix]["source_frame_identity"],
                        "selected_state": attack["record"]["selected_state"], "clean_selected_state": clean["record"]["selected_state"],
                        "case_metrics": {"clean": clean["record"][task + "_metrics"], "attack": attack["record"][task + "_metrics"]},
                        "source_evidence": source_evidence, "source_evidence_sha256": canonical_sha(source_evidence),
                        "metric_reproduction": {"clean": clean["checks"], "attack": attack["checks"]} if task == "tracking" else {"clean": dense_checks["clean"], "attack": dense_checks[objective]},
                        "common_display_metadata": tracking_metadata if task == "tracking" else common,
                        "display_outside_count": None if task == "tracking" else {"GT": common["display_outside_count"]["GT"],
                            "clean": common["display_outside_count"]["clean"], "attack": common["display_outside_count"][objective]},
                        "cloud_rgb_policy": common["cloud_rgb_policy"] if task == "reconstruction" else None,
                        "plain_video": True, "text_overlay": False, "heatmaps_in_video": False,
                        "tracking_query_ids_drawn": False, "numbers_overlay": False,
                        "charts_in_video": False, "legends_in_video": False,
                        "axis_labels_in_video": False, "ticklabels_in_video": False,
                        "display_status_per_frame": frame_statuses,
                        "first_frame_render_seconds": first_render_seconds}
                    write_json(manifest_path, manifest)
                    print(f"Completed {media_id}: decoded128, {width}x{height},10fps,12.8s [{len(manifest['media'])}/64]", flush=True)
            context.conditions.clear()
            context.sources.clear()
        expected = {f"{c['prefix']}_{o}_{task}" for c in selected_clips for o in OBJECTIVES for task in ("tracking", "reconstruction")}
        require(set(manifest["preview_media"] if args.preview_only else manifest["media"]) == expected, "Media IDs differ from the selected canonical clip/objective/task product")
        if not args.preview_only:
            require(len(manifest["media"]) == 64, "Complete output requires all 64 videos")
            for record in manifest["media"].values():
                verified = encoder.probe(output / record["path"], FRAMES, FPS)
                require((verified["width"], verified["height"]) == VIDEO_SIZE[record["task"]], "Final decoded resolution differs")
                require(sha256(output / record["path"]) == record["sha256"], "Completed MP4 bytes changed")
        frozen_after = report_inventory(run / "html_report")
        require(frozen_before == frozen_after, "Original detailed HTML report changed during rendering")
        require(source_sha == {n: sha256(Path(__file__).parent / n) for n in source_names}, "Renderer/import code changed during rendering")
        for (name, size, mtime), digest in context.hashes.items():
            stat = Path(name).stat()
            require((stat.st_size, stat.st_mtime_ns) == (size, mtime), "Previously hashed source file changed during rendering: " + name)
        require(study["input_sha256"] == load_verified_study(run, require_complete=True)["input_sha256"], "Original verified scalar inputs changed during rendering")
        manifest["source_freeze_checks"] = {"passed": True, "renderer_and_imports_unchanged": True,
            "verified_scalar_inputs_unchanged": True, "hashed_source_files_metadata_unchanged": True,
            "hashed_source_files_count": len(context.hashes), "source_evidence_bundle_sha256": canonical_sha({k: v["source_evidence"] for k, v in manifest["media"].items()})}
        manifest["original_report_preservation"] = {"passed": True, "files": len(frozen_before),
            "bytes": sum(v["bytes"] for v in frozen_before.values()), "before_inventory_sha256": canonical_sha(frozen_before),
            "after_inventory_sha256": canonical_sha(frozen_after), "key_files": {n: frozen_before[n] for n in ("index.html", "report_manifest.json", "data/report_data.json", "data/task_coupling_analysis.json")}}
        manifest.update(status="preview_complete" if args.preview_only else "complete", completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status="failed", failed_at_utc=now(), errors=[type(exc).__name__ + ": " + str(exc)], elapsed_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        raise
    print(json.dumps({"status": manifest["status"], "videos": len(manifest["media"]), "manifest": str(manifest_path), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
