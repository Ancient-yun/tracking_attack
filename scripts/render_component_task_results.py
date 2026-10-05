"""Fresh tracking and RGB-colored reconstruction previews of saved PGD results.

CPU only: no model, torch, checkpoint loading, inference, or experiment edits.
All eight videos contain all 128 frames. Dense metrics use every fixed GT-valid
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
from PIL import Image, ImageDraw, ImageFont

from component_projection import ComponentContext, THRESHOLDS, metric_check, require, sha256
from component_report_data import load_verified_study
from render_component_comparisons import CPUEncoder, tracking_frame, write_json
from render_pgd_comparisons import fonts
from render_tracking_comparisons import select_queries, BG, FG, MUTED, CLEAN, ATTACK

OBJECTIVES = ("tracking_3d", "reconstruction_3d")
CLIPS = (("po", "po_mini", "cab_e_3rd_13"),
         ("dr", "ds_mini", "9c43b3-3_obj_source_left_3"))
FRAMES, FPS, WIDTH, HEIGHT = 128, 10.0, 1600, 900
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
                "selection_policy": "Same frame-specific fixed GT-valid raster-stride-eight pixels in all three clouds and both attacks",
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
                "main_detail_crop_policy": "Show the subset inside the explicitly stated GT-detail window; every outside point remains unmodified in the global inset and is counted in n/N",
                "bounds_policy": "Maximum absolute centered coordinate over GT, clean and both attacks' displayed pixels across all 128 frames; no outlier hiding or clipping",
                "view": {"elevation_degrees": 18, "azimuth_degrees": -70, "projection": "orthographic", "static": True},
                "coordinate_basis": "Saved first-camera world XYZ in metres after recorded float32 metric scale; one shared GT-center display translation",
                "heatmap_vmin_m": 0.0, "heatmap_vmax_m": max(maximum, 1e-9),
                "heatmap_limit_policy": "Full fixed-GT-valid residual maximum over clean and both geometry attacks, all 128 frames; no percentile clipping",
                "heatmap_normalization": {"type": "SymLogNorm", "linthresh_m": 0.1, "linscale": 1.0, "base": 10, "vmin_m": 0.0, "vmax_m": max(maximum, 1e-9)},
                "heatmap_invalid_color": "grey", "fixed_GT_mask_for_metrics": True,
                "metric_population": "Every GT-valid pixel across all 128 frames; displayed cloud subset never changes metrics"}
    return raster, center, extent, metadata, metrics


def tracking_image(clean_rgb, attack_rgb, projected, selected, dataset, sequence, objective, t, font_set, ymax):
    # Reuse the established overlay renderer exactly; rearrange its panels for
    # the presentation canvas, without scaling image coordinates independently.
    original, status = tracking_frame(clean_rgb, attack_rgb, projected, selected, dataset, sequence, t, FPS, font_set)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f"{objective}  ·  Tracking 결과", font=font_set[32], fill=FG)
    state = projected["attack_result"]["selected_state"]
    draw.text((20, 58), f"{dataset} / {sequence}     source frame {t:03d}/127     저장 step {state['step']} · 10fps preview", font=font_set[20], fill=MUTED)
    for x, kind, color in ((20, "clean", CLEAN), (812, "attack", ATTACK)):
        m = projected[kind + "_result"]["tracking_metrics"]
        draw.text((x, 96), "Clean" if kind == "clean" else "Attack", font=font_set[30], fill=color)
        draw.text((x, 141), f"APD {m['apd3d_all']:.2f}%    EPE {m['epe_all_m']:.6f} m", font=font_set[26], fill=FG)
        source_x = 20 if kind == "clean" else 812
        canvas.paste(original.crop((source_x, 112, source_x + 768, 544)), (x, 190))
        st = status[kind]
        draw.text((x, 630), f"화면 밖 {st['pred_out_of_frame']}  ·  카메라 뒤 {st['pred_behind_camera']}  ·  비유한 {st['pred_nonfinite']}", font=font_set[20], fill=MUTED)
    draw.text((20, 665), f"초록 GT · 청록 Clean · 주황 Attack  |  동일한 GT 기반 표시 {len(selected)}점 · GT 카메라 투영", font=font_set[21], fill=FG)
    draw.text((20, 703), "전체 평가 query의 프레임별 3D EPE (m) — 표시 점 선택과 별도", font=font_set[21], fill=FG)
    left, right, top, bottom = 100, 1560, 753, 845
    for tick in (0.0, ymax / 2, ymax):
        y = bottom - tick / ymax * (bottom - top)
        draw.line((left, y, right, y), fill=(57, 66, 81))
        draw.text((20, y - 12), f"{tick:.2g}", font=font_set[20], fill=MUTED)
    for kind, color in (("clean", CLEAN), ("attack", ATTACK)):
        values = np.asarray(projected[kind + "_epe_per_frame_m"])
        pts = list(zip(np.linspace(left, right, FRAMES), bottom - values / ymax * (bottom - top)))
        draw.line([(float(x), float(y)) for x, y in pts], fill=color, width=3)
        x, y = pts[t]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    x = left + t / 127 * (right - left)
    draw.line((x, top, x, bottom), fill=FG)
    for f in (0, 32, 64, 96, 127):
        draw.text((left + f / 127 * (right - left) - 10, 858), str(f), font=font_set[20], fill=MUTED)
    return canvas, status


class ReconstructionCanvas:
    def __init__(self, source, conditions, objective, raster, center, extent, metadata, font_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.colors import SymLogNorm
        font_manager.fontManager.addfont(font_path)
        matplotlib.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                                   "axes.unicode_minus": False, "font.size": 19})
        self.plt, self.source, self.conditions, self.objective = plt, source, conditions, objective
        self.raster, self.center, self.extent, self.metadata = raster, center, extent, metadata
        self.fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#101620")
        self.title = self.fig.text(.025, .962, "", color="white", fontsize=28, weight="bold")
        self.subtitle = self.fig.text(.025, .91, "", color="#b8c5d7", fontsize=16)
        self.clouds, self.global_clouds, self.outside_labels = [], [], []
        detail_extent = metadata["local_detail_axis_limits_centered_m"][1]
        for i, name in enumerate(("GT", "Clean", "Attack")):
            ax = self.fig.add_axes([.01 + i * .33, .375, .315, .49], projection="3d", facecolor="#101620")
            self.fig.text(.17 + i * .33, .865, name, color="white", fontsize=25, ha="center")
            ax.set_proj_type("ortho")
            ax.view_init(elev=18, azim=-70)
            ax.set(xlim=(-detail_extent, detail_extent), ylim=(-detail_extent, detail_extent), zlim=(-detail_extent, detail_extent))
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("X (m)", color="white", fontsize=15, labelpad=2)
            ax.set_ylabel("Y (m)", color="white", fontsize=15, labelpad=2)
            ax.set_zlabel("Z (m)", color="white", fontsize=15, labelpad=2)
            ax.tick_params(colors="#b8c5d7", labelsize=12, pad=0)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((.075, .10, .14, 1))
            collection = ax.scatter([], [], [], s=1.8, depthshade=False, marker=".", rasterized=True)
            self.clouds.append(collection)
            self.outside_labels.append(self.fig.text(.17 + i * .33, .832, "", color="#d8e2ef", fontsize=13, ha="center"))
            inset = self.fig.add_axes([.233 + i * .33, .395, .090, .16], projection="3d", facecolor="#101620")
            inset.set_proj_type("ortho")
            inset.view_init(elev=18, azim=-70)
            inset.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
            inset.set_box_aspect((1, 1, 1))
            inset.set_axis_off()
            self.global_clouds.append(inset.scatter([], [], [], s=.6, depthshade=False, marker=".", rasterized=True))
            self.fig.text(.277 + i * .33, .394, f"전체 ±{extent:.3g}m", color="#b8c5d7", fontsize=11, ha="center")
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#858a93")
        self.maps, self.metric_labels = [], []
        norm = SymLogNorm(linthresh=.1, linscale=1, base=10, vmin=0, vmax=metadata["heatmap_vmax_m"])
        for i, name in enumerate(("Clean", "Attack")):
            x = .075 + i * .46
            label = self.fig.text(x, .354, "", color="white", fontsize=21)
            ax = self.fig.add_axes([x, .055, .39, .27], facecolor="#858a93")
            im = ax.imshow(np.zeros((288, 512)), cmap=cmap, norm=norm, interpolation="nearest")
            ax.axis("off")
            self.maps.append(im)
            self.metric_labels.append(label)
        cax = self.fig.add_axes([.949, .069, .015, .235])
        cb = self.fig.colorbar(self.maps[0], cax=cax)
        ticks = [v for v in (0, .1, .3, 1, 3, 10, 30, 100) if v < metadata["heatmap_vmax_m"] and float(norm(v)) <= .90]
        ticks.append(metadata["heatmap_vmax_m"])
        metadata["heatmap_colorbar_ticks_m"] = ticks
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{v:.3g}" for v in ticks])
        cb.ax.tick_params(colors="white", labelsize=14)
        self.fig.text(.942, .350, "EPE (m)", color="white", fontsize=12)
        self.fig.text(.942, .326, "SymLog", color="#b8c5d7", fontsize=11)
        self.fig.text(.026, .009, "공통 GT 상세축 + 전체범위 inset (이상점 보존) | 색상 SymLog: 0.1m 기준 | 3D만 stride 8 · 지표는 전체 valid pixel", color="#b8c5d7", fontsize=14)

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
        for i, (collection, global_collection, points) in enumerate(zip(self.clouds, self.global_clouds, clouds)):
            p = points - self.center
            inside = np.all(np.abs(p) <= detail_extent, axis=1)
            collection._offsets3d = (p[inside, 0], p[inside, 1], p[inside, 2])
            collection.set_facecolors(colors[inside])
            collection.set_edgecolors(colors[inside])
            global_collection._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
            global_collection.set_facecolors(colors)
            global_collection.set_edgecolors(colors)
            self.outside_labels[i].set_text(f"상세축 밖 표시점 {np.count_nonzero(~inside)}/{len(p)} (stride 8)")
        state = conds[self.objective]["record"]["selected_state"]
        self.title.set_text(f"{self.objective}  ·  Reconstruction 결과")
        self.subtitle.set_text(f"{source['dataset']} / {source['sequence']}   source frame {t:03d}/127   저장 step {state['step']}   GT / Clean / Attack RGB-colored pointmaps")
        for i, name in enumerate(("clean", self.objective)):
            c = conds[name]
            m = c["record"]["reconstruction_metrics"]
            self.maps[i].set_data(np.ma.array(frame_error(source, c, t), mask=~valid))
            self.metric_labels[i].set_text(f"{'Clean' if i == 0 else 'Attack'}  APD {m['apd3d']:.2f}%  ·  EPE {m['epe_m']:.6f} m")
        self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy(), "RGB")

    def close(self):
        self.plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-only", action="store_true", help="Render frame-zero layout previews; never mark the eight-video manifest complete")
    args = parser.parse_args()
    started = time.perf_counter()
    run, project, output = args.run_dir.resolve(), args.project_root.resolve(), args.output_dir.resolve()
    require(output.is_relative_to(run / "presentation" / ".build"), "New media must remain inside presentation/.build")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "task_visualization_manifest.json"
    require(not manifest_path.exists(), "Choose a fresh output directory; existing media is not silently overwritten")
    print("Strict completed-study scalar gate and original-report hash inventory", flush=True)
    study = load_verified_study(run, require_complete=True)
    frozen_before = report_inventory(run / "html_report")
    source_names = (Path(__file__).name, "component_projection.py", "component_report_data.py",
                    "render_component_comparisons.py", "render_tracking_comparisons.py", "render_pgd_comparisons.py", "tracking_projection.py")
    source_sha = {n: sha256(Path(__file__).parent / n) for n in source_names}
    context = ComponentContext(run, project, metadata=study["metadata"], records=study["records"])
    font_path, font_set = fonts()
    font_set.update({s: ImageFont.truetype(font_path, s) for s in (16, 21, 26, 30, 32)})
    encoder = CPUEncoder()
    manifest = {"schema_version": 1, "status": "rendering", "started_at_utc": now(), "run_id": run.name,
                "run_signature": study["metadata"]["signature"], "analysis_sha256": study["analysis_sha256"],
                "cpu_only": True, "new_model_inference": False, "GPU_used": False,
                "source_code_sha256": source_sha, "font_path": font_path,
                "frame_count": FRAMES, "fps": FPS, "source_frame_indices": list(range(FRAMES)),
                "preview_fps_is_capture_fps": False, "rgb_difference_panels": False,
                "media": {}, "preview_media": {}, "sequence_common": {}, "errors": []}
    write_json(manifest_path, manifest)
    try:
        for prefix, dataset, sequence in CLIPS:
            print(f"Preparing {dataset}/{sequence}: saved arrays and matched inputs", flush=True)
            source = context.source(dataset, sequence)
            conditions = {o: context.condition(dataset, sequence, o) for o in ("clean", *OBJECTIVES)}
            pairs = {o: context.pair(dataset, sequence, o) for o in OBJECTIVES}
            selected, selection = select_queries(pairs["tracking_3d"])
            other_selected, other_selection = select_queries(pairs["reconstruction_3d"])
            require(np.array_equal(selected, other_selected) and selection == other_selection, "GT query choices differ between objectives")
            ymax = max(float(conditions[o]["epe_per_frame"].max()) for o in ("clean", *OBJECTIVES)) * 1.1
            raster, center, extent, common, dense_checks = prepare_reconstruction(source, conditions)
            manifest["sequence_common"][prefix] = {"dataset": dataset, "sequence": sequence,
                    "tracking_selection": selection, "tracking_epe_chart_ylim_m": [0, ymax],
                    "reconstruction": common, "reconstruction_metric_reproduction": dense_checks}
            write_json(manifest_path, manifest)
            for objective in OBJECTIVES:
                clean, attack = conditions["clean"], conditions[objective]
                for task in ("tracking", "reconstruction"):
                    media_id = f"{prefix}_{objective}_{task}"
                    directory = output / media_id
                    directory.mkdir(exist_ok=True)
                    path = directory / "comparison.mp4"
                    canvas = None if task == "tracking" else ReconstructionCanvas(source, conditions, objective, raster, center, extent, common, font_path)
                    frame_statuses, poster_records = [], {}
                    first_render_seconds = None
                    if args.preview_only:
                        tick = time.perf_counter()
                        if task == "tracking":
                            frame, _ = tracking_image(clean["arrays"]["rgb_float32.npy"][0], attack["arrays"]["rgb_float32.npy"][0],
                                      pairs[objective], selected, dataset, sequence, objective, 0, font_set, ymax)
                        else:
                            frame = canvas.frame(0)
                        first_render_seconds = time.perf_counter() - tick
                        png = directory / "frame_000.png"
                        frame.save(png)
                        manifest["preview_media"][media_id] = {"id": media_id, **asset(png, output), "first_frame_render_seconds": first_render_seconds}
                        if canvas is not None:
                            canvas.close()
                        print(f"  PREVIEW {media_id}: {first_render_seconds:.3f}s/frame, nominal128={first_render_seconds * 128:.1f}s", flush=True)
                        write_json(manifest_path, manifest)
                        continue
                    def frames():
                        nonlocal first_render_seconds
                        for t in range(FRAMES):
                            tick = time.perf_counter()
                            if task == "tracking":
                                frame, status = tracking_image(clean["arrays"]["rgb_float32.npy"][t], attack["arrays"]["rgb_float32.npy"][t],
                                      pairs[objective], selected, dataset, sequence, objective, t, font_set, ymax)
                                frame_statuses.append({"frame": t, **status})
                            else:
                                frame = canvas.frame(t)
                            if t == 0:
                                first_render_seconds = time.perf_counter() - tick
                                print(f"  {media_id} first-frame={first_render_seconds:.3f}s; nominal128 render={first_render_seconds * 128:.1f}s", flush=True)
                            if t in POSTERS:
                                png = directory / f"frame_{t:03d}.png"
                                frame.save(png)
                                poster_records[str(t)] = asset(png, output)
                            if t % 32 == 0:
                                print(f"  {media_id} frame {t}/127", flush=True)
                            yield frame
                    try:
                        probe = encoder.encode(path, frames(), WIDTH, HEIGHT, FPS, FRAMES)
                    finally:
                        if canvas is not None:
                            canvas.close()
                    require(set(poster_records) == {str(t) for t in POSTERS}, "Missing source-frame posters")
                    manifest["media"][media_id] = {"id": media_id, "dataset": dataset, "sequence": sequence,
                       "objective": objective, "task": task, **asset(path, output), "probe": probe,
                       "posters": poster_records, "source_frame_indices": list(range(FRAMES)),
                       "selected_state": attack["record"]["selected_state"], "clean_selected_state": clean["record"]["selected_state"],
                       "case_metrics": {"clean": clean["record"][task + "_metrics"], "attack": attack["record"][task + "_metrics"]},
                       "source_evidence": {"sequence": source["evidence"], "clean": clean["evidence"], "attack": attack["evidence"]},
                       "metric_reproduction": {"clean": clean["checks"], "attack": attack["checks"]} if task == "tracking" else {"clean": dense_checks["clean"], "attack": dense_checks[objective]},
                       "common_display_metadata": {"tracking_selection": selection, "chart_ylim_m": [0, ymax]} if task == "tracking" else common,
                       "display_outside_count": None if task == "tracking" else {"GT": common["display_outside_count"]["GT"], "clean": common["display_outside_count"]["clean"], "attack": common["display_outside_count"][objective]},
                       "cloud_rgb_policy": "Same saved clean RGB colors at identical GT raster pixels for GT/Clean/Attack; geometry alone changes" if task == "reconstruction" else None,
                       "display_status_per_frame": frame_statuses, "first_frame_render_seconds": first_render_seconds}
                    write_json(manifest_path, manifest)
                    print(f"Completed {media_id}: decoded128, {probe['width']}x{probe['height']},10fps,12.8s", flush=True)
            # Release this clip's memory maps/compact arrays before the next one.
            context.conditions.clear()
            context.sources.clear()
        expected = {f"{prefix}_{o}_{task}" for prefix, _, _ in CLIPS for o in OBJECTIVES for task in ("tracking", "reconstruction")}
        require(set(manifest["preview_media"] if args.preview_only else manifest["media"]) == expected, "Expected all eight task previews/videos")
        for record in manifest["media"].values():
            verified = encoder.probe(output / record["path"], FRAMES, FPS)
            require(verified["width"] == WIDTH and verified["height"] == HEIGHT, "Final decoded resolution differs")
            require(sha256(output / record["path"]) == record["sha256"], "Completed MP4 bytes changed")
        frozen_after = report_inventory(run / "html_report")
        require(frozen_before == frozen_after, "Original detailed HTML report changed during rendering")
        require(source_sha == {n: sha256(Path(__file__).parent / n) for n in source_names}, "Renderer/import code changed during rendering")
        manifest["original_report_preservation"] = {"passed": True, "files": len(frozen_before),
              "bytes": sum(v["bytes"] for v in frozen_before.values()), "before_inventory_sha256": canonical_sha(frozen_before),
              "after_inventory_sha256": canonical_sha(frozen_after), "key_files": {n: frozen_before[n] for n in ("index.html", "report_manifest.json", "data/report_data.json", "data/task_coupling_analysis.json")}}
        manifest.update(status="preview_complete" if args.preview_only else "complete", completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status="failed", failed_at_utc=now(), errors=[str(exc)], elapsed_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        raise
    print(json.dumps({"status": manifest["status"], "videos": len(manifest["media"]), "manifest": str(manifest_path), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
