"""Render a completed component campaign from immutable saved arrays, CPU only.

No model, checkpoint, attack, training, CUDA or new prediction is invoked.
Videos contain all 128 saved frames. PNG selection never changes evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from component_projection import ComponentContext, OBJECTIVES, metric_check, read_json, require, sha256
from render_pgd_comparisons import encode_frames, encoder_args, fonts, render_frame as rgb_frame, to_rgb8
from render_tracking_comparisons import (ATTACK, BG, CLEAN, FG, MUTED, CANVAS_HEIGHT, CANVAS_WIDTH,
                                        overlay_panel, render_frame as legacy_tracking_frame, select_queries)

DEFAULT_IMAGE = "sha256:c07e78bf94c671f41392c98e5558cb49296e4187fc7682c51f74467d76539398"
PREVIEW_FRAMES = (0, 64, 127)
COLORS = {"clean": "#222222", "tracking_mse": "#6a4c93", "tracking_3d": "#2478b4",
          "reconstruction_3d": "#df732b", "confidence": "#23875c", "joint_training": "#b94457"}


def renderer_source_hashes():
    directory = Path(__file__).resolve().parent
    return {name: sha256(directory / name) for name in (
        "render_component_comparisons.py", "component_projection.py", "component_report_data.py",
        "render_pgd_comparisons.py", "render_tracking_comparisons.py", "tracking_projection.py",
        "render_official_visualizations.py")}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def asset_record(path, root, kind, **extra):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root), "Asset must stay within output directory")
    return {"kind": kind, "path": path.relative_to(root).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size, **extra}


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always publish only after the complete file has been written.
    temporary = path.with_name(path.stem + ".writing.png")
    figure.savefig(temporary, dpi=150)
    os.replace(temporary, path)
    pyplot().close(figure)


def save_image(image, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".writing.png")
    image.save(temporary)
    os.replace(temporary, path)


class CPUEncoder:
    def __init__(self, image=DEFAULT_IMAGE, ffmpeg=None, ffprobe=None):
        self.image = image
        self.ffmpeg = str(ffmpeg) if ffmpeg else shutil.which("ffmpeg")
        self.ffprobe = str(ffprobe) if ffprobe else shutil.which("ffprobe")
        require(bool(self.ffmpeg) == bool(self.ffprobe), "Supply both ffmpeg and ffprobe, or neither")
        self.image_id = None
        if not self.ffmpeg:
            info = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                                  capture_output=True, text=True, check=True)
            self.image_id = info.stdout.strip()

    def encode(self, path, frames, width, height, fps, count):
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".writing.mp4")
        if self.ffmpeg:
            command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
                       "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                       "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        else:
            command = encoder_args(self.image, path.parent, temporary.name, width, height, fps)
            command[command.index("-pix_fmt", command.index("-c:v")):command.index("-pix_fmt", command.index("-c:v"))] = ["-threads", "2"]
        emitted = 0
        def checked_frames():
            nonlocal emitted
            for frame in frames:
                require(frame.mode == "RGB" and frame.size == (width, height), "Unexpected encoded frame layout")
                emitted += 1
                yield frame
            require(emitted == count, f"Video generator yielded {emitted}, expected {count} frames")
        try:
            encode_frames(command, checked_frames())
            probe = self.probe(temporary, count, fps)
            require((probe["width"], probe["height"]) == (width, height), "Encoded video dimensions differ from requested layout")
            os.replace(temporary, path)
            probe["source_sha256"] = sha256(path)
            return probe
        finally:
            if temporary.exists():
                temporary.unlink()

    def probe(self, path, count, fps):
        path = Path(path).resolve()
        arguments = ["-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                     "stream=width,height,avg_frame_rate,nb_read_frames,codec_name:format=duration", "-of", "json"]
        if self.ffprobe:
            command = [self.ffprobe, *arguments, str(path)]
        else:
            require("," not in str(path.parent), "Docker mount directory contains a comma")
            command = ["docker", "run", "--rm", "--network", "none", "--read-only", "--entrypoint", "ffprobe",
                       "--mount", f"type=bind,source={path.parent},target=/input,readonly", self.image,
                       *arguments, f"/input/{path.name}"]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        evidence = json.loads(process.stdout)
        stream = evidence["streams"][0]
        decoded = int(stream["nb_read_frames"])
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        actual_fps = numerator / denominator
        duration = float(evidence["format"]["duration"])
        require(decoded == count and math.isclose(actual_fps, fps, rel_tol=1e-6, abs_tol=1e-6)
                and abs(duration - count / fps) <= max(0.1, 1 / fps), "Encoded decoded-frame count/FPS/duration differs")
        return {"success": True, "method": "ffprobe_count_frames", "decoded_frames": decoded,
                "fps": actual_fps, "width": int(stream["width"]), "height": int(stream["height"]),
                "duration": duration, "codec": stream["codec_name"], "source_sha256": sha256(path),
                "encoder": "local CPU ffmpeg" if self.ffmpeg else "Docker CPU ffmpeg (no GPU)",
                "encoder_image_id": self.image_id}


def tracking_frame(clean_frame, attack_frame, projected, selected, dataset, sequence, frame_index, fps, font_set):
    """Reuse the established overlay; give the 128-frame chart source indices."""
    canvas, status = legacy_tracking_frame(clean_frame, attack_frame, projected, selected, dataset, sequence, frame_index, fps, font_set)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 735, CANVAS_WIDTH, 965), fill=BG)
    count = len(projected["gt_xy"])
    query_count = np.asarray(projected["gt_xy"]).shape[1]
    clean_epe = np.asarray(projected["clean_epe_per_frame_m"])
    attack_epe = np.asarray(projected["attack_epe_per_frame_m"])
    draw.text((20, 744), f"전체 {query_count}개 점의 프레임별 3D EPE (m) · 평가와 같은 스케일 · 원본 인덱스 0~{count-1}", font=font_set[20], fill=FG)
    draw.text((CANVAS_WIDTH-300, 744), "청록 원본   주황 PGD", font=font_set[18], fill=MUTED)
    left, right, top, bottom = 95, CANVAS_WIDTH-40, 790, 931
    ymax = max(float(clean_epe.max()), float(attack_epe.max()), .1) * 1.10
    for tick in (0, ymax/2, ymax):
        y = bottom - tick/ymax*(bottom-top)
        draw.line((left, y, right, y), fill=(46,57,73))
        draw.text((20, y-11), f"{tick:.2g}", font=font_set[18], fill=MUTED)
    for index in (0,32,64,96,count-1):
        x = left + index/(count-1)*(right-left)
        draw.text((x-10, bottom+6), str(index), font=font_set[16], fill=MUTED)
    for values, color in ((clean_epe,CLEAN),(attack_epe,ATTACK)):
        points = list(zip(np.linspace(left,right,count),bottom-values/ymax*(bottom-top)))
        draw.line([(float(x),float(y)) for x,y in points], fill=color, width=3)
        x,y=points[frame_index]
        draw.ellipse((x-5,y-5,x+5,y+5), fill=color, outline=BG)
    cursor = left + frame_index/(count-1)*(right-left)
    draw.line((cursor,top,cursor,bottom),fill=(209,216,226))
    return canvas,status


def history_plot(condition, path, identity):
    history, record = condition["history"], condition["record"]
    clean = next(row for row in history if row["restart"] == -1)
    iterations = [row for row in history if row["restart"] == 0]
    require([row["step"] for row in iterations] == list(range(21)), "Expected restart 0 steps 0..20")
    selected = record["selected_state"]
    best = next(row for row in history if row["restart"] == selected["restart"] and row["step"] == selected["step"])
    require(np.isclose(best["loss"], record["attack_loss"], rtol=1e-6, atol=1e-6), "Saved best loss differs from selected history")
    plt = pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    steps = [row["step"] for row in iterations]
    axes[0, 0].plot(steps, [row["loss"] for row in iterations], marker=".")
    axes[0, 0].axhline(clean["loss"], color="#777777", linestyle="--", label="clean")
    selected_x = selected["step"] if selected["restart"] == 0 else -1
    axes[0, 0].scatter([selected_x], [best["loss"]], marker="*", s=170, color="#c54832", label=f"saved r={selected['restart']}, step={selected['step']}")
    axes[0, 0].set(xlabel="PGD update index (0=random initial state)", ylabel="This objective's loss (own units)", title="Saved state maximizes attack objective, including clean")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(steps, [row["delta_linf"] for row in iterations], label="L-infinity")
    axes[0, 1].axhline(4 / 255, color="#777777", linestyle="--", label="4/255 bound")
    axes[0, 1].set(xlabel="PGD update index", ylabel="RGB change [0,1] units", title="Perturbation budget")
    axes[0, 1].legend(fontsize=8)
    gradients = [row for row in iterations if "gradient_l2_per_frame" in row]
    require(len(gradients) == 20, "Expected 20 gradient measurements; final step is forward only")
    gradient_values = np.asarray([row["gradient_l2_per_frame"] for row in gradients], float)
    require(gradient_values.shape == (20, 128) and np.isfinite(gradient_values).all() and (gradient_values >= 0).all(), "Invalid saved gradient norms")
    axes[1, 0].plot(range(20), gradient_values.mean(axis=1), label="frame mean")
    axes[1, 0].fill_between(range(20), gradient_values.min(axis=1), gradient_values.max(axis=1), alpha=.2, label="frame min/max")
    axes[1, 0].set(xlabel="Gradient evaluation step", ylabel="Saved RGB gradient L2", title="Recorded norms; zero values are retained")
    axes[1, 0].legend(fontsize=8)
    last_l2 = np.asarray(iterations[-1]["delta_l2_per_frame"], float)
    chosen_l2 = np.asarray(best["delta_l2_per_frame"], float)
    axes[1, 1].plot(range(128), last_l2, label="last step 20")
    axes[1, 1].plot(range(128), chosen_l2, label="saved selected state")
    axes[1, 1].set(xlabel="Source frame 0..127", ylabel="RGB perturbation L2", title="Last and selected iterates may differ")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(f"{identity}\nAll 128 raw frames | 20 PGD updates | one matched derived seed", fontsize=11)
    save_figure(figure, path)


def _temporal_values(values, name, *, positive=False):
    result = np.asarray(values, dtype=np.float64)
    require(result.shape == (128,) and np.isfinite(result).all(), f"{name} must contain all 128 finite frames")
    require((result > 0).all() if positive else (result >= 0).all(), f"{name} has invalid magnitude")
    return result


def _descriptive_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {name: None for name in ("mean", "median", "min", "max")}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


def summarize_normalization(clean_values, attack_values, fixed_gt_values):
    """Export small recorded-scale diagnostics without treating frames as CI samples."""
    clean = _temporal_values(clean_values, "Clean native normalization", positive=True)
    attack = _temporal_values(attack_values, "Attack native normalization", positive=True)
    fixed_gt = _temporal_values(fixed_gt_values, "Fixed GT normalization", positive=True)
    eligible = clean > 1e-12
    values = np.full(128, np.nan, dtype=np.float64)
    np.divide(attack, clean, out=values, where=eligible)
    eligible &= np.isfinite(values)
    valid_values = values[eligible]
    clean_stats, attack_stats = _descriptive_stats(clean), _descriptive_stats(attack)
    ratio_of_means = attack_stats["mean"] / clean_stats["mean"] if clean_stats["mean"] > 1e-12 else None
    if ratio_of_means is not None and not math.isfinite(ratio_of_means):
        ratio_of_means = None
    return {"schema_version": 1, "passed": True, "frame_count": 128,
            "clean": clean_stats, "attack": attack_stats, "fixed_gt": _descriptive_stats(fixed_gt),
            "ratio_of_means": ratio_of_means,
            "framewise_ratio": {**_descriptive_stats(valid_values), "valid_frames": int(eligible.sum()),
                                "excluded_frames": int((~eligible).sum()),
                                "values": [float(value) if valid else None for value, valid in zip(values, eligible)]},
            "per_frame": {"clean": clean.tolist(), "attack": attack.tolist(), "fixed_gt": fixed_gt.tolist()},
            "denominator_policy": "Ratio denominator must be finite and greater than 1e-12; nonfinite ratios are also excluded; exclusions are null.",
            "ratio_mean_policy": "ratio_of_means = mean(attack)/mean(clean); framewise_ratio.mean = mean(attack[t]/clean[t]) over eligible frames.",
            "normalization_definition": "Recorded native per-frame sum of all head2 XYZ point norms divided by 3*H*W; fixed GT uses the recorded same rule.",
            "units": {"prediction_scale": "Raw pre-alignment pointmap coordinate magnitude; not benchmark meters",
                      "fixed_gt_scale": "GT meter-coordinate magnitude", "ratio": "dimensionless"},
            "interpretation_policy": "Prediction and GT raw magnitudes are not a benchmark meter effect or an isolated causal effect. Frame summaries are descriptive, not independent CI samples."}


def summarize_tracking_timeline(clean_values, attack_values, clean_recorded_epe, attack_recorded_epe, query_count):
    """Summarize the already verified all-query frame EPE, using every source time."""
    clean = _temporal_values(clean_values, "Clean tracking EPE")
    attack = _temporal_values(attack_values, "Attack tracking EPE")
    require(isinstance(query_count, (int, np.integer)) and not isinstance(query_count, (bool, np.bool_))
            and query_count > 0, "Tracking timeline requires the complete evaluation query count")
    metric_check(float(clean.mean()), clean_recorded_epe, "Clean frame-mean tracking EPE")
    metric_check(float(attack.mean()), attack_recorded_epe, "Attack frame-mean tracking EPE")
    delta = attack - clean
    segments = [{"source_frame_start": start, "source_frame_end": start + 31, "frame_count": 32,
                 "clean_epe_mean_m": float(clean[start:start+32].mean()),
                 "attack_epe_mean_m": float(attack[start:start+32].mean()),
                 "delta_epe_mean_m": float(delta[start:start+32].mean())} for start in range(0, 128, 32)]
    return {"schema_version": 1, "passed": True, "frame_count": 128, "query_count": int(query_count),
            "evaluation_query_times": 128 * int(query_count), "all_evaluation_query_times": True,
            "clean_epe_mean_m": float(clean.mean()), "attack_epe_mean_m": float(attack.mean()),
            "delta_epe_mean_m": float(delta.mean()), "delta_epe_min_m": float(delta.min()),
            "delta_epe_max_m": float(delta.max()), "increased_frames": int((delta > 0).sum()),
            "equal_frames": int((delta == 0).sum()), "improved_frames": int((delta < 0).sum()),
            "fixed_segments": segments, "frame_mean_matches_whole_clip_epe": True,
            "comparison_tolerance": {"rtol": 1e-6, "atol": 1e-6},
            "evaluation_population_policy": "All saved evaluation queries at all 128 times, including later occlusions; overlay point selection and GT-camera projection do not filter quantitative EPE.",
            "alignment_policy": "Each condition uses saved pred * np.float32(tracking_metrics.scale_all).",
            "count_policy": "Exact signed attack-minus-clean per-frame EPE; >0 increased, ==0 equal, <0 improved. Counts and fixed 32-frame segments describe temporal positions, not independent CI samples."}


def normalization_plot(clean, attack, source, path, identity):
    c = np.asarray(clean["components"]["reconstruction_norm"])
    a = np.asarray(attack["components"]["reconstruction_norm"])
    require(c.shape == a.shape == (128,) and np.isfinite(c).all() and np.isfinite(a).all() and (c > 0).all() and (a > 0).all(), "Invalid native normalization scales")
    plt = pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    frames = np.arange(128)
    for values, label in ((c, "clean prediction head2"), (a, "attack prediction head2"), (source["gt_norm"], "fixed GT")):
        axes[0, 0].plot(frames, values, label=label)
    axes[0, 0].set(xlabel="Source frame", ylabel="Pointmap coordinate magnitude (before alignment)", title="Shared scale affects both native branches")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(frames, a / c)
    axes[0, 1].axhline(1, color="#777777", linestyle="--")
    axes[0, 1].set(xlabel="Source frame", ylabel="Attack / clean prediction scale", title="Association, not isolated normalization causality")
    diagnostics = [condition["record"]["loss_terms"]["diagnostics"] for condition in (clean, attack)]
    confidence = ("track_raw_conf_mean", "track_conf_mean", "reconstruction_conf_mean")
    geometry = ("tracking_l21", "reconstruction_l21")
    for index, (fields, labels) in enumerate(((confidence, ("head1 raw", "head1 effective", "head2")), (geometry, ("head1 L21", "head2 L21")))):
        axis = axes[1, index]
        for offset, diag, label in ((-.17, diagnostics[0], "clean"), (.17, diagnostics[1], "attack")):
            axis.bar(np.arange(len(fields)) + offset, [diag[field] for field in fields], width=.34, label=label)
        axis.set_xticks(range(len(fields)), labels)
        axis.set(ylabel="Confidence score (not probability)" if index == 0 else "Unweighted native L21 (dimensionless)")
        axis.legend(fontsize=8)
    figure.suptitle(f"{identity}\nPrediction: raw pointmap coordinate scale; GT: meter-coordinate magnitude.\nOnly attack/clean ratios and normalized L21 are dimensionless; raw scale gaps are not benchmark meter effects.", fontsize=10)
    save_figure(figure, path)


def reconstruction_frame(source, clean, attack, checks, frame_index, path, identity):
    valid = source["dense_valid"][frame_index]
    gt = source["dense_gt"][frame_index]
    residuals = [np.linalg.norm(condition["arrays"]["reconstruction_float32.npy"][frame_index]
                               * np.float32(condition["record"]["reconstruction_metrics"]["scale"]) - gt, axis=-1)
                 for condition in (clean, attack)]
    plt = pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(13.3, 7.6), constrained_layout=True)
    axes[0, 0].imshow(to_rgb8(clean["arrays"]["rgb_float32.npy"][frame_index]))
    axes[0, 0].set_title("Clean RGB")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#a5a5a5")
    for axis, error, title in zip(axes[0, 1:], residuals, ("Clean reconstruction error", "Attack reconstruction error")):
        image = axis.imshow(np.ma.masked_where(~valid, error), cmap=error_cmap, vmin=0, vmax=checks["vmax_m"])
        axis.set_title(title)
    figure.colorbar(image, ax=list(axes[0, 1:]), label="EPE (m); common clean+all5/all-frame maximum", shrink=.8)
    signed_cmap = plt.get_cmap("RdBu_r").copy()
    signed_cmap.set_bad("#a5a5a5")
    difference = axes[1, 0].imshow(np.ma.masked_where(~valid, residuals[1] - residuals[0]), cmap=signed_cmap,
                                  vmin=-checks["vmax_m"], vmax=checks["vmax_m"])
    axes[1, 0].set_title("Attack minus clean error")
    figure.colorbar(difference, ax=axes[1, 0], label="Signed error change (m)", shrink=.8)
    for axis, condition, label in zip(axes[1, 1:], (clean, attack), ("Clean", "Attack")):
        image = axis.imshow(condition["arrays"]["reconstruction_confidence_float32.npy"][frame_index], cmap="viridis",
                            vmin=checks["confidence_vmin"], vmax=checks["confidence_vmax"])
        axis.set_title(f"{label} head2 confidence score")
    figure.colorbar(image, ax=list(axes[1, 1:]), label="Model score (not probability)", shrink=.8)
    for axis in axes.flat:
        axis.set_axis_off()
    figure.suptitle(f"{identity} | source frame {frame_index}\n512x288 fixed positive-depth GT support; grey=invalid; global median scale", fontsize=11)
    save_figure(figure, path)


def shared_figures(context, dataset, sequence, checks, root, preview_frames, font_set):
    source = context.source(dataset, sequence)
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    evidence = {}
    for objective in ("clean", *OBJECTIVES):
        condition = context.condition(dataset, sequence, objective)
        axis.plot(range(128), condition["epe_per_frame"], color=COLORS[objective], label=objective)
        evidence[objective] = condition["evidence"]
    axis.set(xlabel="Original source frame 0..127", ylabel="Tracking EPE (m), all evaluation queries",
             title=f"{dataset}/{sequence}: clean and independent attacks\nRecorded global scale per condition; all query times including later occlusions")
    axis.legend(fontsize=8, ncol=3)
    axis.grid(alpha=.2)
    timeline_path = root / "timelines" / dataset / sequence / "tracking_epe.png"
    save_figure(figure, timeline_path)
    timeline = {"dataset": dataset, "sequence": sequence, "frame_count": 128, "sources": evidence,
                "assets": [asset_record(timeline_path, root, "tracking_timeline")]}
    geometry = []
    projected = {objective: context.pair(dataset, sequence, objective) for objective in ("tracking_3d", "reconstruction_3d")}
    selected, selection = select_queries(projected["tracking_3d"])
    cols = ("clean", "tracking_3d", "reconstruction_3d")
    for frame_index in preview_frames:
        figure, axes = plt.subplots(3, 3, figsize=(15, 9.7), constrained_layout=True)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#a5a5a5")
        for column, objective in enumerate(cols):
            condition = context.condition(dataset, sequence, objective)
            rgb = condition["arrays"]["rgb_float32.npy"][frame_index]
            axes[0, column].imshow(to_rgb8(rgb))
            state = condition["record"]["selected_state"]
            axes[0, column].set_title(f"{objective}\nsaved r={state['restart']}, step={state['step']}", fontsize=10)
            p = projected["tracking_3d"] if objective in ("clean", "tracking_3d") else projected["reconstruction_3d"]
            panel, _ = overlay_panel(rgb, p, "clean" if objective == "clean" else "attack", frame_index, selected, font_set)
            axes[1, column].imshow(panel)
            error = np.linalg.norm(condition["arrays"]["reconstruction_float32.npy"][frame_index]
                                   * np.float32(condition["record"]["reconstruction_metrics"]["scale"])
                                   - source["dense_gt"][frame_index], axis=-1)
            image = axes[2, column].imshow(np.ma.masked_where(~source["dense_valid"][frame_index], error), cmap=cmap, vmin=0, vmax=checks["vmax_m"])
        for axis in axes.flat:
            axis.set_axis_off()
        figure.colorbar(image, ax=list(axes[2]), label="Reconstruction error (m), shared clean+all5 scale", shrink=.7)
        figure.suptitle(f"{dataset}/{sequence} | same source frame {frame_index} | same GT camera and displayed query IDs\nRows: saved RGB / GT-camera tracking projection / fixed-mask reconstruction error", fontsize=11)
        path = root / "geometry_cases" / dataset / sequence / f"geometry_attack_triptych_frame_{frame_index:03d}.png"
        save_figure(figure, path)
        geometry.append({"dataset": dataset, "sequence": sequence, "frame_index": frame_index, "selection": selection,
                         "sources": {name: context.condition(dataset, sequence, name)["evidence"] for name in cols},
                         "assets": [asset_record(path, root, "geometry_triptych", frame_index=frame_index)]})
    return timeline, geometry


def render_official(context, dataset, sequence, objective, directory, root, encoder, selected):
    # Imports are intentionally deferred. This optional CPU drawing path loads
    # the official plotting module but never a model/checkpoint or model forward.
    from render_official_visualizations import import_official
    require(bool(encoder.ffmpeg), "Optional official visualizer requires local CPU ffmpeg; pass --ffmpeg and --ffprobe")
    import mediapy
    mediapy.set_ffmpeg(encoder.ffmpeg)
    module, official = import_official(context.project / "vendor" / "St4RTrack", context.run)
    source = context.source(dataset, sequence)
    intrinsics = source["intrinsics"].astype(np.float64).copy()
    resize = source["resize"]
    intrinsics *= np.array([resize[0], resize[1], resize[0], resize[1]])
    assets = []
    for name in ("clean", objective):
        condition = context.condition(dataset, sequence, name)
        target = directory / "official" / name
        images = [Image.fromarray(to_rgb8(frame)) for frame in condition["arrays"]["rgb_float32.npy"]]
        np.random.seed(20261002)
        module.visualize_results(images, source["gt"][:, selected].copy(), condition["aligned"][:, selected].copy(),
                                 source["extrinsics"].copy(), intrinsics.copy(), save_path=str(target))
        pyplot().close("all")
        for kind in ("2d", "3d"):
            original_path = target / f"{kind}_tracks.mp4"
            probe = encoder.probe(original_path, 128, 15)
            path = directory / f"official_{name}_{kind}.mp4"
            os.replace(original_path, path)
            assets.append(asset_record(path, root, f"official_{name}_{kind}", probe=probe, official_renderer=official))
    return assets


def render_pair(context, dataset, sequence, objective, root, fps, preview_frames, font_set, encoder, frames_only, official_3d):
    started = time.perf_counter()
    directory = root / "comparisons" / objective / dataset / sequence
    directory.mkdir(parents=True, exist_ok=True)
    projected = context.pair(dataset, sequence, objective)
    source = context.source(dataset, sequence)
    clean, attack = context.condition(dataset, sequence, "clean"), context.condition(dataset, sequence, objective)
    selected, selection = select_queries(projected)
    reconstruction = context.reconstruction_checks(dataset, sequence)
    assets, statuses = [], []
    c_rgb, a_rgb = clean["arrays"]["rgb_float32.npy"], attack["arrays"]["rgb_float32.npy"]
    title = f"{sequence} | {objective}"
    def rgb_frames():
        for frame_index in range(128):
            yield rgb_frame(c_rgb[frame_index], a_rgb[frame_index], clean["record"]["tracking_metrics"],
                            attack["record"]["tracking_metrics"], dataset, title, frame_index, 128, fps, font_set)
    def tracking_frames():
        for frame_index in range(128):
            frame, status = tracking_frame(c_rgb[frame_index], a_rgb[frame_index], projected, selected,
                                           dataset, title, frame_index, fps, font_set)
            statuses.append({"frame_index": frame_index, **status})
            yield frame
    for frame_index in preview_frames:
        path = directory / f"rgb_frame_{frame_index:03d}.png"
        frame = rgb_frame(c_rgb[frame_index], a_rgb[frame_index], clean["record"]["tracking_metrics"],
                          attack["record"]["tracking_metrics"], dataset, title, frame_index, 128, fps, font_set)
        save_image(frame, path)
        assets.append(asset_record(path, root, "rgb", frame_index=frame_index))
        path = directory / f"tracking_frame_{frame_index:03d}.png"
        frame, _ = tracking_frame(c_rgb[frame_index], a_rgb[frame_index], projected, selected,
                                  dataset, title, frame_index, fps, font_set)
        save_image(frame, path)
        assets.append(asset_record(path, root, "tracking", frame_index=frame_index))
        path = directory / f"reconstruction_frame_{frame_index:03d}.png"
        reconstruction_frame(source, clean, attack, reconstruction, frame_index, path, f"{dataset}/{title}")
        assets.append(asset_record(path, root, "reconstruction", frame_index=frame_index))
    history_path, normalization_path = directory / "pgd_history.png", directory / "normalization.png"
    history_plot(attack, history_path, f"{dataset}/{title}")
    normalization_plot(clean, attack, source, normalization_path, f"{dataset}/{title}")
    normalization_summary = summarize_normalization(clean["components"]["reconstruction_norm"],
                                                     attack["components"]["reconstruction_norm"], source["gt_norm"])
    tracking_timeline_summary = summarize_tracking_timeline(projected["clean_epe_per_frame_m"],
                                                           projected["attack_epe_per_frame_m"],
                                                           clean["record"]["tracking_metrics"]["epe_all_m"],
                                                           attack["record"]["tracking_metrics"]["epe_all_m"],
                                                           np.asarray(projected["gt_xy"]).shape[1])
    assets += [asset_record(history_path, root, "pgd_history"), asset_record(normalization_path, root, "normalization")]
    if not frames_only:
        first = next(rgb_frames())
        path = directory / "rgb_comparison.mp4"
        probe = encoder.encode(path, rgb_frames(), *first.size, fps, 128)
        assets.append(asset_record(path, root, "rgb_video", probe=probe, poster=f"comparisons/{objective}/{dataset}/{sequence}/rgb_frame_000.png"))
        path = directory / "tracking_comparison.mp4"
        probe = encoder.encode(path, tracking_frames(), CANVAS_WIDTH, CANVAS_HEIGHT, fps, 128)
        assets.append(asset_record(path, root, "tracking_video", probe=probe, poster=f"comparisons/{objective}/{dataset}/{sequence}/tracking_frame_000.png"))
        if official_3d:
            assets.extend(render_official(context, dataset, sequence, objective, directory, root, encoder, selected))
    else:
        require(not official_3d, "--official-3d cannot be combined with --frames-only")
    timeline, geometry = shared_figures(context, dataset, sequence, reconstruction, root, preview_frames, font_set)
    entry = {"dataset": dataset, "sequence": sequence, "objective": objective, "status": "preview_complete" if frames_only else "complete",
             "frame_count": 128, "source_frame_indices": list(range(128)), "preview_frame_indices": list(preview_frames),
             "preview_fps": fps, "fps_is_original_capture_rate": False, "selection": selection,
             "sources": projected["sources"], "projection_checks": projected["projection_checks"],
             "reconstruction_checks": reconstruction, "display_status_per_frame": statuses,
             "selected_state": attack["record"]["selected_state"], "assets": assets,
             "clean_epe_per_frame_m": projected["clean_epe_per_frame_m"].tolist(),
             "attack_epe_per_frame_m": projected["attack_epe_per_frame_m"].tolist(),
             "normalization_summary": normalization_summary, "tracking_timeline_summary": tracking_timeline_summary,
             "cpu_render_seconds": time.perf_counter() - started, "new_model_inference": False,
             "renderer_source_sha256": renderer_source_hashes()}
    write_json(directory / "visualization.json", entry)
    return entry, timeline, geometry


def manifest_complete(manifest, identities):
    expected = {(dataset, sequence, objective) for dataset, sequence in identities for objective in OBJECTIVES}
    indexed = {(row["dataset"], row["sequence"], row["objective"]): row for row in manifest["clips"]}
    if set(indexed) != expected or len(indexed) != len(manifest["clips"]):
        return False
    for row in indexed.values():
        if row.get("status") != "complete" or row.get("frame_count") != 128:
            return False
        videos = [asset for asset in row["assets"] if asset["kind"] in ("rgb_video", "tracking_video")]
        if len(videos) != 2 or any(not asset.get("probe", {}).get("success") or asset["probe"].get("decoded_frames") != 128 for asset in videos):
            return False
        if not row["projection_checks"]["passed"] or not row["reconstruction_checks"]["passed"]:
            return False
    timeline_keys = {(row["dataset"], row["sequence"]) for row in manifest["timelines"]}
    geometry_keys = {(row["dataset"], row["sequence"], row["frame_index"]) for row in manifest["geometry_cases"]}
    return (timeline_keys == set(identities) and len(manifest["timelines"]) == len(identities)
            and geometry_keys == {(d, s, f) for d, s in identities for f in PREVIEW_FRAMES}
            and len(manifest["geometry_cases"]) == len(identities) * 3 and not manifest["errors"])


def replace_by_key(rows, value, fields):
    key = tuple(value[field] for field in fields)
    return [row for row in rows if tuple(row[field] for field in fields) != key] + [value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument("--clip", help="Optional dataset/sequence; absent means every manifest clip")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--preview-frames", nargs="+", type=int, default=list(PREVIEW_FRAMES))
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--official-3d", action="store_true")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="CPU ffmpeg/ffprobe fallback image; never uses --gpus")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    require(math.isfinite(args.fps) and 0 < args.fps <= 60, "FPS must be finite and in (0,60]")
    require(sorted(set(args.preview_frames)) == list(PREVIEW_FRAMES), "Required QA frame indices are exactly 0,64,127")
    require(not (args.frames_only and args.official_3d), "--official-3d requires videos")
    from component_report_data import load_verified_study
    study = load_verified_study(args.run_dir, args.analysis_dir, require_complete=True)
    root = args.output_dir.resolve()
    run_dir = args.run_dir.resolve()
    require(root != run_dir and not any(root.is_relative_to(run_dir / name) for name in ("conditions", "sequences")), "Output must not overwrite experiment evidence")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "visualization_manifest.json"
    manifest = read_json(path) if path.is_file() else {
        "schema_version": 1, "status": "partial", "created_at_utc": utc_now(), "run_id": run_dir.name,
        "run_signature": study["metadata"]["signature"], "scope": study["analysis"]["scope"],
        "project_root": str(args.project_root.resolve()),
        "analysis_json_sha256": study["analysis_sha256"],
        "artifact_validation_sha256": study["input_sha256"]["artifact_validation.json"],
        "input_scalar_file_sha256": study["input_sha256"],
        "cpu_only": True, "new_model_inference": False, "frame_count": 128,
        "preview_fps_is_original_capture_rate": False, "difference_amplification": 64,
        "clips": [], "timelines": [], "geometry_cases": [], "errors": [], "commands": []}
    require(manifest["run_signature"] == study["metadata"]["signature"], "Output manifest belongs to another run")
    require(manifest.get("analysis_json_sha256") == study["analysis_sha256"]
            and manifest.get("artifact_validation_sha256") == study["input_sha256"]["artifact_validation.json"],
            "Output manifest is bound to different analysis/audit evidence; use a new output directory")
    identities = [tuple(identity) for identity in study["identities"]]
    selected_ids = identities
    if args.clip:
        selected_ids = [identity for identity in identities if "/".join(identity) == args.clip]
        require(len(selected_ids) == 1, "Unknown --clip")
    font_path, font_set = fonts()
    font_set[16] = ImageFont.truetype(font_path, 16)
    encoder = None if args.frames_only else CPUEncoder(args.image, args.ffmpeg, args.ffprobe)
    if args.official_3d:
        require(bool(encoder.ffmpeg), "--official-3d requires local CPU ffmpeg/ffprobe; CPU Docker remains supported for the required comparison videos")
    context = ComponentContext(args.run_dir, args.project_root, metadata=study["metadata"], records=study["records"])
    manifest["status"] = "partial"
    manifest["project_root"] = str(args.project_root.resolve())
    manifest["renderer_source_sha256"] = renderer_source_hashes()
    manifest["commands"].append({"argv": sys.argv, "started_at_utc": utc_now(), "frames_only": args.frames_only})
    write_json(path, manifest)
    try:
        for index, (dataset, sequence) in enumerate(selected_ids, 1):
            print(f"[{index}/{len(selected_ids)}] CPU rendering {args.objective}/{dataset}/{sequence}, all 128 frames", flush=True)
            entry, timeline, geometry = render_pair(context, dataset, sequence, args.objective, root, args.fps,
                                                     PREVIEW_FRAMES, font_set, encoder, args.frames_only, args.official_3d)
            manifest["clips"] = replace_by_key(manifest["clips"], entry, ("dataset", "sequence", "objective"))
            manifest["timelines"] = replace_by_key(manifest["timelines"], timeline, ("dataset", "sequence"))
            for case in geometry:
                manifest["geometry_cases"] = replace_by_key(manifest["geometry_cases"], case, ("dataset", "sequence", "frame_index"))
            manifest["updated_at_utc"] = utc_now()
            write_json(path, manifest)
            context.conditions.clear()
            context.sources.clear()
            gc.collect()
        for relative, digest in study["input_sha256"].items():
            require(sha256(run_dir / relative) == digest, f"Scalar evidence changed while rendering: {relative}")
        require(sha256(Path(study["analysis_dir"]) / "study_analysis.json") == study["analysis_sha256"], "Analysis changed while rendering")
        manifest["status"] = "complete" if manifest_complete(manifest, identities) else "partial"
        manifest["commands"][-1]["completed_at_utc"] = utc_now()
        write_json(path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["errors"].append({"at_utc": utc_now(), "objective": args.objective, "error": f"{type(error).__name__}: {error}"})
        write_json(path, manifest)
        raise
    print(json.dumps({"status": manifest["status"], "rendered_pairs": len(manifest["clips"]), "manifest": str(path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
