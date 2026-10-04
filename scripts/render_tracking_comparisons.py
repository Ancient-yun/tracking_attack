"""Render saved clean/PGD 3D tracks on the corresponding saved RGB, using CPU only.

Both views show the same GT-only spatially selected query IDs. Predictions are
scale-aligned as in evaluation and projected with the ground-truth camera.
This is a diagnostic projection, not a new model prediction of image tracks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_pgd_comparisons import DEFAULT_IMAGE, encode_frames, encoder_args, fonts, load_pair, sha256, to_rgb8
from tracking_projection import load_projected_tracks


GT = (96, 245, 140)
CLEAN = (44, 221, 255)
ATTACK = (255, 173, 59)
FG = (240, 244, 251)
MUTED = (174, 188, 209)
BG = (16, 22, 32)
PANEL_WIDTH = 768
PANEL_HEIGHT = 432
MARGIN = 20
GAP = 24
IMAGE_TOP = 112
CANVAS_WIDTH = 2 * PANEL_WIDTH + GAP + 2 * MARGIN
CANVAS_HEIGHT = 1000
TAIL = 8


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def farthest_points(candidates, query_xy, limit, already=()):
    """Deterministic spatial coverage based only on initial query coordinates."""
    candidates = np.asarray(sorted(set(int(x) for x in candidates)), dtype=np.int64)
    selected = list(already)
    chosen = []
    if not len(candidates) or limit <= 0:
        return chosen
    # The initial central query is deterministic; later points maximize distance
    # to the previously selected points (including the other GT category).
    center = np.array([256.0, 144.0], dtype=np.float64)
    while len(candidates) and len(chosen) < limit:
        points = np.asarray(query_xy[candidates], dtype=np.float64)
        if selected:
            anchors = np.asarray(query_xy[selected], dtype=np.float64)
            distances = ((points[:, None] - anchors[None]) ** 2).sum(axis=-1).min(axis=1)
            candidate_index = int(np.argmax(distances))
        else:
            candidate_index = int(np.argmin(((points - center) ** 2).sum(axis=-1)))
        point_id = int(candidates[candidate_index])
        chosen.append(point_id)
        selected.append(point_id)
        candidates = np.delete(candidates, candidate_index)
    return chosen


def select_queries(projected, count=18):
    query_xy = np.asarray(projected["query_xy"])
    dynamic = np.asarray(projected["dynamic"], dtype=bool)
    eligible = (np.asarray(projected["visibility"])[0].astype(bool)
                & np.asarray(projected["projection_valid"]["gt"])[0].astype(bool)
                & np.isfinite(query_xy).all(axis=-1)
                & (query_xy[:, 0] >= 0) & (query_xy[:, 0] < 512)
                & (query_xy[:, 1] >= 0) & (query_xy[:, 1] < 288))
    dynamic_ids = np.flatnonzero(eligible & dynamic)
    static_ids = np.flatnonzero(eligible & ~dynamic)
    selected = farthest_points(dynamic_ids, query_xy, min(12, count))
    selected.extend(farthest_points(static_ids, query_xy, min(6, count - len(selected)), selected))
    remaining = np.flatnonzero(eligible & ~np.isin(np.arange(len(eligible)), selected))
    selected.extend(farthest_points(remaining, query_xy, count - len(selected), selected))
    if len(selected) < 2:
        raise ValueError("Insufficient initially visible GT queries for comparison")
    return np.asarray(selected, dtype=np.int64), {
        "selection_uses_model_errors": False,
        "selection_uses_attack_results": False,
        "policy": "initially GT-visible, finite on-screen queries; up to 12 dynamic and 6 static; deterministic farthest-point spatial coverage",
        "dynamic_definition": "saved evaluation dynamic mask",
        "eligible_count": int(eligible.sum()),
        "selected_count": len(selected),
        "selected_dynamic_count": int(dynamic[selected].sum()),
        "display_number_to_saved_query_index": {f"{i + 1:02d}": int(point_id) for i, point_id in enumerate(selected)},
        "selected_initial_xy": query_xy[selected].tolist(),
    }


def clipped_segment(start, end, width=PANEL_WIDTH, height=PANEL_HEIGHT):
    """Liang-Barsky clipping: a segment may cross the view, a point is never clamped."""
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return None
    x0, y0 = map(float, start)
    x1, y1 = map(float, end)
    dx, dy = x1 - x0, y1 - y0
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width - 1 - x0), (-dy, y0), (dy, height - 1 - y0)):
        if p == 0:
            if q < 0:
                return None
        else:
            ratio = q / p
            if p < 0:
                lo = max(lo, ratio)
            else:
                hi = min(hi, ratio)
            if lo > hi:
                return None
    return (x0 + lo * dx, y0 + lo * dy), (x0 + hi * dx, y0 + hi * dy)


def line(draw, start, end, color, width=2):
    segment = clipped_segment(start, end)
    if segment is not None:
        draw.line(segment, fill=color, width=width)


def mask_for(projected, key, kind, frame_index, selected):
    if key in projected:
        return np.asarray(projected[key][kind])[frame_index, selected].astype(bool)
    xy = np.asarray(projected[f"{kind}_xy"])[frame_index, selected]
    valid = np.asarray(projected["projection_valid"][kind])[frame_index, selected].astype(bool)
    if key == "in_frame":
        return valid & np.isfinite(xy).all(axis=-1) & (xy[:, 0] >= 0) & (xy[:, 0] < 512) & (xy[:, 1] >= 0) & (xy[:, 1] < 288)
    if key == "behind_camera":
        return ~valid & np.isfinite(xy).all(axis=-1)
    raise KeyError(key)


def dim(color, factor=0.52):
    return tuple(int(value * factor) for value in color)


def overlay_panel(rgb_frame, projected, kind, frame_index, selected, font_set):
    panel = Image.fromarray(to_rgb8(rgb_frame)).resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    factor = PANEL_WIDTH / rgb_frame.shape[2]
    pred_color = CLEAN if kind == "clean" else ATTACK
    gt_xy = np.asarray(projected["gt_xy"])
    pred_xy = np.asarray(projected[f"{kind}_xy"])
    gt_valid = np.asarray(projected["projection_valid"]["gt"], dtype=bool)
    pred_valid = np.asarray(projected["projection_valid"][kind], dtype=bool)
    visibility = np.asarray(projected["visibility"], dtype=bool)
    gt_inside = mask_for(projected, "in_frame", "gt", frame_index, selected)
    pred_inside = mask_for(projected, "in_frame", kind, frame_index, selected)
    behind = mask_for(projected, "behind_camera", kind, frame_index, selected)
    invalid = ~np.isfinite(pred_xy[frame_index, selected]).all(axis=-1) & ~behind
    out = pred_valid[frame_index, selected] & ~pred_inside
    for slot, query in enumerate(selected):
        # Trails are current plus the previous seven frame positions, using the
        # same eight-frame window for GT and prediction in both panels.
        for previous in range(max(0, frame_index - TAIL + 1), frame_index):
            if gt_valid[previous, query] and gt_valid[previous + 1, query]:
                color = dim(GT, 0.75 if visibility[previous, query] and visibility[previous + 1, query] else 0.30)
                line(draw, gt_xy[previous, query] * factor, gt_xy[previous + 1, query] * factor, color)
            if pred_valid[previous, query] and pred_valid[previous + 1, query]:
                line(draw, pred_xy[previous, query] * factor, pred_xy[previous + 1, query] * factor, dim(pred_color, 0.65))
        gt_point, pred_point = gt_xy[frame_index, query] * factor, pred_xy[frame_index, query] * factor
        if gt_valid[frame_index, query] and pred_valid[frame_index, query]:
            line(draw, gt_point, pred_point, dim(pred_color, 0.70), width=1)
        if gt_inside[slot]:
            x, y = map(float, gt_point)
            is_visible = visibility[frame_index, query]
            color = GT if is_visible else dim(GT, 0.42)
            if is_visible:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(5, 20, 8), width=1)
            else:
                draw.line((x - 3, y - 3, x + 3, y + 3), fill=color, width=2)
                draw.line((x - 3, y + 3, x + 3, y - 3), fill=color, width=2)
            label = f"{slot + 1:02d}"
            # Only the label box is kept readable inside the image. Neither GT
            # nor predicted point positions are altered to fit the image.
            label_x = min(max(x + 7, 2), PANEL_WIDTH - 26)
            label_y = min(max(y - 19, 1), PANEL_HEIGHT - 20)
            draw.text((label_x, label_y), label, font=font_set[16], fill=GT, stroke_width=2, stroke_fill=(8, 15, 12))
        if pred_inside[slot]:
            x, y = map(float, pred_point)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(8, 15, 20), width=4)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=pred_color, width=2)
            if not gt_inside[slot] or np.linalg.norm(pred_point - gt_point) > 12:
                label_x = min(max(x + 7, 2), PANEL_WIDTH - 26)
                label_y = min(max(y + 3, 1), PANEL_HEIGHT - 20)
                draw.text((label_x, label_y), f"{slot + 1:02d}", font=font_set[16], fill=pred_color,
                          stroke_width=2, stroke_fill=(8, 15, 20))
    return panel, {
        "gt_visible": int(visibility[frame_index, selected].sum()),
        "gt_occluded": int((~visibility[frame_index, selected]).sum()),
        "pred_in_frame": int(pred_inside.sum()),
        "pred_out_of_frame": int(out.sum()),
        "pred_behind_camera": int(behind.sum()),
        "pred_nonfinite": int(invalid.sum()),
        "out_display_numbers": [f"{i + 1:02d}" for i in np.flatnonzero(out)],
        "behind_display_numbers": [f"{i + 1:02d}" for i in np.flatnonzero(behind)],
        "nonfinite_display_numbers": [f"{i + 1:02d}" for i in np.flatnonzero(invalid)],
    }


def epe_series(projected, kind):
    key = f"{kind}_epe_per_frame_m"
    if key in projected:
        return np.asarray(projected[key], dtype=np.float64)
    return np.asarray(projected[f"{kind}_epe_per_frame"], dtype=np.float64)


def chart(draw, clean_epe, attack_epe, frame_index, font_set, num_queries):
    chart_left, chart_right = 95, CANVAS_WIDTH - 40
    chart_top, chart_bottom = 790, 931
    max_y = max(float(clean_epe.max()), float(attack_epe.max()), 0.1) * 1.10
    draw.text((MARGIN, 744), f"전체 {num_queries}개 추적점의 프레임별 3D 오차 EPE (m) · 평가와 같은 스케일 정렬", font=font_set[20], fill=FG)
    count = len(clean_epe)
    def coordinates(values):
        x = np.linspace(chart_left, chart_right, count)
        y = chart_bottom - np.asarray(values) / max_y * (chart_bottom - chart_top)
        return [(float(xx), float(yy)) for xx, yy in zip(x, y)]
    for tick in (0.0, max_y / 2, max_y):
        y = chart_bottom - tick / max_y * (chart_bottom - chart_top)
        draw.line((chart_left, y, chart_right, y), fill=(46, 57, 73), width=1)
        draw.text((MARGIN, y - 11), f"{tick:.1f}", font=font_set[18], fill=MUTED)
    for number in (1, 16, 32, 48, count):
        x = chart_left + (number - 1) / (count - 1) * (chart_right - chart_left)
        draw.text((x - 10, chart_bottom + 6), str(number), font=font_set[16], fill=MUTED)
    clean_points, attack_points = coordinates(clean_epe), coordinates(attack_epe)
    draw.line(clean_points, fill=CLEAN, width=3)
    draw.line(attack_points, fill=ATTACK, width=3)
    cursor_x = clean_points[frame_index][0]
    draw.line((cursor_x, chart_top, cursor_x, chart_bottom), fill=(209, 216, 226), width=1)
    for points, color in ((clean_points, CLEAN), (attack_points, ATTACK)):
        x, y = points[frame_index]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=BG, width=1)
    draw.text((CANVAS_WIDTH - 300, 744), "청록 원본   주황 PGD", font=font_set[18], fill=MUTED)


def render_frame(clean_frame, attack_frame, projected, selected, dataset, sequence, frame_index, fps, font_set):
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    count = len(projected["gt_xy"])
    clean_result, attack_result = projected["clean_result"], projected["attack_result"]
    draw.text((MARGIN, 10), f"{dataset} / {sequence}  ·  프레임 {frame_index + 1:02d}/{count}", font=font_set[22], fill=FG)
    draw.text((MARGIN, 44), "원본 RGB + 원본 추적", font=font_set[22], fill=CLEAN)
    draw.text((MARGIN + PANEL_WIDTH + GAP, 44), "PGD RGB + 공격 후 추적 (4/255 · 20회)", font=font_set[22], fill=ATTACK)
    for x, result, color in ((MARGIN, clean_result, CLEAN), (MARGIN + PANEL_WIDTH + GAP, attack_result, ATTACK)):
        metrics = result["metrics"]
        draw.text((x, 79), f"전체 APD {metrics['apd3d_all']:.2f}%   ·   전체 EPE {metrics['epe_all_m']:.3f} m", font=font_set[20], fill=color)
    status_per_kind = {}
    for kind, frame, x, color in (("clean", clean_frame, MARGIN, CLEAN), ("attack", attack_frame, MARGIN + PANEL_WIDTH + GAP, ATTACK)):
        panel, status = overlay_panel(frame, projected, kind, frame_index, selected, font_set)
        canvas.paste(panel, (x, IMAGE_TOP))
        status_per_kind[kind] = status
        current_epe = epe_series(projected, kind)[frame_index]
        draw.text((x, 554), f"현재 3D EPE {current_epe:.3f} m  ·  표시 예측 화면 안 {status['pred_in_frame']}/{len(selected)}", font=font_set[20], fill=color)
        draw.text((x, 588), f"화면 밖 OUT {status['pred_out_of_frame']}  ·  카메라 뒤 BEHIND {status['pred_behind_camera']}  ·  GT 가림 {status['gt_occluded']}", font=font_set[18], fill=MUTED)
        labels = []
        for label, key in (("OUT", "out_display_numbers"), ("BEHIND", "behind_display_numbers"), ("NONFINITE", "nonfinite_display_numbers")):
            if status[key]:
                labels.append(f"{label}: {','.join(status[key])}")
        draw.text((x, 619), "   ".join(labels) if labels else "모든 표시 예측이 화면 안에 있습니다.", font=font_set[16], fill=MUTED)
    draw.text((MARGIN, 660), "● 초록: GT   × 어두운 초록: 가려진 GT   ○ 청록/주황: 모델 예측   연결선: GT와 예측 차이   꼬리: 최근 8프레임", font=font_set[18], fill=FG)
    draw.text((MARGIN, 690), f"양쪽 동일한 {len(selected)}개 점 / 동일 번호 · GT 기준 공간 분산 선택 · 3D 예측을 정답 카메라에 투영", font=font_set[18], fill=MUTED)
    chart(draw, epe_series(projected, "clean"), epe_series(projected, "attack"), frame_index, font_set, np.asarray(projected["gt_xy"]).shape[1])
    draw.text((MARGIN, 967), f"모델 입력 512×288을 1.5배 표시 · {fps:g} fps 미리보기 ({count / fps:.2f}초) · GT 카메라 투영 / 실제 촬영 속도 아님", font=font_set[18], fill=MUTED)
    return canvas, status_per_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--frames-only", action="store_true", help="Render QA PNGs and a preview manifest without starting an encoder")
    parser.add_argument("--clip", help="Optional dataset/sequence identifier for layout QA")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 0 < args.fps <= 60:
        parser.error("--fps must be finite and between 0 and 60")
    run_dir, project_root = args.run_dir.resolve(), args.project_root.resolve()
    output_dir = run_dir / "tracking_visualizations"
    output_dir.mkdir(exist_ok=True)
    font_path, font_set = fonts()
    font_set.update({size: ImageFont.truetype(font_path, size) for size in (16,)})
    sequence_dirs = sorted(path.parent.parent for path in (run_dir / "sequences").glob("*/*/pgd_eps4/result.json"))
    if len(sequence_dirs) != 4:
        raise ValueError(f"Expected all four completed PGD clips, found {len(sequence_dirs)}")
    if args.clip:
        sequence_dirs = [path for path in sequence_dirs if "/".join(path.parts[-2:]) == args.clip]
        if not sequence_dirs:
            parser.error(f"Unknown --clip {args.clip}")
    image_id = None
    if not args.frames_only:
        inspection = subprocess.run(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], capture_output=True, text=True, check=True)
        image_id = inspection.stdout.strip()
    manifest = {
        "status": "rendering", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "cpu_only": True, "new_model_inference": False,
        "preview_fps": args.fps, "preview_fps_is_original_capture_rate": False,
        "comparison_panels": ["clean RGB + clean prediction + GT", "PGD RGB + PGD prediction + GT"],
        "difference_amplification": None, "projection_camera": "ground-truth per-frame camera",
        "projection_note": "Diagnostic projection of scale-aligned saved 3D predictions; not model-estimated 2D tracks or camera pose.",
        "selection_uses_model_errors": False, "trailing_positions": TAIL,
        "chart": "all saved queries, per-frame 3D EPE in meters, linear y axis shared by clean and PGD",
        "occlusion": "GT occluded locations are dark green crosses, visible GT locations are green dots; saved evaluator includes all times",
        "offscreen": "Markers are not clamped; crossing segments are clipped to the viewport; OUT/BEHIND query IDs and counts are shown",
        "font_path": font_path,
        "encoder": {"image": args.image, "image_id": image_id, "format": "mp4", "codec": "h264/libx264", "crf": 16, "pixel_format": "yuv420p"},
        "clips": [],
    }
    manifest_path = output_dir / ("preview_manifest.json" if args.frames_only else "manifest.json")
    dump_json(manifest_path, manifest)
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        clean, attacked, clean_result, attack_result, sources = load_pair(sequence_dir)
        if list(clean.shape) != [64, 3, 288, 512]:
            raise ValueError(f"Expected actual 64x3x288x512 model inputs, got {clean.shape}")
        projected = load_projected_tracks(sequence_dir, project_root)
        dataset, sequence = clean_result["dataset"], clean_result["sequence"]
        selected, selection = select_queries(projected)
        base = f"{dataset}__{sequence}__tracking"
        selection_path = output_dir / f"{base}__selection.json"
        projection_path = output_dir / f"{base}__projection_checks.json"
        dump_json(selection_path, selection)
        dump_json(projection_path, {"sources": projected["sources"], "projection_checks": projected["projection_checks"]})
        frame_pngs, status_by_frame = [], []
        def frames():
            for frame_index in range(64):
                frame, status = render_frame(clean[frame_index], attacked[frame_index], projected, selected, dataset, sequence, frame_index, args.fps, font_set)
                if len(status_by_frame) <= frame_index:
                    status_by_frame.append({"frame_index": frame_index, **status})
                yield frame
        for frame_index in (0, 32, 63):
            frame, _ = render_frame(clean[frame_index], attacked[frame_index], projected, selected, dataset, sequence, frame_index, args.fps, font_set)
            preview_path = output_dir / f"{base}__frame_{frame_index:03d}.png"
            frame.save(preview_path)
            frame_pngs.append({"frame_index": frame_index, "path": str(preview_path), "sha256": sha256(preview_path), "bytes": preview_path.stat().st_size})
        video_path = output_dir / f"{base}.mp4"
        if not args.frames_only:
            encode_frames(encoder_args(args.image, output_dir, video_path.name, CANVAS_WIDTH, CANVAS_HEIGHT, args.fps), frames())
        clip_entry = {
            "dataset": dataset, "sequence": sequence, "frame_count": 64,
            "preview_duration_seconds": 64 / args.fps, "source_shape_tchw": list(clean.shape),
            "video_size_wh": [CANVAS_WIDTH, CANVAS_HEIGHT], "sources": sources,
            "track_sources": projected["sources"], "projection_checks": projected["projection_checks"],
            "selection": selection, "selection_path": str(selection_path), "projection_checks_path": str(projection_path),
            "clean_metrics": clean_result["metrics"], "attacked_metrics": attack_result["metrics"],
            "clean_epe_per_frame_m": epe_series(projected, "clean").tolist(),
            "attack_epe_per_frame_m": epe_series(projected, "attack").tolist(),
            "frame_pngs": frame_pngs, "display_status_per_frame": status_by_frame,
        }
        if not args.frames_only:
            clip_entry["comparison_mp4"] = {"path": str(video_path), "sha256": sha256(video_path), "bytes": video_path.stat().st_size}
        manifest["clips"].append(clip_entry)
        dump_json(manifest_path, manifest)
        print(f"[{index}/{len(sequence_dirs)}] {dataset}/{sequence}: {len(selected)} matched queries, 64 frames -> {video_path if not args.frames_only else 'QA PNGs'}", flush=True)
    manifest["status"] = "preview_completed" if args.frames_only else "completed"
    dump_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "clips": len(manifest["clips"]), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
