"""Render synchronized saved RGB comparisons without loading a model.

NumPy memory maps are read-only. Docker is used solely for CPU ffmpeg encoding.
The MP4 files are compressed previews; the float32 NPY files remain authoritative.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_IMAGE = "st4rtrack-pgd:torch2.7.1-cu128"
AMPLIFICATION = 64.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_array(directory: Path, filename: str, result: dict):
    path = directory / filename
    actual_hash = sha256(path)
    recorded_hash = result["artifact_sha256"][filename]
    if actual_hash != recorded_hash:
        raise ValueError(f"Saved artifact hash mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.ndim != 4 or array.shape[1] != 3:
        raise ValueError(f"Expected float32 [T,3,H,W]: {path}, {array.shape}, {array.dtype}")
    return array, {"path": str(path.resolve()), "sha256": actual_hash}


def load_pair(sequence_dir: Path):
    clean_dir, attack_dir = sequence_dir / "clean", sequence_dir / "pgd_eps4"
    clean_result, attack_result = read_json(clean_dir / "result.json"), read_json(attack_dir / "result.json")
    if clean_result["method"] != "clean" or attack_result["method"] != "pgd":
        raise ValueError(f"Unexpected methods in {sequence_dir}")
    if attack_result["epsilon_255"] != 4 or attack_result["steps"] != 20:
        raise ValueError(f"Expected eps=4/255 and 20 steps: {sequence_dir}")
    clean, clean_source = checked_array(clean_dir, "rgb_float32.npy", clean_result)
    attacked, attacked_source = checked_array(attack_dir, "rgb_float32.npy", attack_result)
    clean_delta, clean_delta_source = checked_array(clean_dir, "delta_float32.npy", clean_result)
    attack_delta, attack_delta_source = checked_array(attack_dir, "delta_float32.npy", attack_result)
    if not (clean.shape == attacked.shape == clean_delta.shape == attack_delta.shape):
        raise ValueError(f"Input/delta shapes disagree: {sequence_dir}")
    if clean.shape[0] != 64 or clean_result["metrics"]["num_frames"] != 64 or attack_result["metrics"]["num_frames"] != 64:
        raise ValueError(f"Expected all 64 saved frames: {sequence_dir}")
    maximum_delta = 0.0
    for frame_index in range(clean.shape[0]):
        original, attack = clean[frame_index], attacked[frame_index]
        delta = attack_delta[frame_index]
        for value in (original, attack, delta, clean_delta[frame_index]):
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite saved input: {sequence_dir}, frame {frame_index}")
        if min(float(original.min()), float(attack.min())) < 0 or max(float(original.max()), float(attack.max())) > 1:
            raise ValueError(f"RGB outside [0,1]: {sequence_dir}, frame {frame_index}")
        if np.any(clean_delta[frame_index] != 0):
            raise ValueError(f"Clean delta is nonzero: {sequence_dir}, frame {frame_index}")
        if not np.array_equal(attack - original, delta):
            raise ValueError(f"Saved delta does not equal attacked minus clean: {sequence_dir}, frame {frame_index}")
        maximum_delta = max(maximum_delta, float(np.abs(delta).max()))
    if maximum_delta > 4.0 / 255.0 + 1e-7:
        raise ValueError(f"Saved attack exceeds eps=4/255: {sequence_dir}, L-infinity={maximum_delta}")
    for result in (clean_result, attack_result):
        if not result["saved_input_verification"]["passed"]:
            raise ValueError(f"Recorded saved-input replay did not pass: {sequence_dir}")
    return clean, attacked, clean_result, attack_result, {
        "clean_rgb": clean_source,
        "attacked_rgb": attacked_source,
        "clean_delta": clean_delta_source,
        "attack_delta": attack_delta_source,
        "validation": {"passed": True, "frame_count": 64, "maximum_delta_linf": maximum_delta,
                       "rgb_range": [0, 1], "saved_delta_exact": True, "recorded_model_replay_passed": True},
    }


def fonts():
    candidates = [Path("C:/Windows/Fonts/malgun.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise FileNotFoundError("A rendering font is required: Windows Malgun Gothic or DejaVu Sans")
    return str(path), {size: ImageFont.truetype(str(path), size) for size in (18, 20, 22)}


def to_rgb8(frame):
    return np.rint(np.clip(frame.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def render_frame(original, attacked, clean_metrics, attacked_metrics, dataset, sequence, frame_index, count, fps, font_set):
    _, height, width = original.shape
    margin, gap, top, bottom = 12, 12, 80, 84
    canvas_width = 3 * width + 2 * margin + 2 * gap
    canvas_height = top + height + bottom
    # yuv420p needs even dimensions; padding never changes any input panel.
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (17, 22, 31))
    draw = ImageDraw.Draw(canvas)
    foreground, muted = (239, 243, 249), (183, 195, 214)
    draw.text((margin, 8), f"{dataset} / {sequence}", font=font_set[22], fill=foreground)
    labels = ("원본", "PGD 공격 (4/255 · 20회)", "절대 차이 ×64 (확대 표시)")
    images = (to_rgb8(original), to_rgb8(attacked), to_rgb8(np.abs(attacked - original) * AMPLIFICATION))
    details = (
        f"APD {clean_metrics['apd3d_all']:.2f}%  ·  EPE {clean_metrics['epe_all_m']:.3f}m",
        f"APD {attacked_metrics['apd3d_all']:.2f}%  ·  EPE {attacked_metrics['epe_all_m']:.3f}m",
        "|공격 - 원본| ×64, 밝기 1에서 잘림",
    )
    for panel_index, (label, pixels, detail) in enumerate(zip(labels, images, details)):
        x = margin + panel_index * (width + gap)
        draw.text((x, 45), label, font=font_set[20], fill=foreground)
        canvas.paste(Image.fromarray(pixels), (x, top))
        draw.text((x, top + height + 8), detail, font=font_set[18], fill=muted)
    note = f"프레임 {frame_index + 1:02d}/{count}  ·  미리보기 {fps:g} fps ({count / fps:.1f}초)  ·  MP4는 보기용 압축 영상"
    draw.text((margin, top + height + 43), note, font=font_set[18], fill=muted)
    return canvas


def encoder_args(image, output_dir, filename, width, height, fps):
    if "," in str(output_dir):
        raise ValueError("Docker bind-mount output directory must not contain a comma")
    return [
        "docker", "run", "--rm", "-i", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=128m", "--entrypoint", "ffmpeg",
        "--mount", f"type=bind,source={output_dir},target=/output", image,
        "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"/output/{filename}",
    ]


def encode_frames(args, frames):
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        process.stdin = None
        _, stderr = process.communicate(timeout=120)
    except BaseException:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
            process.stdin = None
        process.kill()
        _, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8", errors="replace"), flush=True)
        raise
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed ({process.returncode}): {stderr.decode('utf-8', errors='replace')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 0 < args.fps <= 60:
        parser.error("--fps must be finite and between 0 and 60")
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    font_path, font_set = fonts()
    inspection = subprocess.run(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], capture_output=True, text=True, check=True)
    sequence_dirs = sorted(path.parent.parent for path in (run_dir / "sequences").glob("*/*/pgd_eps4/result.json"))
    if len(sequence_dirs) != 4:
        raise ValueError(f"Expected all four completed PGD clips, found {len(sequence_dirs)}")
    manifest = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "cpu_only": True, "new_model_inference": False,
        "preview_fps": args.fps, "preview_fps_is_original_capture_rate": False,
        "comparison_panels": ["clean", "pgd_eps4", "clip(abs(pgd-clean)*64,0,1)"],
        "difference_amplification": AMPLIFICATION, "epsilon_255": 4.0, "pgd_steps": 20,
        "font_path": font_path, "encoder": {"image": args.image, "image_id": inspection.stdout.strip(),
        "format": "mp4", "codec": "h264/libx264", "crf": 16, "pixel_format": "yuv420p",
        "note": "Compressed 8-bit previews; saved float32 NPY inputs are the authoritative experimental artifacts."},
        "clips": [],
    }
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        clean, attacked, clean_result, attack_result, sources = load_pair(sequence_dir)
        dataset, sequence = clean_result["dataset"], clean_result["sequence"]
        if (dataset, sequence) != (attack_result["dataset"], attack_result["sequence"]):
            raise ValueError(f"Clean/attack sequence identities disagree: {sequence_dir}")
        filename = f"{dataset}__{sequence}__comparison.mp4"
        preview_path = output_dir / f"{dataset}__{sequence}__first_frame.png"
        count = clean.shape[0]
        def frames():
            for frame_index in range(count):
                yield render_frame(clean[frame_index], attacked[frame_index], clean_result["metrics"],
                                   attack_result["metrics"], dataset, sequence, frame_index, count, args.fps, font_set)
        first_frame = next(frames())
        first_frame.save(preview_path)
        width, height = first_frame.size
        encode_frames(encoder_args(args.image, output_dir, filename, width, height, args.fps), frames())
        video_path = output_dir / filename
        manifest["clips"].append({
            "dataset": dataset, "sequence": sequence, "frame_count": count,
            "preview_duration_seconds": count / args.fps, "source_shape_tchw": list(clean.shape),
            "video_size_wh": [width, height], "sources": sources,
            "clean_metrics": clean_result["metrics"], "attacked_metrics": attack_result["metrics"],
            "comparison_mp4": {"path": str(video_path), "sha256": sha256(video_path), "bytes": video_path.stat().st_size},
            "first_frame_png": {"path": str(preview_path), "sha256": sha256(preview_path), "bytes": preview_path.stat().st_size},
        })
        print(f"[{index}/{len(sequence_dirs)}] {dataset}/{sequence}: {count} frames, {count / args.fps:.1f}s preview -> {video_path}", flush=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "clips": len(manifest["clips"]), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
