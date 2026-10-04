"""Build a portable, single-file HTML slide deck from verified saved results.

Only existing PNGs/MP4s and small JSONs are read. Video transcoding uses CPU
ffmpeg, keeps the full decoded sequence, and never loads a model.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

IMAGE = "sha256:c07e78bf94c671f41392c98e5558cb49296e4187fc7682c51f74467d76539398"
OBJECTIVES = ["tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training"]
PO = "comparisons/{objective}/po_mini/cab_e_3rd_13/"
DR = "comparisons/tracking_3d/ds_mini/9c43b3-3_obj_source_left_3/"
VIDEO_SELECTIONS = {
    "po_rgb": PO.format(objective="tracking_3d") + "rgb_comparison.mp4",
    "po_tracking": PO.format(objective="tracking_3d") + "tracking_comparison.mp4",
    "po_recon_tracking": PO.format(objective="reconstruction_3d") + "tracking_comparison.mp4",
    "dr_tracking": DR + "tracking_comparison.mp4",
}
IMAGE_SELECTIONS = {
    "cover": PO.format(objective="tracking_3d") + "rgb_frame_127.png",
    "scatter": "cross_task_apd_scatter.png",
    "triptych": "geometry_cases/po_mini/cab_e_3rd_13/geometry_attack_triptych_frame_064.png",
    "po_rgb_poster": PO.format(objective="tracking_3d") + "rgb_frame_000.png",
    "po_tracking_poster": PO.format(objective="tracking_3d") + "tracking_frame_000.png",
    "po_recon_tracking_poster": PO.format(objective="reconstruction_3d") + "tracking_frame_000.png",
    "dr_tracking_poster": DR + "tracking_frame_000.png",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def docker_command(entrypoint: str, mounts: list[tuple[Path, str, bool]], args: list[str]):
    command = ["docker", "run", "--rm", "--network", "none", "--read-only", "--entrypoint", entrypoint]
    for source, target, readonly in mounts:
        mount = f"type=bind,source={source.resolve()},target={target}"
        command += ["--mount", mount + (",readonly" if readonly else "")]
    return command + [IMAGE] + args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[1] / "templates/component_presentation.html")
    args = parser.parse_args()
    report, output = args.report_dir.resolve(), args.output.resolve()
    assert not output.is_relative_to(report), "Presentation must not modify the audited report bundle"
    data, manifest = read(report / "data/report_data.json"), read(report / "report_manifest.json")
    assert manifest["status"] == "complete" and manifest["synthetic_only"] is False
    assert data["scope"]["clips"] == 8 and data["scope"]["num_frames"] == 128
    assert data["scope"]["expected_conditions"] == 48 and data["objectives"] == OBJECTIVES
    inventory = manifest["outputs"]
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / ".build/media"
    cache.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def verified(relative: str) -> Path:
        path = (report / relative).resolve()
        assert path.is_relative_to(report) and relative in inventory, relative
        actual = sha(path)
        assert actual == inventory[relative]["sha256"], f"Source hash changed: {relative}"
        inputs[relative] = actual
        return path

    image_urls = {}
    for key, relative in IMAGE_SELECTIONS.items():
        path = verified("assets/" + relative)
        image_urls[key] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    sources = {key: verified("assets/" + relative) for key, relative in VIDEO_SELECTIONS.items()}

    def encode(key: str):
        source = sources[key]
        dest, metadata = cache / (key + ".mp4"), cache / (key + ".json")
        recipe = {"source_sha256": inputs["assets/" + VIDEO_SELECTIONS[key]], "crf": 21, "preset": "veryfast", "pixel_format": "yuv420p", "resize": False}
        reusable = dest.is_file() and metadata.is_file() and read(metadata).get("recipe") == recipe
        if reusable:
            reusable = read(metadata).get("sha256") == sha(dest)
        if not reusable:
            command = docker_command("ffmpeg", [(report, "/source", True), (cache, "/out", False)], [
                "-hide_banner", "-loglevel", "error", "-y", "-i", "/source/assets/" + VIDEO_SELECTIONS[key],
                "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-threads", "4",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "/out/" + dest.name,
            ])
            subprocess.run(command, check=True, capture_output=True, timeout=180)
        probe_command = docker_command("ffprobe", [(cache, "/out", True)], [
            "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
            "stream=nb_read_frames,r_frame_rate,width,height,codec_name,duration", "-of", "json", "/out/" + dest.name,
        ])
        probe = json.loads(subprocess.run(probe_command, check=True, capture_output=True, text=True, timeout=90).stdout)["streams"][0]
        numerator, denominator = map(int, probe["r_frame_rate"].split("/"))
        assert int(probe["nb_read_frames"]) == 128 and numerator / denominator == 10
        assert abs(float(probe["duration"]) - 12.8) < .02 and probe["codec_name"] == "h264"
        result = {"id": key, "path": "assets/" + VIDEO_SELECTIONS[key], "sha256": sha(dest), "bytes": dest.stat().st_size,
                  "recipe": recipe, "probe": {"decoded_frames": 128, "fps": 10, "duration": 12.8,
                  "width": probe["width"], "height": probe["height"], "codec": "h264"}, "cpu_only": True}
        metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"video": key, "bytes": result["bytes"], "frames": 128}), flush=True)
        return key, result, base64.b64encode(dest.read_bytes()).decode("ascii")

    with ThreadPoolExecutor(max_workers=2) as executor:
        videos = dict((key, (proof, encoded)) for key, proof, encoded in executor.map(encode, VIDEO_SELECTIONS))
    aggregates = [next(row for row in data["impact"] if row["dataset"] == "all" and row["objective"] == objective) for objective in OBJECTIVES]
    cases = [row for row in data["conditions"] if (row["dataset"], row["sequence"], row["objective"]) in [
        ("po_mini", "cab_e_3rd_13", "tracking_3d"), ("po_mini", "cab_e_3rd_13", "reconstruction_3d"),
        ("ds_mini", "9c43b3-3_obj_source_left_3", "tracking_3d")]]
    payload = {"run_id": data["run_id"], "scope": data["scope"], "aggregates": aggregates,
               "contrasts": data["contrasts"], "cases": cases, "losses": [
                   {"id": "tracking_mse", "expression": "mean ||s_med P1 - G1||^2"},
                   {"id": "tracking_3d", "expression": "mean(c1_eff * e1)"},
                   {"id": "reconstruction_3d", "expression": "mean(C2 * e2)"},
                   {"id": "confidence", "expression": "-0.2 * (mean log max(c1_eff,1) + mean log C2)"},
                   {"id": "joint_training", "expression": "tracking_3d + reconstruction_3d + confidence"}],
               "sources": {"report_data_sha256": sha(report / "data/report_data.json"), "report_manifest_sha256": sha(report / "report_manifest.json"),
               "loss_structure_sha256": sha(Path(__file__).resolve().parents[1] / "docs/loss_structure.md"), "selected_assets": inputs},
               "media": {key: proof for key, (proof, _) in videos.items()}, "slide_count": 12,
               "selection_policy": "First manifest PO and DR clips, selected without using error magnitude", "standalone": True}
    html = args.template.read_text(encoding="utf-8")
    for key, url in image_urls.items():
        html = html.replace("{{IMAGE_" + key + "}}", url)
    for key, (_, encoded) in videos.items():
        html = html.replace("{{VIDEO_" + key + "}}", encoded)
    html = html.replace("{{PRESENTATION_DATA}}", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    assert "{{IMAGE_" not in html and "{{VIDEO_" not in html and "{{PRESENTATION_DATA}}" not in html
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output)
    proof = {"status": "built", "synthetic_only": False, "output": str(output), "sha256": sha(output), "bytes": output.stat().st_size,
             "standalone": True, "images": len(image_urls), "videos": len(videos), "slides": 12,
             "builder_sha256": sha(Path(__file__)), "template_sha256": sha(args.template), "inputs": payload["sources"], "media": payload["media"]}
    (output.parent / "presentation_manifest.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: proof[key] for key in ["status", "output", "bytes", "slides", "videos", "standalone"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
