"""Build a portable, single-file HTML slide deck from verified saved results.

Saved result JSONs and verified movies supply the data. A four-objective
scientific chart is drawn from saved ratios. CPU ffmpeg keeps all frames;
this builder never loads a model.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html as html_lib
import json
from pathlib import Path
import subprocess

IMAGE = "sha256:c07e78bf94c671f41392c98e5558cb49296e4187fc7682c51f74467d76539398"
SOURCE_OBJECTIVES = ["tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training"]
OBJECTIVES = SOURCE_OBJECTIVES[1:]
LABELS = {"tracking_3d": "Tracking 기하", "reconstruction_3d": "Reconstruction 기하",
          "confidence": "Confidence", "joint_training": "Joint"}


def draw_scatter(coupling, output: Path):
    """Plot only the four presented objectives; preserve original clip ratios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    points = [row for row in coupling["scatter"]["points"] if row["objective"] in OBJECTIVES]
    connections = coupling["scatter"]["connections"]
    assert len(points) == 32 and len(connections) == 8
    assert not [row for row in coupling["scatter"]["excluded"] if row["objective"] in OBJECTIVES]
    colors = dict(zip(OBJECTIVES, ["#167a9a", "#c96c36", "#8863a7", "#62773b"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.55), constrained_layout=True)
    fig.set_facecolor("#f5f3ec")
    ax.set_facecolor("#f5f3ec")
    values = [0.] + [row[key] for row in points for key in ("x", "y")]
    margin = max(5., (max(values) - min(values)) * .07)
    bounds = [min(values) - margin, max(values) + margin]
    for link in connections:
        ax.annotate("", xy=link["to"], xytext=link["from"],
                    arrowprops={"arrowstyle": "->", "color": "#677579", "alpha": .55, "lw": 1.1})
    for objective in OBJECTIVES:
        for dataset, marker in [("po_mini", "o"), ("ds_mini", "^")]:
            selected = [row for row in points if row["objective"] == objective and row["dataset"] == dataset]
            ax.scatter([row["x"] for row in selected], [row["y"] for row in selected],
                       marker=marker, color=colors[objective], s=72, edgecolor="white", linewidth=.6, zorder=3)
    ax.plot(bounds, bounds, "--", color="#a5afb0", lw=1)
    ax.axhline(0, color="#8c9699", lw=.7)
    ax.axvline(0, color="#8c9699", lw=.7)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("Tracking relative APD decrease (%)")
    ax.set_ylabel("Reconstruction relative APD decrease (%)")
    ax.grid(alpha=.15)
    handles = [Line2D([], [], marker="o", linestyle="none", color=colors[o], label=o) for o in OBJECTIVES]
    handles += [Line2D([], [], marker=m, linestyle="none", color="#58656b", label=d)
                for d, m in [("PO", "o"), ("DR", "^")]]
    ax.legend(handles=handles, loc="lower right", fontsize=11, frameon=False)
    fig.savefig(output, dpi=160, metadata={"Software": "Verified four-objective presentation plot"})
    plt.close(fig)
    return {"points": points, "connections": connections, "excluded": [],
            "point_count": 32, "connection_count": 8, "objectives": OBJECTIVES,
            "axis_definition": "100 * (clean APD - attacked APD) / clean APD per clip",
            "connection_definition": "same clip: tracking_3d -> reconstruction_3d; not time progression"}


def comparison_slides(clips, media_records, image_urls):
    sections, groups = [], []
    esc = html_lib.escape
    for clip in clips:
        for task in ("tracking", "reconstruction"):
            slide_index = 7 + len(groups)
            group_id = clip["prefix"] + "_" + task
            title = f'{clip["display_group"]} {clip["clip_index"]} · {task.title()} 결과'
            ids = [f'{clip["prefix"]}_{objective}_{task}' for objective in OBJECTIVES]
            panels = []
            for objective, key in zip(OBJECTIVES, ids):
                state = media_records[key]["selected_state"]
                caption = ('Clean | PGD' if task == "tracking" else 'GT | Clean | PGD')
                panels.append(f'''<figure data-objective="{objective}"><h3>{LABELS[objective]} <small>{objective} · step {state["step"]}</small></h3>
<p class="video-caption">{caption}<button data-video-fullscreen="{key}" aria-label="{LABELS[objective]} 영상 확대">확대</button></p><video muted playsinline preload="none" data-media="{key}" data-objective="{objective}" data-task="{task}" poster="{image_urls[key + '_poster']}" aria-label="{esc(title)} · {LABELS[objective]}"></video><img class="print-poster" src="{image_urls[key + '_poster']}" alt="{esc(title)} · {LABELS[objective]} 첫 프레임">
<div class="case-values" data-task-values data-dataset="{clip['dataset']}" data-sequence="{esc(clip['sequence'])}" data-objective="{objective}" data-task="{task}"></div></figure>''')
            note = ('좌측 Clean, 우측 PGD RGB 위에 같은 GT 선택 query와 예측 trajectory를 표시합니다. 녹색 GT, 청록색 Clean 예측, 주황색 PGD 예측이며 query 번호나 설명 문자는 영상에 넣지 않습니다. 각 조건의 저장된 평가 global median scale을 한 번 적용하여 카메라에 투영합니다. 화면 밖이나 카메라 뒤에 있는 예측점은 RGB 패널에 보이지 않지만 전체 지표와 내장 projection status에 보존됩니다.'
                    if task == "tracking" else
                    'Clean/PGD 예측에 각 조건의 저장된 평가 global median scale을 한 번 적용한 뒤, 원래 meter 좌표의 GT와 동일 중심, 카메라와 상세축에서 비교합니다. 작은 전체범위 inset은 네 공격과 clean의 모든 표시점을 포함합니다. 좌표 clamp나 예측별 중심화 없이 clean RGB 색과 같은 GT-valid stride8 subset을 사용합니다. 상세축 밖 점은 inset에서 확인합니다. 표시 subset과 전체 GT-valid metric 모집단을 구분합니다.')
            foot = ('녹색: GT · 청록색: Clean · 주황색: PGD / 동일 GT query · 화면 밖/카메라 뒤 점은 표시되지 않음'
                    if task == "tracking" else
                    'GT / Clean / PGD 순서 · 공통 GT 상세 범위 · 작은 전체범위 inset에 축 밖 점 포함 · 동일 GT-valid 표시점과 Clean RGB 색')
            sections.append(f'''<section class="slide task-results {task}-comparison" id="slide-{slide_index+1}" aria-hidden="true" data-comparison="{group_id}" data-case="{group_id}" data-dataset="{clip['dataset']}" data-sequence="{esc(clip['sequence'])}" data-task="{task}" data-title="{esc(title)}" data-notes="{esc(note)} 정량 APD/EPE는 전체128프레임의 모든 평가 query 또는 GT-valid pixel입니다. 선택 저장 상태는 최고 공격 목적값이며 마지막 step을 강제하지 않았습니다.">
<h2>{esc(title)}</h2><p class="lead">{esc(clip['sequence'])} · 같은 클립, 네 공격 목적 비교</p><div class="body"><div class="video-quad">{''.join(panels)}</div><div class="pair-controls"></div></div><p class="foot">{foot}</p></section>''')
            groups.append({"id": group_id, "prefix": clip["prefix"], "dataset": clip["dataset"],
                           "sequence": clip["sequence"], "task": task, "slide_index": slide_index, "media_ids": ids})
    return "\n".join(sections), groups


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
    parser.add_argument("--visualization-dir", type=Path, required=True,
                        help="Fresh task-paired videos rendered from saved experiment arrays")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[1] / "templates/component_presentation.html")
    args = parser.parse_args()
    report, output, visual = args.report_dir.resolve(), args.output.resolve(), args.visualization_dir.resolve()
    assert not output.is_relative_to(report), "Presentation must not modify the audited report bundle"
    data, manifest = read(report / "data/report_data.json"), read(report / "report_manifest.json")
    assert manifest["status"] == "complete" and manifest["synthetic_only"] is False
    assert data["scope"]["clips"] == 8 and data["scope"]["num_frames"] == 128
    assert data["scope"]["expected_conditions"] == 48 and data["objectives"] == SOURCE_OBJECTIVES
    visual_manifest_path = visual / "task_visualization_manifest.json"
    visual_manifest = read(visual_manifest_path)
    assert visual_manifest["status"] == "complete"
    assert visual_manifest["cpu_only"] is True and visual_manifest["new_model_inference"] is False
    assert visual_manifest["objectives"] == OBJECTIVES
    assert visual_manifest["plain_video"] is True and visual_manifest["text_overlay"] is False
    clips = visual_manifest["clip_order"]
    assert len(clips) == 8 and len({(c["dataset"], c["sequence"]) for c in clips}) == 8
    selections = {f"{clip['prefix']}_{objective}_{task}": (clip['dataset'], clip['sequence'], objective, task)
                  for clip in clips for objective in OBJECTIVES for task in ("tracking", "reconstruction")}
    media_records = visual_manifest["media"]
    assert set(media_records) == set(selections) and len(media_records) == 64
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

    generated_inputs = {}

    def generated(record):
        path = (visual / record["path"]).resolve()
        assert path.is_relative_to(visual), "Generated asset escaped visualization directory"
        actual = sha(path)
        assert actual == record["sha256"] and path.stat().st_size == record["bytes"], str(path)
        generated_inputs["task_visualization/" + record["path"]] = actual
        return path

    def image_url(path):
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    coupling_path = verified("data/task_coupling_analysis.json")
    charts = output.parent / ".build/charts"
    charts.mkdir(parents=True, exist_ok=True)
    scatter_path = charts / "four_objective_apd_scatter.png"
    scatter = draw_scatter(read(coupling_path), scatter_path)
    scatter.update({"source_data_sha256": sha(coupling_path), "asset_sha256": sha(scatter_path)})
    generated_inputs["presentation_charts/four_objective_apd_scatter.png"] = sha(scatter_path)
    image_urls = {"scatter": image_url(scatter_path)}
    image_urls["cover"] = image_url(generated(media_records[clips[0]["prefix"] + "_tracking_3d_reconstruction"]["posters"]["64"]))
    sources = {}
    for key, identity in selections.items():
        record = media_records[key]
        assert tuple(record[field] for field in ("dataset", "sequence", "objective", "task")) == identity
        assert record["source_frame_indices"] == list(range(128))
        assert record["probe"]["decoded_frames"] == 128
        assert record["probe"]["fps"] == 10 and abs(record["probe"]["duration"] - 12.8) < .02
        assert (record["probe"]["width"], record["probe"]["height"]) == ((1024, 288) if identity[3] == "tracking" else (1600, 600))
        assert record.get("source_evidence"), "Generated movie must bind immutable saved experiment sources"
        sources[key] = generated(record)
        image_urls[key + "_poster"] = image_url(generated(record["posters"]["0"]))

    def encode(key: str):
        source = sources[key]
        dest, metadata = cache / (key + ".mp4"), cache / (key + ".json")
        recipe = {"source_sha256": sha(source), "crf": 21, "preset": "veryfast", "pixel_format": "yuv420p", "resize": False}
        reusable = dest.is_file() and metadata.is_file() and read(metadata).get("recipe") == recipe
        if reusable:
            reusable = read(metadata).get("sha256") == sha(dest)
        if not reusable:
            command = docker_command("ffmpeg", [(visual, "/source", True), (cache, "/out", False)], [
                "-hide_banner", "-loglevel", "error", "-y", "-i", "/source/" + media_records[key]["path"],
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
        assert (probe["width"], probe["height"]) == ((1024, 288) if selections[key][3] == "tracking" else (1600, 600))
        result = {"id": key, "path": "task_visualization/" + media_records[key]["path"], "sha256": sha(dest), "bytes": dest.stat().st_size,
                  "recipe": recipe, "probe": {"decoded_frames": 128, "fps": 10, "duration": 12.8,
                  "width": probe["width"], "height": probe["height"], "codec": "h264"}, "cpu_only": True,
                  **dict(zip(("dataset", "sequence", "objective", "task"), selections[key])),
                  "source_render": media_records[key]}
        metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"video": key, "bytes": result["bytes"], "frames": 128}), flush=True)
        return key, result, base64.b64encode(dest.read_bytes()).decode("ascii")

    with ThreadPoolExecutor(max_workers=2) as executor:
        videos = dict((key, (proof, encoded)) for key, proof, encoded in executor.map(encode, selections))
    aggregates = [next(row for row in data["impact"] if row["dataset"] == "all" and row["objective"] == objective) for objective in OBJECTIVES]
    cases = [next(row for row in data["conditions"] if row["dataset"] == clip["dataset"] and
                  row["sequence"] == clip["sequence"] and row["objective"] == objective)
             for clip in clips for objective in OBJECTIVES]
    assert len(cases) == 32
    sections, comparison_groups = comparison_slides(clips, media_records, image_urls)
    scope = {**data["scope"], "kind": "focused_campaign", "objectives": OBJECTIVES,
             "expected_conditions": 40, "focused_expected_conditions": 40,
             "clean_conditions": 8, "attack_conditions": 32}
    payload = {"run_id": data["run_id"], "scope": scope, "source_scope": data["scope"],
               "objectives": OBJECTIVES, "clips": clips, "comparison_groups": comparison_groups,
               "scatter": scatter, "aggregates": aggregates,
               "contrasts": data["contrasts"], "cases": cases, "losses": [
                   {"id": "tracking_3d", "expression": "mean(c1_eff * e1)"},
                   {"id": "reconstruction_3d", "expression": "mean(C2 * e2)"},
                   {"id": "confidence", "expression": "-0.2 * (mean log max(c1_eff,1) + mean log C2)"},
                   {"id": "joint_training", "expression": "tracking_3d + reconstruction_3d + confidence"}],
               "sources": {"report_data_sha256": sha(report / "data/report_data.json"), "report_manifest_sha256": sha(report / "report_manifest.json"),
               "loss_structure_sha256": sha(Path(__file__).resolve().parents[1] / "docs/loss_structure.md"), "selected_assets": inputs,
               "generated_assets": generated_inputs, "visualization_manifest_sha256": sha(visual_manifest_path)},
               "media": {key: proof for key, (proof, _) in videos.items()}, "slide_count": 24,
               "visualization": visual_manifest,
               "selection_policy": "All eight selected experiment clips, in original manifest order", "standalone": True,
               "presentation_revision": 4, "visualization_revision": 4, "plain_video": True,
               "text_overlay": False, "rgb_absolute_difference_included": False}
    html = args.template.read_text(encoding="utf-8")
    for key, url in image_urls.items():
        html = html.replace("{{IMAGE_" + key + "}}", url)
    html = html.replace("{{COMPARISON_SLIDES}}", sections)
    blocks = '\n'.join(f'<script id="media-{key}" type="application/octet-stream">{encoded}</script>'
                       for key, (_, encoded) in videos.items())
    html = html.replace("{{MEDIA_BLOCKS}}", blocks)
    html = html.replace("{{PRESENTATION_DATA}}", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    assert not any(token in html for token in ("{{IMAGE_", "{{VIDEO_", "{{PRESENTATION_DATA}}", "{{COMPARISON_SLIDES}}", "{{MEDIA_BLOCKS}}"))
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output)
    proof = {"status": "built", "synthetic_only": False, "output": str(output), "sha256": sha(output), "bytes": output.stat().st_size,
             "standalone": True, "images": len(image_urls), "videos": len(videos), "slides": 24,
             "presentation_revision": 4, "visualization_revision": 4, "plain_video": True,
             "builder_sha256": sha(Path(__file__)), "template_sha256": sha(args.template), "inputs": payload["sources"], "media": payload["media"]}
    (output.parent / "presentation_manifest.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: proof[key] for key in ["status", "output", "bytes", "slides", "videos", "standalone"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
