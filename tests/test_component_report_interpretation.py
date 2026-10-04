"""Synthetic interpretation arithmetic only; no real arrays or inference.

Renderer summary helpers receive generated Python scalar lists. No renderer,
Docker, encoder, browser, checkpoint, or real experiment is executed.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from statistics import fmean
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import component_report_data as DATA
import component_report_interpretation as INTERPRET
from render_component_comparisons import summarize_normalization, summarize_tracking_timeline

SPEC = importlib.util.spec_from_file_location("interpretation_scalar_fixture", Path(__file__).with_name("test_component_report_data.py"))
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def visual_fixture(study):
    visual = {"schema_version": 1, "synthetic_only": True, "status": "complete", "errors": [], "cpu_only": True,
              "new_model_inference": False, "frame_count": 128, "run_signature": study["metadata"]["signature"],
              "scope": deepcopy(study["analysis"]["scope"]), "analysis_json_sha256": study["analysis_sha256"],
              "input_scalar_file_sha256": deepcopy(study["input_sha256"]), "clips": [], "timelines": []}
    for dataset, sequence in study["identities"]:
        clean = study["records"][dataset, sequence, "clean"]
        visual["timelines"].append({"dataset": dataset, "sequence": sequence, "frame_count": 128, "synthetic_only": True})
        for oi, objective in enumerate(DATA.OBJECTIVES):
            attack = study["records"][dataset, sequence, objective]
            clean_epe, attack_epe = ([r["tracking_metrics"]["epe_all_m"]] * 128 for r in (clean, attack))
            norm_clean = [float(1 + i // 32) for i in range(128)]
            norm_attack = [v * (.8 + oi * .1) for v in norm_clean]
            visual["clips"].append({"dataset": dataset, "sequence": sequence, "objective": objective, "status": "complete",
                "synthetic_only": True, "frame_count": 128, "source_frame_indices": list(range(128)),
                "clean_epe_per_frame_m": clean_epe, "attack_epe_per_frame_m": attack_epe,
                "tracking_timeline_summary": summarize_tracking_timeline(clean_epe, attack_epe, clean_epe[0], attack_epe[0], 7),
                "normalization_summary": summarize_normalization(norm_clean, norm_attack, [2.] * 128)})
    return visual


@pytest.fixture
def inputs(tmp_path):
    root = FIXTURE.make_fixture(tmp_path / "SYNTHETIC_INTERPRETATION_FULL128")
    study = DATA.load_verified_study(root)
    for (d, s, o), record in study["records"].items():
        record["perturbation_norms"] = {"linf": 0. if o == "clean" else 4 / 255}
    return study, DATA.compute_task_coupling(study), visual_fixture(study)


def refresh(study):
    pairs = study["analysis"]["paired_conditions"]
    for aggregate in study["analysis"]["paired_aggregates"]:
        rows = [p for p in pairs if p["objective"] == aggregate["objective"]
                and (aggregate["dataset"] == "all" or aggregate["dataset"] == p["dataset"])]
        for field in (*DATA.METRICS, *DATA.DIAGNOSTIC_FIELDS):
            for suffix in ("", "_clean", "_attacked"):
                aggregate[field + suffix] = fmean(r[field + suffix] for r in rows)
    for ci in study["analysis"]["bootstrap"]["estimates"]:
        aggregate = next(r for r in study["analysis"]["paired_aggregates"]
                         if (r["dataset"], r["objective"]) == (ci["dataset"], ci["objective"]))
        ci["estimate"] = aggregate[ci["metric"]]
        ci["ci95"] = [ci["estimate"] - 100, ci["estimate"] + 100]
    return study, DATA.compute_task_coupling(study), visual_fixture(study)


def set_attack(study, dataset, sequence, objective, metric, delta):
    pair = next(p for p in study["analysis"]["paired_conditions"]
                if (p["dataset"], p["sequence"], p["objective"]) == (dataset, sequence, objective))
    section, field, convention, _ = DATA.CHANGE_FIELDS[metric]
    pair[metric] = delta
    pair[metric + "_attacked"] = pair[metric + "_clean"] + convention * delta
    study["records"][dataset, sequence, objective][section][field] = pair[metric + "_attacked"]


def set_clean(study, dataset, sequence, metric, value):
    section, field, convention, _ = DATA.CHANGE_FIELDS[metric]
    study["records"][dataset, sequence, "clean"][section][field] = value
    for pair in study["analysis"]["paired_conditions"]:
        if (pair["dataset"], pair["sequence"]) == (dataset, sequence):
            pair[metric + "_clean"] = value
            pair[metric] = convention * (pair[metric + "_attacked"] - value)


def text(result):
    return json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_complete_json_scope_and_readonly_pure_function(inputs, monkeypatch):
    before = deepcopy(inputs)
    def no_io(*args, **kwargs):
        raise AssertionError("Interpretation attempted file I/O")
    monkeypatch.setattr(Path, "open", no_io)
    result = INTERPRET.build_interpretation(*inputs)
    assert inputs == before
    assert result["synthetic_only"] is True
    assert len(result["impact_by_group"]["all"]) == 5
    assert len(result["dataset_comparison"]) == 5
    assert len(result["consistency_by_group"]["all"]) == 20
    assert len(result["timeline_by_clip"]) == 8 and len(result["cases"]) == 40
    assert all(len(t["conditions"]) == 5 for t in result["timeline_by_clip"])
    assert result["scope"]["expected_conditions"] == 48
    assert "per_frame" not in result["cases"][0]["normalization_summary"]
    assert "values" not in result["cases"][0]["normalization_summary"]["framewise_ratio"]
    assert text(result)


def test_every_group_objective_describes_two_tasks_and_four_metrics(inputs):
    result = INTERPRET.build_interpretation(*inputs)
    for group in DATA.GROUPS:
        assert len(result["impact_by_group"][group]) == 5
        for objective, message in zip(DATA.OBJECTIVES, result["impact_by_group"][group]):
            assert objective in message and "Tracking APD" in message and "Reconstruction APD" in message
            assert message.count("EPE") >= 2 and message.count("paired 95% CI") == 4
            assert "clean−attack" in message and "attack−clean" in message
    assert all("PO−DR 기술 차분" in m and "독립 검정은 계산하지 않았습니다" in m for m in result["dataset_comparison"])


def test_negative_improvement_and_aggregate_ratio_definition(inputs):
    study, coupling, visual = inputs
    result = INTERPRET.build_interpretation(*inputs)
    row = next(r for r in coupling["impact_matrix"] if r["dataset"] == "all" and r["objective"] == "confidence")
    average_ratio = 100 * row[DATA.METRICS[0]] / row[DATA.METRICS[0] + "_clean"]
    clip_ratio_mean = fmean(100 * p[DATA.METRICS[0]] / p[DATA.METRICS[0] + "_clean"]
                          for p in study["analysis"]["paired_conditions"] if p["objective"] == "confidence")
    assert abs(average_ratio - clip_ratio_mean) > .01
    message = result["impact_by_group"]["all"][3]
    assert "개선" in message and "-3.00%p" in message
    assert f"상대 감소 {average_ratio:.2f}%" in message
    assert "not mean clip ratios" in result["mean_definition"]


def test_nearzero_apd_omits_connection_and_never_replaces_na_with_zero(inputs):
    study, _, _ = inputs
    set_clean(study, "po_mini", "clip_0", DATA.METRICS[0], 0.)
    result = INTERPRET.build_interpretation(*refresh(study))
    assert result["movement_by_group"]["all"]["excluded_clips"] == 1
    assert result["movement_by_group"]["po_mini"]["valid_clips"] == 3
    assert all(c["sequence"] != "clip_0" or c["dataset"] != "po_mini" for c in result["movement_by_group"]["all"]["clips"])


def test_zero_epe_denominator_stays_na_in_each_group(inputs):
    study, _, _ = inputs
    for dataset, sequence in study["identities"]:
        set_clean(study, dataset, sequence, DATA.METRICS[2], 0.)
    result = INTERPRET.build_interpretation(*refresh(study))
    assert all("N/A배" in m for messages in result["impact_by_group"].values() for m in messages)


def test_exact_signed_consistency_and_extreme_ids_not_rounded(inputs):
    study, _, _ = inputs
    for dataset, sequence in study["identities"]:
        set_attack(study, dataset, sequence, "tracking_mse", DATA.METRICS[0], [-1., 0., 2., 4.][int(sequence[-1])])
    result = INTERPRET.build_interpretation(*refresh(study))
    row = result["consistency_by_group"]["all"][0]
    assert row["counters"] == {"worsened": 4, "unchanged": 2, "improved": 2}
    assert row["median"] == 1 and row["min"] == -1 and row["max"] == 4
    assert row["min_clips"] == ["po_mini/clip_0", "ds_mini/clip_0"]
    assert row["max_clips"] == ["po_mini/clip_3", "ds_mini/clip_3"]
    assert "결과 기반 탐색" in row["summary"]
    for dataset, sequence in study["identities"]:
        set_attack(study, dataset, sequence, "tracking_mse", DATA.METRICS[0], 1e-10)
    result = INTERPRET.build_interpretation(*refresh(study))
    assert result["consistency_by_group"]["all"][0]["counters"]["worsened"] == 8


def test_direct_ci_zero_inclusion_and_both_improved_semantics(inputs):
    study, _, _ = inputs
    for dataset, sequence in study["identities"]:
        for objective, deltas in (("tracking_3d", (-2., -1., -.01, -.01)), ("reconstruction_3d", (-4., -3., -.03, -.02))):
            for metric, delta in zip(DATA.METRICS, deltas):
                set_attack(study, dataset, sequence, objective, metric, delta)
    study, coupling, visual = refresh(study)
    coupling["geometry_contrasts"]["estimates"][0]["ci95"] = [-1., 3.]
    result = INTERPRET.build_interpretation(study, coupling, visual)
    assert "둘 다 clean보다 개선" in result["contrasts_by_group"]["all"][0]
    assert "A의 개선폭이 더 작습니다" in result["contrasts_by_group"]["all"][0]
    assert "CI가 0을 포함하므로" in result["contrasts_by_group"]["all"][0]
    assert any("A가 두 과제" in m for m in result["contrasts_by_group"]["all"])


def test_apd_epe_sign_conflict_is_explained_not_hidden(inputs):
    study, _, _ = inputs
    for dataset, sequence in study["identities"]:
        for objective, deltas in (("tracking_3d", (6., -1., -.01, .06)), ("reconstruction_3d", (2., 4., .05, .03))):
            for metric, delta in zip(DATA.METRICS, deltas):
                set_attack(study, dataset, sequence, objective, metric, delta)
    result = INTERPRET.build_interpretation(*refresh(study))
    assert "APD와 EPE의 변화 방향이 다르므로" in result["impact_by_group"]["all"][1]
    assert any("APD와 EPE 직접 차이의 부호 양상이 다릅니다" in m for m in result["contrasts_by_group"]["all"])


def test_arrow_to_minus_from_and_strata_are_preserved(inputs):
    result = INTERPRET.build_interpretation(*inputs)
    all_rows = result["movement_by_group"]["all"]["clips"]
    assert len(all_rows) == 8
    for row in all_rows:
        assert row["delta_x_pct"] == row["to"][0] - row["from"][0]
        assert row["delta_y_pct"] == row["to"][1] - row["from"][1]
        assert row["from_objective"] == "tracking_3d" and row["to_objective"] == "reconstruction_3d"
    assert len(result["movement_by_group"]["po_mini"]["clips"]) == len(result["movement_by_group"]["ds_mini"]["clips"]) == 4
    assert sum(result["movement_by_group"]["all"]["direction_counts"].values()) == 8


def test_selected_clean_initial_early_and_final_counts_and_budget(inputs):
    study, _, _ = inputs
    for objective, state in zip(DATA.OBJECTIVES[:3], ({"restart": -1, "step": 0}, {"restart": 0, "step": 0}, {"restart": 0, "step": 7})):
        study["records"]["po_mini", "clip_0", objective]["selected_state"] = state
    result = INTERPRET.build_interpretation(*inputs)["input_budget"]
    assert result["max_linf_255"] == 4 and result["within_budget"] == result["attacks"] == 40
    assert (result["selected_clean_count"], result["selected_initial_count"], result["selected_early_count"], result["selected_final_count"]) == (1, 1, 1, 37)


def test_normalization_ratio_mean_distinction_and_na_frames(inputs):
    study, coupling, visual = inputs
    for clip in visual["clips"]:
        c = [1e-14] * 32 + [1.] * 32 + [2.] * 32 + [8.] * 32
        a = [1.] * 128
        clip["normalization_summary"] = summarize_normalization(c, a, [2.] * 128)
    result = INTERPRET.build_interpretation(*inputs)
    norm = result["cases"][0]["normalization_summary"]
    assert norm["framewise_ratio"]["excluded_frames"] == 32
    assert norm["framewise_ratio"]["valid_frames"] == 96
    assert norm["ratio_of_means"] != norm["framewise_ratio"]["mean"]
    assert "raw pre-alignment" in result["cases"][0]["summary"][2]
    assert "frame별 비율 평균" in result["cases"][0]["summary"][2]


def test_all_na_norm_ratios_remain_null(inputs):
    for clip in inputs[2]["clips"]:
        clip["normalization_summary"] = summarize_normalization([1e-14] * 128, [1.] * 128, [2.] * 128)
    result = INTERPRET.build_interpretation(*inputs)
    norm = result["cases"][0]["normalization_summary"]
    assert norm["ratio_of_means"] is None and norm["framewise_ratio"]["median"] is None
    assert "N/A" in result["cases"][0]["summary"][2]
    assert "NaN" not in text(result) and "Infinity" not in text(result)


def test_shared_clean_timeline_cannot_change_even_when_whole_clip_mean_matches(inputs):
    clip = inputs[2]["clips"][1]
    clean = clip["clean_epe_per_frame_m"]
    clean[0], clean[1] = clean[0] + .01, clean[1] - .01
    clip["tracking_timeline_summary"] = summarize_tracking_timeline(clean, clip["attack_epe_per_frame_m"],
                                                                    fmean(clean), fmean(clip["attack_epe_per_frame_m"]), 7)
    with pytest.raises(DATA.ReportValidationError, match="same clean EPE timeline"):
        INTERPRET.build_interpretation(*inputs)


def test_native_confidence_change_can_disagree_with_benchmark_improvement(inputs):
    result = INTERPRET.build_interpretation(*inputs)
    message = result["diagnostics_by_group"]["all"][3]
    assert "Tracking native L21은 증가, benchmark EPE는 감소" in message
    assert "Reconstruction native L21은 증가, benchmark EPE는 감소" in message


@pytest.mark.parametrize("mutation", [
    lambda s, c, v: s.update(complete_verified=False),
    lambda s, c, v: s["analysis"]["scope"].update(num_frames=64),
    lambda s, c, v: s["resume"].update(status="running"),
    lambda s, c, v: s["campaign"].update(status="paused"),
    lambda s, c, v: v.update(new_model_inference=True),
    lambda s, c, v: v.update(run_signature="stale"),
    lambda s, c, v: c.update(analysis_sha256="stale"),
    lambda s, c, v: v["clips"].pop(),
    lambda s, c, v: v["clips"][0].update(frame_count=64),
    lambda s, c, v: v["clips"][0]["tracking_timeline_summary"].update(frame_count=64),
    lambda s, c, v: v["clips"][0].update(clean_epe_per_frame_m=[.2] * 64),
    lambda s, c, v: v["clips"][0]["normalization_summary"].update(frame_count=64),
    lambda s, c, v: v["clips"][0]["normalization_summary"]["framewise_ratio"].update(excluded_frames=1),
    lambda s, c, v: v["clips"][0]["tracking_timeline_summary"].update(increased_frames=64),
    lambda s, c, v: v["clips"][0]["tracking_timeline_summary"]["fixed_segments"][0].update(source_frame_end=63),
    lambda s, c, v: v["clips"][0]["normalization_summary"]["units"].update(prediction_scale="meters"),
    lambda s, c, v: c["geometry_contrasts"].update(direction="B minus A"),
    lambda s, c, v: c["geometry_contrasts"]["estimates"][0].update(unit="relative_percent"),
    lambda s, c, v: s["records"]["po_mini", "clip_0", "tracking_mse"]["perturbation_norms"].update(linf=.2),
])
def test_incomplete_stale_or_inconsistent_export_is_refused(inputs, mutation):
    mutation(*inputs)
    with pytest.raises(DATA.ReportValidationError):
        INTERPRET.build_interpretation(*inputs)


def test_scientific_limits_and_native_confidence_explanation(inputs):
    result = INTERPRET.build_interpretation(*inputs)
    diagnostics = result["diagnostics_by_group"]["all"][3]
    assert "raw tracking confidence" in diagnostics and "effective tracking confidence" in diagnostics
    assert "비가중 tracking L21" in diagnostics and "Tracking EPE 증가" in diagnostics
    combined = text(result)
    for invalid_claim in ("학습 loss 중요도가 입증", "가장 중요한 학습 loss", "통계적으로 확정", "전체 데이터셋에서도", "confidence 확률"):
        assert invalid_claim not in combined
    assert "인과적 중요도" in combined and "검증하지 않았습니다" in combined
    assert "독립 통계 표본 수가 아닙니다" in combined
    assert "목적값 단위가 다르므로 raw loss" in combined
