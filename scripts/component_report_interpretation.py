"""Describe verified PGD outcomes using scalar evidence only.

No files, saved arrays, models, or networks are opened here. The caller must
first use load_verified_study and validate the complete renderer manifest.
Numbers are descriptive observations on the eight selected clips; bootstrap
intervals retain their original clip-level interpretation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from statistics import fmean, median

from component_report_data import (DATASETS, DIAGNOSTIC_FIELDS, GROUPS, METRICS,
                                   OBJECTIVES, ReportValidationError, safe_ratio)


LABELS = {"all": "전체 8클립", "po_mini": "PO 4클립", "ds_mini": "DR 4클립"}
METRIC_LABELS = {
    METRICS[0]: "Tracking APD 감소", METRICS[1]: "Reconstruction APD 감소",
    METRICS[2]: "Tracking EPE 증가", METRICS[3]: "Reconstruction EPE 증가",
}
UNITS = {m: "%p" if "apd" in m else "m" for m in METRICS}
LIMITS = [
    "선정 개발 8클립·동일 checkpoint·단일 base attack seed·restart 1의 고정 모델 입력 공격 민감성 비교입니다.",
    "모델 재학습, 학습 loss 제거의 인과적 중요도, training weight 우열, 전체 데이터셋 일반화, 전역 최악 공격은 검증하지 않았습니다.",
    "CI는 PO/DR 내부 clip 복원 추출의 조건부 변동이며 scene 독립성·scene cluster·attack seed/model/GT 불확실성·다중 비교를 보정하지 않습니다.",
    "동일 8클립의 반복 공격 40조건과 128프레임·dense pixel은 독립 통계 표본 수가 아닙니다.",
    "Native L21은 무차원, confidence는 확률이 아닌 모델 score입니다. 공유 입력·표현·head2 normalization 변화의 관련성은 개별 요소의 독립 인과 효과를 증명하지 않습니다.",
    "서로 다른 과제의 모집단·정렬과 목적값 단위가 다르므로 raw loss 크기로 공격 강도나 과제 연결 계수를 비교하지 않습니다.",
]


def require(condition, message):
    if not condition:
        raise ReportValidationError(message)


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"Invalid interpretation scalar: {label}")
    return float(value)


def same(first, second, label):
    require(math.isclose(number(first, label), number(second, label), rel_tol=1e-6, abs_tol=1e-6),
            f"Interpretation scalar mismatch: {label}")


def index(rows, fields, label):
    require(isinstance(rows, list), f"Missing {label}")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and all(k in row for k in fields), f"Malformed {label}")
        key = tuple(row[k] for k in fields)
        require(key not in result, f"Duplicate {label}: {key}")
        result[key] = row
    return result


def fmt(value, digits=2, signed=False):
    return "N/A" if value is None else format(value, f"+.{digits}f" if signed else f".{digits}f")


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def damage(value):
    return "손상" if value > 0 else "개선" if value < 0 else "변화 없음"


def ci_text(ci, digits, unit):
    require(isinstance(ci, list) and len(ci) == 2, "Missing interpretation CI")
    lo, hi = (number(v, "CI endpoint") for v in ci)
    require(lo <= hi, "Reversed interpretation CI")
    return f"paired 95% CI {fmt(lo, digits, True)}~{fmt(hi, digits, True)}{unit}"


def ratio_fields(row):
    result = {}
    for family in ("tracking", "reconstruction"):
        apd = family + "_apd_drop_pp"
        value = safe_ratio(row[apd], row[apd + "_clean"])
        result[family + "_apd_relative_drop_pct"] = None if value is None else 100 * value
        epe = family + "_epe_increase_m"
        result[family + "_epe_ratio"] = safe_ratio(row[epe + "_attacked"], row[epe + "_clean"])
    return result


def _validate(study, coupling, visual):
    require(study.get("complete_verified") is True, "Interpretation requires strict completed-study verification")
    analysis, metadata = study["analysis"], study["metadata"]
    scope = analysis["scope"]
    require(scope.get("kind") == "full_campaign" and scope.get("clips") == 8
            and scope.get("dataset_counts") == {d: 4 for d in DATASETS}
            and scope.get("num_frames") == 128 and scope.get("steps") == 20
            and scope.get("epsilon_255") == 4 and scope.get("expected_conditions") == 48
            and scope.get("all_frames") is True and scope.get("objectives") == list(OBJECTIVES),
            "Interpretation scope must be all 48 conditions / PO4+DR4 / 128 frames / PGD20 / epsilon4")
    require(analysis.get("status") in ("complete", "complete_with_recorded_failures")
            and study["campaign"].get("status") == study["resume"].get("status") == "complete",
            "Both campaign controllers must be complete")
    require(analysis["validation"].get("completed_conditions") == 48
            and analysis["validation"].get("raw_frame_verified_clips") == 8
            and analysis["validation"].get("errors") == [] and analysis["validation"].get("missing_conditions") == []
            and study["audit"].get("passed") is True and study["audit"].get("source_files_verified") is True,
            "Interpretation needs complete audit and scalar-validation evidence")
    identities = [tuple(v) for v in study["identities"]]
    require(len(identities) == len(set(identities)) == 8
            and Counter(d for d, _ in identities) == {d: 4 for d in DATASETS}, "Unbalanced/duplicate interpretation clips")
    expected = {(d, s, o) for d, s in identities for o in OBJECTIVES}
    records = study["records"]
    require(set(records) == expected | {(d, s, "clean") for d, s in identities}, "Interpretation needs exactly 48 original results")
    for key, record in records.items():
        require(record["tracking_metrics"].get("num_frames") == record["reconstruction_metrics"].get("num_frames") == 128,
                f"Original interpretation metric is not 128 frames: {key}")
    pairs = index(analysis["paired_conditions"], ("dataset", "sequence", "objective"), "paired conditions")
    require(set(pairs) == expected, "Interpretation needs 40 unique paired attacks")
    impacts = index(coupling["impact_matrix"], ("dataset", "objective"), "impact matrix")
    require(set(impacts) == {(g, o) for g in GROUPS for o in OBJECTIVES}, "Interpretation needs 15 group/objective impacts")
    require(coupling.get("run_signature") == visual.get("run_signature") == metadata["signature"]
            and coupling.get("analysis_sha256") == visual.get("analysis_json_sha256") == study["analysis_sha256"]
            and coupling.get("scope") == visual.get("scope") == scope
            and coupling.get("input_scalar_sha256") == visual.get("input_scalar_file_sha256") == study["input_sha256"],
            "Interpretation inputs do not bind the same verified scalar study")
    require(visual.get("status") == "complete" and visual.get("errors") == []
            and visual.get("cpu_only") is True and visual.get("new_model_inference") is False
            and visual.get("frame_count") == 128, "Interpretation requires completed CPU-only 128-frame renderer evidence")
    clips = index(visual["clips"], ("dataset", "sequence", "objective"), "rendered clips")
    require(set(clips) == expected, "Interpretation needs all 40 rendered clips")
    for key, clip in clips.items():
        require(clip.get("status") == "complete" and clip.get("frame_count") == 128
                and clip.get("source_frame_indices") == list(range(128)), f"Rendered interpretation clip is incomplete: {key}")
    for (group, objective), row in impacts.items():
        children = [p for (d, _, o), p in pairs.items() if o == objective and (group == "all" or d == group)]
        require(row.get("paired_clips") == len(children) == (8 if group == "all" else 4), "Interpretation group support differs")
        for field in (*METRICS, *DIAGNOSTIC_FIELDS):
            for suffix in ("", "_clean", "_attacked"):
                same(row[field + suffix], fmean(number(p[field + suffix], field) for p in children), field)
        for metric in METRICS:
            ci_text(row["ci95"][metric], 2, UNITS[metric])
        for field, ratio in ratio_fields(row).items():
            require(row.get(field) is None if ratio is None else row.get(field) is not None,
                    f"Interpretation ratio availability differs: {field}")
            if ratio is not None:
                same(row[field], ratio, field)
    return identities, pairs, impacts, clips


def _impact(group, row):
    text = []
    for family in ("tracking", "reconstruction"):
        apd, epe = family + "_apd_drop_pp", family + "_epe_increase_m"
        text.append(f"{family.capitalize()} APD {fmt(row[apd + '_clean'])}→{fmt(row[apd + '_attacked'])}% "
                    f"(clean−attack {fmt(row[apd], 2, True)}%p, {damage(row[apd])}; "
                    f"상대 감소 {fmt(row[family + '_apd_relative_drop_pct'])}%; {ci_text(row['ci95'][apd], 2, '%p')}). "
                    f"EPE {fmt(row[epe + '_clean'], 3)}→{fmt(row[epe + '_attacked'], 3)}m "
                    f"(attack−clean {fmt(row[epe], 3, True)}m, {damage(row[epe])}; "
                    f"{fmt(row[family + '_epe_ratio'])}배; {ci_text(row['ci95'][epe], 3, 'm')}).")
        if sign(row[apd]) * sign(row[epe]) == -1:
            text.append(f"{family.capitalize()}은 APD와 EPE의 변화 방향이 다르므로 두 지표를 함께 해석합니다.")
    return f"{LABELS[group]} · {row['objective']}: " + " ".join(text)


def _dataset_comparison(impacts):
    rows = []
    for objective in OBJECTIVES:
        po, dr = impacts["po_mini", objective], impacts["ds_mini", objective]
        details = []
        for metric in METRICS:
            digits, unit = (2, "%p") if "apd" in metric else (3, "m")
            details.append(f"{METRIC_LABELS[metric]} PO {fmt(po[metric], digits, True)}{unit} "
                           f"({ci_text(po['ci95'][metric], digits, unit)}), DR {fmt(dr[metric], digits, True)}{unit} "
                           f"({ci_text(dr['ci95'][metric], digits, unit)}); PO−DR 기술 차분 "
                           f"{fmt(po[metric] - dr[metric], digits, True)}{unit}")
        rows.append(objective + ": " + "; ".join(details) + ". Dataset 간 차이에 대한 별도 CI나 독립 검정은 계산하지 않았습니다.")
    return rows


def _relative_comparison(a, b):
    if a == b:
        return "두 공격의 clean 대비 변화가 같습니다"
    leader = "A" if a > b else "B"
    if a < 0 and b < 0:
        return f"둘 다 clean보다 개선됐으며 {leader}의 개선폭이 더 작습니다"
    if min(a, b) < 0:
        return f"한 공격은 개선됐으며 {leader}의 clean 대비 변화가 손상 방향으로 더 큽니다"
    return f"{leader}의 clean 대비 손상이 더 큽니다"


def _contrasts(coupling, impacts):
    require(coupling["geometry_contrasts"].get("direction") == "tracking_3d - reconstruction_3d",
            "Direct contrast direction must stay A minus B")
    rows = index(coupling["geometry_contrasts"]["estimates"], ("dataset", "metric"), "direct contrasts")
    require(set(rows) == {(g, m) for g in GROUPS for m in METRICS}, "All twelve direct contrasts are required")
    result = {}
    for group in GROUPS:
        a, b = impacts[group, "tracking_3d"], impacts[group, "reconstruction_3d"]
        messages = []
        for metric in METRICS:
            row = rows[group, metric]
            require(row.get("paired_clips") == row.get("common_clips") == (8 if group == "all" else 4)
                    and row.get("unit") == ("percentage_points" if "apd" in metric else "metres"), "Direct contrast support/unit differs")
            same(row["estimate"], a[metric] - b[metric], "direct contrast")
            digits, unit = (2, "%p") if "apd" in metric else (3, "m")
            ci = row["ci95"]
            uncertainty = "CI가 0을 포함하므로 상대 차이는 불확실합니다" if ci[0] <= 0 <= ci[1] else "CI가 0을 포함하지 않는 관측 차이입니다"
            messages.append(f"{LABELS[group]} · {METRIC_LABELS[metric]}: A clean 대비 {fmt(a[metric], digits, True)}{unit} "
                            f"({damage(a[metric])}), B {fmt(b[metric], digits, True)}{unit} ({damage(b[metric])}). "
                            f"A−B {fmt(row['estimate'], digits, True)}{unit} ({ci_text(ci, digits, unit)}). "
                            f"{_relative_comparison(a[metric], b[metric])}. {uncertainty}.")
        for family, metrics in (("APD", METRICS[:2]), ("EPE", METRICS[2:])):
            signs = tuple(sign(rows[group, m]["estimate"]) for m in metrics)
            pattern = {(1, -1): "각 목적이 자기 과제 지표에서 상대 변화가 더 큰 양상",
                       (1, 1): "A가 두 과제 지표 모두에서 상대 변화가 더 큰 양상",
                       (-1, -1): "B가 두 과제 지표 모두에서 상대 변화가 더 큰 양상",
                       (-1, 1): "상대 과제 목적이 해당 과제 지표에서 변화가 더 큰 양상"}.get(signs, "한 지표 이상에서 차이 점 추정이 0인 양상")
            uncertain = any(rows[group, m]["ci95"][0] <= 0 <= rows[group, m]["ci95"][1] for m in metrics)
            messages.append(f"{LABELS[group]} · {family} 두 과제 점 추정: {pattern}. "
                            + ("한 지표 이상 CI가 0을 포함하므로 선택성의 차이는 불확실합니다. " if uncertain else "")
                            + "개선폭을 포함한 상대 비교이며, 위의 각 공격 clean 대비 값을 함께 읽습니다.")
        apd_signs = tuple(sign(rows[group, m]["estimate"]) for m in METRICS[:2])
        epe_signs = tuple(sign(rows[group, m]["estimate"]) for m in METRICS[2:])
        if apd_signs != epe_signs:
            messages.append(f"{LABELS[group]}에서는 APD와 EPE 직접 차이의 부호 양상이 다릅니다. 한 지표만으로 목적별 과제 선택성을 단정하지 않습니다.")
        result[group] = messages
    return result


def _consistency(identities, pairs):
    result = {}
    for group in GROUPS:
        rows = []
        for objective in OBJECTIVES:
            children = [(d, s, pairs[d, s, objective]) for d, s in identities if group == "all" or d == group]
            for metric in METRICS:
                values = [number(p[metric], metric) for _, _, p in children]
                counts = {"worsened": sum(v > 0 for v in values), "unchanged": sum(v == 0 for v in values),
                          "improved": sum(v < 0 for v in values)}
                minimum, maximum = min(values), max(values)
                min_ids = [f"{d}/{s}" for d, s, p in children if p[metric] == minimum]
                max_ids = [f"{d}/{s}" for d, s, p in children if p[metric] == maximum]
                digits = 2 if "apd" in metric else 3
                rows.append({"dataset": group, "objective": objective, "metric": metric, "unit": UNITS[metric],
                             "paired_clips": len(values), "counters": counts, "mean": fmean(values), "median": median(values),
                             "min": minimum, "max": maximum, "min_clips": min_ids, "max_clips": max_ids,
                             "summary": f"{objective} · {METRIC_LABELS[metric]}: 손상 {counts['worsened']}/{len(values)}, "
                                        f"개선 {counts['improved']}/{len(values)}, 동일 {counts['unchanged']}/{len(values)}클립. "
                                        f"중앙값 {fmt(median(values), digits, True)}{UNITS[metric]}, "
                                        f"범위 {fmt(minimum, digits, True)}~{fmt(maximum, digits, True)}{UNITS[metric]}. "
                                        "최대·최소 사례는 결과 기반 탐색이며 전체 클립 분포와 함께 제시합니다."})
        result[group] = rows
    return result


def _movement(identities, pairs, coupling):
    connections = index(coupling["scatter"]["connections"], ("dataset", "sequence"), "scatter connections")
    result = {}
    for group in GROUPS:
        clips, excluded = [], 0
        for dataset, sequence in identities:
            if group != "all" and dataset != group:
                continue
            a, b = (ratio_fields(pairs[dataset, sequence, o]) for o in ("tracking_3d", "reconstruction_3d"))
            xy = [a["tracking_apd_relative_drop_pct"], a["reconstruction_apd_relative_drop_pct"],
                  b["tracking_apd_relative_drop_pct"], b["reconstruction_apd_relative_drop_pct"]]
            if any(v is None for v in xy):
                require((dataset, sequence) not in connections, "N/A clip unexpectedly has a scatter connection")
                excluded += 1
                continue
            require((dataset, sequence) in connections, "Valid paired clip is missing its scatter connection")
            connection = connections[dataset, sequence]
            for actual, expected in zip(connection["from"] + connection["to"], xy):
                same(actual, expected, "scatter connection")
            dx, dy = xy[2] - xy[0], xy[3] - xy[1]
            def direction(value, family):
                return f"{family} 상대 APD 손상 " + ("증가" if value > 0 else "감소" if value < 0 else "동일")
            clips.append({"dataset": dataset, "sequence": sequence, "from_objective": "tracking_3d",
                          "to_objective": "reconstruction_3d", "from": xy[:2], "to": xy[2:],
                          "delta_x_pct": dx, "delta_y_pct": dy,
                          "summary": f"{dataset}/{sequence}: 목적 A→B에서 Δx {fmt(dx, 2, True)}%, Δy {fmt(dy, 2, True)}%; "
                                     f"{direction(dx, 'tracking')}, {direction(dy, 'reconstruction')}. "
                                     "PGD 진행이나 누적 공격을 뜻하지 않습니다."})
        n = len(clips)
        counts = {"tracking_down_reconstruction_up": sum(c["delta_x_pct"] < 0 < c["delta_y_pct"] for c in clips),
                  "both_up": sum(c["delta_x_pct"] > 0 and c["delta_y_pct"] > 0 for c in clips),
                  "both_down": sum(c["delta_x_pct"] < 0 and c["delta_y_pct"] < 0 for c in clips),
                  "tracking_up_reconstruction_down": sum(c["delta_x_pct"] > 0 > c["delta_y_pct"] for c in clips),
                  "axis_equal": sum(c["delta_x_pct"] == 0 or c["delta_y_pct"] == 0 for c in clips)}
        result[group] = {"clips": clips, "excluded_clips": excluded, "valid_clips": n, "direction_counts": counts,
                         "summary": [f"{LABELS[group]} · A→B 이동을 계산한 동일 클립 {n}개, clean APD 분모 N/A 제외 {excluded}개. "
                                     f"Tracking 손상 감소/reconstruction 손상 증가 {counts['tracking_down_reconstruction_up']}/{n}, "
                                     f"양쪽 증가 {counts['both_up']}/{n}, 양쪽 감소 {counts['both_down']}/{n}, "
                                     f"반대 이동 {counts['tracking_up_reconstruction_down']}/{n}, 한 축 이상 동일 {counts['axis_equal']}/{n}. "
                                     "클립별 상대 감소율 좌표의 기술적 비교이며 과제별 물리적 손상 동등성·독립 40표본을 뜻하지 않습니다."]}
    return result


def _diagnostics(impacts):
    result = {}
    labels = {"tracking_native_l21_change": "비가중 tracking L21", "reconstruction_native_l21_change": "비가중 reconstruction L21",
              "track_raw_conf_mean_change": "raw tracking confidence", "track_conf_mean_change": "effective tracking confidence",
              "reconstruction_conf_mean_change": "reconstruction confidence"}
    for group in GROUPS:
        messages = []
        for objective in OBJECTIVES:
            row = impacts[group, objective]
            text = [f"{label} {fmt(row[field + '_clean'], 3)}→{fmt(row[field + '_attacked'], 3)} "
                    f"(attack−clean {fmt(row[field], 3, True)})" for field, label in labels.items()]
            benchmark = [f"{METRIC_LABELS[m]} {fmt(row[m], 2 if 'apd' in m else 3, True)}{UNITS[m]} ({damage(row[m])})" for m in METRICS]
            associations = []
            for family in ("tracking", "reconstruction"):
                native = row[family + "_native_l21_change"]
                native_action = "증가" if native > 0 else "감소" if native < 0 else "동일"
                epe = row[family + "_epe_increase_m"]
                epe_action = "증가" if epe > 0 else "감소" if epe < 0 else "동일"
                associations.append(f"{family.capitalize()} native L21은 {native_action}, benchmark EPE는 {epe_action}로 관찰됐습니다")
            messages.append(f"{LABELS[group]} · {objective}: " + "; ".join(text) + ". 같은 조건의 benchmark는 "
                            + "; ".join(benchmark) + ". " + "; ".join(associations)
                            + ". Native는 무차원, confidence는 score이며 m 또는 확률이 아닙니다.")
        messages.append("위 변화는 저장된 같은 조건의 동반 관찰입니다. Confidence가 weighted 목적값과 관련되더라도 native 기하 오차·benchmark 손상과 동일한 값이 아니며, raw loss 절댓값이나 gradient norm으로 목적의 중요도를 순위화하지 않습니다.")
        result[group] = messages
    return result


def _input_budget(identities, records):
    norms, states = [], Counter()
    for dataset, sequence in identities:
        for objective in OBJECTIVES:
            record = records[dataset, sequence, objective]
            linf = number(record["perturbation_norms"]["linf"], "perturbation L-infinity")
            require(linf >= 0, "Negative perturbation norm")
            norms.append(linf)
            selected = record["selected_state"]
            restart, step = selected["restart"], selected["step"]
            require(type(restart) is int and type(step) is int and (restart == -1 and step == 0 or restart == 0 and 0 <= step <= 20),
                    "Invalid selected attack state")
            states["clean" if restart == -1 else "initialization" if step == 0 else "final" if step == 20 else "early"] += 1
    limit, tolerance = 4 / 255, 1e-7
    within = sum(v <= limit + tolerance for v in norms)
    require(within == 40, "Perturbation exceeds verified input budget")
    maximum = max(norms)
    return {"max_linf": maximum, "max_linf_255": maximum * 255, "epsilon_linf": limit, "validation_atol": tolerance,
            "within_budget": within, "attacks": 40, "selected_clean_count": states["clean"],
            "selected_initial_count": states["initialization"], "selected_early_count": states["early"],
            "selected_final_count": states["final"], "selected_state_counts": dict(states),
            "summary": [f"전체40공격 최대 L∞는 {maximum:.8f} (×255 {maximum * 255:.6f})이며 ε4/255 검증 허용오차 1e−7 내 {within}/40조건입니다.",
                        f"저장 상태: clean {states['clean']}, random 초기 상태 {states['initialization']}, 중간 iterate {states['early']}, 마지막 step20 {states['final']}조건. "
                        "저장 상태는 clean 포함 해당 공격 목적값의 최고 상태이며 최저 APD·최고 benchmark EPE로 선택하지 않았습니다."]}


def build_interpretation(study, coupling, visual_manifest):
    """Return JSON-serializable Korean descriptions; never read or mutate inputs."""
    try:
        identities, pairs, impacts, clips = _validate(study, coupling, visual_manifest)
        timeline, cases = _visual_summaries(study, visual_manifest, clips)
        return {"schema_version": 1, "synthetic_only": study["analysis"].get("synthetic_only", False) is True,
                "run_signature": study["metadata"]["signature"], "analysis_sha256": study["analysis_sha256"],
                "scope": deepcopy(study["analysis"]["scope"]), "input_scalar_sha256": deepcopy(study["input_sha256"]),
                "mean_definition": "equal clip weights; all=0.5*PO+0.5*DR; aggregate ratios use mean clean/attack, not mean clip ratios",
                "impact_by_group": {g: [_impact(g, impacts[g, o]) for o in OBJECTIVES] for g in GROUPS},
                "dataset_comparison": _dataset_comparison(impacts), "contrasts_by_group": _contrasts(coupling, impacts),
                "consistency_by_group": _consistency(identities, pairs), "movement_by_group": _movement(identities, pairs, coupling),
                "diagnostics_by_group": _diagnostics(impacts), "input_budget": _input_budget(identities, study["records"]),
                "timeline_by_clip": timeline, "cases": cases, "limits": list(LIMITS)}
    except ReportValidationError:
        raise
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as exc:
        raise ReportValidationError(f"Malformed verified interpretation input: {exc}") from exc


def _values(values, label, *, positive=False):
    require(isinstance(values, list) and len(values) == 128, f"{label} must contain all 128 source frames")
    result = [number(v, label) for v in values]
    require(all(v > 0 if positive else v >= 0 for v in result), f"Invalid magnitude in {label}")
    return result


def _stats(values):
    return {"mean": fmean(values), "median": median(values), "min": min(values), "max": max(values)}


def _check_stats(actual, values, label):
    for field, expected in _stats(values).items():
        same(actual[field], expected, label + "/" + field)


def _normalization_summary(summary):
    require(summary.get("schema_version") == 1 and summary.get("passed") is True and summary.get("frame_count") == 128,
            "Normalization summary must verify 128 frames")
    units = summary.get("units", {})
    require(isinstance(units.get("prediction_scale"), str)
            and "pre-alignment pointmap coordinate magnitude" in units["prediction_scale"].lower()
            and units.get("fixed_gt_scale") == "GT meter-coordinate magnitude" and units.get("ratio") == "dimensionless",
            "Normalization units must distinguish raw prediction scale and meter GT coordinates")
    per_frame = summary["per_frame"]
    clean, attack, gt = (_values(per_frame[k], "normalization " + k, positive=True) for k in ("clean", "attack", "fixed_gt"))
    for key, values in (("clean", clean), ("attack", attack), ("fixed_gt", gt)):
        _check_stats(summary[key], values, "normalization " + key)
    expected = [safe_ratio(a, c) for c, a in zip(clean, attack)]
    ratios = summary["framewise_ratio"]
    require(isinstance(ratios["values"], list) and len(ratios["values"]) == 128, "Normalization ratios must cover 128 frames")
    for actual, value in zip(ratios["values"], expected):
        require((actual is None) == (value is None), "Normalization N/A ratio availability differs")
        if value is not None:
            same(actual, value, "framewise normalization ratio")
    valid = [v for v in expected if v is not None]
    require(ratios["valid_frames"] == len(valid) and ratios["excluded_frames"] == 128 - len(valid),
            "Normalization denominator support differs")
    if valid:
        _check_stats(ratios, valid, "framewise normalization ratio")
    else:
        require(all(ratios[k] is None for k in ("mean", "median", "min", "max")), "Empty ratios must stay N/A")
    mean_ratio = safe_ratio(fmean(attack), fmean(clean))
    require((summary["ratio_of_means"] is None) == (mean_ratio is None), "Normalization ratio of means availability differs")
    if mean_ratio is not None:
        same(summary["ratio_of_means"], mean_ratio, "normalization ratio of means")
    result = deepcopy({k: v for k, v in summary.items() if k != "per_frame"})
    result["framewise_ratio"].pop("values", None)
    return result, gt


def _tracking_summary(summary, clip, clean_record, attack_record):
    require(summary.get("schema_version") == 1 and summary.get("passed") is True and summary.get("frame_count") == 128,
            "Tracking timeline summary must verify 128 frames")
    q = summary.get("query_count")
    require(type(q) is int and q > 0 and summary.get("evaluation_query_times") == 128 * q
            and summary.get("all_evaluation_query_times") is True, "Timeline must describe all evaluation queries and 128 frames")
    clean = _values(clip["clean_epe_per_frame_m"], "clean tracking timeline")
    attack = _values(clip["attack_epe_per_frame_m"], "attack tracking timeline")
    delta = [a - c for c, a in zip(clean, attack)]
    for field, expected in (("clean_epe_mean_m", fmean(clean)), ("attack_epe_mean_m", fmean(attack)),
                            ("delta_epe_mean_m", fmean(delta)), ("delta_epe_min_m", min(delta)), ("delta_epe_max_m", max(delta))):
        same(summary[field], expected, "timeline " + field)
    same(fmean(clean), clean_record["tracking_metrics"]["epe_all_m"], "clean whole-clip timeline EPE")
    same(fmean(attack), attack_record["tracking_metrics"]["epe_all_m"], "attack whole-clip timeline EPE")
    for field, count in (("increased_frames", sum(v > 0 for v in delta)), ("equal_frames", sum(v == 0 for v in delta)),
                         ("improved_frames", sum(v < 0 for v in delta))):
        require(summary[field] == count, "Timeline signed frame count differs")
    segments = summary["fixed_segments"]
    require(isinstance(segments, list) and len(segments) == 4, "Four pre-fixed 32-frame temporal segments required")
    for start, segment in zip((0, 32, 64, 96), segments):
        require(segment.get("source_frame_start") == start and segment.get("source_frame_end") == start + 31
                and segment.get("frame_count") == 32, "Timeline segment must preserve source indices and 32 frames")
        for field, values in (("clean_epe_mean_m", clean), ("attack_epe_mean_m", attack), ("delta_epe_mean_m", delta)):
            same(segment[field], fmean(values[start:start + 32]), "timeline segment " + field)
    result = deepcopy(summary)
    result["minimum_delta_source_frames"] = [i for i, value in enumerate(delta) if value == min(delta)]
    result["maximum_delta_source_frames"] = [i for i, value in enumerate(delta) if value == max(delta)]
    return result


def _visual_summaries(study, visual, clips):
    """Cross-check small scalar exports, then omit their raw 128-value lists."""
    timelines = index(visual["timelines"], ("dataset", "sequence"), "timeline clips")
    identities = [tuple(v) for v in study["identities"]]
    require(set(timelines) == set(identities) and all(r.get("frame_count") == 128 for r in timelines.values()),
            "Eight full 128-frame timeline clips are required")
    result, cases = [], []
    for dataset, sequence in identities:
        conditions, fixed_gt, clean_norm, clean_epe, query_count = [], None, None, None, None
        clean_record = study["records"][dataset, sequence, "clean"]
        for objective in OBJECTIVES:
            clip = clips[dataset, sequence, objective]
            record = study["records"][dataset, sequence, objective]
            tracking = _tracking_summary(clip["tracking_timeline_summary"], clip, clean_record, record)
            norm, current_gt = _normalization_summary(clip["normalization_summary"])
            current_clean = clip["normalization_summary"]["per_frame"]["clean"]
            if fixed_gt is None:
                fixed_gt, clean_norm = current_gt, current_clean
                clean_epe, query_count = clip["clean_epe_per_frame_m"], tracking["query_count"]
            else:
                require(current_gt == fixed_gt and current_clean == clean_norm,
                        "All objectives must share unchanged GT and clean normalization magnitudes")
                require(clip["clean_epe_per_frame_m"] == clean_epe and tracking["query_count"] == query_count,
                        "All objectives must share the same clean EPE timeline and evaluation queries")
            parts = [f"frame{r['source_frame_start']}~{r['source_frame_end']}: ΔEPE "
                     f"{fmt(r['delta_epe_mean_m'], 3, True)}m" for r in tracking["fixed_segments"]]
            temporal = [f"{dataset}/{sequence} · {objective}: 전체128프레임·{tracking['query_count']}개 평가 query의 "
                        f"tracking EPE {fmt(tracking['clean_epe_mean_m'], 3)}→{fmt(tracking['attack_epe_mean_m'], 3)}m, "
                        f"평균 변화 {fmt(tracking['delta_epe_mean_m'], 3, True)}m. "
                        f"프레임 EPE 증가 {tracking['increased_frames']}/128, 개선 {tracking['improved_frames']}/128, "
                        f"동일 {tracking['equal_frames']}/128; Δ 범위 "
                        f"{fmt(tracking['delta_epe_min_m'], 3, True)}~{fmt(tracking['delta_epe_max_m'], 3, True)}m.",
                        "; ".join(parts) + ". 고정 시간 구간의 기술적 분포이며 128개의 독립 표본이나 프레임 CI가 아닙니다."]
            ratios = norm["framewise_ratio"]
            normalization = (f"예측 head2 norm 평균 {fmt(norm['clean']['mean'], 3)}→{fmt(norm['attack']['mean'], 3)} "
                            f"(raw pre-alignment 좌표 크기), frame별 attack/clean 비율 중앙값 {fmt(ratios['median'], 3)}, "
                            f"범위 {fmt(ratios['min'], 3)}~{fmt(ratios['max'], 3)}, "
                            f"유효 {ratios['valid_frames']}/128·분모 N/A {ratios['excluded_frames']}. "
                            f"평균 attack/평균 clean의 비율은 {fmt(norm['ratio_of_means'], 3)}로 frame별 비율 평균 "
                            f"{fmt(ratios['mean'], 3)}과 구분합니다. "
                            f"고정 GT norm 평균은 {fmt(norm['fixed_gt']['mean'], 3)} (GT meter 좌표 크기)이며 모든 목적에서 동일합니다. "
                            "Norm 비율은 무차원이며 norm 변화 자체를 meter benchmark 손상이나 독립 인과 효과로 해석하지 않습니다.")
            conditions.append({"objective": objective, "tracking_timeline_summary": tracking, "summary": temporal})
            diagnostics = record["loss_terms"]["diagnostics"]
            clean_diag = clean_record["loss_terms"]["diagnostics"]
            diagnostic = "; ".join(f"{field} {fmt(clean_diag[field], 3)}→{fmt(diagnostics[field], 3)}"
                                    for field in DIAGNOSTIC_FIELDS.values())
            cases.append({"dataset": dataset, "sequence": sequence, "objective": objective,
                          "tracking_timeline_summary": tracking, "normalization_summary": norm,
                          "summary": [*temporal, normalization,
                                      diagnostic + ". 비가중 native L21과 raw/effective confidence의 동반 변화이며 meter 오차·확률과 구분합니다."]})
        result.append({"dataset": dataset, "sequence": sequence, "frame_count": 128, "conditions": conditions})
    return result, cases
