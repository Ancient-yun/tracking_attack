"""Deterministic roster extension preserves the measured/frozen comparisons."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("component_roster", ROOT / "scripts/build_loss_component_roster.py")
roster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roster)


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def inputs():
    full = read("docker/manifests/synthetic_full.json")
    original = read("docker/manifests/pgd_4clips.json")
    base = read("docker/manifests/loss_components_14clips.json")
    return full, original, base


def test_all_supported_counts_are_nested_balanced_and_preserve_sha_bytes():
    full, original, base = inputs()
    eligible = base["selection_details"]["eligible_sequences_in_source_order"]
    previous = base["entries"]
    source = {(e["dataset"], e["sequence"]): e for e in full["entries"]}
    for count in (16, 18, 20, 22, 24, 26, 28, 96):
        selected, drawn, continuation = roster.extend_entries(full["entries"], original["entries"], base["entries"], eligible, count)
        assert selected[:len(previous)] == previous
        assert selected[:14] == base["entries"]
        assert len(selected) == len({(e["dataset"], e["sequence"]) for e in selected}) == count
        assert all(sum(e["dataset"] == dataset for e in selected) == count // 2 for dataset in roster.DATASETS)
        assert drawn == base["selection_details"]["sampled_six_additions"]
        assert all(e == source[e["dataset"], e["sequence"]] for e in selected)
        assert all(e["sequence"] in eligible[e["dataset"]] for e in selected)
        assert all(len(continuation[dataset]) == 40 for dataset in roster.DATASETS)
        previous = selected


@pytest.mark.parametrize("count", [14, 15, 17, 98, True])
def test_bad_counts_or_unavailable_rosters_are_rejected(count):
    full, original, base = inputs()
    with pytest.raises(ValueError):
        roster.extend_entries(full["entries"], original["entries"], base["entries"],
                              base["selection_details"]["eligible_sequences_in_source_order"], count)


def test_changed_base_or_gt_population_cannot_silently_change_selection():
    full, original, base = inputs()
    changed = copy.deepcopy(base["entries"])
    changed[2], changed[3] = changed[3], changed[2]
    with pytest.raises(ValueError, match="existing14"):
        roster.extend_entries(full["entries"], original["entries"], changed,
                              base["selection_details"]["eligible_sequences_in_source_order"], 18)


def test_frozen_outputs_are_idempotent_and_never_overwrite_changed_files(tmp_path):
    (tmp_path / "docker/manifests").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    manifest = {"created_at_utc": "first", "campaign_expected_clips": 18, "entries": []}
    config = {"campaign_expected_clips": 18}
    paths = roster.save_pair(tmp_path, 18, manifest, config)
    before = [Path(path).read_bytes() for path in paths]
    same = {**manifest, "created_at_utc": "later"}
    roster.save_pair(tmp_path, 18, same, config)
    assert before == [Path(path).read_bytes() for path in paths]
    with pytest.raises(FileExistsError):
        roster.save_pair(tmp_path, 18, {**manifest, "entries": ["changed"]}, config)
    assert before == [Path(path).read_bytes() for path in paths]


def test_runtime_optimization_is_explicit_and_preserves_other_config_fields(tmp_path):
    (tmp_path / "configs").mkdir()
    base = read("configs/loss_components.json")
    optimized = {**base, "runtime_optimization": True}
    original_path = tmp_path / "configs/loss_components.json"
    optimized_path = tmp_path / "configs/loss_components_optimized.json"
    original_path.write_text(json.dumps(base), encoding="utf-8")
    optimized_path.write_text(json.dumps(optimized), encoding="utf-8")
    result, source = roster.campaign_config(tmp_path)
    assert result == base
    assert "runtime_optimization" not in result
    assert source == {"configs/loss_components.json": roster.file_sha256(original_path)}
    result, source = roster.campaign_config(tmp_path, runtime_optimization=True)
    assert result == optimized
    assert source == {"configs/loss_components.json": roster.file_sha256(original_path),
                      "configs/loss_components_optimized.json": roster.file_sha256(optimized_path)}
    optimized_path.write_text(json.dumps({**optimized, "steps": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="differ only"):
        roster.campaign_config(tmp_path, runtime_optimization=True)
    # The default never reads or adopts a changed optimized config.
    assert roster.campaign_config(tmp_path)[0] == base


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_optimized_switch_rejects_non_boolean_enabled_flag(tmp_path, value):
    (tmp_path / "configs").mkdir()
    base = {"seed": 20261002, "steps": 20}
    (tmp_path / "configs/loss_components.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "configs/loss_components_optimized.json").write_text(
        json.dumps({**base, "runtime_optimization": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="differ only"):
        roster.campaign_config(tmp_path, runtime_optimization=True)


def test_all_frame_smaller_rosters_preserve_each_eligible_old_prefix_and_original_four():
    full, original, base = inputs()
    eligible = {dataset: [entry["sequence"] for entry in full["entries"] if entry["dataset"] == dataset]
                for dataset in roster.DATASETS}
    previous = []
    for count in (4, 6, 8, 10, 12, 14, 16, 28):
        selected, details = roster.all_frame_entries(full["entries"], original["entries"], base["entries"], eligible, count)
        assert selected[:len(previous)] == previous
        for dataset in roster.DATASETS:
            wanted = [entry for entry in base["entries"] if entry["dataset"] == dataset][:count//2]
            assert [entry for entry in selected if entry["dataset"] == dataset][:len(wanted)] == wanted
        assert all(entry in selected for entry in original["entries"])
        assert details["ineligible_original_four_ids"] == []
        assert all(value == [] for value in details["ineligible_original14_ids"].values())
        previous = selected


def test_full_frame_eligibility_replaces_ineligible_old_slots_without_reusing64_eligibility():
    full, original, base = inputs()
    eligible = {dataset: [entry["sequence"] for entry in full["entries"] if entry["dataset"] == dataset]
                for dataset in roster.DATASETS}
    # Full128 motion can invalidate a previously usable64-frame population.
    invalid = [entry for entry in base["entries"] if entry["dataset"] == "po_mini"][2]
    eligible["po_mini"].remove(invalid["sequence"])
    selected, details = roster.all_frame_entries(full["entries"], original["entries"], base["entries"], eligible, 8)
    assert invalid not in selected and all(entry in selected for entry in original["entries"])
    assert details["ineligible_original14_ids"]["po_mini"] == [invalid["sequence"]]
    replacement = details["replaced_original14_slots"]["po_mini"][0]
    assert replacement["ineligible_old_sequence"] == invalid["sequence"]
    assert replacement["replacement_sequence"] == details["full_seeded_fallback_permutation"]["po_mini"][0]
    assert [entry for entry in selected if entry["dataset"] == "po_mini"][2]["sequence"] == replacement["replacement_sequence"]
    selected10, _ = roster.all_frame_entries(full["entries"], original["entries"], base["entries"], eligible, 10)
    assert selected10[:8] == selected
    # An ineligible original clip is expressly disclosed rather than preserved incorrectly.
    eligible["po_mini"].remove(original["entries"][0]["sequence"])
    selected, details = roster.all_frame_entries(full["entries"], original["entries"], base["entries"], eligible, 8)
    assert original["entries"][0] not in selected
    assert details["ineligible_original_four_ids"] == ["po_mini/" + original["entries"][0]["sequence"]]


def test_all_frames_outputs_never_collide_with_historical64_files(tmp_path):
    (tmp_path / "docker/manifests").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    historical = {"created_at_utc": "first", "campaign_expected_clips": 18, "num_frames": 64}
    paths64 = roster.save_pair(tmp_path, 18, historical, {"num_frames": 64})
    before = [Path(path).read_bytes() for path in paths64]
    allframes = {**historical, "num_frames": 128, "all_frames": True, "campaign_expected_frames": 128}
    paths128 = roster.save_pair(tmp_path, 18, allframes, {"num_frames": 128, "all_frames": True})
    assert all(Path(path).name == "loss_components_18clips_allframes.json" for path in paths128)
    assert before == [Path(path).read_bytes() for path in paths64]
