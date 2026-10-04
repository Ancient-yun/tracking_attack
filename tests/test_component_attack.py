"""CPU-double tests for native components, multiple-output VJP and PGD."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from st4rtrack_pgd.component_attack import (
    OBJECTIVES, OUTPUT_KEYS, component_attack_video, component_loss_and_gradient,
    component_loss_terms, predict_components,
)
from st4rtrack_pgd.pgd_video import AttackConfig, loss_and_gradient


class ToyComponentPair:
    def __init__(self, dense_gt, dense_valid):
        self.dense_gt = dense_gt
        self.dense_valid = dense_valid

    def points(self, first, other, t):
        a, b = first.mean((-1, -2)), other.mean((-1, -2))
        bases = a.new_tensor([[0.11, 0.2, 1.3], [0.3, 0.15, 2.2],
                              [0.2, 0.4, 3.4], [0.4, 0.3, 4.7]])
        offset = a * a.new_tensor([0.24, 0.11, 0.17]) + b * a.new_tensor([0.12, 0.21, 0.09]) + a * b * 0.04
        tracks = bases + a.new_tensor([0.6, 0.8, 1.1, 1.3])[:, None] * offset + 0.02 * t
        native = tracks[[2, 0, 3]] + b * 0.031
        c1 = 1 + torch.exp(a.sum() * 0.08 + b.sum() * a.new_tensor([0.05, 0.09, 0.04]) + t * 0.03)
        head2 = bases + 0.16 * a + 0.23 * b + 0.015 * t
        c2 = 1 + torch.exp(a.sum() * a.new_tensor([0.07, 0.09, 0.03, 0.06]) + b.sum() * 0.11)
        # Upstream valid=None uses H*W*3 as nnz, preserving the /3 quirk.
        norm = head2.norm(dim=-1).sum() / (head2.numel() + 1e-8)
        gt_norm = self.dense_gt[t].norm(dim=-1).sum() / (self.dense_gt[t].numel() + 1e-8)
        residual = head2 / norm.clamp_min(1e-8) - self.dense_gt[t] / gt_norm.clamp_min(1e-8)
        distances = residual.norm(dim=-1)[self.dense_valid[t]]
        confidence = c2[self.dense_valid[t]]
        return {
            "tracks": tracks, "native_tracks": native, "track_conf": c1,
            "reconstruction_norm": norm,
            "reconstruction_regression_sum": (distances * confidence).sum(),
            "reconstruction_log_conf_sum": confidence.log().sum(),
            "reconstruction_l21_sum": distances.sum(),
            "reconstruction_conf_sum": confidence.sum(),
        }

    def __call__(self, rgb, t):
        return self.points(rgb[0], rgb[t], t)


def fixture(dynamic=True):
    rgb = torch.linspace(0.13, 0.79, 36, dtype=torch.float64).reshape(3, 3, 2, 2)
    dense_gt = rgb.new_tensor([[0.15, 0.1, 1.2], [0.2, 0.2, 2.8],
                              [0.3, 0.5, 3.9], [0.5, 0.1, 5.3]])[None].repeat(3, 1, 1)
    dense_gt += torch.arange(3, dtype=rgb.dtype)[:, None, None] * 0.047
    dense_valid = torch.tensor([[True, False, True, False], [True, True, True, True],
                                [False, True, False, False]])
    pair = ToyComponentPair(dense_gt, dense_valid)
    outputs = predict_components(pair, rgb)
    targets = {
        "tracks": outputs["tracks"] * 1.29 + rgb.new_tensor([0.03, -0.07, 0.11]),
        "valid": torch.ones((3, 4), dtype=torch.bool),
        "native_tracks": outputs["native_tracks"] * 1.16 + rgb.new_tensor([0.02, 0.07, -0.1]),
        "native_track_valid": torch.ones((3, 3), dtype=torch.bool),
        "training_dynamic": torch.tensor([False, False, bool(dynamic)]),
        "reconstruction_gt_norm": dense_gt.norm(dim=-1).sum(-1) / (dense_gt[0].numel() + 1e-8),
        "reconstruction_valid_counts": dense_valid.sum(-1),
    }
    return pair, rgb, targets


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_full_and_recompute_match_all_outputs_and_gradients(objective):
    pair, rgb, targets = fixture()
    full = component_loss_and_gradient(pair, rgb, targets, objective, mode="full")
    replay = component_loss_and_gradient(pair, rgb, targets, objective)
    assert full.loss == pytest.approx(replay.loss, abs=1e-13)
    assert full.scale == pytest.approx(replay.scale, abs=1e-13)
    assert full.terms == replay.terms
    for name in OUTPUT_KEYS:
        torch.testing.assert_close(full.outputs[name], replay.outputs[name], rtol=0, atol=0)
    torch.testing.assert_close(full.gradient, replay.gradient, rtol=1e-12, atol=1e-12)
    assert (replay.gradient.flatten(1).norm(dim=1) > 0).all()
    assert not rgb.requires_grad


def test_mse_control_matches_original_engine_with_separate_native_queries():
    pair, rgb, targets = fixture()
    old = loss_and_gradient(lambda video, t: pair(video, t)["tracks"], rgb, targets["tracks"], targets["valid"])
    new = component_loss_and_gradient(pair, rgb, targets, "tracking_mse")
    assert old.loss == new.loss
    assert old.scale == new.scale
    torch.testing.assert_close(old.gradient, new.gradient, rtol=1e-12, atol=1e-12)
    assert new.outputs["tracks"].shape[1] != new.outputs["native_tracks"].shape[1]


def test_native_term_split_uses_total_pixel_count_and_detached_max():
    pair, rgb, targets = fixture()
    out = predict_components(pair, rgb)
    terms = component_loss_terms(out, targets)
    weights = out["track_conf"].clone()
    weights[:, 2] = 5 * out["track_conf"][:, :2].max()
    error = out["native_tracks"] / out["reconstruction_norm"][:, None, None] - targets["native_tracks"] / targets["reconstruction_gt_norm"][:, None, None]
    track = (error.norm(dim=-1) * weights).mean()
    recon = out["reconstruction_regression_sum"].sum() / 7
    conf = -0.2 * (weights.clamp_min(1).log().mean() + out["reconstruction_log_conf_sum"].sum() / 7)
    torch.testing.assert_close(terms["tracking_regression"], track)
    torch.testing.assert_close(terms["reconstruction_regression"], recon)
    torch.testing.assert_close(terms["confidence"], conf)
    torch.testing.assert_close(terms["total"], track + recon + conf)
    wrong_frame_mean = (out["reconstruction_regression_sum"] / targets["reconstruction_valid_counts"]).mean()
    assert float((wrong_frame_mean - recon).abs()) > 1e-3


def test_dynamic_confidence_stop_gradient_is_preserved():
    pair, rgb, targets = fixture()
    outputs = {name: value.clone().requires_grad_() for name, value in predict_components(pair, rgb).items()}
    terms = component_loss_terms(outputs, targets)
    grad = torch.autograd.grad(terms["total"], outputs["track_conf"])[0]
    assert torch.equal(grad[:, 2], torch.zeros_like(grad[:, 2]))
    l21 = (outputs["native_tracks"] / outputs["reconstruction_norm"][:, None, None] - targets["native_tracks"] / targets["reconstruction_gt_norm"][:, None, None]).norm(dim=-1)
    expected = (l21[:, :2] - 0.2 / outputs["track_conf"][:, :2]) / 9
    torch.testing.assert_close(grad[:, :2], expected)


def test_head2_normalization_and_confidence_are_gradient_paths():
    pair, rgb, targets = fixture()
    good = component_loss_and_gradient(pair, rgb, targets, "tracking_3d", mode="full")

    def detached_norm(video, t):
        outputs = pair(video, t)
        outputs["reconstruction_norm"] = outputs["reconstruction_norm"].detach()
        return outputs

    wrong = component_loss_and_gradient(detached_norm, rgb, targets, "tracking_3d", mode="full")
    assert float((wrong.gradient - good.gradient).abs().max()) > 1e-3
    with pytest.raises(RuntimeError, match="reconstruction_norm.*detached"):
        component_loss_and_gradient(detached_norm, rgb, targets, "tracking_3d")
    confidence_gradient = component_loss_and_gradient(pair, rgb, targets, "confidence").gradient
    assert float(confidence_gradient.abs().max()) > 1e-3


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_rgb_finite_difference_on_static_support(objective):
    # No dynamic reweighting here: recomputing detached max in numeric probes
    # would intentionally include a path absent from official autograd.
    pair, rgb, targets = fixture(dynamic=False)
    analytic = component_loss_and_gradient(pair, rgb, targets, objective).gradient
    key = OBJECTIVES[objective]
    h = 1e-6
    for index in [(0, 0, 0, 0), (0, 2, 1, 1), (1, 1, 1, 0), (2, 2, 0, 1)]:
        plus, minus = rgb.clone(), rgb.clone()
        plus[index] += h
        minus[index] -= h
        hi = component_loss_terms(predict_components(pair, plus), targets)[key]
        lo = component_loss_terms(predict_components(pair, minus), targets)[key]
        assert float(analytic[index]) == pytest.approx(float((hi - lo) / (2 * h)), abs=2e-8)


def test_shared_frame_zero_includes_both_self_pair_roles():
    pair, rgb, targets = fixture()
    compact = {name: value.clone().requires_grad_() for name, value in predict_components(pair, rgb).items()}
    loss = component_loss_terms(compact, targets)["total"]
    upstream = torch.autograd.grad(loss, tuple(compact.values()), allow_unused=True)
    expected = torch.zeros_like(rgb)
    roles = []
    for t in range(len(rgb)):
        first, second = rgb[0].clone().requires_grad_(), rgb[t].clone().requires_grad_()
        outputs = pair.points(first, second, t)
        active = [(outputs[name], grad[t]) for name, grad in zip(compact, upstream) if grad is not None]
        grad_first, grad_second = torch.autograd.grad(tuple(item[0] for item in active), (first, second),
                                                    tuple(item[1] for item in active))
        expected[0] += grad_first
        expected[t] += grad_second
        if t == 0:
            roles = [grad_first, grad_second]
    actual = component_loss_and_gradient(pair, rgb, targets, "joint_training").gradient
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert all(float(value.abs().max()) > 0 for value in roles)


def test_replay_checks_confidence_and_statistic_not_only_tracks():
    pair, rgb, targets = fixture()
    calls = 0

    def stateful(video, t):
        nonlocal calls
        calls += 1
        outputs = pair(video, t)
        outputs["track_conf"] = outputs["track_conf"] + calls * 0.01
        return outputs

    with pytest.raises(RuntimeError, match="replay changed.*track_conf"):
        component_loss_and_gradient(stateful, rgb, targets, "tracking_mse")


def test_nonfinite_detached_and_rgb_disconnected_fail():
    pair, rgb, targets = fixture()

    def nonfinite(video, t):
        outputs = pair(video, t)
        outputs["reconstruction_norm"] = outputs["reconstruction_norm"] * float("nan")
        return outputs

    with pytest.raises(FloatingPointError, match="reconstruction_norm"):
        component_loss_and_gradient(nonfinite, rgb, targets, "joint_training")
    with pytest.raises(RuntimeError, match="detached"):
        component_loss_and_gradient(lambda video, t: {k: v.detach() for k, v in pair(video, t).items()},
                                    rgb, targets, "confidence")
    with pytest.raises(RuntimeError, match="disconnected"):
        component_loss_and_gradient(lambda video, t: {k: v.detach().requires_grad_() for k, v in pair(video, t).items()},
                                    rgb, targets, "confidence")


def test_zero_objective_derivative_does_not_hide_a_detached_forward():
    pair, rgb, targets = fixture()
    targets["tracks"] = predict_components(pair, rgb)["tracks"].clone()
    with pytest.raises(RuntimeError, match="tracks.*detached"):
        component_loss_and_gradient(lambda video, t: {k: v.detach() for k, v in pair(video, t).items()},
                                    rgb, targets, "tracking_mse")


def test_zero_native_gradient_is_valid():
    pair, rgb, targets = fixture(dynamic=False)

    def constant_confidence(video, t):
        outputs = pair(video, t)
        outputs["track_conf"] = outputs["track_conf"] * 0 + 2
        outputs["reconstruction_log_conf_sum"] = outputs["reconstruction_log_conf_sum"] * 0 + 0.7
        return outputs

    replay = component_loss_and_gradient(constant_confidence, rgb, targets, "confidence")
    full = component_loss_and_gradient(constant_confidence, rgb, targets, "confidence", mode="full")
    torch.testing.assert_close(replay.gradient, torch.zeros_like(rgb), rtol=0, atol=0)
    torch.testing.assert_close(replay.gradient, full.gradient, rtol=0, atol=0)


def test_pgd_bounds_incumbent_callback_state_and_reload():
    pair, rgb, targets = fixture()
    callbacks = []
    config = AttackConfig(eps=0.03, steps=3, restarts=2, seed=31)
    result = component_attack_video(pair, rgb, targets, "joint_training", config, progress_callback=callbacks.append)
    assert float(result.delta.abs().max()) <= config.eps + 1e-14
    assert ((result.rgb >= 0) & (result.rgb <= 1)).all()
    assert result.loss == max(row["loss"] for row in result.history)
    assert result.terms["total"] == result.loss
    assert result.elapsed_seconds > 0
    assert len(callbacks) == len(result.history) == 9
    chosen = [row for row in result.history if all(row[k] == v for k, v in result.selected_state.items())]
    assert chosen[0]["loss"] == result.loss
    for row in result.history:
        assert row["objective"] == "joint_training"
        assert len(row["delta_linf_per_frame"]) == len(rgb)
        json.dumps(row, allow_nan=False)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "rgb.npy"
        np.save(path, result.rgb.numpy())
        replay = predict_components(pair, torch.from_numpy(np.load(path)))
        for name in OUTPUT_KEYS:
            torch.testing.assert_close(replay[name], result.outputs[name], rtol=0, atol=0)
        replay_terms = component_loss_terms(replay, targets)
        assert float(replay_terms["total"]) == result.loss


def test_same_seed_shares_initial_noise_across_objectives():
    pair, rgb, targets = fixture()
    starts = []
    for objective in ("tracking_3d", "confidence", "tracking_mse"):
        visited = []

        def capture(video, t):
            if t == 0:
                visited.append(video.detach().clone())
            return pair(video, t)

        result = component_attack_video(capture, rgb, targets, objective, AttackConfig(eps=0.03, steps=1, seed=78))
        starts.append(visited[1])  # after the clean objective forward
        assert result.loss >= result.history[0]["loss"]
    for start in starts[1:]:
        torch.testing.assert_close(start, starts[0], rtol=0, atol=0)


def test_zero_epsilon_returns_exact_clean_and_no_gradient_forward():
    pair, rgb, targets = fixture()
    result = component_attack_video(pair, rgb, targets, "joint_training", AttackConfig(eps=0))
    torch.testing.assert_close(result.rgb, rgb, rtol=0, atol=0)
    torch.testing.assert_close(result.delta, torch.zeros_like(rgb), rtol=0, atol=0)
    assert len(result.history) == 1
    assert result.selected_state == {"restart": -1, "step": 0}


def test_gt_masks_are_fixed_and_no_static_support_is_explicit():
    pair, rgb, targets = fixture()
    targets["training_dynamic"][:] = True
    with pytest.raises(ValueError, match="no static support"):
        component_loss_terms(predict_components(pair, rgb), targets)
    targets["training_dynamic"][:] = False
    targets["native_tracks"] = targets["native_tracks"].requires_grad_()
    with pytest.raises(ValueError, match="fixed"):
        component_loss_terms(predict_components(pair, rgb), targets)


@pytest.mark.parametrize("objective", list(OBJECTIVES))
@pytest.mark.parametrize("masked", [False, True])
def test_frame_local_vjp_matches_full_and_generic_recompute(objective, masked):
    pair, rgb, targets = fixture()
    if masked:
        targets["valid"][0, 1] = False
        targets["tracks"][0, 1] = float("nan")
        targets["native_track_valid"][1, 0] = False
        targets["native_tracks"][1, 0] = float("nan")
    calls = []

    def frames(first, other, t):
        calls.append((t, first is other, tuple(first.shape), tuple(other.shape)))
        return pair.points(first, other, t)

    full = component_loss_and_gradient(pair, rgb, targets, objective, mode="full")
    generic = component_loss_and_gradient(pair, rgb, targets, objective)
    local = component_loss_and_gradient(pair, rgb, targets, objective, pair_frames_fn=frames)
    for reference in (full, generic):
        torch.testing.assert_close(local.gradient, reference.gradient, rtol=1e-12, atol=1e-12)
        assert local.loss == reference.loss
        assert local.scale == reference.scale
        assert local.terms == reference.terms
        for name in OUTPUT_KEYS:
            torch.testing.assert_close(local.outputs[name], reference.outputs[name], rtol=0, atol=0)
    assert calls == [(0, True, (3, 2, 2), (3, 2, 2)),
                     (1, False, (3, 2, 2), (3, 2, 2)),
                     (2, False, (3, 2, 2), (3, 2, 2))]
    assert not rgb.requires_grad


@pytest.mark.parametrize("unused_role", ["first", "other"])
def test_frame_local_vjp_allows_one_disconnected_role(unused_role):
    pair, rgb, targets = fixture()

    def frames(first, other, t):
        return pair.points(first.detach() if unused_role == "first" else first,
                           other.detach() if unused_role == "other" else other, t)

    generic = lambda video, t: frames(video[0], video[t], t)
    reference = component_loss_and_gradient(generic, rgb, targets, "joint_training", mode="full")
    local = component_loss_and_gradient(generic, rgb, targets, "joint_training", pair_frames_fn=frames)
    torch.testing.assert_close(local.gradient, reference.gradient, rtol=1e-12, atol=1e-12)
    assert float(local.gradient[0].abs().max()) > 0
    if unused_role == "other":
        torch.testing.assert_close(local.gradient[1:], torch.zeros_like(rgb[1:]), rtol=0, atol=0)


@pytest.mark.parametrize("name", list(OUTPUT_KEYS))
def test_grouped_frame_local_validation_reports_nonfinite_field(name):
    pair, rgb, targets = fixture()

    def frames(first, other, t):
        values = pair.points(first, other, t)
        values[name] = values[name] * float("nan")
        return values

    with pytest.raises(FloatingPointError, match=name):
        component_loss_and_gradient(pair, rgb, targets, "joint_training", pair_frames_fn=frames)


def test_frame_local_replay_and_detachment_checks_remain_fail_fast():
    pair, rgb, targets = fixture()
    calls = []

    def mismatch(first, other, t):
        calls.append(t)
        values = pair.points(first, other, t)
        values["track_conf"] = values["track_conf"] + 0.01
        return values

    with pytest.raises(RuntimeError, match="frame 0: track_conf"):
        component_loss_and_gradient(pair, rgb, targets, "tracking_mse", pair_frames_fn=mismatch)
    assert calls == [0]

    def detached(first, other, t):
        return {name: value.detach() for name, value in pair.points(first, other, t).items()}

    with pytest.raises(RuntimeError, match="active component track_conf is detached at frame 0"):
        component_loss_and_gradient(pair, rgb, targets, "confidence", pair_frames_fn=detached)

    def disconnected(first, other, t):
        return {name: value.detach().requires_grad_() for name, value in pair.points(first, other, t).items()}

    with pytest.raises(RuntimeError, match="disconnected from RGB input at frame 0"):
        component_loss_and_gradient(pair, rgb, targets, "confidence", pair_frames_fn=disconnected)


def test_frame_local_nonfinite_backward_is_rejected():
    pair, rgb, targets = fixture()

    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return gradient * float("nan")

    def frames(first, other, t):
        values = pair.points(first, other, t)
        values["track_conf"] = BadBackward.apply(values["track_conf"])
        return values

    with pytest.raises(FloatingPointError, match="RGB gradient.*frame 0"):
        component_loss_and_gradient(pair, rgb, targets, "confidence", pair_frames_fn=frames)


@pytest.mark.parametrize("tolerance_multiple, passes", [(0.5, True), (1.5, False)])
def test_grouped_replay_keeps_original_tolerance(tolerance_multiple, passes):
    pair, rgb, targets = fixture()

    def frames(first, other, t):
        values = pair.points(first, other, t)
        name = "reconstruction_log_conf_sum"
        tolerance = values[name].detach().abs() * 1e-5 + 1e-6
        values[name] = values[name] + tolerance_multiple * tolerance
        return values

    if passes:
        original = component_loss_and_gradient(pair, rgb, targets, "tracking_mse")
        local = component_loss_and_gradient(pair, rgb, targets, "tracking_mse", pair_frames_fn=frames)
        torch.testing.assert_close(local.gradient, original.gradient, rtol=1e-12, atol=1e-12)
    else:
        with pytest.raises(RuntimeError, match="frame 0: reconstruction_log_conf_sum"):
            component_loss_and_gradient(pair, rgb, targets, "tracking_mse", pair_frames_fn=frames)


def test_full_reference_ignores_optional_frame_replay_function():
    pair, rgb, targets = fixture()

    def forbidden(*args):
        raise AssertionError("full reference must call the generic pair function")

    original = component_loss_and_gradient(pair, rgb, targets, "joint_training", mode="full")
    explicit = component_loss_and_gradient(pair, rgb, targets, "joint_training", mode="full", pair_frames_fn=forbidden)
    torch.testing.assert_close(original.gradient, explicit.gradient, rtol=0, atol=0)


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_frame_local_pgd_preserves_noise_selection_outputs_and_history(objective):
    pair, rgb, targets = fixture()
    config = AttackConfig(eps=0.03, steps=3, restarts=2, seed=31)
    original = component_attack_video(pair, rgb, targets, objective, config)
    callbacks = []
    local = component_attack_video(pair, rgb, targets, objective, config,
                                   pair_frames_fn=pair.points, progress_callback=callbacks.append)
    torch.testing.assert_close(local.rgb, original.rgb, rtol=0, atol=0)
    torch.testing.assert_close(local.delta, original.delta, rtol=0, atol=0)
    assert local.loss == original.loss
    assert local.terms == original.terms
    assert local.selected_state == original.selected_state
    assert callbacks == local.history
    assert len(local.history) == len(original.history)
    for actual, expected in zip(local.history, original.history):
        for name in actual:
            if name.startswith("gradient_"):
                assert actual[name] == pytest.approx(expected[name], rel=1e-12, abs=1e-12)
            else:
                assert actual[name] == expected[name]


def test_frame_local_zero_epsilon_never_calls_gradient_replay():
    pair, rgb, targets = fixture()

    def forbidden(*args):
        raise AssertionError("zero epsilon must not calculate a gradient")

    result = component_attack_video(pair, rgb, targets, "joint_training", AttackConfig(eps=0), pair_frames_fn=forbidden)
    torch.testing.assert_close(result.rgb, rgb, rtol=0, atol=0)
