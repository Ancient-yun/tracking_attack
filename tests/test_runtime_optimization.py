"""Exact CPU checks against real official modules, without model downloads/GPU."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from st4rtrack_pgd.common import sha256
from st4rtrack_pgd.component_attack import component_loss_and_gradient
from st4rtrack_pgd.component_forward import NativeComponentForward, native_pointmap_norm
from st4rtrack_pgd.runtime_optimization import FixedPositionGetter3D, apply_runtime_optimizations
from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward
from st4rtrack_pgd.worldtrack_adapter import ensure_vendor_importable
from test_component_forward import toy_targets


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "St4RTrack"


def small_official_forward():
    ensure_vendor_importable(VENDOR)
    from dust3r.model import AsymmetricCroCo3DStereo
    torch.manual_seed(57)
    model = AsymmetricCroCo3DStereo(
        pos_embed="RoPE100", patch_embed_cls="PatchEmbedDust3R", img_size=(32, 16),
        head_type="linear", head_type1="linear", output_mode="pts3d", landscape_only=False,
        enc_embed_dim=32, enc_depth=1, enc_num_heads=4,
        dec_embed_dim=32, dec_depth=1, dec_num_heads=4,
    )
    target = toy_targets()
    dense = torch.rand(2, 16, 32, 3) + torch.tensor([0., 0., 2.])
    mask = torch.ones((2, 16, 32), dtype=torch.bool)
    mask[0, 0, 0] = False
    mask[1, 0, :7] = False
    target = replace(target, reconstruction_gt=dense, reconstruction_valid=mask,
                     reconstruction_gt_norm=native_pointmap_norm(dense),
                     reconstruction_valid_counts=mask.flatten(1).sum(1),
                     depth=dense[..., 2])
    base = St4RTrackForward(model, target.native_query_xy.double() + .2, vendor_root=VENDOR)
    rgb = torch.rand(2, 3, 16, 32) * .5 + .25
    return base, target, rgb


def test_fixed_positions_exactly_match_original_and_cache_all_dimensions():
    ensure_vendor_importable(VENDOR)
    from models.blocks import PositionGetter3D
    original = PositionGetter3D()
    optimized = FixedPositionGetter3D((18, 32))
    for batch in (1, 2):
        for length in (2, 4):
            expected = original(batch, length, 18, 32, torch.device("cpu"))
            actual = optimized(batch, length, 18, 32, torch.device("cpu"))
            assert torch.equal(expected, actual)
            assert optimized(batch, length, 18, 32, torch.device("cpu")) is actual
    assert len(optimized.cache) == 4
    with pytest.raises(ValueError, match="different|differs"):
        optimized(1, 2, 20, 32, torch.device("cpu"))


def test_real_official_rope_values_gradients_state_dict_and_restore_are_exact():
    base, target, rgb = small_official_forward()
    model = base.model
    values = torch.randn(2, 4, 2, 8, requires_grad=True)
    positions = torch.tensor([[[0, 0], [0, 1]]] * 2)
    expected = model.rope_enc(values, positions)
    expected_grad = torch.autograd.grad(expected.square().sum(), values)[0]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    source_before = sha256(VENDOR / "croco" / "models" / "pos_embed.py")
    original_getter = model.patch_embed.position_getter3d
    handle = apply_runtime_optimizations(model, image_hw=(16, 32), vendor_root=VENDOR)
    assert apply_runtime_optimizations(model, image_hw=(16, 32), vendor_root=VENDOR) is handle
    actual = model.rope_enc(values, positions)
    actual_grad = torch.autograd.grad(actual.square().sum(), values)[0]
    assert torch.equal(expected, actual)
    assert torch.equal(expected_grad, actual_grad)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert source_before == sha256(VENDOR / "croco" / "models" / "pos_embed.py")
    handle.restore()
    assert model.patch_embed.position_getter3d is original_getter
    assert "forward" not in model.rope_enc.__dict__
    assert torch.equal(expected, model.rope_enc(values, positions))
    handle.restore()  # Idempotent, and no model/source state is changed.


def test_optimized_native_pair_values_local_vjp_and_official_oracle_match():
    base, target, rgb = small_official_forward()
    baseline = NativeComponentForward(base, target)
    before, before_maps, before_conf = baseline.predict_with_maps(rgb)
    generic = component_loss_and_gradient(baseline.pair_fn, rgb, target, "joint_training")
    optimized = NativeComponentForward(base, target, optimize_runtime=True)
    try:
        after, after_maps, after_conf = optimized.predict_with_maps(rgb)
        assert all(torch.equal(before[name], after[name]) for name in before)
        assert torch.equal(before_maps, after_maps) and torch.equal(before_conf, after_conf)
        local = component_loss_and_gradient(
            optimized.pair_fn, rgb, target, "joint_training", pair_frames_fn=optimized.pair_frames_fn,
        )
        torch.testing.assert_close(generic.gradient, local.gradient, atol=1e-6, rtol=1e-6)
        assert generic.loss == local.loss
        assert optimized._frame_view(rgb[0])["true_shape"].device.type == "cpu"
        assert optimized.check_official_loss_parity(rgb)["passed"]
        assert optimized.metadata["runtime_optimization"]["enabled"]
        assert optimized.metadata["runtime_optimization"]["fixed_valid_flat_indices"]
        assert len(optimized._gt_cache) == 1
    finally:
        optimized.restore_runtime()
    assert not optimized.metadata["runtime_optimization"]["enabled"]
    assert optimized._gt_cache == {}
    restored = optimized.predict_components(rgb)
    assert all(torch.equal(before[name], restored[name]) for name in before)


def test_pair_frame_api_matches_video_and_accumulates_shared_self_pair():
    base, target, rgb = small_official_forward()
    forward = NativeComponentForward(base, target)
    video = rgb.clone().requires_grad_()
    expected = forward(video, 1)
    actual = forward.pair_frames_fn(video[0], video[1], 1)
    for name in expected:
        assert torch.equal(expected[name], actual[name])
    expected_grad = torch.autograd.grad(expected["native_tracks"].sum(), video)[0]
    first, other = rgb[0].clone().requires_grad_(), rgb[1].clone().requires_grad_()
    pair = forward.pair_frames_fn(first, other, 1)
    first_grad, other_grad = torch.autograd.grad(pair["native_tracks"].sum(), (first, other))
    assert torch.equal(expected_grad, torch.stack((first_grad, other_grad)))
    self_video = rgb.clone().requires_grad_()
    self_gradient = torch.autograd.grad(forward(self_video, 0)["native_tracks"].sum(), self_video)[0]
    shared = rgb[0].clone().requires_grad_()
    shared_gradient = torch.autograd.grad(forward.pair_frames_fn(shared, shared, 0)["native_tracks"].sum(), shared)[0]
    assert torch.equal(self_gradient[0], shared_gradient) and self_gradient[1].eq(0).all()
    with pytest.raises(ValueError, match="same leaf"):
        forward.pair_frames_fn(shared, shared.clone(), 0)


def test_runtime_rejects_temporal_variant_and_wrong_fixed_grid():
    base, target, rgb = small_official_forward()
    base.model.arch_mode = "TempDust3r"
    with pytest.raises(ValueError, match="VanillaDust3r"):
        apply_runtime_optimizations(base.model, image_hw=(16, 32), vendor_root=VENDOR)
    base.model.arch_mode = "VanillaDust3r"
    handle = apply_runtime_optimizations(base.model, image_hw=(16, 32), vendor_root=VENDOR)
    try:
        with pytest.raises(ValueError, match="different fixed runtime grid"):
            apply_runtime_optimizations(base.model, image_hw=(32, 32), vendor_root=VENDOR)
    finally:
        handle.restore()
