"""Native pointmap loss, fixed benchmark GT, and official-source oracle checks."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

from st4rtrack_pgd.component_forward import (
    NativeComponentForward, NativeTargets, check_official_component_oracle,
    gradient_repeat_diagnostics, load_component_sequence, native_confidence_terms, native_pointmap_norm,
    rasterize_native_queries, source_frame_counts, stack_component_outputs,
)
from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "St4RTrack"


class TwoHeadToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, view1, view2):
        a, b = view1["img"], view2["img"]
        p1 = (a + 2 * b).permute(0, 2, 3, 1) + a.new_tensor([.4, .7, 2.0])
        p2 = (.3 * a + .8 * b).permute(0, 2, 3, 1) + a.new_tensor([.5, .8, 2.3])
        return {"pts3d": p1 * self.weight, "conf": 1 + a[:, 0].exp()}, {
            "pts3d_in_other_view": p2 * self.weight, "conf": 1 + b[:, 1].exp(),
        }


def toy_targets() -> NativeTargets:
    t, h, w = 2, 4, 6
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dense = torch.stack((x.float() / 4, y.float() / 4, torch.full_like(x, 2.0).float()), -1)
    dense = dense.unsqueeze(0).repeat(t, 1, 1, 1)
    dense[1, ..., 0] += .1
    dense[0, 0, 0] = torch.tensor([0., 0., 0.])
    dense[1, 0, 0] = torch.tensor([.1, 0., 0.])  # invalid depth retains camera origin
    valid = torch.ones((t, h, w), dtype=torch.bool)
    valid[:, 0, 0] = False
    valid[1, 0, 1] = False  # Unequal frame counts must use one global pixel mean.
    xy = torch.tensor([[1, 1], [4, 2], [5, 3]])
    tracks = dense[:, xy[:, 1], xy[:, 0]].clone()
    tracks[1, 0, 0] += .3
    tracks[1, 1:, 0] -= .1
    motion = (tracks[-1] - tracks[0]).abs().sum(-1) / (tracks[0].norm(dim=-1) + 1e-8)
    return NativeTargets(
        native_tracks=tracks, native_track_valid=torch.ones((t, len(xy)), dtype=torch.bool),
        training_dynamic=motion > motion.mean(), native_query_xy=xy,
        native_point_indices=torch.arange(len(xy)), tracks=tracks.clone(),
        valid=torch.ones((t, len(xy)), dtype=torch.bool), reconstruction_gt=dense,
        reconstruction_valid=valid, reconstruction_gt_norm=native_pointmap_norm(dense),
        reconstruction_valid_counts=valid.flatten(1).sum(1),
        camera_c2w=torch.eye(4).unsqueeze(0).repeat(t, 1, 1),
        intrinsics=torch.eye(3).unsqueeze(0).repeat(t, 1, 1),
        depth=torch.full((t, h, w), 2.), metadata={},
    )


def toy_forward() -> tuple[NativeComponentForward, torch.Tensor]:
    targets = toy_targets()
    base = St4RTrackForward(TwoHeadToy(), targets.native_query_xy.double() + .2, vendor_root=VENDOR)
    video = torch.linspace(.52, .84, 2 * 3 * 4 * 6).reshape(2, 3, 4, 6).requires_grad_()
    return NativeComponentForward(base, targets), video


def test_native_norm_preserves_upstream_xyz_count_and_all_pixels():
    maps = torch.full((2, 4, 6, 3), 1.).requires_grad_()
    norm = native_pointmap_norm(maps)
    torch.testing.assert_close(norm, torch.full((2,), 3 ** .5 / 3))
    norm.sum().backward()
    assert (maps.grad > 0).all()
    # No positive-depth mask is applied to the norm: invalid camera-origin
    # points affect the scale even when absent from regression supervision.
    changed = maps.detach().clone()
    changed[:, 0, 0] = 0
    assert (native_pointmap_norm(changed) < norm).all()


def test_native_raster_rounds_clamps_deduplicates_and_orders_row_major():
    xy = torch.tensor([[1.8, 1.8], [2.1, 2.2], [5.9, 3.9], [.5, .5]])
    tracks = torch.arange(2 * 4 * 3).float().reshape(2, 4, 3)
    pixels, selected, indices = rasterize_native_queries(xy, tracks, (6, 4), torch.tensor([10, 11, 12, 13]))
    assert pixels.tolist() == [[0, 0], [2, 2], [5, 3]]
    assert indices.tolist() == [13, 11, 12]
    torch.testing.assert_close(selected, tracks[:, [3, 1, 2]])


def test_pair_preserves_both_heads_confidence_and_shared_anchor_gradients():
    forward, video = toy_forward()
    self_output = forward(video, 0)
    self_output["native_tracks"].sum().backward()
    assert (video.grad[0, :, forward.targets.native_query_xy[:, 1], forward.targets.native_query_xy[:, 0]] == 6).all()
    assert video.grad[1].eq(0).all()
    assert forward.model.weight.grad is None and not forward.model.weight.requires_grad
    video.grad = None
    outputs = stack_component_outputs([forward(video, t) for t in range(len(video))])
    assert outputs["reconstruction_norm"].requires_grad
    assert outputs["track_conf"].requires_grad
    assert outputs["reconstruction_regression_sum"].requires_grad
    terms = native_confidence_terms(outputs, forward.targets)
    terms["joint"].backward()
    assert torch.isfinite(video.grad).all() and (video.grad.abs().flatten(1).sum(1) > 0).all()
    # Head2 scale couples unqueried pixels to head1's native regression.
    assert video.grad[:, :, 0, 1].abs().sum() > 0


def test_full_active_loss_and_input_gradient_match_unchanged_official_ast():
    forward, video = toy_forward()
    report = forward.check_official_loss_parity(video, atol=2e-5, rtol=2e-5)
    assert report["passed"]
    assert report["gradient_max_abs_error"] < 2e-5
    assert all(value > 0 for value in report["gradient_per_frame_l1"])
    assert report["oracle"]["full_training_loss_module_imported"] is False


def test_captured_maps_cpu_oracle_and_saved_rgb_replay():
    forward, video = toy_forward()
    packed, reconstruction, confidence = forward.predict_with_maps(video)
    assert reconstruction.device.type == "cpu" and not reconstruction.requires_grad
    report = check_official_component_oracle(packed, reconstruction, confidence, forward.targets, vendor_root=VENDOR)
    assert report["passed"]
    # Exact float replay does not change queries, GT masks or reductions.
    buffer = BytesIO()
    torch.save(video.detach(), buffer)
    buffer.seek(0)
    replay = torch.load(buffer, weights_only=True)
    repeated, repeated_maps, repeated_conf = forward.predict_with_maps(replay)
    for key in packed:
        assert torch.equal(packed[key], repeated[key])
    assert torch.equal(reconstruction, repeated_maps)
    assert torch.equal(confidence, repeated_conf)


def test_dense_loader_normalizes_camera_and_preserves_invalid_camera_origins(tmp_path):
    from PIL import Image
    buffer = BytesIO()
    Image.new("RGB", (160, 90), (60, 120, 180)).save(buffer, format="JPEG")
    images = np.array([buffer.getvalue()] * 2, dtype=object)
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    extrinsics[:, 0, 3] = [3., 4.]
    cam_tracks = np.array([[[4, 0, 2], [3, 0, 2]], [[5, 0, 2], [4.1, 0, 2]]], dtype=np.float32)
    depth = np.full((2, 90, 160), 2., dtype=np.float32)
    depth[:, 0, 0] = 0
    source = tmp_path / "synthetic_dense.npz"
    np.savez(source, images_jpeg_bytes=images, tracks_XYZ=cam_tracks,
             fx_fy_cx_cy=np.array([10., 10., 80., 45.]),
             visibility=np.array([[True, True], [False, False]]),
             extrinsics_w2c=extrinsics, depth_map=depth)
    sequence, targets = load_component_sequence(source, num_frames=2, vendor_root=VENDOR)
    assert tuple(sequence.rgb.shape) == (2, 3, 288, 512)
    assert targets.reconstruction_gt.shape == (2, 288, 512, 3)
    torch.testing.assert_close(targets.camera_c2w[0], torch.eye(4))
    torch.testing.assert_close(targets.reconstruction_gt[:, 0, 0], torch.tensor([[0., 0., 0.], [-1., 0., 0.]]))
    assert not targets.reconstruction_valid[:, 0, 0].any()
    assert targets.native_track_valid.all() and sequence.valid.all()
    assert targets["native_tracks"] is targets.native_tracks
    assert targets.metadata["tensor_sha256"] == targets.tensor_hashes()
    assert targets.metadata["normalization"].startswith("per-frame sum(ALL")
    assert targets.training_dynamic.tolist() == [True, False]  # row-major: x=128 precedes x=144
    expected_counts = {name: 2 for name in (
        "images_jpeg_bytes", "tracks_XYZ", "visibility", "depth_map", "extrinsics_w2c",
    )}
    for metadata in (sequence.metadata, targets.metadata):
        assert metadata["original_frame_count"] == 2
        assert metadata["source_frame_counts"] == expected_counts
        assert metadata["all_frames_used"] is True
        assert metadata["used_frame_indices"] == [0, 1]
    prefix, prefix_targets = load_component_sequence(source, num_frames=1, vendor_root=VENDOR)
    for metadata in (prefix.metadata, prefix_targets.metadata):
        assert metadata["original_frame_count"] == 2
        assert metadata["all_frames_used"] is False
        assert metadata["used_frame_indices"] == [0]


@pytest.mark.parametrize("mismatched_field", [
    "images_jpeg_bytes", "tracks_XYZ", "visibility", "depth_map", "extrinsics_w2c",
])
def test_loader_rejects_raw_frame_count_mismatch_before_decoding(tmp_path, mismatched_field):
    # Headers suffice even for intentionally undecodable JPEG payloads. A
    # matching one-frame prefix must not hide disagreement after that prefix.
    arrays = {
        "images_jpeg_bytes": np.array([b"not-jpeg"] * 2, dtype=object),
        "tracks_XYZ": np.ones((2, 3, 3), dtype=np.float32),
        "visibility": np.ones((2, 3), dtype=bool),
        "depth_map": np.ones((2, 4, 6), dtype=np.float32),
        "extrinsics_w2c": np.tile(np.eye(4), (2, 1, 1)),
        "fx_fy_cx_cy": np.ones(4),
    }
    arrays[mismatched_field] = arrays[mismatched_field][:1]
    source = tmp_path / "frame_count_mismatch.npz"
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="source frame counts disagree"):
        source_frame_counts(source)
    with pytest.raises(ValueError, match="source frame counts disagree"):
        load_component_sequence(source, num_frames=1, vendor_root=VENDOR)


def test_loader_requires_dense_gt_instead_of_silently_using_sparse_surrogate(tmp_path):
    # Required temporal fields are checked in headers before RGB decoding.
    from PIL import Image
    buffer = BytesIO()
    Image.new("RGB", (160, 90)).save(buffer, format="JPEG")
    path = tmp_path / "sparse_only.npz"
    np.savez(path, images_jpeg_bytes=np.array([buffer.getvalue()], dtype=object),
             tracks_XYZ=np.array([[[0., 0., 2.]]]), fx_fy_cx_cy=np.array([10., 10., 80., 45.]),
             visibility=np.ones((1, 1), dtype=bool), extrinsics_w2c=np.eye(4)[None])
    with pytest.raises(ValueError, match="Dense native GT"):
        load_component_sequence(path, num_frames=1, vendor_root=VENDOR)


def test_repeat_noise_diagnostics_preserve_strict_failures_but_measure_small_noise():
    # Near-zero elements may fail strict absolute tolerance while the full
    # gradient discrepancy is small and independently reproduced as noise.
    positive = torch.tensor([[1., 6e-5], [.5, 6e-5]])
    negative = torch.tensor([[1., -6e-5], [.5, -6e-5]])
    report = gradient_repeat_diagnostics([positive, negative], [negative, positive])
    assert report["passed"]
    assert report["native_repeat_relative_l2"] > 0
    assert report["strict_allclose"]["passed"] is False
    assert report["strict_allclose"]["failed_elements"] == 2
    assert report["strict_allclose"]["max_abs_error"] > 1e-4
    assert report["cross_relative_l2_max"] < report["cross_relative_l2_allowed"]
    assert report["retain_graph"] == [True, True, True, False]


def test_repeat_noise_gate_rejects_source_bias_beyond_measured_repeat_noise():
    native = torch.tensor([[1., 3e-5]])
    official = torch.tensor([[1., 0.]])
    report = gradient_repeat_diagnostics([native, native], [official, official])
    assert not report["passed"]
    assert report["relative_l2_cap_passed"]  # Small does not imply correct.
    assert not report["cross_within_repeat_noise_passed"]
    assert report["repeat_relative_l2_max"] == 0
    assert report["cross_relative_l2_allowed"] == 1e-5


def test_repeat_noise_cap_rejects_unbounded_noise_even_when_cross_matches_noise():
    positive, negative = torch.tensor([[1., .02]]), torch.tensor([[1., -.02]])
    report = gradient_repeat_diagnostics([positive, negative], [negative, positive])
    assert not report["passed"]
    assert report["cross_within_repeat_noise_passed"]
    assert not report["relative_l2_cap_passed"]
    with pytest.raises(ValueError, match="<=2e-3"):
        gradient_repeat_diagnostics([positive, negative], [negative, positive], relative_l2_cap=.1)
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        gradient_repeat_diagnostics([positive * float("nan"), negative], [negative, positive])


def test_cpu_native_oracle_cannot_relax_strict_gradient_parity():
    forward, video = toy_forward()
    with pytest.raises(ValueError, match="CPU remains strict"):
        forward.check_official_loss_parity(video, repeat_noise_check=True)
