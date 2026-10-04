"""Coordinate/query and input-gradient tests independent of learned weights."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward, normalize_rgb
from st4rtrack_pgd.worldtrack_adapter import load_sequence, sample_query_tracks, select_queries


class PairModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, view1, view2):
        point_map = (view1["img"] + 2 * view2["img"]) * self.weight
        return {"pts3d": point_map.permute(0, 2, 3, 1)}, {}


def test_official_normalization_roundtrip():
    rgb = torch.arange(256).float().div(255)
    normalized = normalize_rgb(rgb)
    reconstructed = normalized.mul(0.5).add(0.5)
    assert torch.equal(normalize_rgb(reconstructed), normalized)


def test_query_selection_uses_initial_visibility_and_world_motion():
    uv = np.array([[[1.9, 2.8], [8.0, 2.0], [2.0, 2.0]], [[1, 2], [8, 2], [2, 2]]])
    gt = np.array([[[0, 0, 2], [0, 0, 2], [0, 0, 2]], [[0.02, 0, 2], [0, 0, 2], [1, 0, 2]]])
    vis = np.array([[True, True, False], [False, False, True]])
    xy, selected, dynamic, indices = select_queries(uv, gt, vis, (8, 4), (16, 8))
    np.testing.assert_array_equal(indices, [0])  # border is out of bounds
    np.testing.assert_allclose(xy, [[3.8, 5.6]])
    np.testing.assert_array_equal(dynamic, [True])
    np.testing.assert_array_equal(selected, gt[:, :1])


def test_sampler_truncates_xy_and_retains_output_gradient():
    maps = torch.arange(2 * 8 * 16 * 3).float().reshape(2, 8, 16, 3).requires_grad_()
    xy = torch.tensor([[3.8, 5.6]], dtype=torch.float64)
    selected = sample_query_tracks(maps, xy)
    assert torch.equal(selected, maps[:, 5, 3].unsqueeze(1))
    selected.sum().backward()
    assert maps.grad[:, 5, 3].eq(1).all()
    assert maps.grad.sum() == 6
    with pytest.raises(ValueError, match="outside"):
        sample_query_tracks(maps, torch.tensor([[-0.1, 0]]))


def test_pair_gradients_share_anchor_and_self_pair_branches():
    model = PairModel()
    forward = St4RTrackForward(model, torch.tensor([[3.8, 5.6]]))
    rgb = torch.full((2, 3, 8, 16), 0.7, requires_grad=True)
    forward(rgb, 0).sum().backward()
    assert rgb.grad[0, :, 5, 3].eq(6).all()
    assert rgb.grad[1].eq(0).all()
    assert model.weight.grad is None and not model.weight.requires_grad
    rgb.grad = None
    forward(rgb, 1).sum().backward()
    assert rgb.grad[0, :, 5, 3].eq(2).all()
    assert rgb.grad[1, :, 5, 3].eq(4).all()
    assert torch.isfinite(rgb.grad).all()


def test_official_loader_normalizes_world_and_keeps_later_occluded_queries(tmp_path):
    # Runs through the actual unmodified vendor loader/preprocessor. The camera
    # moves +1 in W2C translation; a stationary world point moves +1 in camera x.
    from PIL import Image
    jpeg = BytesIO()
    Image.new("RGB", (160, 90), (60, 120, 180)).save(jpeg, format="JPEG")
    images = np.array([jpeg.getvalue(), jpeg.getvalue()], dtype=object)
    cam = np.array([[[4, 0, 2], [3, 0, 2]], [[5, 0, 2], [4.1, 0, 2]]], dtype=np.float32)
    extrinsics = np.tile(np.eye(4), (2, 1, 1))
    extrinsics[:, 0, 3] = [3, 4]
    path = tmp_path / "synthetic.npz"
    np.savez(path, images_jpeg_bytes=images, tracks_XYZ=cam,
             fx_fy_cx_cy=np.array([10, 10, 80, 45]),
             visibility=np.array([[True, True], [False, False]]), extrinsics_w2c=extrinsics)
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "St4RTrack"
    sequence = load_sequence(path, num_frames=2, vendor_root=vendor_root)
    assert tuple(sequence.rgb.shape) == (2, 3, 288, 512)
    torch.testing.assert_close(sequence.gt_tracks[:, 0], torch.tensor([[4., 0, 2], [4, 0, 2]]))
    assert sequence.valid.all()
    assert sequence.dynamic.tolist() == [False, True]
    assert torch.equal(normalize_rgb(sequence.rgb[0]), sequence.official_views[0]["img"][0])
    np.testing.assert_allclose(sequence.metadata["first_extrinsic_w2c"], np.eye(4))
