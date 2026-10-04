"""Differentiable sequence-global median alignment used by St4RTrack.

The reference is ``dust3r/track_eval_util.py::_compute_scale_factor_global``:
the scale is median(||GT||) / median(||prediction||), with norms clamped at
1e-12. NumPy's median averages the two central values for an even count;
``torch.median`` does not, so this module implements that convention explicitly.
The scale remains inside the autograd graph. There is no confidence weighting.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_tracks(pred: Tensor, gt: Tensor, valid: Tensor) -> None:
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"pred must have shape [T,N,3], got {tuple(pred.shape)}")
    if gt.shape != pred.shape:
        raise ValueError(f"GT shape {tuple(gt.shape)} differs from prediction {tuple(pred.shape)}")
    if valid.shape != pred.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must be a fixed boolean [T,N] GT mask")
    if pred.device != gt.device or pred.device != valid.device:
        raise ValueError("prediction, GT, and valid mask must be on the same device")
    if not pred.is_floating_point() or not gt.is_floating_point():
        raise TypeError("prediction and GT must be floating point")
    if not bool(valid.any()):
        raise ValueError("GT valid mask selects no points")
    # A bad prediction is an explicit failure even when a GT entry is masked.
    # This avoids quietly hiding a numerical model failure behind an eval mask.
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError("predicted 3D tracks contain NaN or Inf")
    if not bool(torch.isfinite(gt[valid]).all()):
        raise FloatingPointError("GT contains NaN or Inf at a valid position")


def numpy_style_median(values: Tensor) -> Tensor:
    """A differentiable scalar median with NumPy's even-count convention.

    At ties this uses the valid piecewise subgradient selected by ``sort``.
    Numerical derivative checks should therefore use untied interior values.
    """
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("median requires a nonempty one-dimensional tensor")
    ordered = values.sort().values
    midpoint = ordered.numel() // 2
    if ordered.numel() % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) * 0.5


def global_median_scale(pred: Tensor, gt: Tensor, valid: Tensor, eps: float = 1e-12) -> Tensor:
    """One common scale over all GT-valid frame/query positions.

    A mask consisting of the official retained queries at all evaluated times
    reproduces the upstream global scale exactly (within floating precision).
    The caller establishes the mask once; it must not change with predictions
    or confidence. Occluded points can remain valid per the upstream protocol.
    """
    _validate_tracks(pred, gt, valid)
    if eps <= 0:
        raise ValueError("scale epsilon must be positive")
    gt_norm = torch.linalg.vector_norm(gt[valid], dim=-1).clamp_min(eps)
    pred_norm = torch.linalg.vector_norm(pred[valid], dim=-1).clamp_min(eps)
    scale = numpy_style_median(gt_norm) / numpy_style_median(pred_norm).clamp_min(eps)
    if not bool(torch.isfinite(scale)):
        raise FloatingPointError("global median scale is not finite")
    return scale


def aligned_tracking_loss(pred: Tensor, gt: Tensor, valid: Tensor) -> Tensor:
    """Mean squared 3D distance after differentiable global median alignment."""
    scale = global_median_scale(pred, gt, valid)
    # Select before subtracting: invalid GT may contain NaN padding.
    residual = pred[valid] * scale - gt[valid]
    loss = residual.square().sum(dim=-1).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("aligned squared tracking loss is not finite")
    return loss
