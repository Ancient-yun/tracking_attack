"""All-frame RGB attacks with exact two-pass sequence-loss gradients.

``pair_fn(rgb, t)`` receives [T,3,H,W] RGB in [0,1] and returns [N,3]
tracks for the model pair (rgb[0], rgb[t]). For t=0 both views MUST use the
same rgb[0] tensor. No detach/no_grad may be applied inside this callable.
It must be deterministic: freeze the model, call eval(), and disable TTA.

Recompute stores just compact query tracks, differentiates the global loss
including its scale, and replays one pair at a time. Every replay differentiates
against the same video leaf, so all frame-0 contributions (including both
inputs of pair (0,0)) are added. No per-pair alignment is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

import torch
from torch import Tensor

from .tracking_loss import aligned_tracking_loss, global_median_scale

PairFn = Callable[[Tensor, int], Tensor]
GradientMode = Literal["recompute", "full"]
AttackMethod = Literal["clean", "noise", "fgsm", "pgd"]


@dataclass(frozen=True)
class AttackConfig:
    method: AttackMethod = "pgd"
    eps: float = 4.0 / 255.0
    steps: int = 20
    step_size: float | None = None
    restarts: int = 1
    seed: int = 0
    random_start: bool = True
    gradient_mode: GradientMode = "recompute"
    check_replay: bool = True
    record_per_frame: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"clean", "noise", "fgsm", "pgd"}:
            raise ValueError(f"unknown attack method: {self.method}")
        if not 0.0 <= self.eps <= 1.0:
            raise ValueError("eps must be finite and in [0,1]")
        if self.steps < 1 or not isinstance(self.steps, int):
            raise ValueError("steps must be a positive integer")
        if self.restarts < 1 or not isinstance(self.restarts, int):
            raise ValueError("restarts must be a positive integer")
        if self.step_size is not None and (not 0.0 <= self.step_size <= 1.0):
            raise ValueError("step_size must be finite and in [0,1]")
        if self.gradient_mode not in {"recompute", "full"}:
            raise ValueError("gradient_mode must be 'recompute' or 'full'")


@dataclass
class GradientResult:
    loss: float
    scale: float
    tracks: Tensor
    gradient: Tensor


@dataclass
class AttackResult:
    rgb: Tensor
    delta: Tensor
    tracks: Tensor
    loss: float
    scale: float
    history: list[dict]
    elapsed_seconds: float


def _validate_rgb(rgb: Tensor) -> None:
    if rgb.ndim != 4 or rgb.shape[1] != 3 or any(size < 1 for size in rgb.shape):
        raise ValueError(f"RGB must have nonempty shape [T,3,H,W], got {tuple(rgb.shape)}")
    if not rgb.is_floating_point():
        raise TypeError("RGB must be a floating point tensor in [0,1]")
    if not bool(torch.isfinite(rgb).all()):
        raise FloatingPointError("RGB contains NaN or Inf")
    if bool((rgb < 0).any()) or bool((rgb > 1).any()):
        raise ValueError("RGB is outside the attack space [0,1]")


def _pair_tracks(pair_fn: PairFn, rgb: Tensor, t: int) -> Tensor:
    points = pair_fn(rgb, t)
    if not isinstance(points, Tensor):
        raise TypeError(f"pair_fn(frame={t}) must return a tensor")
    if points.ndim != 2 or points.shape[-1] != 3 or points.shape[0] == 0:
        raise ValueError(f"pair_fn(frame={t}) returned {tuple(points.shape)}; expected [N,3]")
    if points.device != rgb.device:
        raise ValueError(f"pair_fn(frame={t}) moved tracks off the input device")
    if not bool(torch.isfinite(points).all()):
        raise FloatingPointError(f"pair_fn(frame={t}) returned NaN or Inf")
    return points


@torch.no_grad()
def predict_tracks(pair_fn: PairFn, rgb: Tensor) -> Tensor:
    """Compact [T,N,3] prediction without retaining model graphs."""
    _validate_rgb(rgb)
    pieces = [_pair_tracks(pair_fn, rgb, t).detach() for t in range(rgb.shape[0])]
    if any(points.shape != pieces[0].shape for points in pieces):
        raise ValueError("query count changed between frame pairs")
    return torch.stack(pieces)


def _checked_gradient(output: Tensor, input_rgb: Tensor, grad_outputs: Tensor | None = None) -> Tensor:
    if not output.requires_grad:
        raise RuntimeError("model tracks are detached; the pair forward must retain input gradients")
    gradient = torch.autograd.grad(output, input_rgb, grad_outputs=grad_outputs, allow_unused=True)[0]
    if gradient is None:
        raise RuntimeError("pair forward is disconnected from RGB input")
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("input gradient contains NaN or Inf")
    return gradient


def loss_and_gradient(
    pair_fn: PairFn,
    rgb: Tensor,
    gt: Tensor,
    valid: Tensor,
    *,
    mode: GradientMode = "recompute",
    check_replay: bool = True,
) -> GradientResult:
    """Return the exact gradient of the common sequence-aligned objective.

    ``full`` retains every pair graph for small-clip verification. ``recompute``
    uses one graph at a time. Inputs and outputs are detached from caller graphs;
    model parameter .grad buffers are never populated by this function.
    Replay equality is checked to detect dropout/TTA/stateful callables.
    """
    _validate_rgb(rgb)
    if mode not in {"recompute", "full"}:
        raise ValueError("gradient mode must be 'recompute' or 'full'")
    with torch.enable_grad():
        rgb_leaf = rgb.detach().requires_grad_(True)
        if mode == "full":
            tracks = torch.stack([_pair_tracks(pair_fn, rgb_leaf, t) for t in range(rgb.shape[0])])
            loss = aligned_tracking_loss(tracks, gt, valid)
            scale = global_median_scale(tracks, gt, valid)
            gradient = _checked_gradient(loss, rgb_leaf)
            return GradientResult(float(loss.detach()), float(scale.detach()), tracks.detach(), gradient.detach())

        # Forward values are cheap to retain: only N query points per pair.
        compact = predict_tracks(pair_fn, rgb_leaf).requires_grad_(True)
        loss = aligned_tracking_loss(compact, gt, valid)
        scale = global_median_scale(compact, gt, valid)
        output_gradient = torch.autograd.grad(loss, compact)[0].detach()
        if not bool(torch.isfinite(output_gradient).all()):
            raise FloatingPointError("compact track gradient contains NaN or Inf")
        gradient = torch.zeros_like(rgb_leaf)
        for t in range(rgb.shape[0]):
            replay = _pair_tracks(pair_fn, rgb_leaf, t)
            if replay.shape != compact[t].shape:
                raise ValueError(f"replayed query shape changed at frame {t}")
            if check_replay and not torch.allclose(replay.detach(), compact[t], rtol=1e-5, atol=1e-6):
                error = float((replay.detach() - compact[t]).abs().max())
                raise RuntimeError(
                    f"pair replay changed at frame {t} (max difference {error:.3g}); "
                    "disable dropout/TTA and make the forward deterministic"
                )
            gradient.add_(_checked_gradient(replay, rgb_leaf, output_gradient[t]))
            # No graph for any previous pair survives this iteration.
            del replay
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("summed all-frame gradient contains NaN or Inf")
        return GradientResult(float(loss.detach()), float(scale.detach()), compact.detach(), gradient.detach())


def project_rgb(candidate: Tensor, clean_rgb: Tensor, eps: float) -> Tensor:
    """Intersect the per-pixel L-infinity ball with valid RGB [0,1]."""
    if candidate.shape != clean_rgb.shape:
        raise ValueError("candidate and clean RGB must have identical shape")
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0,1]")
    return torch.maximum(torch.minimum(candidate, clean_rgb + eps), clean_rgb - eps).clamp(0.0, 1.0)


def _evaluate(pair_fn: PairFn, rgb: Tensor, gt: Tensor, valid: Tensor) -> tuple[Tensor, float, float]:
    tracks = predict_tracks(pair_fn, rgb)
    with torch.no_grad():
        loss = float(aligned_tracking_loss(tracks, gt, valid))
        scale = float(global_median_scale(tracks, gt, valid))
    return tracks, loss, scale


def _history_entry(
    restart: int,
    step: int,
    rgb: Tensor,
    clean: Tensor,
    loss: float,
    scale: float,
    gradient: Tensor | None,
    per_frame: bool,
) -> dict:
    delta_flat = (rgb - clean).flatten(1)
    entry = {"restart": restart, "step": step, "loss": loss, "scale": scale,
             "delta_linf": float(delta_flat.abs().max())}
    if per_frame:
        entry["delta_linf_per_frame"] = delta_flat.abs().amax(dim=1).cpu().tolist()
        entry["delta_l2_per_frame"] = torch.linalg.vector_norm(delta_flat, dim=1).cpu().tolist()
        if gradient is not None:
            grad_flat = gradient.flatten(1)
            entry["gradient_linf_per_frame"] = grad_flat.abs().amax(dim=1).cpu().tolist()
            entry["gradient_l2_per_frame"] = torch.linalg.vector_norm(grad_flat, dim=1).cpu().tolist()
    return entry


def attack_video(
    pair_fn: PairFn,
    clean_rgb: Tensor,
    gt: Tensor,
    valid: Tensor,
    config: AttackConfig | None = None,
    *,
    progress_callback: Callable[[dict], None] | None = None,
) -> AttackResult:
    """Run clean, uniform noise, FGSM, or projected all-frame sign-gradient ascent.

    PGD selects the highest objective over clean, every restart initialization,
    and every iterate. This prevents a poor restart or later step from replacing
    a stronger incumbent. Noise is one unbiased baseline sample, not optimized.
    FGSM returns its actual one eps-sized step even if that step lowers the loss.
    eps=0 returns exact clean.
    An optional callback receives each history entry, including the clean
    baseline, restart initializations, and final states; this module prints
    nothing by default. The callback must treat the entry as read-only.
    """
    config = config or AttackConfig()
    _validate_rgb(clean_rgb)
    started = perf_counter()
    clean = clean_rgb.detach().clone()
    tracks, clean_loss, clean_scale = _evaluate(pair_fn, clean, gt, valid)
    best_rgb, best_tracks = clean.clone(), tracks
    best_loss, best_scale = clean_loss, clean_scale
    history: list[dict] = []

    def record(entry: dict) -> None:
        history.append(entry)
        if progress_callback is not None:
            progress_callback(dict(entry))

    record(_history_entry(-1, 0, clean, clean, clean_loss, clean_scale, None, config.record_per_frame))

    def result() -> AttackResult:
        delta = best_rgb - clean
        # Tolerance accounts only for addition/subtraction rounding in float RGB.
        tolerance = 8 * torch.finfo(clean.dtype).eps
        if float(delta.abs().max()) > config.eps + tolerance:
            raise RuntimeError("internal error: saved attack exceeds its L-infinity bound")
        _validate_rgb(best_rgb)
        return AttackResult(best_rgb.detach(), delta.detach(), best_tracks.detach(),
                            best_loss, best_scale, history, perf_counter() - started)

    if config.method == "clean" or config.eps == 0:
        return result()

    generator = torch.Generator(device=clean.device).manual_seed(config.seed)

    def random_rgb() -> Tensor:
        noise = torch.empty_like(clean).uniform_(-config.eps, config.eps, generator=generator)
        return project_rgb(clean + noise, clean, config.eps).detach()

    if config.method == "noise":
        best_rgb = random_rgb()
        best_tracks, best_loss, best_scale = _evaluate(pair_fn, best_rgb, gt, valid)
        record(_history_entry(0, 0, best_rgb, clean, best_loss, best_scale, None, config.record_per_frame))
        return result()

    restarts = config.restarts if config.method == "pgd" else 1
    steps = config.steps if config.method == "pgd" else 1
    step_size = (config.step_size if config.step_size is not None else config.eps / 4.0)
    if config.method == "fgsm":
        step_size = config.eps

    for restart in range(restarts):
        current = random_rgb() if config.method == "pgd" and config.random_start else clean.clone()
        for step in range(steps):
            derivative = loss_and_gradient(pair_fn, current, gt, valid,
                                           mode=config.gradient_mode, check_replay=config.check_replay)
            record(_history_entry(restart, step, current, clean, derivative.loss,
                                  derivative.scale, derivative.gradient, config.record_per_frame))
            if derivative.loss > best_loss:
                best_rgb, best_tracks = current.clone(), derivative.tracks.clone()
                best_loss, best_scale = derivative.loss, derivative.scale
            current = project_rgb(current + step_size * derivative.gradient.sign(), clean, config.eps).detach()
        current_tracks, current_loss, current_scale = _evaluate(pair_fn, current, gt, valid)
        record(_history_entry(restart, steps, current, clean, current_loss, current_scale,
                              None, config.record_per_frame))
        if config.method == "fgsm" or current_loss > best_loss:
            best_rgb, best_tracks = current.clone(), current_tracks
            best_loss, best_scale = current_loss, current_scale
    return result()
