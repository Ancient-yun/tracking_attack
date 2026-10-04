"""Compact native-loss components and exact all-frame RGB attacks.

The evaluation tracks and training raster queries are distinct populations.
Each pair returns evaluation ``tracks``, ``native_tracks``, native head-1
confidence and differentiable head-2 sufficient statistics. ``reconstruction_norm``
is the actual upstream normalization factor (mean head-2 norm divided by three
for the pinned source's unmasked coordinate-count quirk). Reconstruction
statistics already contain the upstream per-frame normalization and mask.
This module never substitutes evaluation median alignment for native training
normalization. The median MSE remains a separate control objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping

import torch
from torch import Tensor

from .pgd_video import AttackConfig, _validate_rgb, project_rgb
from .tracking_loss import aligned_tracking_loss, global_median_scale

ComponentPairFn = Callable[[Tensor, int], dict[str, Tensor]]
ComponentFramesFn = Callable[[Tensor, Tensor, int], dict[str, Tensor]]
OUTPUT_KEYS = (
    "tracks", "native_tracks", "track_conf", "reconstruction_norm",
    "reconstruction_regression_sum", "reconstruction_log_conf_sum",
    "reconstruction_l21_sum", "reconstruction_conf_sum",
)
RECON_SUM_KEYS = OUTPUT_KEYS[4:]
OBJECTIVES = {
    "tracking_mse": "tracking_mse",
    "tracking_3d": "tracking_regression",
    "reconstruction_3d": "reconstruction_regression",
    "confidence": "confidence",
    "joint_training": "total",
}


@dataclass
class ComponentGradientResult:
    loss: float
    scale: float
    tracks: Tensor
    outputs: dict[str, Tensor]
    terms: dict
    gradient: Tensor


@dataclass
class ComponentAttackResult:
    rgb: Tensor
    delta: Tensor
    tracks: Tensor
    outputs: dict[str, Tensor]
    loss: float
    scale: float
    terms: dict
    history: list[dict]
    elapsed_seconds: float
    selected_state: dict[str, int]


def _validate_outputs(outputs: Mapping[str, Tensor], *, stacked: bool, device=None) -> None:
    if not isinstance(outputs, Mapping):
        raise TypeError("component pair must return a tensor dictionary")
    missing = set(OUTPUT_KEYS) - set(outputs)
    if missing:
        raise ValueError(f"component outputs missing fields: {sorted(missing)}")
    for name in OUTPUT_KEYS:
        value = outputs[name]
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(f"component output {name} must be a floating tensor")
        if device is not None and value.device != device:
            raise ValueError(f"component output {name} moved off the RGB device")
    # Keep one host decision per pair rather than synchronizing after every
    # scalar reduction. Shape/type/device checks remain immediate metadata.
    value_checks = torch.stack([
        *(torch.isfinite(outputs[name]).all() for name in OUTPUT_KEYS),
        (outputs["track_conf"] > 0).all(),
        (outputs["reconstruction_norm"] >= 0).all(),
    ]).detach().cpu().tolist()
    for name, finite in zip(OUTPUT_KEYS, value_checks):
        if not finite:
            raise FloatingPointError(f"component output {name} contains NaN or Inf")
    rank = 3 if stacked else 2
    for name in ("tracks", "native_tracks"):
        value = outputs[name]
        if value.ndim != rank or value.shape[-1] != 3 or value.numel() == 0:
            raise ValueError(f"component {name} has invalid shape {tuple(value.shape)}")
    if outputs["track_conf"].shape != outputs["native_tracks"].shape[:-1]:
        raise ValueError("track_conf shape must match the native query population")
    if not value_checks[len(OUTPUT_KEYS)]:
        raise ValueError("native track confidence must be positive")
    scalar_shape = (outputs["tracks"].shape[0],) if stacked else ()
    for name in OUTPUT_KEYS[3:]:
        if outputs[name].shape != scalar_shape:
            raise ValueError(f"component {name} must have shape {scalar_shape}")
    if stacked and outputs["native_tracks"].shape[0] != outputs["tracks"].shape[0]:
        raise ValueError("evaluation and native track frame counts differ")
    if not value_checks[len(OUTPUT_KEYS) + 1]:
        raise ValueError("head-2 normalization cannot be negative")


def _validate_targets(outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor]) -> None:
    expected = {"tracks", "valid", "native_tracks", "native_track_valid",
                "training_dynamic", "reconstruction_gt_norm", "reconstruction_valid_counts"}
    missing = expected - set(targets)
    if missing:
        raise ValueError(f"component targets missing fields: {sorted(missing)}")
    device = outputs["tracks"].device
    for name in expected:
        if not isinstance(targets[name], Tensor) or targets[name].device != device:
            raise ValueError(f"target {name} must be a tensor on the output device")
        if targets[name].requires_grad:
            raise ValueError(f"GT target {name} must be fixed, without gradients")
    for points, mask in (("tracks", "valid"), ("native_tracks", "native_track_valid")):
        if targets[points].shape != outputs[points].shape:
            raise ValueError(f"target {points} shape differs from its prediction")
        if not targets[points].is_floating_point():
            raise TypeError(f"target {points} must be floating point")
        if targets[mask].dtype != torch.bool or targets[mask].shape != targets[points].shape[:-1]:
            raise ValueError(f"target {mask} must be the fixed boolean query mask")
        if not bool(targets[mask].any()):
            raise ValueError(f"target {mask} selects no points")
        if not bool(torch.isfinite(targets[points][targets[mask]]).all()):
            raise FloatingPointError(f"target {points} contains NaN or Inf at valid points")
    dynamic = targets["training_dynamic"]
    if dynamic.dtype != torch.bool or dynamic.shape != outputs["native_tracks"].shape[1:2]:
        raise ValueError("training_dynamic must be a fixed boolean native-query vector")
    frame_shape = outputs["tracks"].shape[:1]
    for name in ("reconstruction_gt_norm", "reconstruction_valid_counts"):
        value = targets[name]
        if value.shape != frame_shape or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"target {name} must be a finite nonnegative [T] tensor")
    if not bool(targets["reconstruction_valid_counts"].sum() > 0):
        raise ValueError("reconstruction mask selects no valid depth pixels")


def component_loss_terms(
    outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor], *,
    alpha: float = 0.2, reweight_scale: float = 5.0,
) -> dict:
    """Native active terms plus a separate median-aligned MSE control.

    Native head-1 masks/query rasterization and dynamic classification are
    established by the data adapter. Dynamic confidence is replaced by a
    detached multiple of the maximum static confidence over the whole sequence,
    matching upstream sequence-mode ``ConfLoss``. Reconstruction sums are
    divided by the total valid pixel count, never averaged frame by frame.
    """
    if not (0 < alpha < float("inf")) or not (0 < reweight_scale < float("inf")):
        raise ValueError("alpha and reweight_scale must be finite and positive")
    _validate_outputs(outputs, stacked=True)
    _validate_targets(outputs, targets)
    native_mask = targets["native_track_valid"]
    dynamic = targets["training_dynamic"].unsqueeze(0).expand_as(native_mask)[native_mask]
    raw_conf = outputs["track_conf"][native_mask]
    if not bool((~dynamic).any()):
        raise ValueError("native confidence reweighting has no static support")
    static_max = raw_conf[~dynamic].max().detach()
    weights = torch.where(dynamic, static_max * reweight_scale, raw_conf)
    norm = outputs["reconstruction_norm"].clamp_min(1e-8)[:, None, None]
    gt_norm = targets["reconstruction_gt_norm"].clamp_min(1e-8)[:, None, None]
    # Mask before subtraction so unused GT padding cannot contaminate losses.
    normalized_pred = (outputs["native_tracks"] / norm)[native_mask]
    normalized_gt = (targets["native_tracks"] / gt_norm)[native_mask]
    distances = torch.linalg.vector_norm(normalized_pred - normalized_gt, dim=-1)
    tracking_regression = (distances * weights).mean()
    track_log = torch.log(weights.clamp_min(1)).mean()
    count = targets["reconstruction_valid_counts"].sum()
    reconstruction_regression = outputs["reconstruction_regression_sum"].sum() / count
    reconstruction_log = outputs["reconstruction_log_conf_sum"].sum() / count
    confidence = -alpha * (track_log + reconstruction_log)
    total = tracking_regression + reconstruction_regression + confidence
    terms = {
        "tracking_mse": aligned_tracking_loss(outputs["tracks"], targets["tracks"], targets["valid"]),
        "tracking_regression": tracking_regression,
        "reconstruction_regression": reconstruction_regression,
        "confidence": confidence,
        "total": total,
        "diagnostics": {
            "tracking_l21": distances.mean(),
            "reconstruction_l21": outputs["reconstruction_l21_sum"].sum() / count,
            "track_conf_mean": weights.mean(),
            "track_raw_conf_mean": raw_conf.mean(),
            "static_conf_max_detached": static_max,
            "reconstruction_conf_mean": outputs["reconstruction_conf_sum"].sum() / count,
            "track_log_conf_mean": track_log,
            "reconstruction_log_conf_mean": reconstruction_log,
            "native_track_count": native_mask.sum().to(distances.dtype),
            "reconstruction_count": count.to(distances.dtype),
        },
    }
    finite_terms = [(name, item) for name, value in terms.items()
                    for item in (value.values() if isinstance(value, dict) else (value,))]
    finite_checks = torch.stack([torch.isfinite(item).all() for _, item in finite_terms]).detach().cpu().tolist()
    for (name, _), finite in zip(finite_terms, finite_checks):
        if not finite:
            raise FloatingPointError(f"component loss {name} is not finite")
    return terms


def _json_terms(terms: dict) -> dict:
    return {name: _json_terms(value) if isinstance(value, dict) else float(value.detach())
            for name, value in terms.items()}


def _objective(terms: dict, objective: str) -> Tensor:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown component objective: {objective}")
    return terms[OBJECTIVES[objective]]


def _pair_outputs(pair_fn: ComponentPairFn, rgb: Tensor, t: int) -> dict[str, Tensor]:
    outputs = pair_fn(rgb, t)
    _validate_outputs(outputs, stacked=False, device=rgb.device)
    return {name: outputs[name] for name in OUTPUT_KEYS}


def _stack_outputs(pieces: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    for name in OUTPUT_KEYS:
        if any(piece[name].shape != pieces[0][name].shape for piece in pieces):
            raise ValueError(f"component output {name} shape changed between frame pairs")
    return {name: torch.stack([piece[name] for piece in pieces]) for name in OUTPUT_KEYS}


@torch.no_grad()
def predict_components(pair_fn: ComponentPairFn, rgb: Tensor) -> dict[str, Tensor]:
    """Retain only compact values; no pair model graphs survive this call."""
    _validate_rgb(rgb)
    return _stack_outputs([_pair_outputs(pair_fn, rgb, t) for t in range(len(rgb))])


def _detach_outputs(outputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach() for name, value in outputs.items()}


def component_loss_and_gradient(
    pair_fn: ComponentPairFn, rgb: Tensor, targets: Mapping[str, Tensor], objective: str,
    *, mode: str = "recompute", check_replay: bool = True,
    alpha: float = 0.2, reweight_scale: float = 5.0,
    pair_frames_fn: ComponentFramesFn | None = None,
) -> ComponentGradientResult:
    """Exact multi-output VJP, including head-2 normalization and confidence.

    Unused fields have no VJP. Zero native gradients remain valid; missing RGB
    connections and nonfinite values are explicit failures. Both input roles of
    pair (0,0) and every shared frame-zero contribution are added automatically.
    An explicit ``pair_frames_fn(first, other, t)`` enables frame-local replay
    inputs in recompute mode. Generic pair functions and the full-graph
    reference continue to receive the entire video.
    """
    _validate_rgb(rgb)
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown component objective: {objective}")
    if mode not in {"full", "recompute"}:
        raise ValueError("component gradient mode must be 'full' or 'recompute'")
    with torch.enable_grad():
        leaf = rgb.detach().requires_grad_(True)
        if mode == "full":
            compact = _stack_outputs([_pair_outputs(pair_fn, leaf, t) for t in range(len(rgb))])
        else:
            compact = {name: value.detach().requires_grad_(True)
                       for name, value in predict_components(pair_fn, leaf).items()}
        terms = component_loss_terms(compact, targets, alpha=alpha, reweight_scale=reweight_scale)
        loss = _objective(terms, objective)
        scale = global_median_scale(compact["tracks"], targets["tracks"], targets["valid"])
        if not loss.requires_grad:
            raise RuntimeError("component objective is detached from model outputs")
        if mode == "full":
            gradient = torch.autograd.grad(loss, leaf, allow_unused=True)[0]
            if gradient is None:
                raise RuntimeError("component forward is disconnected from RGB input")
        else:
            vjps = torch.autograd.grad(loss, tuple(compact.values()), allow_unused=True)
            upstream = dict(zip(compact, vjps))
            used_vjps = [(name, value) for name, value in upstream.items() if value is not None]
            finite_vjps = torch.stack([torch.isfinite(value).all() for _, value in used_vjps]).detach().cpu().tolist()
            for (name, _), finite in zip(used_vjps, finite_vjps):
                if not finite:
                    raise FloatingPointError(f"compact VJP {name} contains NaN or Inf")
            gradient = torch.zeros_like(leaf)
            for t in range(len(rgb)):
                if pair_frames_fn is None:
                    inputs = (leaf,)
                    replay = _pair_outputs(pair_fn, leaf, t)
                else:
                    first = rgb[0].detach().requires_grad_(True)
                    other = first if t == 0 else rgb[t].detach().requires_grad_(True)
                    inputs = (first,) if t == 0 else (first, other)
                    replay = pair_frames_fn(first, other, t)
                    _validate_outputs(replay, stacked=False, device=rgb.device)
                    replay = {name: replay[name] for name in OUTPUT_KEYS}
                active_outputs, active_vjps = [], []
                for name in OUTPUT_KEYS:
                    if replay[name].shape != compact[name][t].shape:
                        raise ValueError(f"component replay shape changed at frame {t}: {name}")
                if check_replay:
                    replay_checks = torch.stack([
                        torch.isclose(replay[name].detach(), compact[name][t], rtol=1e-5, atol=1e-6).all()
                        for name in OUTPUT_KEYS
                    ]).detach().cpu().tolist()
                    for name, matches in zip(OUTPUT_KEYS, replay_checks):
                        if matches:
                            continue
                        error = float((replay[name].detach() - compact[name][t]).abs().max())
                        raise RuntimeError(f"component replay changed at frame {t}: {name} (max difference {error:.3g})")
                for name in OUTPUT_KEYS:
                    if upstream[name] is None:
                        continue
                    vjp = upstream[name][t].detach()
                    if not replay[name].requires_grad:
                        empty_reconstruction = (name in RECON_SUM_KEYS and
                            not bool(targets["reconstruction_valid_counts"][t] > 0) and
                            not bool(replay[name] != 0))
                        if empty_reconstruction:
                            continue
                        raise RuntimeError(f"active component {name} is detached at frame {t}")
                    active_outputs.append(replay[name])
                    active_vjps.append(vjp)
                if active_outputs:
                    pieces = torch.autograd.grad(tuple(active_outputs), inputs,
                                                 grad_outputs=tuple(active_vjps), allow_unused=True)
                    connected_pieces = [piece for piece in pieces if piece is not None]
                    if not connected_pieces:
                        raise RuntimeError(f"component forward is disconnected from RGB input at frame {t}")
                    if not bool(torch.stack([torch.isfinite(piece).all() for piece in connected_pieces]).all()):
                        raise FloatingPointError(f"component RGB gradient contains NaN or Inf at frame {t}")
                    if pair_frames_fn is None:
                        gradient.add_(pieces[0])
                    else:
                        # For t=0 the same leaf was passed into both input
                        # roles, so its one VJP already contains both paths.
                        if pieces[0] is not None:
                            gradient[0].add_(pieces[0])
                        if t != 0 and pieces[1] is not None:
                            gradient[t].add_(pieces[1])
                del replay
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("summed component RGB gradient contains NaN or Inf")
        detached = _detach_outputs(compact)
        return ComponentGradientResult(float(loss.detach()), float(scale.detach()), detached["tracks"],
                                       detached, _json_terms(terms), gradient.detach())


@torch.no_grad()
def _evaluate(pair_fn, rgb, targets, objective, alpha, reweight_scale):
    outputs = predict_components(pair_fn, rgb)
    terms = component_loss_terms(outputs, targets, alpha=alpha, reweight_scale=reweight_scale)
    loss = float(_objective(terms, objective))
    scale = float(global_median_scale(outputs["tracks"], targets["tracks"], targets["valid"]))
    return outputs, loss, scale, _json_terms(terms)


def _history_entry(restart, step, rgb, clean, loss, scale, terms, gradient, per_frame):
    flat = (rgb - clean).flatten(1)
    entry = {"restart": restart, "step": step, "loss": loss, "scale": scale,
             "terms": terms, "delta_linf": float(flat.abs().max())}
    if per_frame:
        entry["delta_linf_per_frame"] = flat.abs().amax(dim=1).cpu().tolist()
        entry["delta_l2_per_frame"] = torch.linalg.vector_norm(flat, dim=1).cpu().tolist()
        if gradient is not None:
            grad = gradient.flatten(1)
            entry["gradient_linf_per_frame"] = grad.abs().amax(dim=1).cpu().tolist()
            entry["gradient_l2_per_frame"] = torch.linalg.vector_norm(grad, dim=1).cpu().tolist()
    return entry


def component_attack_video(
    pair_fn: ComponentPairFn, clean_rgb: Tensor, targets: Mapping[str, Tensor], objective: str,
    config: AttackConfig | None = None, *, alpha: float = 0.2, reweight_scale: float = 5.0,
    progress_callback: Callable[[dict], None] | None = None,
    pair_frames_fn: ComponentFramesFn | None = None,
) -> ComponentAttackResult:
    """Projected ascent with an objective-specific, clean-inclusive incumbent.

    The RNG depends only on AttackConfig.seed, never the objective ID, enabling
    identical restart noise across all five loss ablations. Elapsed time is
    synchronized for CUDA inputs and covers all forwards inside this function.
    """
    config = config or AttackConfig()
    _validate_rgb(clean_rgb)
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown component objective: {objective}")
    if clean_rgb.device.type == "cuda":
        torch.cuda.synchronize(clean_rgb.device)
    started = perf_counter()
    clean = clean_rgb.detach().clone()
    best_outputs, best_loss, best_scale, best_terms = _evaluate(pair_fn, clean, targets, objective, alpha, reweight_scale)
    best_rgb = clean.clone()
    selected_state = {"restart": -1, "step": 0}
    history = []

    def record(restart, step, current, loss, scale, terms, gradient=None):
        entry = _history_entry(restart, step, current, clean, loss, scale, terms, gradient, config.record_per_frame)
        entry["objective"] = objective
        history.append(entry)
        if progress_callback is not None:
            progress_callback(dict(entry))

    record(-1, 0, clean, best_loss, best_scale, best_terms)

    def result():
        delta = best_rgb - clean
        tolerance = 8 * torch.finfo(clean.dtype).eps
        if float(delta.abs().max()) > config.eps + tolerance:
            raise RuntimeError("saved component attack exceeds its L-infinity bound")
        _validate_rgb(best_rgb)
        if clean.device.type == "cuda":
            torch.cuda.synchronize(clean.device)
        return ComponentAttackResult(best_rgb.detach(), delta.detach(), best_outputs["tracks"], best_outputs,
                                     best_loss, best_scale, best_terms, history, perf_counter() - started,
                                     dict(selected_state))

    if config.method == "clean" or config.eps == 0:
        return result()
    generator = torch.Generator(device=clean.device).manual_seed(config.seed)

    def random_rgb():
        delta = torch.empty_like(clean).uniform_(-config.eps, config.eps, generator=generator)
        return project_rgb(clean + delta, clean, config.eps).detach()

    if config.method == "noise":
        best_rgb = random_rgb()
        best_outputs, best_loss, best_scale, best_terms = _evaluate(pair_fn, best_rgb, targets, objective, alpha, reweight_scale)
        selected_state = {"restart": 0, "step": 0}
        record(0, 0, best_rgb, best_loss, best_scale, best_terms)
        return result()

    restarts = config.restarts if config.method == "pgd" else 1
    steps = config.steps if config.method == "pgd" else 1
    step_size = config.eps if config.method == "fgsm" else (
        config.step_size if config.step_size is not None else config.eps / 4)
    for restart in range(restarts):
        current = random_rgb() if config.method == "pgd" and config.random_start else clean.clone()
        for step in range(steps):
            derivative = component_loss_and_gradient(pair_fn, current, targets, objective,
                mode=config.gradient_mode, check_replay=config.check_replay, alpha=alpha, reweight_scale=reweight_scale,
                pair_frames_fn=pair_frames_fn)
            record(restart, step, current, derivative.loss, derivative.scale, derivative.terms, derivative.gradient)
            if derivative.loss > best_loss:
                best_rgb, best_outputs = current.clone(), _detach_outputs(derivative.outputs)
                best_loss, best_scale, best_terms = derivative.loss, derivative.scale, derivative.terms
                selected_state = {"restart": restart, "step": step}
            current = project_rgb(current + step_size * derivative.gradient.sign(), clean, config.eps).detach()
        outputs, loss, scale, terms = _evaluate(pair_fn, current, targets, objective, alpha, reweight_scale)
        record(restart, steps, current, loss, scale, terms)
        if config.method == "fgsm" or loss > best_loss:
            best_rgb, best_outputs = current.clone(), outputs
            best_loss, best_scale, best_terms = loss, scale, terms
            selected_state = {"restart": restart, "step": steps}
    return result()
