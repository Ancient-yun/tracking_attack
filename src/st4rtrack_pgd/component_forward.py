"""The two active native training losses on fixed WorldTrack targets.

The author Seqmode checkpoint uses confidence-weighted, scale-normalized L21
regression of both pointmap heads. This module retains both heads' input
gradients and the pinned source's normalization quirks. Native loss queries
are rasterized independently from the official evaluation's truncated queries.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import numpy as np
import torch
from torch import Tensor

from .common import OFFICIAL_COMMIT, sha256
from .st4rtrack_forward import St4RTrackForward, normalize_rgb
from .runtime_optimization import apply_runtime_optimizations, runtime_optimization_metadata
from .worldtrack_adapter import (
    WorldTrackSequence, ensure_vendor_importable, load_sequence, sample_query_tracks,
)


def tensor_sha256(value: Tensor) -> str:
    """Hash dtype, shape and exact contiguous CPU bytes, without a byte copy."""
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(f"{array.dtype}:{array.shape}:".encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def native_pointmap_norm(pointmaps: Tensor) -> Tensor:
    """The actual upstream avg_dis scale: sum(norm)/(3*H*W), per frame.

    losses.py passes masks positionally into a signature with an extra
    traj_mask argument, leaving valid2=None. misc.invalid_to_zeros then counts
    xyz components, rather than pixels. Both effects are intentionally retained.
    """
    if pointmaps.ndim not in (3, 4) or pointmaps.shape[-1] != 3:
        raise ValueError("Expected [H,W,3] or [T,H,W,3] pointmaps")
    # Flatten and divide once, matching the official reduction operation order.
    nnz = pointmaps.shape[-3] * pointmaps.shape[-2] * 3
    return (pointmaps.norm(dim=-1).flatten(-2).sum(-1) / (nnz + 1e-8)).clamp_min(1e-8)


def rasterize_native_queries(
    query_xy: Tensor, world_tracks: Tensor, image_wh: tuple[int, int],
    point_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Round/clamp initial queries; keep the last GT at colliding pixels.

    The original datasets assign their query array into a dense sparse map.
    This reproduces its last-assignment collision rule explicitly, then returns
    points in row-major pixel order, as Boolean indexing in ConfLoss does.
    """
    if query_xy.ndim != 2 or query_xy.shape[-1] != 2:
        raise ValueError("query_xy must have shape [N,2]")
    if world_tracks.ndim != 3 or world_tracks.shape[1:] != (len(query_xy), 3):
        raise ValueError("world_tracks must have shape [T,N,3]")
    w, h = image_wh
    if min(w, h) <= 0 or len(query_xy) == 0:
        raise ValueError("Native rasterization needs positive dimensions and queries")
    if not torch.isfinite(query_xy).all() or not torch.isfinite(world_tracks).all():
        raise ValueError("Native GT queries contain nonfinite values")
    pixels = query_xy.detach().cpu().round().long()
    pixels[:, 0].clamp_(0, w - 1)
    pixels[:, 1].clamp_(0, h - 1)
    flat = pixels[:, 1] * w + pixels[:, 0]
    # Explicit last writer avoids relying on undefined GPU duplicate scatter.
    last = {int(pixel): i for i, pixel in enumerate(flat.tolist())}
    selected = torch.tensor([last[pixel] for pixel in sorted(last)], dtype=torch.long)
    source = torch.arange(len(query_xy)) if point_indices is None else point_indices.cpu()
    return pixels[selected], world_tracks[:, selected.to(world_tracks.device)], source[selected]


@dataclass
class NativeTargets(Mapping[str, Any]):
    """Fixed GT context; attribute and mapping access are both supported."""

    native_tracks: Tensor                    # [T,M,3], first-camera world
    native_track_valid: Tensor               # [T,M], all initial queries
    training_dynamic: Tensor                 # [M], endpoint relative L1 > mean
    native_query_xy: Tensor                  # [M,2], rounded integer pixel
    native_point_indices: Tensor             # [M], original NPZ query index
    tracks: Tensor                           # [T,N,3], official eval GT
    valid: Tensor                            # [T,N], official eval membership
    reconstruction_gt: Tensor                # [T,H,W,3], dense world GT
    reconstruction_valid: Tensor             # [T,H,W], positive finite depth
    reconstruction_gt_norm: Tensor           # [T], upstream scale /3
    reconstruction_valid_counts: Tensor      # [T]
    camera_c2w: Tensor                       # [T,4,4], normalized first camera
    intrinsics: Tensor                       # [T,3,3], model-grid OpenCV K
    depth: Tensor                            # [T,H,W], NN depth
    metadata: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        if key not in {field.name for field in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(field.name for field in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def to(self, device: str | torch.device) -> "NativeTargets":
        return replace(self, **{
            field.name: getattr(self, field.name).to(device)
            for field in fields(self) if isinstance(getattr(self, field.name), Tensor)
        })

    def tensor_hashes(self) -> dict[str, str]:
        return {
            field.name: tensor_sha256(getattr(self, field.name))
            for field in fields(self) if isinstance(getattr(self, field.name), Tensor)
        }


def source_frame_counts(path: str | Path) -> dict[str, int]:
    """Read released temporal array lengths from NPY headers, without payloads.

    The official RGB loader slices its arrays before returning them. Inspecting
    ZIP members first preserves the full source count and catches mismatches
    outside the requested prefix, including object-encoded JPEG arrays.
    """
    required = ("images_jpeg_bytes", "tracks_XYZ", "visibility", "depth_map", "extrinsics_w2c")
    counts = {}
    with ZipFile(Path(path).resolve()) as archive:
        members = set(archive.namelist())
        missing = [name for name in required if f"{name}.npy" not in members]
        if missing:
            raise ValueError(f"Dense native GT needs released temporal NPZ fields {missing}")
        for name in required:
            with archive.open(f"{name}.npy") as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise ValueError(f"Unsupported released NPY header version {version}: {name}")
                shape, _fortran, _dtype = readers[version](stream)
            if not shape or shape[0] <= 0:
                raise ValueError(f"Released temporal field {name} has no frames: {shape}")
            counts[name] = int(shape[0])
    if len(set(counts.values())) != 1:
        raise ValueError(f"Released source frame counts disagree: {counts}")
    return counts


def load_component_sequence(
    path: str | Path, num_frames: int = 64, image_size: int = 512,
    vendor_root: str | Path | None = None,
) -> tuple[WorldTrackSequence, NativeTargets]:
    """Use official RGB/evaluation preprocessing and dense WorldTrack depth GT.

    Dense depth is resized with INTER_NEAREST to the actual RGB/model grid,
    usually 512x288. K's x/y rows scale with that grid, without introducing a
    crop or changing the existing evaluator's pixel convention. This applies
    the native loss to released benchmark GT; it does not recreate training
    augmentation, training mesh populations, or TTA pseudo-label generation.
    """
    frame_counts = source_frame_counts(path)
    original_frame_count = frame_counts["depth_map"]
    sequence = load_sequence(path, num_frames, image_size, vendor_root)
    root = ensure_vendor_importable(vendor_root)
    import cv2

    t_count, _, h, w = sequence.rgb.shape
    if t_count != min(num_frames, original_frame_count):
        raise ValueError("Loaded RGB count differs from the requested source prefix")
    frame_provenance = {
        "original_frame_count": original_frame_count,
        "source_frame_counts": frame_counts,
        "all_frames_used": t_count == original_frame_count,
        "used_frame_indices": list(range(t_count)),
    }
    sequence.metadata.update(frame_provenance)
    ow, oh = sequence.metadata["original_wh"]
    source_path = Path(path).resolve()
    with np.load(source_path, allow_pickle=True) as archive:
        required = {"depth_map", "extrinsics_w2c", "fx_fy_cx_cy"}
        if not required.issubset(archive.files):
            raise ValueError(f"Dense native GT needs NPZ fields {sorted(required)}")
        depth_source = archive["depth_map"]
        if depth_source.ndim != 3 or depth_source.shape[1:] != (oh, ow) or len(depth_source) < t_count:
            raise ValueError("Depth maps do not match decoded RGB dimensions/frames")
        # Only model-grid depth and points are retained; do not construct a
        # multi-gigabyte full-resolution [T,H_original,W_original,3] array.
        depths = np.stack([
            cv2.resize(depth_source[i].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            for i in range(t_count)
        ])
        del depth_source
        extrinsics = np.array(archive["extrinsics_w2c"][:t_count], dtype=np.float32, copy=True)
        original_k = np.array(archive["fx_fy_cx_cy"], dtype=np.float64, copy=True)
    if extrinsics.shape != (t_count, 4, 4) or original_k.shape != (4,):
        raise ValueError("Expected static fx_fy_cx_cy[4] and extrinsics[T,4,4]")
    if not np.isfinite(depths).all() or not np.isfinite(extrinsics).all() or not np.isfinite(original_k).all():
        # Native normalization uses all pixels, including invalid depth. NaN
        # padding would poison the literal upstream objective, so reject it.
        raise ValueError("Nonfinite dense GT cannot use native all-pixel normalization")
    if (original_k[:2] <= 0).any():
        raise ValueError("Focal lengths must be positive")
    extrinsics = extrinsics @ np.linalg.inv(extrinsics[0])
    c2w = np.linalg.inv(extrinsics)
    fx, fy, cx, cy = original_k * np.array([w / ow, h / oh, w / ow, h / oh])
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    camera = np.stack(((u - cx) * depths / fx, (v - cy) * depths / fy, depths), axis=-1).astype(np.float32)
    world = np.einsum("tij,thwj->thwi", c2w[:, :3, :3], camera) + c2w[:, None, None, :3, 3]
    valid_depth = (depths > 0) & np.isfinite(world).all(axis=-1)
    dense_world = torch.from_numpy(np.ascontiguousarray(world)).float()
    dense_valid = torch.from_numpy(np.ascontiguousarray(valid_depth))
    counts = dense_valid.flatten(1).sum(1)
    if (counts == 0).any():
        raise ValueError("At least one frame has no valid reconstruction depth")
    native_xy, native_tracks, native_indices = rasterize_native_queries(
        sequence.query_xy, sequence.gt_tracks, (w, h),
        torch.tensor(sequence.metadata["selected_point_indices"], dtype=torch.long),
    )
    movement = (native_tracks[-1] - native_tracks[0]).abs().sum(-1)
    movement = movement / (native_tracks[0].norm(dim=-1) + 1e-8)
    dynamic = movement > movement.mean()
    k = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    metadata = {
        **frame_provenance,
        "official_commit": OFFICIAL_COMMIT,
        "native_objective": "ConfLoss(Regr3D(L21,norm_mode='avg_dis',velo_loss=True),alpha=.2,velo_weight=0,reweight_mode='max',reweight_scale=5)",
        "active_components": ["confidence_tracking_3d", "confidence_reconstruction_3d"],
        "source_path": str(source_path),
        "dense_gt_source": "WorldTrack depth_map + fx_fy_cx_cy + normalized extrinsics_w2c",
        "gt_pointmap_grid_wh": [w, h],
        "rgb_policy": sequence.metadata["preprocessing"],
        "depth_resize": "cv2.INTER_NEAREST to final official RGB grid",
        "intrinsics_resize": "x/y rows scaled independently by final/original dimensions; evaluator pixel convention",
        "training_adapter_difference": "WorldTrack GT and evaluator RGB; training half-pixel camera conversion/augmentation and original mesh population are not recreated",
        "native_query_rule": "initial eval query population, torch.round+clamp, last original query per collision, row-major Boolean-mask order",
        "native_query_count": len(native_xy),
        "eval_query_count": len(sequence.query_xy),
        "native_collision_count": len(sequence.query_xy) - len(native_xy),
        "native_mask_rule": "initial-frame queries remain selected at every frame, including occlusion",
        "native_dynamic_rule": "L1(world[-1]-world[0])/(norm(world[0])+1e-8) > population mean",
        "confidence_reweight": "dynamic confidence = 5*max(current predicted static confidence over all frames).detach(); static confidence stays differentiable",
        "normalization": "per-frame sum(ALL head2 point norms)/(3*H*W), clamped >=1e-8; both upstream positional-mask and xyz-count behavior preserved",
        "reconstruction_valid_rule": "depth>0 and finite world point; no evaluation floating-point filter",
        "source_sha256": {
            relative: sha256(root / relative) for relative in (
                "dust3r/losses.py", "dust3r/utils/geometry.py", "dust3r/utils/misc.py",
                "scripts_run/train_seq_reweight.sh", "dust3r/datasets/pointodyssey.py",
                "dust3r/datasets/dynamic_replica.py",
            )
        },
    }
    targets = NativeTargets(
        native_tracks=native_tracks.contiguous(),
        native_track_valid=torch.ones(native_tracks.shape[:2], dtype=torch.bool),
        training_dynamic=dynamic, native_query_xy=native_xy,
        native_point_indices=native_indices, tracks=sequence.gt_tracks,
        valid=sequence.valid, reconstruction_gt=dense_world,
        reconstruction_valid=dense_valid,
        reconstruction_gt_norm=native_pointmap_norm(dense_world),
        reconstruction_valid_counts=counts,
        camera_c2w=torch.from_numpy(np.ascontiguousarray(c2w)).float(),
        intrinsics=k.unsqueeze(0).repeat(t_count, 1, 1),
        depth=torch.from_numpy(np.ascontiguousarray(depths)).float(), metadata=metadata,
    )
    metadata["tensor_sha256"] = targets.tensor_hashes()
    metadata["rgb_sha256"] = tensor_sha256(sequence.rgb)
    metadata["evaluation_query_xy_sha256"] = tensor_sha256(sequence.query_xy)
    sequence.metadata["component_targets"] = metadata
    return sequence, targets


def stack_component_outputs(outputs: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    if not outputs:
        raise ValueError("No component outputs")
    return {key: torch.stack([item[key] for item in outputs]) for key in outputs[0]}


def native_confidence_terms(
    outputs: Mapping[str, Tensor] | Sequence[Mapping[str, Tensor]],
    targets: NativeTargets, *, alpha: float = .2, reweight_scale: float = 5.,
) -> dict[str, Tensor]:
    """Complete active checkpoint objective; also useful as an oracle target."""
    prediction = stack_component_outputs(outputs) if not isinstance(outputs, Mapping) else outputs
    points = prediction["native_tracks"]
    device = points.device
    valid = targets.native_track_valid.to(device)
    gt = targets.native_tracks.to(device)
    pred_scale = prediction["reconstruction_norm"][:, None, None]
    gt_scale = targets.reconstruction_gt_norm.to(device)[:, None, None]
    l21 = (points / pred_scale - gt / gt_scale).norm(dim=-1)
    dynamic = targets.training_dynamic.to(device).unsqueeze(0).expand_as(valid)
    conf = prediction["track_conf"]
    if not bool((valid & ~dynamic).any()):
        raise ValueError("Native max5 confidence reweighting requires static GT queries")
    static_max = conf[valid & ~dynamic].max().detach()
    weights = torch.where(dynamic, torch.full_like(conf, reweight_scale * static_max), conf)
    tracking = (l21 * weights - alpha * weights.clamp_min(1).log())[valid].mean()
    reconstruction_count = targets.reconstruction_valid_counts.to(device).sum()
    reconstruction = (
        prediction["reconstruction_regression_sum"].sum()
        - alpha * prediction["reconstruction_log_conf_sum"].sum()
    ) / reconstruction_count
    return {
        "tracking": tracking, "reconstruction": reconstruction,
        "joint": tracking + reconstruction,
        "tracking_l21": l21[valid].mean(),
        "reconstruction_l21": prediction["reconstruction_l21_sum"].sum() / reconstruction_count,
    }


def load_official_active_loss_oracle(vendor_root: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Compile unchanged official loss class AST nodes without TTA-only imports.

    CameraLoss imports optional training dependencies, including PyTorch3D.
    Its inactive branches are irrelevant to the checkpoint's two active losses.
    We execute original class/function nodes (no rewritten loss expressions)
    and import the actual official geometry helpers. Source hashes expose this
    exact validation route; this is not a claim that the full training package
    imported successfully.
    """
    root = ensure_vendor_importable(vendor_root)
    source = root / "dust3r" / "losses.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"Sum", "BaseCriterion", "LLoss", "L21Loss", "Criterion", "MultiLoss", "Regr3D", "ConfLoss"}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError("Pinned native loss definitions are missing")
    geometry = importlib.import_module("dust3r.utils.geometry")
    namespace = {
        "torch": torch, "nn": torch.nn, "copy": copy, "deepcopy": deepcopy,
        "inv": geometry.inv, "geotrf": geometry.geotrf,
        "normalize_pointcloud_seq": geometry.normalize_pointcloud_seq,
    }
    subset = ast.Module(body=selected, type_ignores=[])
    exec(compile(subset, str(source), "exec"), namespace)
    criterion = namespace["ConfLoss"](
        namespace["Regr3D"](namespace["L21Loss"](), norm_mode="avg_dis", velo_loss=True),
        alpha=.2, velo_weight=0, pose_weight=0, traj_weight=0, align3d_weight=0,
        depth_weight=0, cotracker=False, reweight_mode="max", reweight_scale=5.,
    )
    return criterion, {
        "mode": "unchanged official active-loss AST + actual official geometry import",
        "full_training_loss_module_imported": False,
        "selected_nodes": [node.name for node in selected],
        "losses_source_sha256": sha256(source),
        "selected_ast_sha256": hashlib.sha256(ast.dump(subset, include_attributes=False).encode()).hexdigest(),
        "geometry_source_sha256": sha256(root / "dust3r" / "utils" / "geometry.py"),
        "misc_source_sha256": sha256(root / "dust3r" / "utils" / "misc.py"),
    }


def _official_oracle_inputs(
    native_tracks: Tensor, track_conf: Tensor, reconstruction: Tensor,
    reconstruction_conf: Tensor, targets: NativeTargets,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Build sparse anchor maps exactly in native Boolean-index order."""
    t_count, h, w, _ = reconstruction.shape
    device = reconstruction.device
    xy = targets.native_query_xy.to(device)
    sparse = reconstruction.new_zeros((t_count, h, w, 3))
    sparse[:, xy[:, 1], xy[:, 0]] = targets.native_tracks.to(device)
    traj_mask = torch.zeros((t_count, h, w), device=device, dtype=torch.bool)
    traj_mask[:, xy[:, 1], xy[:, 0]] = True
    gt1 = {
        "pts3d": targets.reconstruction_gt[:1].to(device).expand(t_count, -1, -1, -1),
        "camera_pose": torch.eye(4, device=device, dtype=reconstruction.dtype).unsqueeze(0).repeat(t_count, 1, 1),
        "traj_ptc": sparse, "traj_mask": traj_mask,
        "valid_mask": targets.reconstruction_valid[:1].to(device).expand(t_count, -1, -1),
        "supervised_label": torch.ones(t_count, device=device),
    }
    gt2 = {
        "pts3d": targets.reconstruction_gt.to(device),
        "valid_mask": targets.reconstruction_valid.to(device),
        "supervised_label": torch.ones(t_count, device=device),
    }
    # Head1 is only read at native raster queries. Filling other pixels with
    # zero does not change the active native loss or any tested derivative.
    pointmap1 = reconstruction.new_zeros((t_count, h, w, 3))
    pointmap1[:, xy[:, 1], xy[:, 0]] = native_tracks
    conf1 = reconstruction.new_ones((t_count, h, w))
    conf1[:, xy[:, 1], xy[:, 0]] = track_conf
    return gt1, gt2, {"pts3d": pointmap1, "conf": conf1}, {
        "pts3d_in_other_view": reconstruction, "conf": reconstruction_conf,
    }


def check_official_component_oracle(
    packed_outputs: Mapping[str, Tensor], reconstruction: Tensor,
    reconstruction_conf: Tensor, targets: NativeTargets, *,
    vendor_root: str | Path | None = None, atol: float = 2e-5, rtol: float = 2e-5,
) -> dict[str, Any]:
    """CPU validation of captured maps, native loss values and head derivatives.

    This requires no model, checkpoint, CUDA, or second inference. Derivatives
    are compared with respect to sampled head1 XYZ/confidence and all head2
    XYZ/confidence values. Input-RGB gradient parity is a separate bounded check
    in NativeComponentForward.check_official_loss_parity.
    """
    oracle, metadata = load_official_active_loss_oracle(vendor_root)
    gt = targets.to("cpu")
    native_tracks = packed_outputs["native_tracks"].detach().cpu().clone().requires_grad_(True)
    track_conf = packed_outputs["track_conf"].detach().cpu().clone().requires_grad_(True)
    dense = reconstruction.detach().cpu().clone().requires_grad_(True)
    conf2 = reconstruction_conf.detach().cpu().clone().requires_grad_(True)
    norm = native_pointmap_norm(dense)
    residual = (dense / norm[:, None, None, None]
                - gt.reconstruction_gt / gt.reconstruction_gt_norm[:, None, None, None]).norm(dim=-1)
    valid = gt.reconstruction_valid
    stats = {
        "native_tracks": native_tracks, "track_conf": track_conf,
        "reconstruction_norm": norm,
        "reconstruction_regression_sum": (conf2 * residual).masked_fill(~valid, 0).flatten(1).sum(1),
        "reconstruction_log_conf_sum": conf2.log().masked_fill(~valid, 0).flatten(1).sum(1),
        "reconstruction_l21_sum": residual.masked_fill(~valid, 0).flatten(1).sum(1),
        "reconstruction_conf_sum": conf2.masked_fill(~valid, 0).flatten(1).sum(1),
    }
    for name in stats:
        torch.testing.assert_close(stats[name].detach(), packed_outputs[name].detach().cpu(), atol=atol, rtol=rtol)
    terms = native_confidence_terms(stats, gt)
    reference, details = oracle(*_official_oracle_inputs(native_tracks, track_conf, dense, conf2, gt))
    torch.testing.assert_close(terms["joint"], reference, atol=atol, rtol=rtol)
    variables = (native_tracks, track_conf, dense, conf2)
    own_gradients = torch.autograd.grad(terms["joint"], variables, retain_graph=True)
    official_gradients = torch.autograd.grad(reference, variables)
    errors = {}
    for name, own, official in zip(("head1_xyz", "head1_conf", "head2_xyz", "head2_conf"), own_gradients, official_gradients):
        if not torch.isfinite(own).all() or not torch.isfinite(official).all():
            raise FloatingPointError(f"Native oracle {name} gradients contain NaN/Inf")
        torch.testing.assert_close(own, official, atol=atol, rtol=rtol)
        errors[name] = float((own - official).abs().max())
    for name, official_key in (("tracking", "conf_loss_1"), ("reconstruction", "conf_loss_2")):
        torch.testing.assert_close(terms[name].detach(), reference.new_tensor(details[official_key]), atol=atol, rtol=rtol)
    return {
        "passed": True, "device": "cpu", "num_frames": len(dense),
        "loss_abs_error": float((terms["joint"] - reference).abs().detach()),
        "head_gradient_max_abs_error": errors,
        "components": {name: float(value.detach()) for name, value in terms.items()},
        "official_details": details, "oracle": metadata, "atol": atol, "rtol": rtol,
    }


class NativeGradientParityError(AssertionError):
    """A failed RGB-gradient gate with machine-readable measured diagnostics."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__("Native RGB gradient parity failed: " + json.dumps(diagnostics, sort_keys=True, allow_nan=False))


def gradient_repeat_diagnostics(
    native_gradients: Sequence[Tensor], official_gradients: Sequence[Tensor], *,
    atol: float = 2e-5, rtol: float = 2e-5,
    relative_l2_cap: float = 1e-3, noise_multiplier: float = 1.5,
    cross_relative_floor: float = 1e-5,
) -> dict[str, Any]:
    """Measure repeat noise without changing any gradient or loss formula.

    Exactly two gradients from each unchanged graph are required. Norms are
    measured in float64 on CPU so diagnostic reductions do not introduce GPU
    reduction noise. The gate uses the worst of four native/official cross
    comparisons and cannot pass merely by allowing unlimited repeat noise.
    """
    if len(native_gradients) != 2 or len(official_gradients) != 2:
        raise ValueError("Repeat diagnostics require two native and two official gradients")
    if not 0 < relative_l2_cap <= 2e-3:
        raise ValueError("RGB gradient relative L2 cap must be positive and <=2e-3")
    if noise_multiplier != 1.5 or cross_relative_floor != 1e-5:
        raise ValueError("The pinned repeat-noise gate uses multiplier=1.5 and floor=1e-5")
    originals = [*native_gradients, *official_gradients]
    if any(value.shape != originals[0].shape for value in originals):
        raise ValueError("Repeat gradient shapes differ")
    if any(not bool(torch.isfinite(value).all()) for value in originals):
        raise FloatingPointError("Native oracle repeat gradients contain NaN/Inf")
    cpu = [value.detach().cpu().double() for value in originals]

    def relative_l2(left: Tensor, right: Tensor) -> float:
        denominator = max(float(left.norm()), float(right.norm()), 1e-12)
        return float((left - right).norm()) / denominator

    native_repeat = relative_l2(cpu[0], cpu[1])
    official_repeat = relative_l2(cpu[2], cpu[3])
    cross = [relative_l2(cpu[i], cpu[j]) for i in (0, 1) for j in (2, 3)]
    maximum_repeat = max(native_repeat, official_repeat)
    maximum_cross = max(cross)
    allowed_cross = max(cross_relative_floor, noise_multiplier * maximum_repeat)
    cap_passed = max(maximum_repeat, maximum_cross) <= relative_l2_cap
    noise_gate_passed = maximum_cross <= allowed_cross
    # Preserve the original strict allclose result; repeat-noise evidence never
    # changes its tolerances or relabels a strict failure as a strict pass.
    strict_passed, strict_message = True, None
    try:
        torch.testing.assert_close(native_gradients[0], official_gradients[0], atol=atol, rtol=rtol)
    except AssertionError as error:
        strict_passed, strict_message = False, str(error)
    error = (cpu[0] - cpu[2]).abs()
    mismatch = error > (atol + rtol * cpu[2].abs())
    reference_nonzero = cpu[2].abs() > 0
    element_relative = error[reference_nonzero] / cpu[2].abs()[reference_nonzero]
    return {
        "mode": "same_graph_native_twice_official_twice",
        "backward_calls": 4,
        "retain_graph": [True, True, True, False],
        "relative_l2_denominator": "max(norm(left),norm(right),1e-12), measured float64 CPU",
        "native_repeat_relative_l2": native_repeat,
        "official_repeat_relative_l2": official_repeat,
        "cross_relative_l2": cross[0],
        "cross_relative_l2_all_four": cross,
        "cross_relative_l2_max": maximum_cross,
        "repeat_relative_l2_max": maximum_repeat,
        "relative_l2_cap": relative_l2_cap,
        "cross_relative_l2_allowed": allowed_cross,
        "noise_multiplier": noise_multiplier,
        "cross_relative_floor": cross_relative_floor,
        "relative_l2_cap_passed": cap_passed,
        "cross_within_repeat_noise_passed": noise_gate_passed,
        "passed": cap_passed and noise_gate_passed,
        "strict_allclose": {
            "passed": strict_passed, "atol": atol, "rtol": rtol,
            "failed_elements": int(mismatch.sum()),
            "total_elements": error.numel(),
            "failed_fraction": float(mismatch.double().mean()),
            "max_abs_error": float(error.max()),
            "max_relative_error_nonzero_reference": float(element_relative.max()) if element_relative.numel() else 0.,
            "assertion_message": strict_message,
        },
        "cross_relative_l2_per_frame": [
            relative_l2(cpu[0][t], cpu[2][t]) for t in range(len(cpu[0]))
        ],
    }


class NativeComponentForward:
    """One frozen pair forward exposing native-loss sufficient statistics."""

    def __init__(self, base: St4RTrackForward, gt_context: NativeTargets, *, optimize_runtime: bool = False) -> None:
        self.base = base
        self.model = base.model
        self.targets = gt_context
        self.gt_context = gt_context
        self.vendor_root = base.vendor_root
        self.checkpoint = base.checkpoint
        self.optimize_runtime = bool(optimize_runtime)
        self.runtime_handle = None
        self._gt_cache: dict[torch.device, dict[str, Any]] = {}
        self._cpu_true_shape = torch.tensor(gt_context.reconstruction_gt.shape[1:3]).unsqueeze(0)
        if len(gt_context.native_tracks) != len(gt_context.reconstruction_gt):
            raise ValueError("Native track and reconstruction frame counts differ")
        if self.optimize_runtime:
            self.runtime_handle = apply_runtime_optimizations(
                self.model, image_hw=tuple(gt_context.reconstruction_gt.shape[1:3]), vendor_root=self.vendor_root,
            )
            self._cached_gt(self.device)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.base.metadata, "native_component_targets": self.targets.metadata,
            "runtime_optimization": {
                **runtime_optimization_metadata(self.model),
                "component_optimized": self.optimize_runtime,
                "cpu_true_shape": self.optimize_runtime,
                "fixed_gt_device_cache": self.optimize_runtime,
                "fixed_valid_flat_indices": self.optimize_runtime,
                "pair_frames_api": True,
            },
        }

    def restore_runtime(self) -> None:
        """Restore model methods and this wrapper's baseline preprocessing."""
        if self.runtime_handle is not None:
            self.runtime_handle.restore()
        self.optimize_runtime = False
        self._gt_cache.clear()

    def _cached_gt(self, device: torch.device) -> dict[str, Any]:
        device = torch.device(device)
        if device not in self._gt_cache:
            mask = self.targets.reconstruction_valid.detach().cpu().flatten(1)
            h, w = self.targets.reconstruction_gt.shape[1:3]
            eval_xy = self.base.query_xy.detach().cpu().long()
            if not ((eval_xy[:, 0] >= 0) & (eval_xy[:, 0] < w)
                    & (eval_xy[:, 1] >= 0) & (eval_xy[:, 1] < h)).all():
                raise ValueError("Fixed evaluation pixels are outside the model grid")
            self._gt_cache[device] = {
                "gt": self.targets.reconstruction_gt.to(device),
                "gt_norm": self.targets.reconstruction_gt_norm.to(device),
                "native_xy": self.targets.native_query_xy.to(device),
                "eval_xy": eval_xy.to(device),
                "indices": [row.nonzero(as_tuple=True)[0].to(device) for row in mask],
                "all_valid": [bool(row.all()) for row in mask],
            }
        return self._gt_cache[device]

    def _frame_view(self, frame: Tensor) -> dict[str, Any]:
        if self.optimize_runtime:
            # PatchEmbedDust3R ignores true_shape; the unchanged official head
            # wrapper reads CPU H,W. Identical constant CPU shape avoids a
            # GPU -> CPU .tolist synchronization without changing image values.
            return {"img": normalize_rgb(frame).unsqueeze(0), "true_shape": self._cpu_true_shape}
        return self.base._view(frame)

    def raw_predictions_frames(self, frame0: Tensor, framet: Tensor, t: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        expected_hw = tuple(self.targets.reconstruction_gt.shape[1:3])
        if not 0 <= t < len(self.targets.native_tracks):
            raise ValueError("Frame index differs from fixed native GT")
        for frame in (frame0, framet):
            if frame.ndim != 3 or frame.shape[0] != 3 or not frame.is_floating_point():
                raise ValueError("Pair frames must be floating [3,H,W]")
            if frame.device != self.device or tuple(frame.shape[-2:]) != expected_hw:
                raise ValueError("Pair frame device/resolution differs from model and fixed GT")
        if t == 0 and frame0 is not framet:
            raise ValueError("Self-pair t=0 must receive the same leaf in both frame roles")
        view1, view2 = self._frame_view(frame0), self._frame_view(framet)
        with torch.autocast(device_type=frame0.device.type, enabled=False):
            return self.model(view1, view2)

    def raw_predictions(self, rgb: Tensor, t: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("rgb must be floating [T,3,H,W]")
        if len(rgb) != len(self.targets.native_tracks) or not 0 <= t < len(rgb):
            raise ValueError("RGB frame count/index differs from fixed native GT")
        if rgb.device != self.device:
            raise ValueError("RGB and model devices differ")
        if tuple(rgb.shape[-2:]) != tuple(self.targets.reconstruction_gt.shape[1:3]):
            raise ValueError("RGB resolution differs from fixed native GT")
        anchor = rgb[0]
        return self.raw_predictions_frames(anchor, anchor if t == 0 else rgb[t], t)

    def _components_from_predictions(self, pred1: Mapping[str, Tensor], pred2: Mapping[str, Tensor], t: int) -> dict[str, Tensor]:
        p1, p2 = pred1["pts3d"], pred2["pts3d_in_other_view"]
        if p1.ndim != 4 or p1.shape[0] != 1 or p2.shape != p1.shape:
            raise ValueError("Expected matching single-pair pointmaps [1,H,W,3]")
        p2 = p2[0]
        conf2 = pred2["conf"][0]
        norm = native_pointmap_norm(p2)
        if pred1["conf"].shape != p1.shape[:-1] or conf2.shape != p2.shape[:-1]:
            raise ValueError("Confidence maps do not match pointmaps")
        if self.optimize_runtime:
            cache = self._cached_gt(p2.device)
            xy, eval_xy = cache["native_xy"], cache["eval_xy"]
            flat_pred = p2.reshape(-1, 3)
            flat_gt = cache["gt"][t].reshape(-1, 3)
            flat_conf = conf2.reshape(-1)
            if not cache["all_valid"][t]:
                indices = cache["indices"][t]
                flat_pred = flat_pred.index_select(0, indices)
                flat_gt = flat_gt.index_select(0, indices)
                flat_conf = flat_conf.index_select(0, indices)
            residual = (flat_pred / norm - flat_gt / cache["gt_norm"][t]).norm(dim=-1)
            confidence = flat_conf
            eval_tracks = p1[0, eval_xy[:, 1], eval_xy[:, 0]]
        else:
            xy = self.targets.native_query_xy.to(p1.device)
            gt = self.targets.reconstruction_gt[t].to(p2.device)
            gt_norm = self.targets.reconstruction_gt_norm[t].to(p2.device)
            valid = self.targets.reconstruction_valid[t].to(p2.device)
            residual = (p2[valid] / norm - gt[valid] / gt_norm).norm(dim=-1)
            confidence = conf2[valid]
            eval_tracks = sample_query_tracks(p1[0], self.base.query_xy)
        return {
            "tracks": eval_tracks,
            "native_tracks": p1[0, xy[:, 1], xy[:, 0]],
            "track_conf": pred1["conf"][0, xy[:, 1], xy[:, 0]],
            "reconstruction_norm": norm,
            "reconstruction_regression_sum": (confidence * residual).sum(),
            "reconstruction_log_conf_sum": confidence.log().sum(),
            "reconstruction_l21_sum": residual.sum(),
            "reconstruction_conf_sum": confidence.sum(),
        }

    def pair_fn(self, rgb: Tensor, t: int) -> dict[str, Tensor]:
        return self._components_from_predictions(*self.raw_predictions(rgb, t), t)

    def pair_frames_fn(self, frame0: Tensor, framet: Tensor, t: int) -> dict[str, Tensor]:
        """Pair-local RGB VJP; t=0 shares one identical leaf for both roles."""
        return self._components_from_predictions(*self.raw_predictions_frames(frame0, framet, t), t)

    def __call__(self, rgb: Tensor, t: int) -> dict[str, Tensor]:
        return self.pair_fn(rgb, t)

    def predict(self, rgb: Tensor) -> Tensor:
        """Existing official evaluation tracks, including attacked RGB replay."""
        return self.base.predict(rgb)

    @torch.no_grad()
    def predict_components(self, rgb: Tensor) -> dict[str, Tensor]:
        """Capture sufficient statistics on CPU, releasing every pair graph."""
        return stack_component_outputs([
            {key: value.cpu() for key, value in self.pair_fn(rgb, t).items()}
            for t in range(len(rgb))
        ])

    @torch.no_grad()
    def predict_with_maps(self, rgb: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """Capture packed statistics and full head2 XYZ/conf maps on CPU."""
        outputs, maps, confidences = [], [], []
        for t in range(len(rgb)):
            pred1, pred2 = self.raw_predictions(rgb, t)
            outputs.append({key: value.cpu() for key, value in self._components_from_predictions(pred1, pred2, t).items()})
            maps.append(pred2["pts3d_in_other_view"][0].cpu())
            confidences.append(pred2["conf"][0].cpu())
        return stack_component_outputs(outputs), torch.stack(maps), torch.stack(confidences)

    def official_predict(self, sequence: WorldTrackSequence) -> Tensor:
        return self.base.official_predict(sequence)

    def check_clean_parity(self, sequence: WorldTrackSequence, **kwargs: Any) -> dict[str, Any]:
        return self.base.check_clean_parity(sequence, **kwargs)

    def check_official_loss_parity(
        self, rgb: Tensor, *, atol: float = 2e-5, rtol: float = 2e-5,
        loss_fn: Callable[[Mapping[str, Tensor], NativeTargets], Tensor] | None = None,
        repeat_noise_check: bool = False, relative_l2_cap: float = 1e-3,
    ) -> dict[str, Any]:
        """Compare full active objective value and input gradient with upstream.

        This is intended for a bounded two-frame toy/real validation. Both
        formulas consume the identical raw model outputs, avoiding duplicate
        model graphs while independently checking scale, confidence, masks and
        population reductions. The root runner controls any CUDA invocation.
        CPU/default checks retain strict elementwise gradient parity. Explicit
        CUDA repeat-noise mode measures two backwards per identical loss graph
        and only passes a bounded cross difference consistent with repeat noise.
        """
        if repeat_noise_check and self.device.type != "cuda":
            raise ValueError("Repeat-noise mode is only available for real CUDA RGB validation; CPU remains strict")
        oracle, oracle_metadata = load_official_active_loss_oracle(self.vendor_root)
        video = rgb.detach().clone().to(self.device).requires_grad_(True)
        raw = [self.raw_predictions(video, t) for t in range(len(video))]
        outputs = stack_component_outputs([
            self._components_from_predictions(p1, p2, t) for t, (p1, p2) in enumerate(raw)
        ])
        target = self.targets.to(self.device)
        proposed_terms = native_confidence_terms(outputs, target)
        proposed = proposed_terms["joint"] if loss_fn is None else loss_fn(outputs, target)
        reconstruction = torch.cat([pair[1]["pts3d_in_other_view"] for pair in raw])
        reconstruction_conf = torch.cat([pair[1]["conf"] for pair in raw])
        reference, reference_details = oracle(*_official_oracle_inputs(
            outputs["native_tracks"], outputs["track_conf"], reconstruction,
            reconstruction_conf, target,
        ))
        # Loss values and individual active source components remain strict
        # gates regardless of GPU backward repeat-noise diagnostics.
        torch.testing.assert_close(proposed, reference, atol=atol, rtol=rtol)
        for name, official_key in (("tracking", "conf_loss_1"), ("reconstruction", "conf_loss_2")):
            torch.testing.assert_close(proposed_terms[name].detach(), proposed.new_tensor(reference_details[official_key]), atol=atol, rtol=rtol)
        gradient_diagnostics = None
        if repeat_noise_check:
            native_gradients = [
                torch.autograd.grad(proposed, video, retain_graph=True)[0],
                torch.autograd.grad(proposed, video, retain_graph=True)[0],
            ]
            official_gradients = [
                torch.autograd.grad(reference, video, retain_graph=True)[0],
                torch.autograd.grad(reference, video, retain_graph=False)[0],
            ]
            gradient_diagnostics = gradient_repeat_diagnostics(
                native_gradients, official_gradients, atol=atol, rtol=rtol,
                relative_l2_cap=relative_l2_cap,
            )
            if not gradient_diagnostics["passed"]:
                raise NativeGradientParityError(gradient_diagnostics)
            proposed_gradient, reference_gradient = native_gradients[0], official_gradients[0]
        else:
            proposed_gradient = torch.autograd.grad(proposed, video, retain_graph=True)[0]
            reference_gradient = torch.autograd.grad(reference, video)[0]
            if not torch.isfinite(proposed_gradient).all() or not torch.isfinite(reference_gradient).all():
                raise FloatingPointError("Native oracle gradients contain NaN/Inf")
            torch.testing.assert_close(proposed_gradient, reference_gradient, atol=atol, rtol=rtol)
        return {
            "passed": True, "num_frames": len(video), "atol": atol, "rtol": rtol,
            "proposed_loss": float(proposed.detach()), "official_loss": float(reference.detach()),
            "loss_abs_error": float((proposed - reference).abs().detach()),
            "gradient_max_abs_error": float((proposed_gradient - reference_gradient).abs().max()),
            "gradient_per_frame_l1": proposed_gradient.abs().flatten(1).sum(1).detach().cpu().tolist(),
            "gradient_parity_mode": "bounded_cuda_repeat_noise" if repeat_noise_check else "strict_allclose",
            "gradient_repeat_diagnostics": gradient_diagnostics,
            "components": {key: float(value.detach()) for key, value in proposed_terms.items()},
            "official_details": reference_details, "oracle": oracle_metadata,
        }
