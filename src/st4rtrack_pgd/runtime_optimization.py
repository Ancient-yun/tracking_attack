"""Reversible fixed-grid optimizations of the unchanged official FP32 model.

Only VanillaDust3r / Python RoPE2D / PatchEmbedDust3R are supported. No weights,
attention backend, dtype, or vendor file are changed. The original RoPE tensor
operations remain identical; the fixed patch-grid maximum replaces a repeated
GPU scalar read, and constant positions use a correctly keyed device cache.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch import Tensor

from .common import OFFICIAL_COMMIT, sha256
from .worldtrack_adapter import ensure_vendor_importable


class FixedPositionGetter3D:
    """Same Cartesian positions and layout, cached by all shape/device fields."""

    def __init__(self, patch_hw: tuple[int, int]) -> None:
        self.patch_hw = patch_hw
        self.cache: dict[tuple[Any, ...], Tensor] = {}

    def __call__(self, b: int, t: int, h: int, w: int, device: torch.device) -> Tensor:
        if (h, w) != self.patch_hw:
            raise ValueError("Runtime position grid differs from the configured fixed grid")
        key = (b, t, h, w, torch.device(device))
        if key not in self.cache:
            # CPU construction avoids a GPU reduction or dynamic scalar read.
            positions = torch.cartesian_prod(torch.arange(t), torch.arange(h), torch.arange(w))
            self.cache[key] = positions.view(1, t * h * w, 3).expand(b, -1, -1).clone().to(device)
        # The caller only reads positions; retaining the identical cached
        # tensor avoids repeated allocation. No position enters autograd.
        return self.cache[key]


class RuntimeOptimizationHandle:
    def __init__(self, model: torch.nn.Module, image_hw: tuple[int, int], vendor_root: Path) -> None:
        if getattr(model, "arch_mode", None) != "VanillaDust3r":
            raise ValueError("Fixed-grid runtime optimization requires VanillaDust3r")
        patch = getattr(model, "patch_embed", None)
        if patch is None or type(patch).__name__ != "PatchEmbedDust3R":
            raise ValueError("Fixed-grid runtime optimization requires official PatchEmbedDust3R")
        ropes = [getattr(model, name, None) for name in ("rope_enc", "rope_dec")]
        if any(rope is None or type(rope).__name__ != "RoPE2D" for rope in ropes):
            raise ValueError("Fixed-grid runtime optimization requires official Python RoPE2D")
        h, w = image_hw
        ph, pw = patch.patch_size
        if min(h, w) <= 0 or h % ph or w % pw:
            raise ValueError("Image resolution must be positive and divisible by patch size")
        self.model, self.image_hw = model, image_hw
        self.patch_hw = (h // ph, w // pw)
        self.seq_len = max(self.patch_hw)
        self.enabled = True
        self._rope_original = [(rope, rope.forward, "forward" in rope.__dict__) for rope in ropes]
        self._position_original = patch.position_getter3d
        patch.position_getter3d = FixedPositionGetter3D(self.patch_hw)
        n_tokens, seq_len = self.patch_hw[0] * self.patch_hw[1], self.seq_len

        def fixed_grid_forward(rope: torch.nn.Module, tokens: Tensor, positions: Tensor) -> Tensor:
            if tokens.size(3) % 2 or positions.ndim != 3 or positions.shape[-1] != 2:
                raise ValueError("Fixed-grid RoPE requires original 2D position/head layout")
            if positions.shape[1] != n_tokens or tokens.shape[2] != n_tokens:
                raise ValueError("Fixed-grid RoPE received a different token grid")
            # These are the original pos_embed.py:146-154 operations, with the
            # already-known seq_len replacing int(positions.max())+1 only.
            d = tokens.size(3) // 2
            cos, sin = rope.get_cos_sin(d, seq_len, tokens.device, tokens.dtype)
            y, x = tokens.chunk(2, dim=-1)
            y = rope.apply_rope1d(y, positions[:, :, 0], cos, sin)
            x = rope.apply_rope1d(x, positions[:, :, 1], cos, sin)
            return torch.cat((y, x), dim=-1)

        for rope in ropes:
            rope.forward = MethodType(fixed_grid_forward, rope)
        self.metadata = {
            "enabled": True, "official_commit": OFFICIAL_COMMIT,
            "restriction": "VanillaDust3r + Python RoPE2D + PatchEmbedDust3R",
            "image_hw": list(image_hw), "patch_grid_hw": list(self.patch_hw),
            "rope_seq_len": self.seq_len,
            "fixed_grid_rope": "same tensor operations; CPU-constant seq_len; no positions.max scalar read",
            "position_cache": "Cartesian t,y,x unchanged; keyed by b,t,h,w,device",
            "precision": "unchanged FP32; no AMP/TF32/attention backend change",
            "vendor_files_modified": False,
            "source_sha256": {
                relative: sha256(vendor_root / relative) for relative in (
                    "croco/models/pos_embed.py", "croco/models/blocks.py",
                    "dust3r/patch_embed.py", "dust3r/model.py",
                )
            },
            "optimization_source_sha256": sha256(Path(__file__)),
        }
        model._st4rtrack_runtime_handle = self

    def restore(self) -> None:
        if not self.enabled:
            return
        for rope, original, had_instance_forward in self._rope_original:
            if had_instance_forward:
                rope.forward = original
            else:
                del rope.forward
        self.model.patch_embed.position_getter3d = self._position_original
        self.enabled = False
        self.metadata = {**self.metadata, "enabled": False, "restored": True}
        if getattr(self.model, "_st4rtrack_runtime_handle", None) is self:
            del self.model._st4rtrack_runtime_handle

    def __enter__(self) -> "RuntimeOptimizationHandle":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.restore()


def apply_runtime_optimizations(
    model: torch.nn.Module, *, image_hw: tuple[int, int], vendor_root: str | Path | None = None,
) -> RuntimeOptimizationHandle:
    """Apply once or reuse the existing compatible handle; restore is explicit."""
    current = getattr(model, "_st4rtrack_runtime_handle", None)
    if current is not None and current.enabled:
        if current.image_hw != tuple(image_hw):
            raise ValueError("The model already has a different fixed runtime grid")
        return current
    return RuntimeOptimizationHandle(model, tuple(image_hw), ensure_vendor_importable(vendor_root))


def runtime_optimization_metadata(model: torch.nn.Module) -> dict[str, Any]:
    current = getattr(model, "_st4rtrack_runtime_handle", None)
    return dict(current.metadata) if current is not None else {"enabled": False}
