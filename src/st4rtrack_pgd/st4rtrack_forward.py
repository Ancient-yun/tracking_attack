"""Frozen St4RTrack pair inference with gradients to every input frame."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from .worldtrack_adapter import (
    WorldTrackSequence,
    ensure_vendor_importable,
    sample_query_tracks,
)


def normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """The official torchvision Normalize((.5,)*3, (.5,)*3)."""
    return rgb.sub(0.5).div(0.5)


def _load_pth_model(model_module: Any, checkpoint_path: str) -> torch.nn.Module:
    """Official .pth construction with checked weight compatibility.

    Upstream load_model uses strict=False. Keeping its exact constructor
    adjustments but loading strictly makes an incompatible/misnamed checkpoint
    a visible failure instead of leaving randomly initialized model layers.
    These are author-distributed trusted checkpoints: like the official loader,
    loading .pth allows pickled argument objects and its saved constructor.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"args", "model"} <= checkpoint.keys():
        raise ValueError("Expected an official .pth checkpoint with args and model fields")
    expression = getattr(checkpoint["args"], "model", None)
    if not isinstance(expression, str) or not expression.startswith("AsymmetricCroCo3DStereo("):
        raise ValueError("Checkpoint does not specify the official AsymmetricCroCo3DStereo constructor")
    expression = expression.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in expression:
        expression = expression[:-1] + ", landscape_only=False)"
    else:
        expression = expression.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    if "landscape_only=False" not in expression:
        raise ValueError("Checkpoint landscape setting could not match the official loader")
    model = eval(expression, vars(model_module))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


class St4RTrackForward:
    """A callable ``pair_fn(rgb[T,3,H,W], t) -> tracks[N,3]``.

    The pinned evaluator tracks with ``pred1['pts3d']`` for pairs ``(0,t)``,
    including ``(0,0)``. Both views in that self-pair consume the same rgb[0]
    tensor so its gradient accumulates through both model branches. Inputs are
    already resized; no PIL conversion, detach, NumPy conversion or no_grad
    occurs along this path. The model's weights stay frozen and TTA is unused.

    The official sequence positional encoding depends on batch size. This
    implementation intentionally uses one pair at a time and compares clean
    outputs against ``dust3r.inference.inference(..., batch_size=1)``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        query_xy: torch.Tensor,
        *,
        vendor_root: str | Path | None = None,
        checkpoint: str | None = None,
    ) -> None:
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if query_xy.ndim != 2 or query_xy.shape[-1] != 2:
            raise ValueError("query_xy must have shape [N,2]")
        self.query_xy = query_xy.detach().clone()
        self.vendor_root = Path(vendor_root).resolve() if vendor_root else None
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        query_xy: torch.Tensor,
        *,
        vendor_root: str | Path | None = None,
        device: str | torch.device = "cuda",
        local_files_only: bool = True,
    ) -> "St4RTrackForward":
        """Load a .pth or local HF folder; never silently substitute a model.

        An explicit Hugging Face repository ID is accepted only when
        ``local_files_only=False``. A local model.safetensors file is resolved to
        its config.json-containing folder because the official loader treats
        arbitrary files as torch .pth checkpoints.
        """
        root = ensure_vendor_importable(vendor_root)
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        model_module = importlib.import_module("dust3r.model")
        model_class = model_module.AsymmetricCroCo3DStereo
        checkpoint_text = str(checkpoint)
        candidate = Path(checkpoint).expanduser()
        if candidate.is_file() and candidate.suffix.lower() == ".safetensors":
            if candidate.name != "model.safetensors":
                raise ValueError("HF loader expects a file named model.safetensors")
            if not (candidate.parent / "config.json").is_file():
                raise FileNotFoundError(f"Missing HF config.json beside {candidate}")
            candidate = candidate.parent
        if candidate.exists():
            if candidate.is_dir() and not (candidate / "config.json").is_file():
                raise FileNotFoundError(f"Missing HF config.json inside {candidate}")
            checkpoint_text = str(candidate.resolve())
        elif local_files_only:
            raise FileNotFoundError(f"Local checkpoint is missing: {candidate}")
        if candidate.is_file():
            if candidate.suffix.lower() not in (".pth", ".pt"):
                raise ValueError("Expected official .pth/.pt or HF model.safetensors/config.json")
            model = _load_pth_model(model_module, checkpoint_text)
        else:
            model = model_class.from_pretrained(
                checkpoint_text, local_files_only=local_files_only, strict=True,
            )
        model = model.to(selected_device)
        return cls(model, query_xy, vendor_root=root, checkpoint=checkpoint_text)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            try:
                return next(self.model.buffers()).device
            except StopIteration:
                return self.query_xy.device

    @property
    def metadata(self) -> dict[str, Any]:
        croco_args = getattr(self.model, "croco_args", {})
        return {
            "checkpoint": self.checkpoint,
            "model_class": type(self.model).__name__,
            "arch_mode": getattr(self.model, "arch_mode", croco_args.get("arch_mode")),
            "rope_mode": getattr(self.model, "rope_mode", croco_args.get("rope_mode")),
            "official_clean_batch_size": 1,
            "weight_loading": "strict",
            "frozen_weights": True,
            "tta": False,
        }

    @staticmethod
    def _view(image: torch.Tensor) -> dict[str, Any]:
        # Match actual model input in official inference(batch_size=1). The
        # official wrapper drops NumPy true_shape and the model derives it from
        # img. Providing its identical shape as a tensor preserves that result.
        return {
            "img": normalize_rgb(image).unsqueeze(0),
            "true_shape": torch.tensor(image.shape[-2:], device=image.device).unsqueeze(0),
        }

    def pair_fn(self, rgb: torch.Tensor, t: int) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("rgb must be a floating tensor with shape [T,3,H,W]")
        if not 0 <= t < len(rgb):
            raise IndexError(f"Frame {t} outside sequence of length {len(rgb)}")
        if rgb.device != self.device:
            raise ValueError(f"RGB is on {rgb.device}, model is on {self.device}")
        # Construct both views from the shared video tensor; for t=0 this is
        # literally the same frame, with gradients through both normalizations.
        view1, view2 = self._view(rgb[0]), self._view(rgb[t])
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            pred1, _pred2 = self.model(view1, view2)
        point_map = pred1["pts3d"]
        if point_map.ndim != 4 or point_map.shape[0] != 1:
            raise ValueError(f"Unexpected pred1['pts3d'] shape: {tuple(point_map.shape)}")
        return sample_query_tracks(point_map[0], self.query_xy)

    def __call__(self, rgb: torch.Tensor, t: int) -> torch.Tensor:
        return self.pair_fn(rgb, t)

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return [T,N,3]; caller controls whether the graph is retained."""
        return torch.stack([self.pair_fn(rgb, t) for t in range(len(rgb))])

    @torch.no_grad()
    def official_predict(self, sequence: WorldTrackSequence) -> torch.Tensor:
        """Clean reference from the unchanged official inference wrapper."""
        ensure_vendor_importable(self.vendor_root)
        if not sequence.official_views:
            raise ValueError("load_sequence must supply official clean reference views")
        inference = importlib.import_module("dust3r.inference").inference
        # The upstream wrapper mutates combined tensors but not original dicts.
        output = inference(
            list(sequence.official_views), self.model, self.device,
            batch_size=1, verbose=False, anchor_view=0,
        )
        maps = torch.cat([item["pred1"]["pts3d"] for item in output], dim=0)
        if len(maps) != len(sequence.rgb):
            raise ValueError("Official inference returned a different number of frames")
        return sample_query_tracks(maps, sequence.query_xy)

    @torch.no_grad()
    def check_clean_parity(
        self,
        sequence: WorldTrackSequence,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> dict[str, float | bool]:
        """Fail before attacks if custom and official clean predictions differ."""
        rgb = sequence.rgb.to(self.device)
        custom = self.predict(rgb)
        reference = self.official_predict(sequence)
        if not torch.isfinite(custom).all() or not torch.isfinite(reference).all():
            raise FloatingPointError("Clean inference contains NaN or Inf")
        max_abs = float((custom - reference).abs().max().item())
        torch.testing.assert_close(custom, reference, atol=atol, rtol=rtol)
        return {"passed": True, "max_abs_error": max_abs, "atol": atol, "rtol": rtol}
