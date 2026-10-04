"""Record actual runtime and exercise CUDA autograd when a GPU is available."""
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    result = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_architectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "author_environment": {"torch": "2.5.1", "cuda": "12.1", "os": "Linux"},
        "environment_note": "Local RTX5080 (Blackwell) uses torch2.7.1 CUDA12.8. This is a documented deviation from the author's Linux torch2.5.1 CUDA12.1 runtime.",
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["compute_capability"] = torch.cuda.get_device_capability(0)
        x = torch.rand((256, 256), device="cuda", requires_grad=True)
        x.square().mean().backward()
        torch.cuda.synchronize()
        result["cuda_autograd_verified"] = bool(torch.isfinite(x.grad).all())
        result["total_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["nvidia_smi_error"] = str(exc)
    (root / "assets").mkdir(exist_ok=True)
    (root / "assets" / "environment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
