#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_PYTHON_INSTALL_DIR="$task_root/.tools/python"
export UV_CACHE_DIR="$task_root/.cache/uv"
export HF_HOME="$task_root/.cache/huggingface"
export TORCH_HOME="$task_root/.cache/torch"
export MPLCONFIGDIR="$task_root/.cache/matplotlib"
task_uv="$task_root/.tools/uv/uv"
mkdir -p "$task_root/.tools/uv" "$UV_CACHE_DIR" "$task_root/assets"
if [[ ! -x "$task_uv" ]]; then
    curl -fL --retry 3 'https://github.com/astral-sh/uv/releases/download/0.12.22/uv-x86_64-unknown-linux-gnu.tar.gz' -o "$task_root/.tools/uv.tar.gz"
    tar -xzf "$task_root/.tools/uv.tar.gz" -C "$task_root/.tools/uv" --strip-components=1
fi
"$task_uv" python install 3.12 --no-bin
[[ -x "$task_root/.venv/bin/python" ]] || "$task_uv" venv --python 3.12 "$task_root/.venv"
task_python="$task_root/.venv/bin/python"
# cu128 supports Blackwell. Pass TORCH_PLATFORM=cu121 only on pre-Blackwell GPUs
# when intentionally reproducing the author's torch2.5.1 CUDA12.1 environment.
task_platform="${TORCH_PLATFORM:-cu128}"
if [[ "$task_platform" == cu121 ]]; then
    "$task_uv" pip install --python "$task_python" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
else
    "$task_uv" pip install --python "$task_python" torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$task_platform"
fi
"$task_uv" pip install --python "$task_python" -r "$task_root/scripts/requirements-runtime.txt"
"$task_uv" pip install --python "$task_python" --no-deps -e "$task_root"
"$task_python" "$task_root/scripts/verify_environment.py"
"$task_uv" pip freeze --python "$task_python" > "$task_root/assets/environment-freeze.txt"
if [[ "${1:-}" == --download-assets ]]; then
    "$task_python" "$task_root/scripts/download_assets.py"
fi
printf 'Ready: %s\n' "$task_python"
