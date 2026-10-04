# syntax=docker/dockerfile:1
FROM python:3.12.15-slim-bookworm@sha256:9901e0a8d75037d8242ed43155cbcb2d1f61be1356383d8054afb59fd50e39c4

LABEL org.opencontainers.image.title="St4RTrack full-frame PGD" \
      org.opencontainers.image.description="RTX 5080 CUDA 12.8 runtime; default command is a bounded smoke test" \
      org.opencontainers.image.source="https://github.com/HavenFeng/St4RTrack" \
      org.opencontainers.image.revision="0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=4 \
    MPLBACKEND=Agg \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    XDG_CACHE_HOME=/workspace/.cache/xdg \
    HOME=/workspace/.cache/home

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && apt-get clean \
    && python -m pip install --no-cache-dir uv==0.12.22

COPY docker/runtime-requirements.txt docker/torch-constraints.txt /tmp/dependencies/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --system -c /tmp/dependencies/torch-constraints.txt -r /tmp/dependencies/runtime-requirements.txt

WORKDIR /workspace
# Clone in Linux rather than copying Windows Git settings/line endings.
# Keep .git: the experiment verifies the official commit and clean source.
RUN git clone --recursive https://github.com/HavenFeng/St4RTrack.git vendor/St4RTrack \
    && git -C vendor/St4RTrack checkout --detach 0f9a3f44a7ebac76600cd31ec9eea5228ad7db91 \
    && git -C vendor/St4RTrack submodule update --init --recursive \
    && test "$(git -C vendor/St4RTrack rev-parse HEAD)" = 0f9a3f44a7ebac76600cd31ec9eea5228ad7db91 \
    && git -C vendor/St4RTrack diff --exit-code

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY configs/ configs/
COPY docker/ docker/
RUN uv pip install --system --no-deps -e . \
    && python -m compileall -q src scripts \
    && mkdir -p assets/checkpoints data/worldtrack_release runs/docker .cache/home \
    && python -m pip freeze > docker/image-environment-freeze.txt

CMD ["python", "scripts/docker_smoke.py", "--output", "runs/docker/smoke/report.json", "--timeout-seconds", "240"]
