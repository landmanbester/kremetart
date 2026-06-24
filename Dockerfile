# CUDA 13 runtime base.
#
# The [full] stack is a CUDA-13 GPU stack: holoscan-cu13 -> cupy-cuda13x, plus nvidia-cublas
# (13.x). NVIDIA documents that the CUDA runtime is required even for CPU-only Holoscan
# pipelines, so a plain python:slim base can never run these commands natively. The "runtime"
# flavour (not "base") additionally ships the CUDA math libraries cupy links against at
# runtime -- cuBLAS, cuSOLVER (cp.linalg.solve in operators/dft_lm.py), cuFFT, cuRAND,
# cuSPARSE -- so we do not have to assemble them by hand.
#
# Ubuntu 24.04 => glibc 2.39, which is what NVIDIA recommends for the holoscan-cu13 wheel
# (the wheel itself is manylinux_2_35). Bump the patch (13.0.x) freely; keep the major at 13
# so it matches holoscan-cu13 / cupy-cuda13x.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# Make the GPU visible when launched under the NVIDIA Container Toolkit. The runtime base
# already sets these; re-affirmed here as documentation. NB: this image only *uses* the GPU
# when the container is started with GPU access (`docker/podman run --gpus all`, or
# `apptainer/singularity --nv`). Without that the cupy/holoscan pipeline has no CUDA device.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda

# System packages (the cuda-* ones come from the NVIDIA CUDA apt repo already configured in the
# base image; the -13-0 suffix must track the base image's CUDA major.minor):
#   ffmpeg              -> core/smoovie.py encodes the rendered PNG frames to mp4 via subprocess;
#                          smoovie hard-fails ("ffmpeg not found on PATH") without it.
#   cuda-cudart-dev-13-0 + cuda-cccl-13-0
#                       -> CUDA headers (cuda_runtime.h, CUB/Thrust/libcu++). cupy JIT-compiles its
#                          elementwise/reduction kernels at runtime via NVRTC and needs these; the
#                          "runtime" base ships the CUDA *libraries* but not the *headers*, so
#                          without them every cupy kernel (e.g. operators/dft_lm.py's cp.linalg.solve)
#                          dies with "Failed to find CUDA headers". A few MB -- far leaner than the
#                          full "-devel" base. cupy locates them via CUDA_HOME (set below).
#   libgomp1            -> OpenMP runtime that the numpy/scipy/healpy wheels dlopen.
#   ca-certificates     -> TLS for pip and for the TART catalogue API (smoovie --overlay-catalog).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        cuda-cudart-dev-13-0 \
        cuda-cccl-13-0 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast package installation.
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

# The CUDA runtime base ships no Python, and Ubuntu 24.04 would otherwise give Python 3.12.
# Several [full] deps (tart-tools, netcdf4, tart[all], arcae, xarray-ms, msv4-utils) are gated
# on `python_version >= '3.11'`, so we let uv provision a standalone Python 3.11 (matching
# .python-version) into a venv, independent of the base image's system Python.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv python install 3.11 \
    && uv venv --python 3.11 /opt/venv

# Copy package files (LICENSE is required by pyproject's license-files glob).
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install package with full dependencies into the venv (VIRTUAL_ENV is honoured by uv).
RUN uv pip install --no-cache ".[full]"

# Finalize wheel_axle wheels (holoscan-cu13) at build time. These wheels cannot ship
# symlinks, so they defer creating their shared-library SONAME links (e.g.
# libholoscan_core.so.4 -> libholoscan_core.so.4.3.0) to a first-import "finalize" step
# driven by a .pth file. That step needs to write to site-packages -- which is read-only
# when the image runs under apptainer/singularity, so finalize fails there and
# `import holoscan` dies with "libholoscan_core.so.4: cannot open shared object file".
# Triggering site processing now (writable build layer) bakes the symlinks in, writes the
# axle.done marker, and self-removes the .pth, so the runtime import is clean and read-only-safe.
RUN python -c "import holoscan"

# Make CLI available
CMD ["kremetart", "--help"]
