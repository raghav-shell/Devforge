# Official base image for DevForge "Can It Play DOOM?".
# The private scorer runs agents in exactly this image (tag announced by organizers).
#
#   docker build -t devforge-doom .
#   python validate.py --docker --image devforge-doom     # validate a submission inside it
#
# torch and jax ship their own CUDA runtime via pip wheels; only the host NVIDIA driver is needed.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # JAX preallocates 75% of VRAM by default; agents share one GPU. The scorer caps JAX at the
    # 2 GB budget (XLA_PYTHON_CLIENT_MEM_FRACTION, computed from the host GPU) when it starts agents,
    # and sets the one-thread variables from doom_env.limits.THREAD_ENV (1 CPU per agent).
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git git-lfs curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN uv venv --python 3.13 /opt/venv

# Exact versions of every package: docker/requirements.lock (from docker/requirements.in).
# Bump together with doom_env.CONTRACT_VERSION and the scorer's pin.
COPY docker/requirements.lock /opt/requirements.lock
RUN uv pip install -r /opt/requirements.lock

COPY env /opt/doom_env_src
RUN uv pip install /opt/doom_env_src

WORKDIR /workspace
CMD ["python"]
