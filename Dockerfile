ARG REPO=248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland
ARG BASE_TAG=base
FROM ${REPO}:${BASE_TAG}

SHELL ["/bin/bash", "-c"]

RUN condax install uv
RUN condax install s5cmd

ENV LANG=en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install python3-dev libnuma1 cmake -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY pyproject.toml /workdir/pyproject.toml
COPY uv.lock /workdir/uv.lock
COPY .python-version /workdir/.python-version
COPY 3rdparty/Megatron-LM/pyproject.toml /workdir/3rdparty/Megatron-LM/pyproject.toml
COPY 3rdparty/Megatron-LM/setup.py /workdir/3rdparty/Megatron-LM/setup.py
COPY 3rdparty/Megatron-LM/megatron/core/__init__.py /workdir/3rdparty/Megatron-LM/megatron/core/__init__.py
COPY 3rdparty/Megatron-LM/megatron/core/package_info.py /workdir/3rdparty/Megatron-LM/megatron/core/package_info.py
COPY 3rdparty/Megatron-LM/megatron/core/datasets/Makefile /workdir/3rdparty/Megatron-LM/megatron/core/datasets/Makefile
COPY 3rdparty/Megatron-LM/megatron/core/datasets/helpers.cpp /workdir/3rdparty/Megatron-LM/megatron/core/datasets/helpers.cpp
COPY src/megatron/bridge/__init__.py /workdir/src/megatron/bridge/__init__.py
COPY src/megatron/bridge/package_info.py /workdir/src/megatron/bridge/package_info.py
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN --mount=type=cache,id=uv-cache2,target=/root/.cache/uv <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
conda activate base
cd /workdir
uv sync --only-group build
SH

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
conda activate base
source /workdir/.venv/bin/activate
J="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$J CTEST_PARALLEL_LEVEL=$J NPY_NUM_BUILD_JOBS=$J \
       CARGO_BUILD_JOBS=$J MAX_JOBS=$J MAKEFLAGS="-j$J -l$J" NINJAFLAGS="-j $J" \
       NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
       TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
echo "$MAX_JOBS"
cd /workdir
uv sync
SH

RUN <<'EOF' cat >> /root/.bashrc
source /workdir/.venv/bin/activate
EOF

COPY . /workdir

WORKDIR /workdir
