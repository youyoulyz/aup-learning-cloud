# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Use a verified ROCm PyTorch image as base
ARG BASE_IMAGE=ghcr.io/amdresearch/auplc-base:latest
FROM ${BASE_IMAGE}

# GPU target and ROCm version — override at build time
#
# Supported GPU_TARGET values:
#   gfx110x       RDNA 3   — gfx1100/1101/1102/1103 (dGPU)
#   gfx1150       RDNA 3.5 — Radeon 890M (Strix Point iGPU)
#   gfx1151       RDNA 3.5 — Radeon 8060S (Strix Halo iGPU)
#   gfx120x       RDNA 4   — Radeon 9x series (dGPU)
#
# The tarball and wheel URLs are derived from GPU_TARGET automatically.
# Use ROCM_TARBALL_URL / PYTORCH_INDEX_URL to override for edge cases.
ARG GPU_TARGET=gfx1151
ARG ROCM_VERSION=7.11.0

# IMPORTANT: base image ends with USER jovyan. pip install as jovyan silently
# falls back to ~/.local/, which gets masked by the PVC mount at runtime.
# Always switch to root before pip install to ensure system-wide installation.
USER root

# Install course-specific packages system-wide
RUN pip3 install --no-cache-dir \
    pandas \
    numpy>=1.21.0 \
    matplotlib>=3.5.0 \
    plotly>=5.10.0 \
    PyYAML>=6.0 \
    jsonschema>=4.0.0 \
    seaborn>=0.12.0 \
    bokeh>=3.0.0 \
    ipywidgets>=8.0.0 \
    jinja2>=3.0.0 \
    markdown>=3.4.0 \
    astunparse==1.6.2 \
    colorlover \
    dash-bootstrap-components \
    dash-svg \
    dash>=3.0.0 \
    kaleido==0.2.1 \
    matplotlib \
    plotext \
    plotille \
    pymongo \
    setuptools \
    sqlalchemy>=2.0.42 \
    tabulate \
    textual \
    textual_plotext \
    textual-fspicker>=0.4.3 \
    tqdm \
    perfetto \
    torch-tb-profiler


RUN apt-get update && \
    # 先安装基础工具
    apt-get install -y --no-install-recommends \
    vim git wget pciutils build-essential cmake libsqlite3-dev python3-dev python3-pip clang && \
    # 强制安装 ROCm 特定架构包，解决文件冲突
    apt-get install -y -o Dpkg::Options::="--force-overwrite" --no-install-recommends \
    amdrocm-hipblas-common-dev-${GPU_TARGET} \
    amdrocm-blas-${GPU_TARGET} \
    amdrocm-blas-dev-${GPU_TARGET} \
    amdrocm-profiler-base \
    amdrocm-profiler \
    amdrocm-developer-tools && \
    rm -rf /var/lib/apt/lists/*
USER root

# Copy related rocm notebooks into the docker
COPY ./course_data /opt/workspace/SHT-PROF



RUN chown -R jovyan:1000 /opt/workspace
USER jovyan
WORKDIR /opt/workspace/SHT-PROF
