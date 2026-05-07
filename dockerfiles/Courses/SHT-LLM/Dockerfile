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

# IMPORTANT: base image ends with USER jovyan. pip install as jovyan silently
# falls back to ~/.local/, which gets masked by the PVC mount at runtime.
# Always switch to root before pip install to ensure system-wide installation.
USER root

# Install course-specific packages system-wide
RUN pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    transformers \
    datasets>=2.14.0 \
    tokenizers>=0.15.0 \
    accelerate>=0.25.0 \
    evaluate>=0.4.0 \
    # Parameter-efficient fine-tuning for L8 and L10
    peft>=0.7.0 \
    # Text processing and NLP utilities
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    # Visualization
    matplotlib>=3.7.0 \
    seaborn>=0.11.0 \
    plotly>=5.0.0 \
    kaleido>=0.2.1 \
    # Graph visualization for L3 neural network fundamentals
    torchviz>=0.0.2 \
    graphviz>=0.20.1 \
    # Image processing
    Pillow>=8.3.0 \
    imageio>=2.9.0
    # Additional dependencies for advanced features

# 预先升级基础构建工具
RUN pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --upgrade setuptools

RUN pip3 install --no-cache-dir  -i  https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package  \
    requests>=2.28.0 \
    typing-extensions>=4.4.0 \
    # LLM15: Quantization and Evaluation
    gptqmodel>=1.0.0 \
    autoawq>=0.2.0 \
    #optimum>=1.15.0 \
    llmcompressor>=0.1.0 \
    # Quantization utilities
    bitsandbytes>=0.41.0 \
    # GGUF support (llama.cpp Python bindings)
    llama-cpp-python>=0.2.0 \
    # Perplexity and evaluation metrics
    rouge_score>=0.1.2 \
    sacrebleu>=2.0.0 \
    optimum>=1.24.0

# FashionMNIST
RUN mkdir -p /opt/workspace/LLM/data/FashionMNIST/raw && cd /opt/workspace/LLM/data/FashionMNIST/raw && \
    wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz && \
    wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz && \
    wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz && \
    wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

# Download DialoGPT-small
RUN mkdir -p /opt/workspace/LLM/models/microsoft/DialoGPT-small && cd /opt/workspace/LLM/models/microsoft/DialoGPT-small && \
    wget https://hf-mirror.com/microsoft/DialoGPT-small/resolve/main/config.json && \
    wget https://hf-mirror.com/microsoft/DialoGPT-small/resolve/main/pytorch_model.bin && \
    wget https://hf-mirror.com/microsoft/DialoGPT-small/resolve/main/vocab.json && \
    wget https://hf-mirror.com/microsoft/DialoGPT-small/resolve/main/merges.txt && \
    wget https://hf-mirror.com/microsoft/DialoGPT-small/resolve/main/tokenizer_config.json

# Download Yahama
RUN mkdir -p /opt/workspace/LLM/data/yahma/alpaca-cleaned && cd /opt/workspace/LLM/data/yahma/alpaca-cleaned && \
    wget https://hf-mirror.com/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json

# Copy related rocm notebooks into the docker
COPY ./course_data /opt/workspace/LLM

RUN chown -R jovyan:1000 /opt/workspace
USER jovyan
WORKDIR /opt/workspace/LLM
