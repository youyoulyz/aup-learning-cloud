#!/usr/bin/env bash
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

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALLER="$ROOT/auplc-installer"

grep -q 'verify_sha256()' "$INSTALLER"
grep -q 'HELM_LINUX_AMD64_SHA256=' "$INSTALLER"
grep -q 'K9S_LINUX_AMD64_DEB_SHA256=' "$INSTALLER"
grep -q 'ROCM_DEVICE_PLUGIN_SHA256=' "$INSTALLER"
grep -q 'ROCM_DEVICE_PLUGIN_COMMIT=' "$INSTALLER"
grep -q 'ROCM_LABELLER_SHA256=' "$INSTALLER"
grep -Fq "INSTALL_K3S_VERSION=\"\${K3S_VERSION}\"" "$INSTALLER"

if grep -q 'ROCM_DEVICE_PLUGIN_COMMIT="master"' "$INSTALLER"; then
    echo 'FAIL: ROCm device plugin still tracks master instead of a pinned commit'
    exit 1
fi

if grep -q 'curl -sfL https://get.k3s.io |' "$INSTALLER"; then
    echo 'FAIL: k3s still uses pipe-to-shell'
    exit 1
fi

if grep -q 'kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml' "$INSTALLER"; then
    echo 'FAIL: ROCm plugin still applies remote URL directly'
    exit 1
fi

if grep -q 'kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-labeller.yaml' "$INSTALLER"; then
    echo 'FAIL: ROCm labeller still applies remote URL directly'
    exit 1
fi

grep -q 'verify_sha256 /tmp/helm-linux-amd64.tar.gz' "$INSTALLER"
grep -q 'verify_sha256 /tmp/k9s_linux_amd64.deb' "$INSTALLER"
grep -q 'verify_sha256 /tmp/k8s-ds-amdgpu-dp.yaml' "$INSTALLER"
grep -q 'verify_sha256 /tmp/k8s-ds-amdgpu-labeller.yaml' "$INSTALLER"

echo 'Installer integrity checks present.'
