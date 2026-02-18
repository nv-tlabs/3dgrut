#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit on error
set -e

# Default CUDA version is 12.8 (for modern GPUs including Blackwell)
# For older GPUs, use: CUDA_VERSION=11.8 ./install_env_uv.sh
CUDA_VERSION=${CUDA_VERSION:-"12.8"}

echo "=========================================="
echo "3DGRUT Installation Script (uv)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  CUDA_VERSION: $CUDA_VERSION"
echo ""

# ==========================================
# Step 1: Check prerequisites
# ==========================================
echo "[1/8] Checking prerequisites..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  - uv: $(uv --version)"

# Check for nvcc (CUDA toolkit)
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit $CUDA_VERSION"
    echo "  For Ubuntu: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo "  - nvcc: $NVCC_VERSION"

# Determine maximum supported GCC version based on CUDA version
# CUDA 11.8 supports up to gcc-11, CUDA 12.x supports up to gcc-13
NVCC_MAJOR=$(echo $NVCC_VERSION | cut -d '.' -f 1)
if [ "$NVCC_MAJOR" -ge 12 ]; then
    MAX_GCC_VERSION=13
else
    MAX_GCC_VERSION=11
fi

# Check GCC version
gcc_version=$(gcc -dumpversion | cut -d '.' -f 1)
if [ "$gcc_version" -gt "$MAX_GCC_VERSION" ]; then
    # Try to find a compatible GCC version
    GCC_FOUND=false
    for v in $(seq $MAX_GCC_VERSION -1 11); do
        if command -v gcc-$v &> /dev/null && command -v g++-$v &> /dev/null; then
            GCC_PATH=$(which gcc-$v)
            GXX_PATH=$(which g++-$v)
            GCC_FOUND=true
            echo "  - gcc: Using gcc-$v (system gcc is $gcc_version)"
            break
        fi
    done
    if [ "$GCC_FOUND" = false ]; then
        echo "ERROR: Default gcc version is $gcc_version (>$MAX_GCC_VERSION), and no compatible gcc found."
        echo "  CUDA $NVCC_VERSION requires GCC <= $MAX_GCC_VERSION. Install a compatible version:"
        echo "    sudo apt-get install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION"
        exit 1
    fi
else
    GCC_PATH=$(which gcc)
    GXX_PATH=$(which g++)
    echo "  - gcc: $gcc_version"
fi

# Set TORCH_CUDA_ARCH_LIST and PyTorch index URL based on CUDA version
# We support: 11.8 (older GPUs) and 12.x (modern GPUs)
if [ "$CUDA_VERSION" = "11.8" ]; then
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0+PTX"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    CUDA_MAJOR_TARGET=11
elif [[ "$CUDA_VERSION" == 12* ]]; then
    # CUDA 12.x - use cu124 wheels (compatible with 12.4+)
    # PyTorch cu124 supports up to sm_90 (Hopper/Ada)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    CUDA_MAJOR_TARGET=12
else
    echo "ERROR: Unsupported CUDA version: $CUDA_VERSION"
    echo "  Supported versions: 11.8, 12.x (e.g., 12.4, 12.6, 12.8)"
    exit 1
fi

# Verify system CUDA matches requested version
if [ "$NVCC_MAJOR" != "$CUDA_MAJOR_TARGET" ]; then
    echo "WARNING: System CUDA ($NVCC_VERSION) does not match requested CUDA ($CUDA_VERSION)"
    echo "  Proceeding with system CUDA $NVCC_VERSION"
fi

echo ""

# ==========================================
# Step 2: Create virtual environment
# ==========================================
echo "[2/8] Creating virtual environment..."

if [ -d ".venv" ]; then
    echo "  Virtual environment already exists at .venv"
    echo "  To recreate, remove it first: rm -rf .venv"
else
    uv venv .venv --python 3.11
    echo "  Created .venv with Python 3.11"
fi

# Activate environment
source .venv/bin/activate
echo "  Activated .venv"
echo ""

# Set compiler environment variables
export CC=$GCC_PATH
export CXX=$GXX_PATH

# ==========================================
# Step 3: Install PyTorch with CUDA
# ==========================================
echo "[3/8] Installing PyTorch with CUDA $CUDA_VERSION..."

uv pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL
echo "  PyTorch installed"
echo ""

# ==========================================
# Step 4: Install numpy (must be <2.0)
# ==========================================
echo "[4/8] Installing numpy<2.0..."

uv pip install "numpy<2.0"
echo "  numpy installed"
echo ""

# ==========================================
# Step 5: Install Kaolin
# ==========================================
echo "[5/8] Installing Kaolin..."

if [ "$CUDA_MAJOR_TARGET" = "11" ]; then
    # Use pre-built wheel for CUDA 11.8
    uv pip install kaolin==0.17.0 --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html
    echo "  Kaolin installed from wheel"
else
    # Build Kaolin from source for CUDA 12.x
    echo "  Building Kaolin from source (no pre-built wheel for CUDA 12.x)..."

    rm -rf thirdparty/kaolin
    git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git thirdparty/kaolin
    pushd thirdparty/kaolin > /dev/null

    # Pin to a fixed commit for reproducibility
    git checkout c2da967b9e0d8e3ebdbd65d3e8464d7e39005203

    # Apply fix for CUDA 12.x compatibility
    sed -i 's!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type()!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type()!g' kaolin/csrc/render/spc/raytrace_cuda.cu

    # Install build dependencies
    uv pip install ninja imageio imageio-ffmpeg
    uv pip install -r tools/viz_requirements.txt -r tools/requirements.txt -r tools/build_requirements.txt

    # Build and install
    IGNORE_TORCH_VER=1 python setup.py install

    popd > /dev/null
    rm -rf thirdparty/kaolin
    echo "  Kaolin built and installed"
fi
echo ""

# ==========================================
# Step 6: Initialize git submodules
# ==========================================
echo "[6/8] Initializing git submodules..."

git submodule update --init --recursive
echo "  Submodules initialized"
echo ""

# ==========================================
# Step 7: Install git dependencies and project
# ==========================================
echo "[7/8] Installing dependencies and project..."

# Install fused-ssim (git dependency)
uv pip install --no-build-isolation "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

# Install ppisp (git dependency)
uv pip install --no-build-isolation "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.1"

# Install the project with dependencies from pyproject.toml
uv pip install --no-build-isolation -e .
echo "  Project installed"
echo ""

# ==========================================
# Step 8: Install tiny-cuda-nn
# ==========================================
echo "[8/8] Installing tiny-cuda-nn..."

pushd thirdparty/tiny-cuda-nn/bindings/torch > /dev/null
uv pip install --no-build-isolation .
popd > /dev/null
echo "  tiny-cuda-nn installed"
echo ""

# ==========================================
# Done!
# ==========================================
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source activate_env.sh"
echo ""
echo "To verify the installation:"
echo "  python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
echo "  python -c \"import kaolin; print(f'Kaolin: {kaolin.__version__}')\""
echo ""
