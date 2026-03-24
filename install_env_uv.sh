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
echo "[1/9] Checking prerequisites..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  - uv: $(uv --version)"

# Determine maximum supported GCC version based on CUDA version
# CUDA 11.8 supports up to gcc-11, CUDA 12.x supports up to gcc-13
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d '.' -f 1)
if [ "$CUDA_MAJOR" -ge 12 ]; then
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
        echo "  CUDA $CUDA_VERSION requires GCC <= $MAX_GCC_VERSION. Install a compatible version:"
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
    CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
    CUDA_FULL_VERSION="11.8.0"
elif [ "$CUDA_VERSION" = "12.4" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    CUDA_MAJOR_TARGET=12
    CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run"
    CUDA_FULL_VERSION="12.4.1"
elif [ "$CUDA_VERSION" = "12.6" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    CUDA_MAJOR_TARGET=12
    CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run"
    CUDA_FULL_VERSION="12.6.3"
elif [ "$CUDA_VERSION" = "12.8" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    CUDA_MAJOR_TARGET=12
    CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run"
    CUDA_FULL_VERSION="12.8.1"
else
    echo "ERROR: Unsupported CUDA version: $CUDA_VERSION"
    echo "  Supported versions: 11.8, 12.4, 12.6, 12.8"
    exit 1
fi

echo ""

# ==========================================
# Step 2: Create virtual environment
# ==========================================
echo "[2/9] Creating virtual environment..."

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
# Step 3: Set up CUDA toolkit
# ==========================================
# Use system CUDA if available and matching, otherwise download locally.
# To force local install even with system CUDA: FORCE_LOCAL_CUDA=1 ./install_env_uv.sh
CUDA_LOCAL_DIR="$(pwd)/.venv/cuda-${CUDA_FULL_VERSION}"
USE_SYSTEM_CUDA=false

if [ "${FORCE_LOCAL_CUDA:-0}" != "1" ]; then
    SYSTEM_NVCC=$(command -v nvcc 2>/dev/null || true)
    if [ -n "$SYSTEM_NVCC" ]; then
        SYSTEM_CUDA_VER=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        SYSTEM_CUDA_MAJOR=$(echo "$SYSTEM_CUDA_VER" | cut -d '.' -f 1)
        if [ "$SYSTEM_CUDA_MAJOR" = "$CUDA_MAJOR_TARGET" ]; then
            USE_SYSTEM_CUDA=true
            # Derive CUDA_HOME from nvcc location (e.g., /usr/local/cuda/bin/nvcc -> /usr/local/cuda)
            export CUDA_HOME="$(dirname "$(dirname "$SYSTEM_NVCC")")"
        fi
    fi
fi

if [ "$USE_SYSTEM_CUDA" = true ]; then
    echo "[3/9] Using system CUDA toolkit at $CUDA_HOME (nvcc $SYSTEM_CUDA_VER)"
else
    if [ -x "$CUDA_LOCAL_DIR/bin/nvcc" ]; then
        echo "[3/9] Local CUDA toolkit already installed at $CUDA_LOCAL_DIR"
    else
        echo "[3/9] Installing local CUDA toolkit ${CUDA_FULL_VERSION}..."
        CUDA_RUNFILE="/tmp/cuda_${CUDA_FULL_VERSION}_linux.run"

        if [ ! -f "$CUDA_RUNFILE" ]; then
            echo "  Downloading CUDA ${CUDA_FULL_VERSION} toolkit (~4GB)..."
            wget -q --show-progress -O "$CUDA_RUNFILE" "$CUDA_RUNFILE_URL"
        else
            echo "  Using cached runfile at $CUDA_RUNFILE"
        fi

        echo "  Extracting toolkit to $CUDA_LOCAL_DIR (this may take a few minutes)..."
        chmod +x "$CUDA_RUNFILE"
        "$CUDA_RUNFILE" --toolkit --toolkitpath="$CUDA_LOCAL_DIR" --silent --no-man-page --override
        echo "  CUDA ${CUDA_FULL_VERSION} toolkit installed locally"
    fi

    export CUDA_HOME="$CUDA_LOCAL_DIR"
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

NVCC_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo "  - nvcc: $NVCC_VERSION ($( [ "$USE_SYSTEM_CUDA" = true ] && echo "system" || echo "local" ))"
echo ""

# ==========================================
# Step 4: Install PyTorch with CUDA
# ==========================================
echo "[4/9] Installing PyTorch with CUDA $CUDA_VERSION..."

uv pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL
echo "  PyTorch installed"
echo ""

# ==========================================
# Step 5: Install numpy (must be <2.0)
# ==========================================
echo "[5/9] Installing numpy<2.0..."

uv pip install "numpy<2.0"
echo "  numpy installed"
echo ""

# ==========================================
# Step 6: Install Kaolin
# ==========================================
echo "[6/9] Installing Kaolin..."

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
# Step 7: Initialize git submodules
# ==========================================
echo "[7/9] Initializing git submodules..."

git submodule update --init --recursive
echo "  Submodules initialized"
echo ""

# ==========================================
# Step 8: Install git dependencies and project
# ==========================================
echo "[8/9] Installing dependencies and project..."

# Install fused-ssim (git dependency)
uv pip install --no-build-isolation "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

# Install ppisp (git dependency)
uv pip install --no-build-isolation "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.1"

# Install the project with dependencies from pyproject.toml
uv pip install --no-build-isolation -e .
echo "  Project installed"
echo ""

# ==========================================
# Step 9: Install tiny-cuda-nn
# ==========================================
echo "[9/9] Installing tiny-cuda-nn..."

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
