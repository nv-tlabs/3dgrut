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

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [VENV_NAME] [--help]

Full end-to-end installation script for 3DGRUT using uv. Installs all Python
dependencies, Kaolin, and extra requirements into a .venv (or an active conda
environment).

Arguments:
  VENV_NAME           Prompt name for the virtual environment (default: 3dgrut)

Environment Variables:
  CUDA_HOME           Path to CUDA toolkit (auto-detected when not set)
  TORCH_CUDA_ARCH_LIST
                      Override the GPU architectures to build for. When unset,
                      the list is derived automatically from the installed
                      PyTorch wheel (minimum sm_70).

What this script does:
  1. Initializes git submodules
  2. Detects system CUDA toolkit (requires nvcc or CUDA_HOME)
  3. Creates .venv with Python 3.11 (skipped inside an active conda env)
  4. Pins PyTorch version constraints and configures the PyTorch index
  5. Installs the project and its dependencies (uv pip install -e .[gui])
  6. Detects TORCH_CUDA_ARCH_LIST from the installed PyTorch wheel
  7. Installs Kaolin (pre-built wheel for CUDA <=12, built from source for CUDA 13+)
  8. Installs extra requirements from requirements_extra.txt

Notes:
  - Run inside an active conda environment (created with scripts/create_conda.sh)
    to let conda manage the CUDA toolkit instead of relying on the system nvcc.
  - To set up a .venv with a self-contained CUDA toolkit, run
    scripts/create_venv_cuda.sh first, then activate it before running this script.

Examples:
  $(basename "$0")                          # venv=3dgrut, auto-detect CUDA
  $(basename "$0") myenv                    # venv=myenv,  auto-detect CUDA
  CUDA_HOME=/usr/local/cuda-12.8 $(basename "$0")
  conda activate 3dgrut && $(basename "$0") # use conda-managed CUDA
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

VENV_NAME=${1:-"3dgrut"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ensure_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" ]] && [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "  Using CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  if command -v nvcc &>/dev/null; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
    echo "  Set CUDA_HOME=${CUDA_HOME} (from nvcc in PATH)"
    return 0
  fi
  if [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
    echo "  Set CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  local best=""
  for d in /usr/local/cuda-*; do
    if [[ -d "$d" && -x "$d/bin/nvcc" ]]; then
      best="$d"
    fi
  done
  if [[ -n "$best" ]]; then
    export CUDA_HOME="$best"
    echo "  Set CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

error_with_color_and_exit() {
    echo -e "\033[31m${@}\033[0m" >&2
    exit 1
}

# ==========================================
# Step 1: Initialize git submodules
# ==========================================
echo "[1/8] Initializing git submodules..."
git submodule update --init --recursive
echo ""

# ==========================================
# Step 2: Detect CUDA
# ==========================================
echo "[2/8] Detecting CUDA toolkit..."
if ! ensure_cuda_home; then
    error_with_color_and_exit \
        "ERROR: CUDA toolkit not found and CUDA_HOME is not set. CUDA            \n" \
        "  Toolkit is required to build the project. You can install it          \n" \
        "  using the system package manager (e.g., apt, yum, etc.), or           \n" \
        "  create virtual environment with CUDA toolkit installed.               \n" \
        "                                                                        \n" \
        "To create a conda environment with CUDA toolkit installed, run:         \n" \
        "                                                                        \n" \
        "    CUDA_VERSION=<version> bash scripts/create_conda.sh <env_name>      \n" \
        "                                                                        \n" \
        "To create a VENV with CUDA toolkit installed, run:                      \n" \
        "                                                                        \n" \
        "    CUDA_VERSION=<version> bash scripts/create_venv_cuda.sh <env_name>  \n" \
        "                                                                        \n" \
        "For more details, see the README.md file."
fi

# ==========================================
# Step 3: Create virtual environment
# ==========================================
echo "[3/8] Checking virtual environment..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. We use uv to install python dependencies."
    echo "Please install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  - uv: $(uv --version)"

if [ -z "${CONDA_PREFIX:-}" ]; then
    if [ -d ".venv" ]; then
        # Either using UV installed CUDA or system CUDA but with variables persisted in the venv
        echo "  Virtual environment already exists at .venv"
        echo "  To recreate, remove it first: rm -rf .venv"
    else
        export CUDA_VERSION="$($CUDA_HOME/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
        export CC="${CC:-$(which gcc)}"
        export CXX="${CXX:-$(which g++)}"
        # This should be a symlink to the actual python interpreter, don't use realpath
        export UV_PYTHON="$(pwd)/.venv/bin/python"
        export UV_PROJECT_ENVIRONMENT="$(pwd)/.venv"
        # Compute CUDA related environment variables
        source ${SCRIPT_DIR}/scripts/cuda_helper.sh
        # Using system CUDA if available
        uv venv .venv --python 3.11 --prompt $VENV_NAME
        echo "  Created .venv with Python 3.11"
        # Persist environment variables in the venv
        bash ${SCRIPT_DIR}/scripts/persist_env_vars_in_venv.sh
    fi
    source .venv/bin/activate
    echo "  Activated .venv"
    echo ""
else
    echo "  Running in a pre-configured conda environment, conda manages CUDA toolkit installation"
    echo ""
fi

# Check all environment variables (script will fail if any are not set)
echo "Environment variables:"
echo "  CC: $CC"
echo "  CXX: $CXX"
echo "  CUDA_VERSION: $CUDA_VERSION"
echo "  CUDA_FULL_VERSION: $CUDA_FULL_VERSION"
echo "  CUDA_MAJOR_TARGET: $CUDA_MAJOR_TARGET"
echo "  CUDA_MINOR_TARGET: $CUDA_MINOR_TARGET"
echo "  UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
echo "  UV_PYTHON: $UV_PYTHON"
echo "  TORCH_INDEX_URL: $TORCH_INDEX_URL"
echo "  TORCH_VERSION: $TORCH_VERSION"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo ""

# ==========================================
# Step 4: Set constraints and index
# ==========================================
echo "[4/8] Setting constraints and index..."
if [ -n "${TORCH_VERSION:-}" ]; then
    echo "torch${TORCH_VERSION}" > "$UV_PROJECT_ENVIRONMENT/constraints.txt"
fi
export UV_CONSTRAINT="$UV_PROJECT_ENVIRONMENT/constraints.txt"
echo "  UV constraint file: $UV_CONSTRAINT"

export UV_INDEX="${UV_INDEX:-} pytorch=${TORCH_INDEX_URL}"
echo "  PyTorch index: ${TORCH_INDEX_URL}"
echo ""

# ==========================================
# Step 5: Install project and dependencies
# ==========================================
echo "[5/8] Installing project and dependencies..."
uv pip install -e .[gui]
echo ""

# ==========================================
# Step 7: Build Kaolin from source (CUDA 13+ only)
# ==========================================
if [ "${CUDA_MAJOR_TARGET:-0}" -le 12 ]; then
    echo "[7/8] Kaolin installed from wheel"
    # version is of form 2.8.0+cu128
    actual_torch_version=$(uv pip show torch | grep Version | awk '{print $2}' | sed 's/+cu.*//')
    kaolin_find_link="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${actual_torch_version}_cu${CUDA_MAJOR_TARGET}${CUDA_MINOR_TARGET}.html"
    # append to UV_FIND_LINKS
    export UV_FIND_LINKS="${UV_FIND_LINKS:+${UV_FIND_LINKS}:}${kaolin_find_link}"
    echo "  Kaolin find link: ${kaolin_find_link}"
    echo ""
else
    echo "[7/8] Building Kaolin from source (no pre-built wheel for CUDA ${CUDA_MAJOR_TARGET}.x)..."

    # Clone the repository and remove the existing one if it exists
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

    # Clean up``
    popd > /dev/null
    rm -rf thirdparty/kaolin
    echo "  Kaolin built and installed"
    echo ""
fi
uv pip install -e .[playground]

# ==========================================
# Step 8: Install extra requirements
# ==========================================
echo "[8/8] Installing extra requirements..."
uv pip install --no-build-isolation -r requirements_extra.txt
echo ""

# ==========================================
# Done!
# ==========================================

echo "Verifying the installation..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import kaolin; print(f'Kaolin: {kaolin.__version__}')"
python -c "import ppisp; print(f'PPISP: {ppisp.__version__}')"
python -c "from fused_ssim import fused_ssim; print(f'Fused-SSIM: ready')"

echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "  source .venv/bin/activate"
else
    echo "  conda activate $CONDA_DEFAULT_ENV"
fi
echo ""
