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

Create a uv virtual environment (.venv) with a CUDA toolkit for 3DGRUT.
The CUDA toolkit is reused from the system when the major version matches,
and downloaded locally into .venv/cuda-<version>/ otherwise.

Arguments:
  VENV_NAME           Prompt name for the virtual environment (default: 3dgrut)

Environment Variables:
  CUDA_VERSION        CUDA toolkit version to use (default: 12.8)
                      Supported values: 11.8 (or 11), 12.4, 12.6, 12.8 (or 12), 13.0 (or 13)
  FORCE_LOCAL_CUDA    Set to 1 to download and install CUDA locally even when
                      a matching system nvcc is found (default: 0)

What this script does:
  1. Checks that uv is installed
  2. Creates .venv with Python 3.11 (skips if it already exists)
  3. Detects or downloads the CUDA toolkit for the requested version
  4. Persists CUDA/build environment variables into the venv activation hooks

Examples:
  $(basename "$0")                              # venv=3dgrut, CUDA=12.8
  $(basename "$0") myenv                        # venv=myenv,  CUDA=12.8
  CUDA_VERSION=12.4 $(basename "$0")            # venv=3dgrut, CUDA=12.4
  FORCE_LOCAL_CUDA=1 $(basename "$0") myenv     # force local CUDA download
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

VENV_NAME=${1:-"3dgrut"}

# Default CUDA version is 12.8 (for modern GPUs including Blackwell)
# For older GPUs, use: CUDA_VERSION=11.8 ./scripts/create_venv_cuda.sh
CUDA_VERSION=${CUDA_VERSION:-"12.8"}

echo "=========================================="
echo "3DGRUT Installation Script (uv)"
echo "=========================================="
SCRIPT_DIR=$(dirname $(realpath $0))
source $SCRIPT_DIR/cuda_helper.sh

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
echo ""

# ==========================================
# Step 2: Create virtual environment
# ==========================================
echo "[2/9] Creating virtual environment..."

if [ -d ".venv" ]; then
    # Either using UV installed CUDA or system CUDA but with variables persisted in the venv
    echo "  Virtual environment already exists at .venv"
    echo "  To recreate, remove it first: rm -rf .venv"
else
    uv venv .venv --python 3.11 --prompt $VENV_NAME
    echo "  Created .venv with Python 3.11"
fi
# Activate the virtual environment
source .venv/bin/activate
echo "  Activated .venv"
echo ""

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
CUDA_SOURCE=$([ "$USE_SYSTEM_CUDA" = true ] && echo "system" || echo "local")
echo "  - nvcc: $NVCC_VERSION ($CUDA_SOURCE)"
echo ""

bash ${SCRIPT_DIR}/persist_env_vars_in_venv.sh
