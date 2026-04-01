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

# NOTE: this script will create a conda environment with CUDA toolkit based on the CUDA_VERSION
# specified. It will also set the CC, CXX, and TORCH_CUDA_ARCH_LIST environment variables correctly
# within the conda environment.

usage() {
    cat <<EOF
Usage: $(basename "$0") [CONDA_ENV] [--help]

Create a conda environment with CUDA toolkit and configure build environment
variables for 3DGRUT.

Arguments:
  CONDA_ENV       Name of the conda environment to create (default: 3dgrut)

Environment Variables:
  CUDA_VERSION    CUDA toolkit version to install (default: 12.8)
                  Supported values: 11.8 (or 11), 12.4, 12.6, 12.8 (or 12), 13.0 (or 13)

What this script does:
  1. Resolves CUDA version to a full toolkit version and conda channel
  2. Finds or installs a GCC version compatible with the chosen CUDA toolkit
  3. Creates the conda environment (skips if it already exists)
  4. Persists CC, CXX, TORCH_CUDA_ARCH_LIST and other build vars in the env
  5. Installs cuda-toolkit, cmake, and ninja via conda
  6. Installs OpenGL headers (mesa-libgl-devel) for the playground

Examples:
  $(basename "$0")                        # env=3dgrut, CUDA=12.8
  $(basename "$0") myenv                  # env=myenv,  CUDA=12.8
  CUDA_VERSION=12.4 $(basename "$0")      # env=3dgrut, CUDA=12.4
  CUDA_VERSION=11 $(basename "$0") myenv  # env=myenv,  CUDA=11.8
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CONDA_ENV=${1:-"3dgrut"}
CUDA_VERSION=${CUDA_VERSION:-"12.8"}

echo "=========================================="
echo "Conda Environment Setup"
echo "=========================================="
echo ""

SCRIPT_DIR=$(dirname $(realpath $0))
source $SCRIPT_DIR/cuda_helper.sh

echo "Configuration:"
echo "  CONDA_ENV:            $CONDA_ENV"
echo "  CUDA_VERSION:         $CUDA_VERSION (toolkit $CUDA_FULL_VERSION)"
echo "  CUDA_CONDA_CHANNEL:   $CUDA_CONDA_CHANNEL"
echo "  MAX_GCC_VERSION:      $MAX_GCC_VERSION"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo ""

# ------------------------------------------
# Step 2: Find a compatible GCC version
# ------------------------------------------
gcc_version=$("$GCC_PATH" -dumpversion | cut -d '.' -f 1)
echo ""

# ------------------------------------------
# Step 3: Create and activate conda environment
# ------------------------------------------
eval "$(conda shell.bash hook)"

CONDA_ENV_PATH=$(conda env list | sed -E -n "s/^${CONDA_ENV}[[:space:]]+\*?[[:space:]]*(.*)$/\1/p")
if [ -z "${CONDA_ENV_PATH}" ]; then
    echo "Conda environment '${CONDA_ENV}' not found, creating it..."
    conda create --name "${CONDA_ENV}" -y python=3.11
else
    echo "NOTE: Conda environment '${CONDA_ENV}' already exists at ${CONDA_ENV_PATH}, skipping creation"
fi
conda activate "$CONDA_ENV"

# ------------------------------------------
# Step 4: Persist CC, CXX, and TORCH_CUDA_ARCH_LIST in the conda env
# ------------------------------------------
echo "Setting conda env vars: CC=$GCC_PATH CXX=$GXX_PATH"
conda env config vars set "CC=$GCC_PATH" "CXX=$GXX_PATH" \
    "CUDA_MAJOR_TARGET=$CUDA_MAJOR_TARGET" \
    "CUDA_MINOR_TARGET=$CUDA_MINOR_TARGET" \
    "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
    "TORCH_INDEX_URL=$TORCH_INDEX_URL" \
    "TORCH_VERSION=$TORCH_VERSION" \
    "UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX" \
    "UV_PYTHON=$(which python)"

conda deactivate
conda activate "$CONDA_ENV"

# Verify the compiler is correct after reactivation
actual_gcc_version=$("$CC" -dumpversion | cut -d '.' -f 1)
if [ "$actual_gcc_version" -gt "$MAX_GCC_VERSION" ]; then
    echo "ERROR: CC=$CC reports GCC $actual_gcc_version which exceeds max $MAX_GCC_VERSION for CUDA $CUDA_FULL_VERSION"
    exit 1
fi
echo "  Verified: CC=$CC (GCC $actual_gcc_version), TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo ""

# ------------------------------------------
# Step 5: Install CUDA toolkit and build tools
# ------------------------------------------
echo "Installing CUDA toolkit from $CUDA_CONDA_CHANNEL ..."

if [ "$CUDA_MAJOR" -ge 12 ]; then
    conda install -y cuda-toolkit cmake ninja "gcc_linux-64=$gcc_version" -c "$CUDA_CONDA_CHANNEL"
else
    conda install -y cuda-toolkit cmake ninja -c "$CUDA_CONDA_CHANNEL"
fi

# ------------------------------------------
# Step 6: Install OpenGL headers for the playground
# ------------------------------------------
# Use --override-channels to avoid conflicts with nvidia channel's cuda-toolkit spec
conda install -c conda-forge --override-channels mesa-libgl-devel-cos7-x86_64 -y

echo ""
echo "=========================================="
echo "Conda environment '${CONDA_ENV}' is ready"
echo "=========================================="
echo ""
echo "To activate: conda activate ${CONDA_ENV}"
