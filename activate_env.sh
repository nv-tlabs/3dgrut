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

# Activation script for 3DGRUT uv environment
# Usage: source activate_env.sh

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv directory not found."
    echo "Run ./install_env_uv.sh first to create the environment."
    return 1 2>/dev/null || exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Detect local CUDA toolkit installed by install_env_uv.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_LOCAL_DIR=$(find "$SCRIPT_DIR/.venv" -maxdepth 1 -name "cuda-*" -type d 2>/dev/null | head -1)

if [ -n "$CUDA_LOCAL_DIR" ] && [ -x "$CUDA_LOCAL_DIR/bin/nvcc" ]; then
    export CUDA_HOME="$CUDA_LOCAL_DIR"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# Set CUDA architecture list based on nvcc version
NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
if [ -z "$NVCC_VERSION" ]; then
    echo "WARNING: nvcc not found. TORCH_CUDA_ARCH_LIST not set."
    echo "  Run ./install_env_uv.sh to install a local CUDA toolkit."
else
    CUDA_MAJOR=$(echo $NVCC_VERSION | cut -d '.' -f 1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
    else
        export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0+PTX"
    fi
fi

# Set GCC to a compatible version if system GCC is too new
# CUDA 12.x supports up to gcc-13, CUDA 11.x supports up to gcc-11
gcc_version=$(gcc -dumpversion 2>/dev/null | cut -d '.' -f 1)
if [ -n "$gcc_version" ] && [ -n "$CUDA_MAJOR" ]; then
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        MAX_GCC=13
    else
        MAX_GCC=11
    fi
    if [ "$gcc_version" -gt "$MAX_GCC" ]; then
        GCC_FOUND=false
        for v in $(seq $MAX_GCC -1 11); do
            if command -v gcc-$v &> /dev/null && command -v g++-$v &> /dev/null; then
                export CC=$(which gcc-$v)
                export CXX=$(which g++-$v)
                GCC_FOUND=true
                break
            fi
        done
        if [ "$GCC_FOUND" = false ]; then
            # No compatible GCC found — allow nvcc to use the system compiler
            export NVCC_APPEND_FLAGS="--allow-unsupported-compiler"
            export TORCH_NVCC_FLAGS="--allow-unsupported-compiler"
        fi
    fi
fi

echo "3DGRUT environment activated"
[ -n "$CUDA_HOME" ] && echo "  CUDA_HOME=$CUDA_HOME"
[ -n "$TORCH_CUDA_ARCH_LIST" ] && echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
[ -n "$CC" ] && echo "  CC=$CC"
[ -n "$CXX" ] && echo "  CXX=$CXX"
[ -n "$NVCC_APPEND_FLAGS" ] && echo "  NVCC_APPEND_FLAGS=$NVCC_APPEND_FLAGS"
