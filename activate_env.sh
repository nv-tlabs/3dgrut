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

# Set CUDA architecture list based on installed CUDA version
NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
if [ -z "$NVCC_VERSION" ]; then
    echo "WARNING: nvcc not found in PATH. TORCH_CUDA_ARCH_LIST not set."
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
        for v in $(seq $MAX_GCC -1 11); do
            if command -v gcc-$v &> /dev/null; then
                export CC=$(which gcc-$v)
                export CXX=$(which g++-$v)
                break
            fi
        done
    fi
fi

echo "3DGRUT environment activated"
[ -n "$TORCH_CUDA_ARCH_LIST" ] && echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
[ -n "$CC" ] && echo "  CC=$CC"
[ -n "$CXX" ] && echo "  CXX=$CXX"
