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

#!/bin/bash

# Exit on error
set -e


CONDA_ENV=${1:-"3dgrut"}

# parse an optional second arg WITH_GCC11 to also manually use gcc-11 within the environment
WITH_GCC11=false
if [ $# -ge 2 ]; then
    if [ "$2" = "WITH_GCC11" ]; then
        WITH_GCC11=true
    fi
fi


# Test if we have GCC<=11, and early-out if not
if [ ! "$WITH_GCC11" = true ]; then
    # Make sure gcc is at most 11 for nvcc compatibility
    gcc_version=$(gcc -dumpversion)
    if [ "$gcc_version" -gt 11 ]; then
        echo "Default gcc version $gcc_version is higher than 11. See note about installing gcc-11 (you may need 'sudo apt-get install gcc-11 g++-11') and rerun with ./install.sh 3dgrut WITH_GCC11"
        exit 1
    fi

fi


# If we're going to set gcc11, make sure it is available
if [ "$WITH_GCC11" = true ]; then
    # Ensure gcc-11 is on path
    if ! command -v gcc-11 2>&1 >/dev/null
    then
        echo "gcc-11 could not be found. Perhaps you need to run 'sudo apt-get install gcc-11 g++-11'?"
        exit 1
    fi
    if ! command -v g++-11 2>&1 >/dev/null
    then
        echo "g++-11 could not be found. Perhaps you need to run 'sudo apt-get install gcc-11 g++-11'?"
        exit 1
    fi

    GCC_11_PATH=$(which gcc-11)
    GXX_11_PATH=$(which g++-11)
fi

# Create and activate conda environment
eval "$(conda shell.bash hook)"
conda create -n $CONDA_ENV python=3.11 -y
conda activate $CONDA_ENV

# Set CC and CXX variables to gcc11 in the conda env
if [ "$WITH_GCC11" = true ]; then
    echo "Setting CC=$GCC_11_PATH and CXX=$GXX_11_PATH in conda environment"

    conda env config vars set CC=$GCC_11_PATH CXX=$GXX_11_PATH

    conda deactivate
    conda activate $CONDA_ENV

    # Make sure it worked
    gcc_version=$($CC -dumpversion)
    if [ "$gcc_version" -gt 11 ]; then
        echo "gcc version $gcc_version is still higher than 11, setting gcc-11 failed"
    fi
fi

# Install CUDA and PyTorch dependencies
conda install -y cuda-toolkit -c nvidia/label/cuda-11.8.0
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 "numpy<2.0" -c pytorch -c nvidia/label/cuda-11.8.0
conda install -y cmake ninja -c nvidia/label/cuda-11.8.0
# Install OpenGL headers for the playground
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 -y 

# Initialize git submodules and install Python requirements
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .

echo "Setup completed successfully!"