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

CUDA_VERSION=${CUDA_VERSION:-"11.8.0"}

# Verify user arguments
echo "Arguments:"
echo "  CONDA_ENV: $CONDA_ENV"
echo "  WITH_GCC11: $WITH_GCC11"
echo "  CUDA_VERSION: $CUDA_VERSION"
echo ""

# Make sure TORCH_CUDA_ARCH_LIST matches the pytorch wheel setting.
# Reference: https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh#L54
#
# (cuda11) $ python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
# 11.8 ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_37', 'sm_90', 'compute_37']
#
# (cuda12) $ python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
# 12.8 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
#
# Check if CUDA_VERSION is supported
if [ "$CUDA_VERSION" = "11.8.0" ]; then
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0";
elif [ "$CUDA_VERSION" = "12.8.1" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0";
else
    echo "Unsupported CUDA version: $CUDA_VERSION, available options are 11.8.0 and 12.8.1"
    exit 1
fi
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Test if we have GCC<=11, and early-out if not
if [ ! "$WITH_GCC11" = true ]; then
    # Make sure gcc is at most 11 for nvcc compatibility
    gcc_version=$(gcc -dumpversion)
    if [ "$gcc_version" -gt 11 ]; then
        echo "Default gcc version $gcc_version is higher than 11. See note about installing gcc-11 (you may need 'sudo apt-get install gcc-11 g++-11') and rerun with ./install_env.sh 3dgrut WITH_GCC11"
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
GCC_VERSION=$($GCC_11_PATH -dumpversion | cut -d '.' -f 1)

# Create and activate conda environment
eval "$(conda shell.bash hook)"

# Finds the path of the environment if the environment already exists
CONDA_ENV_PATH=$(conda env list | sed -E -n "s/^${CONDA_ENV}[[:space:]]+\*?[[:space:]]*(.*)$/\1/p")
if [ -z "${CONDA_ENV_PATH}" ]; then
  echo "Conda environment '${CONDA_ENV}' not found, creating it"
  conda create --name ${CONDA_ENV} -y python=3.11
else
  echo "NOTE: Conda environment '${CONDA_ENV}' already exists at ${CONDA_ENV_PATH}, skipping environment creation"
fi
conda activate $CONDA_ENV

# Set CC and CXX variables to gcc11 in the conda env
if [ "$WITH_GCC11" = true ]; then
    echo "Setting CC=$GCC_11_PATH and CXX=$GXX_11_PATH in conda environment"

    conda env config vars set CC=$GCC_11_PATH CXX=$GXX_11_PATH

    conda deactivate
    conda activate $CONDA_ENV

    # Make sure it worked
    gcc_version=$($CC -dumpversion | cut -d '.' -f 1)
    echo "gcc_version=$gcc_version"
    if [ "$gcc_version" -gt 11 ]; then
        echo "gcc version $gcc_version is still higher than 11, setting gcc-11 failed"
        exit 1
    fi
fi

conda env config vars set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
conda deactivate
conda activate $CONDA_ENV

# Install CUDA and PyTorch dependencies
# CUDA 11.8 supports until compute capability 9.0
if [ "$CUDA_VERSION" = "11.8.0" ]; then
    echo "Installing CUDA 11.8.0 ..."
    conda install -y cuda-toolkit cmake ninja -c nvidia/label/cuda-11.8.0
    conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 "numpy<2.0" -c pytorch -c nvidia/label/cuda-11.8.0
    pip3 install --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html kaolin==0.17.0

# CUDA 12.8 supports compute capability 10.0 and 12.0
elif [ "$CUDA_VERSION" = "12.8.1" ]; then
    echo "Installing CUDA 12.8.1 ..."
    conda install -y cuda-toolkit cmake ninja gcc_linux-64=$GCC_VERSION -c nvidia/label/cuda-12.8.1
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    pip3 install --force-reinstall "numpy<2"

    # TODO move to using wheel once kaolin is available
    rm -fr thirdparty/kaolin
    git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git thirdparty/kaolin
    pushd thirdparty/kaolin
    pip install --upgrade pip
    pip install --no-cache-dir ninja imageio imageio-ffmpeg
    pip install --no-cache-dir        \
        -r tools/viz_requirements.txt \
        -r tools/requirements.txt     \
        -r tools/build_requirements.txt
    IGNORE_TORCH_VER=1 python setup.py develop
    popd
    rm -fr thirdparty/kaolin

# Unsupported CUDA version
else
    echo "Unsupported CUDA version: $CUDA_VERSION, available options are 11.8.0 and 12.8.1"
    exit 1
fi

# Install OpenGL headers for the playground
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 -y 

# Initialize git submodules and install Python requirements
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .

echo "Setup completed successfully!"
