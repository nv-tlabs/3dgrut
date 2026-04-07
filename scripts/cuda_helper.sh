#!/bin/bash

set -euo pipefail

# ------------------------------------------
# Step 1: Resolve CUDA major version to full version and config
# ------------------------------------------
# The user specifies a major version (11, 12, 13); the script picks the
# concrete toolkit version, conda channel, arch list, and max GCC.
#
# TORCH_CUDA_ARCH_LIST must match the pytorch wheel build settings.
# Reference: https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh#L54
# Verify with: python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
case "$CUDA_VERSION" in
    11.8.0 | 11.8 | 11)
        export CUDA_FULL_VERSION="11.8.0"
        export CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_FULL_VERSION}/local_installers/cuda_${CUDA_FULL_VERSION}_520.61.05_linux.run"
        export MAX_GCC_VERSION=11
        export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0+PTX"
        # Pin PyTorch to 2.4.0 — latest version with pre-built Kaolin cu118 wheel
        TORCH_VERSION="==2.4.0"
        ;;
    12.4.1 | 12.4)
        export CUDA_FULL_VERSION="12.4.1"
        export CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_FULL_VERSION}/local_installers/cuda_${CUDA_FULL_VERSION}_550.54.15_linux.run"
        export MAX_GCC_VERSION=13
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX"
        TORCH_VERSION="==2.6.0"
        ;;
    12.6.3 | 12.6)
        export CUDA_FULL_VERSION="12.6.3"
        export CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_FULL_VERSION}/local_installers/cuda_${CUDA_FULL_VERSION}_560.35.05_linux.run"
        export MAX_GCC_VERSION=13
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        TORCH_VERSION="==2.8.0"
        ;;
    12.8.1 | 12.8 | 12)
        export CUDA_FULL_VERSION="12.8.1"
        export CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_FULL_VERSION}/local_installers/cuda_${CUDA_FULL_VERSION}_570.124.06_linux.run"
        export MAX_GCC_VERSION=14
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        TORCH_VERSION="==2.8.0"
        ;;
    13.0.2 | 13.0 | 13)
        export CUDA_FULL_VERSION="13.0.2"
        export MAX_GCC_VERSION=16
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.9;9.0;10.0;12.0+PTX"
        ;;
    *)
        echo "ERROR: Unsupported CUDA version: $CUDA_VERSION"
        echo "  Available: 11.8, 12.4, 12.6, 12.8, 13.0"
        exit 1
        ;;
esac
echo "  - cuda: $CUDA_FULL_VERSION"

# Marketing major.minor from the resolved toolkit (e.g. 11.8.0 -> 11, 8), not from CUDA_VERSION aliases like "12".
CUDA_MAJOR_TARGET=$(echo "$CUDA_FULL_VERSION" | cut -d. -f1)
CUDA_MINOR_TARGET=$(echo "$CUDA_FULL_VERSION" | cut -d. -f2)
export CUDA_MAJOR_TARGET CUDA_MINOR_TARGET
export CUDA_CONDA_CHANNEL="nvidia/label/cuda-${CUDA_FULL_VERSION}"

# PyTorch release URL for the given CUDA version
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_MAJOR_TARGET}${CUDA_MINOR_TARGET}"
export TORCH_VERSION=${TORCH_VERSION:-}
# export TORCHVISION_VERSION=${TORCHVISION_VERSION:-}
# export TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-}

# Check GCC version — find or install a compatible version
gcc_version=$(gcc -dumpversion | cut -d '.' -f 1)
if [ "$gcc_version" -gt "$MAX_GCC_VERSION" ]; then
    # Try to find an already-installed compatible GCC version
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
        echo "  - gcc: $gcc_version is too new for CUDA $CUDA_VERSION (requires GCC <= $MAX_GCC_VERSION)"
        echo "    Installing gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION..."
        sudo apt-get install -y gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION
        GCC_PATH=$(which gcc-$MAX_GCC_VERSION)
        GXX_PATH=$(which g++-$MAX_GCC_VERSION)
        echo "  - gcc: Installed and using gcc-$MAX_GCC_VERSION"
    fi
else
    GCC_PATH=$(which gcc)
    GXX_PATH=$(which g++)
    echo "  - gcc: $gcc_version"
fi

echo ""

export GCC_PATH
export GXX_PATH

