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

# Windows port of scripts/cuda_helper.sh
# Resolves CUDA major version to full version, torch index, and arch list.
#
# TORCH_CUDA_ARCH_LIST must match the pytorch wheel build settings.
# Reference: https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh#L54

$ErrorActionPreference = "Stop"

switch -Regex ($env:CUDA_VERSION) {
    "^(11\.8\.0|11\.8|11)$" {
        $env:CUDA_FULL_VERSION      = "11.8.0"
        $env:TORCH_CUDA_ARCH_LIST   = "7.0;7.5;8.0;8.6;9.0+PTX"
        $script:TORCH_VERSION       = "==2.4.0"
    }
    "^(12\.4\.1|12\.4)$" {
        $env:CUDA_FULL_VERSION      = "12.4.1"
        $env:TORCH_CUDA_ARCH_LIST   = "7.5;8.0;8.6;9.0+PTX"
        $script:TORCH_VERSION       = "==2.6.0"
    }
    "^(12\.6\.3|12\.6)$" {
        $env:CUDA_FULL_VERSION      = "12.6.3"
        $env:TORCH_CUDA_ARCH_LIST   = "7.5;8.0;8.6;9.0+PTX"
        $script:TORCH_VERSION       = "==2.8.0"
    }
    "^(12\.8\.1|12\.8|12)$" {
        $env:CUDA_FULL_VERSION      = "12.8.1"
        $env:TORCH_CUDA_ARCH_LIST   = "7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        $script:TORCH_VERSION       = "==2.8.0"
    }
    "^(13\.0\.2|13\.0|13)$" {
        $env:CUDA_FULL_VERSION      = "13.0.2"
        $env:TORCH_CUDA_ARCH_LIST   = "7.5;8.0;8.9;9.0;10.0;12.0+PTX"
        $script:TORCH_VERSION       = ""
    }
    default {
        Write-Host "ERROR: Unsupported CUDA version: $env:CUDA_VERSION" -ForegroundColor Red
        Write-Host "  Available: 11.8, 12.4, 12.6, 12.8, 13.0"
        exit 1
    }
}

Write-Host "  - cuda: $env:CUDA_FULL_VERSION"

$parts = $env:CUDA_FULL_VERSION.Split(".")
$env:CUDA_MAJOR_TARGET = $parts[0]
$env:CUDA_MINOR_TARGET = $parts[1]

$env:TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu$($env:CUDA_MAJOR_TARGET)$($env:CUDA_MINOR_TARGET)"
$env:TORCH_VERSION   = $script:TORCH_VERSION

Write-Host ""
