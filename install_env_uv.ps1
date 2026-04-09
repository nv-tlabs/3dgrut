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

<#
.SYNOPSIS
    Full end-to-end installation script for 3DGRUT on Windows using uv.

.DESCRIPTION
    Installs all Python dependencies, Kaolin, and extra requirements into a .venv.

    What this script does:
      1. Initializes git submodules
      2. Detects system CUDA toolkit (requires nvcc in PATH or CUDA_HOME)
      3. Detects Visual Studio C++ compiler (cl.exe, cmake, ninja)
      4. Creates .venv with Python 3.11
      5. Pins PyTorch version constraints and configures the PyTorch index
      6. Installs the project and its dependencies (uv pip install -e .[dev,gui])
      7. Installs Kaolin (pre-built wheel for CUDA <=12)
      8. Installs extra requirements from requirements_extra.txt
      9. Installs slangc

.PARAMETER VenvName
    Prompt name for the virtual environment (default: 3dgrut)

.EXAMPLE
    .\install_env_uv.ps1
    .\install_env_uv.ps1 -VenvName myenv
    $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"; .\install_env_uv.ps1
#>

param (
    [string]$VenvName = "3dgrut"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Error-And-Exit {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
    exit 1
}

$env:DISTUTILS_USE_SDK=1

# ==========================================
# Step 1: Initialize git submodules
# ==========================================
Write-Host "[1/9] Initializing git submodules..."
git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { Write-Error-And-Exit "git submodule update failed" }
Write-Host ""

# ==========================================
# Step 2: Detect CUDA
# ==========================================
Write-Host "[2/9] Detecting CUDA toolkit..."

if ($env:CUDA_HOME -and (Test-Path (Join-Path $env:CUDA_HOME "bin\nvcc.exe"))) {
    Write-Host "  Using CUDA_HOME=$env:CUDA_HOME"
} else {
    $nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvccPath) {
        $env:CUDA_HOME = Split-Path (Split-Path $nvccPath.Source)
        Write-Host "  Set CUDA_HOME=$env:CUDA_HOME (from nvcc in PATH)"
    } else {
        $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $cudaRoot) {
            $best = Get-ChildItem $cudaRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
            if ($best -and (Test-Path (Join-Path $best.FullName "bin\nvcc.exe"))) {
                $env:CUDA_HOME = $best.FullName
                Write-Host "  Set CUDA_HOME=$env:CUDA_HOME (auto-detected)"
            }
        }
    }
}

if (-not $env:CUDA_HOME -or -not (Test-Path (Join-Path $env:CUDA_HOME "bin\nvcc.exe"))) {
    Write-Error-And-Exit @"
ERROR: CUDA toolkit not found and CUDA_HOME is not set.
  Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
  or set CUDA_HOME to your CUDA installation directory.
"@
}

$nvccOutput = & "$env:CUDA_HOME\bin\nvcc.exe" --version 2>&1 | Out-String
$match = [regex]::Match($nvccOutput, "release (\d+\.\d+)")
if (-not $match.Success) { Write-Error-And-Exit "Could not parse CUDA version from nvcc output" }
$env:CUDA_VERSION = $match.Groups[1].Value
Write-Host "  CUDA version: $env:CUDA_VERSION"
Write-Host ""

# ==========================================
# Step 3: Detect Visual Studio build tools
# ==========================================
Write-Host "[3/9] Detecting Visual Studio build tools..."

# CUDA 12.x officially supports MSVC from VS 2017 through VS 2022.
$script:CudaCompatibleVsIds = @("2017", "2019", "2022", "15", "16", "17")

function Get-VsVersionFromPath([string]$Path) {
    $parts = $Path.Replace("/", "\") -split "\\"
    for ($i = 0; $i -lt $parts.Length; $i++) {
        if ($parts[$i] -eq "Microsoft Visual Studio" -and ($i + 1) -lt $parts.Length) {
            return $parts[$i + 1]
        }
    }
    return ""
}

# Resolve glob patterns against VS install directories, preferring CUDA-compatible versions.
function Find-InVsInstall([string[]]$Patterns) {
    $compatible = $null
    $fallback = $null
    foreach ($pattern in $Patterns) {
        foreach ($p in (Resolve-Path $pattern -ErrorAction SilentlyContinue)) {
            if (-not $fallback) { $fallback = $p.Path }
            if (-not $compatible -and
                ($script:CudaCompatibleVsIds -contains (Get-VsVersionFromPath $p.Path))) {
                $compatible = $p.Path
            }
        }
    }
    if ($compatible) { return $compatible }
    return $fallback
}

# If the MSVC environment is already fully configured (Developer Command Prompt,
# vcvarsall.bat, or ilammy/msvc-dev-cmd in CI), respect it. We verify both cl.exe
# on PATH and LIB containing x64 paths to catch partial setups that cause x86/x64
# linker mismatches.
$existingCl = Get-Command cl.exe -ErrorAction SilentlyContinue
$libIsX64 = $env:LIB -and ($env:LIB -match "\\x64|\\amd64")

if ($existingCl -and $libIsX64) {
    $clDir = Split-Path $existingCl.Source
    Write-Host "  cl.exe:    $clDir (environment already configured)"
} else {
    $vcvarsall = Find-InVsInstall @(
        "C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat"
    )
    if (-not $vcvarsall) {
        Write-Error-And-Exit @"
ERROR: Visual Studio C++ compiler not found.
  Install Visual Studio Build Tools from:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
  Include the 'Desktop development with C++' workload.
"@
    }
    Write-Host "  Initializing MSVC x64 environment via vcvarsall.bat..."
    foreach ($line in (cmd /c "`"$vcvarsall`" x64 >nul 2>&1 && set" 2>&1)) {
        if ($line -match "^([^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    $existingCl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $existingCl) { Write-Error-And-Exit "vcvarsall.bat ran but cl.exe is still not on PATH." }
    $clDir = Split-Path $existingCl.Source
    Write-Host "  cl.exe:    $clDir"
}

# cmake and ninja are normally available after vcvarsall.bat; fall back to VS directory search.
foreach ($tool in @(
    @{ Name = "cmake"; Required = $true },
    @{ Name = "ninja"; Required = $false }
)) {
    $cmd = Get-Command $tool.Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        $dir = Find-InVsInstall @(
            "C:\Program Files\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\*\$($tool.Name).exe",
            "C:\Program Files (x86)\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\*\$($tool.Name).exe"
        )
        if ($dir) {
            $dir = Split-Path $dir
            Write-Host "  $($tool.Name):     $dir"
            $env:Path = "$dir;$env:Path"
        } elseif ($tool.Required) {
            Write-Error-And-Exit "$($tool.Name) not found. Install CMake or Visual Studio Build Tools."
        } else {
            Write-Host "  WARNING: $($tool.Name) not found, builds may be slower" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  $($tool.Name):     $(Split-Path $cmd.Source)"
    }
}

# Check if detected VS is newer than CUDA officially supports
$detectedVsId = Get-VsVersionFromPath $clDir
if ($detectedVsId -and -not ($script:CudaCompatibleVsIds -contains $detectedVsId)) {
    Write-Host "  WARNING: VS '$detectedVsId' is newer than CUDA $($env:CUDA_VERSION) officially supports (2017-2022)." -ForegroundColor Yellow
    Write-Host "  Setting --allow-unsupported-compiler for nvcc." -ForegroundColor Yellow
    $env:NVCC_APPEND_FLAGS = "--allow-unsupported-compiler"
    $env:CUDAFLAGS = "--allow-unsupported-compiler"
}

$env:Path = "$env:CUDA_HOME\bin;$env:Path"
Write-Host ""

# ==========================================
# Step 4: Resolve CUDA version config
# ==========================================
Write-Host "[4/9] Resolving CUDA configuration..."
. "$ScriptDir\scripts\cuda_helper.ps1"

# ==========================================
# Step 5: Create virtual environment
# ==========================================
Write-Host "[5/9] Checking virtual environment..."

$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Error-And-Exit @"
ERROR: uv is not installed.
  Install it with: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
"@
}
Write-Host "  - uv: $(uv --version)"

$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) ".venv"
$env:UV_PYTHON = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"

if (Test-Path ".venv") {
    Write-Host "  Virtual environment already exists at .venv"
    Write-Host "  To recreate, remove it first: Remove-Item -Recurse -Force .venv"
} else {
    uv venv .venv --python 3.11 --prompt $VenvName
    if ($LASTEXITCODE -ne 0) { Write-Error-And-Exit "Failed to create virtual environment" }
    Write-Host "  Created .venv with Python 3.11"
}

Write-Host ""
Write-Host "Environment variables:"
Write-Host "  CUDA_HOME:              $env:CUDA_HOME"
Write-Host "  CUDA_VERSION:           $env:CUDA_VERSION"
Write-Host "  CUDA_FULL_VERSION:      $env:CUDA_FULL_VERSION"
Write-Host "  CUDA_MAJOR_TARGET:      $env:CUDA_MAJOR_TARGET"
Write-Host "  CUDA_MINOR_TARGET:      $env:CUDA_MINOR_TARGET"
Write-Host "  UV_PROJECT_ENVIRONMENT: $env:UV_PROJECT_ENVIRONMENT"
Write-Host "  UV_PYTHON:              $env:UV_PYTHON"
Write-Host "  TORCH_INDEX_URL:        $env:TORCH_INDEX_URL"
Write-Host "  TORCH_VERSION:          $env:TORCH_VERSION"
Write-Host "  TORCH_CUDA_ARCH_LIST:   $env:TORCH_CUDA_ARCH_LIST"
Write-Host ""

# Persist env vars into the venv activation script
& "$ScriptDir\scripts\persist_env_vars_in_venv.ps1"

# ==========================================
# Step 6: Set constraints and index
# ==========================================
Write-Host "[6/9] Setting constraints and index..."

$constraintFile = Join-Path $env:UV_PROJECT_ENVIRONMENT "constraints.txt"
if ($env:TORCH_VERSION) {
    Set-Content -Path $constraintFile -Value "torch$env:TORCH_VERSION"
} else {
    Set-Content -Path $constraintFile -Value ""
}
$env:UV_CONSTRAINT = $constraintFile
Write-Host "  UV constraint file: $env:UV_CONSTRAINT"

$env:UV_INDEX = "pytorch=$env:TORCH_INDEX_URL"
Write-Host "  PyTorch index: $env:TORCH_INDEX_URL"
Write-Host ""

# ==========================================
# Step 7: Install project and dependencies
# ==========================================
Write-Host "[7/9] Installing project and dependencies..."
uv pip install -e ".[dev,gui]"
if ($LASTEXITCODE -ne 0) { Write-Error-And-Exit "Failed to install project dependencies" }
Write-Host ""

# ==========================================
# Step 8: Install Kaolin
# ==========================================
if ([int]$env:CUDA_MAJOR_TARGET -le 12) {
    Write-Host "[8/9] Installing Kaolin from pre-built wheel..."
    $torchInfo = uv pip show torch 2>&1 | Out-String
    $versionMatch = [regex]::Match($torchInfo, "Version:\s+(\S+)")
    $fullTorchVersion = $versionMatch.Groups[1].Value
    $actualTorchVersion = $fullTorchVersion -replace '\+cu.*', ''
    $kaolinFindLink = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${actualTorchVersion}_cu$($env:CUDA_MAJOR_TARGET)$($env:CUDA_MINOR_TARGET).html"
    Write-Host "  Kaolin find link: $kaolinFindLink"
    $env:UV_FIND_LINKS = $kaolinFindLink
} else {
    Write-Error-And-Exit "CUDA $($env:CUDA_MAJOR_TARGET).x requires building Kaolin from source, which is not yet supported on Windows."
}

uv pip install -e ".[playground]"
if ($LASTEXITCODE -ne 0) { Write-Error-And-Exit "Failed to install Kaolin / playground dependencies" }
Write-Host ""

# ==========================================
# Step 9a: Install extra requirements
# ==========================================
Write-Host "[9/9] Installing extra requirements..."
uv pip install --no-cache --no-build-isolation -r requirements_extra.txt
if ($LASTEXITCODE -ne 0) { Write-Error-And-Exit "Failed to install extra requirements" }
Write-Host ""

# ==========================================
# Step 9b: Install slangc
# ==========================================
Write-Host "Installing slangc..."
& "$ScriptDir\scripts\install_slangc.ps1"
if ($LASTEXITCODE -ne 0) { Write-Error-And-Exit "Failed to install slangc" }
Write-Host ""

# ==========================================
# Verify
# ==========================================
Write-Host "Verifying the installation..."
& $env:UV_PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
& $env:UV_PYTHON -c "import kaolin; print(f'Kaolin: {kaolin.__version__}')"
& $env:UV_PYTHON -c "import ppisp; print(f'PPISP: {ppisp.__version__}')"
& $env:UV_PYTHON -c "from fused_ssim import fused_ssim; print('Fused-SSIM: ready')"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:"
Write-Host "  .venv\Scripts\activate"
Write-Host ""
