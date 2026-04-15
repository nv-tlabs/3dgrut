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

# Windows port of scripts/install_slangc.sh
# Installs slangc from the official source into UV_PROJECT_ENVIRONMENT\Scripts

$ErrorActionPreference = "Stop"

Write-Host "Installing slangc from the official source"
Write-Host "  environment: $env:UV_PROJECT_ENVIRONMENT"

$SLANGC_VERSION = "2026.5.2"

$slangcExe = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\slangc.exe"

if (Test-Path $slangcExe) {
    Write-Host "Slangc found at $slangcExe"
    Write-Host "  Checking version..."
    $actualVersion = & $slangcExe -version 2>&1
    if ($actualVersion -ne $SLANGC_VERSION) {
        Write-Host "  ERROR: Slangc version is incorrect, expected $SLANGC_VERSION, got $actualVersion" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Slangc is already installed and matches the expected version"
    exit 0
}

$downloadUrl = "https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-windows-x86_64.zip"
$zipPath = Join-Path $env:TEMP "slang-${SLANGC_VERSION}-windows-x86_64.zip"
$extractPath = Join-Path $env:TEMP "slang-extract"

if (-not (Test-Path $zipPath)) {
    Write-Host "  Downloading slangc $SLANGC_VERSION..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing
}

Write-Host "  Extracting..."
if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

$scriptsDir = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts"
if (-not (Test-Path $scriptsDir)) {
    Write-Host "  ERROR: $scriptsDir does not exist, the virtual environment may not have been created correctly." -ForegroundColor Red
    exit 1
}

Copy-Item "$extractPath\bin\*" -Destination $scriptsDir -Force -Recurse
Write-Host "Slangc installed at $scriptsDir\slangc.exe"
