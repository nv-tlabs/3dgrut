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

# Windows port of scripts/persist_env_vars_in_venv.sh
# Persists 3DGRUT environment variables into .venv\Scripts\Activate.ps1
# so they are restored automatically on activation and cleaned up on deactivate.

$ErrorActionPreference = "Stop"

$activateScript = ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: $activateScript not found" -ForegroundColor Red
    exit 1
}

$banner = "# --- 3DGRUT env vars -------"
$content = Get-Content $activateScript -Raw

if ($content.Contains($banner)) {
    Write-Host "  Environment variables already persisted in $activateScript"
    return
}

# Find the VS compiler directory for PATH persistence
$clExe = Get-Command cl.exe -ErrorAction SilentlyContinue
$vsCompilerDir = if ($clExe) { Split-Path $clExe.Source } else { "" }
$cmakeExe = Get-Command cmake.exe -ErrorAction SilentlyContinue
$cmakeDir = if ($cmakeExe) { Split-Path $cmakeExe.Source } else { "" }
$ninjaExe = Get-Command ninja.exe -ErrorAction SilentlyContinue
$ninjaDir = if ($ninjaExe) { Split-Path $ninjaExe.Source } else { "" }

$envBlock = @"

$banner

# Save pre-existing values so deactivate can restore them
`$global:_3DGRUT_OLD_VARS = @{}
function Set-3dGrutVar([string]`$Name, [string]`$Value) {
    `$global:_3DGRUT_OLD_VARS[`$Name] = if (Test-Path "env:`$Name") { (Get-Item "env:`$Name").Value } else { `$null }
    Set-Item "env:`$Name" `$Value
}

Set-3dGrutVar "CUDA_HOME"              "$env:CUDA_HOME"
Set-3dGrutVar "CUDA_VERSION"           "$env:CUDA_VERSION"
Set-3dGrutVar "CUDA_FULL_VERSION"      "$env:CUDA_FULL_VERSION"
Set-3dGrutVar "CUDA_MAJOR_TARGET"      "$env:CUDA_MAJOR_TARGET"
Set-3dGrutVar "CUDA_MINOR_TARGET"      "$env:CUDA_MINOR_TARGET"
Set-3dGrutVar "TORCH_CUDA_ARCH_LIST"   "$env:TORCH_CUDA_ARCH_LIST"
Set-3dGrutVar "TORCH_INDEX_URL"        "$env:TORCH_INDEX_URL"
Set-3dGrutVar "TORCH_VERSION"          "$env:TORCH_VERSION"
Set-3dGrutVar "UV_PYTHON"              "$env:UV_PYTHON"
Set-3dGrutVar "UV_PROJECT_ENVIRONMENT" "$env:UV_PROJECT_ENVIRONMENT"
$(if ($env:NVCC_APPEND_FLAGS) {
"Set-3dGrutVar `"NVCC_APPEND_FLAGS`"    `"$env:NVCC_APPEND_FLAGS`"`nSet-3dGrutVar `"CUDAFLAGS`"              `"$env:CUDAFLAGS`""
})

# Add CUDA and VS build tools to PATH
`$global:_3DGRUT_OLD_PATH = `$env:PATH
`$extraPaths = @("$($env:CUDA_HOME)\bin", "$vsCompilerDir", "$cmakeDir", "$ninjaDir") | Where-Object { `$_ -ne "" }
`$env:PATH = (`$extraPaths -join ";") + ";" + `$env:PATH

Remove-Item function:Set-3dGrutVar

# Wrap deactivate to undo 3DGRUT env vars
`$global:_3DGRUT_ORIG_DEACTIVATE = `$function:deactivate
function global:deactivate([switch] `$NonDestructive) {
    foreach (`$kv in `$global:_3DGRUT_OLD_VARS.GetEnumerator()) {
        if (`$null -ne `$kv.Value) {
            Set-Item "env:`$(`$kv.Key)" `$kv.Value
        } else {
            Remove-Item "env:`$(`$kv.Key)" -ErrorAction SilentlyContinue
        }
    }
    `$env:PATH = `$global:_3DGRUT_OLD_PATH
    & `$global:_3DGRUT_ORIG_DEACTIVATE -NonDestructive:`$NonDestructive
}

# --- end 3DGRUT env vars ---
"@

Add-Content -Path $activateScript -Value $envBlock

Write-Host "  Persisted environment variables in $($activateScript):"
Write-Host "      CUDA_HOME  CUDA_VERSION  CUDA_FULL_VERSION"
Write-Host "      CUDA_MAJOR_TARGET  CUDA_MINOR_TARGET"
Write-Host "      TORCH_CUDA_ARCH_LIST  TORCH_INDEX_URL  TORCH_VERSION"
Write-Host "      UV_PYTHON  UV_PROJECT_ENVIRONMENT"
Write-Host ""
