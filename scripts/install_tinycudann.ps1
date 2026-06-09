# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$PythonExe = if ($env:UV_PYTHON -and (Test-Path $env:UV_PYTHON)) { $env:UV_PYTHON } else { "python" }
$BindingDir = Join-Path $RootDir "thirdparty\tiny-cuda-nn\bindings\torch"

if (-not (Test-Path $BindingDir)) {
    Write-Host "ERROR: thirdparty/tiny-cuda-nn is missing. Run git submodule update --init --recursive first." -ForegroundColor Red
    exit 1
}

if (-not $env:TCNN_CUDA_ARCHITECTURES) {
    $ArchScript = @'
import os
import re
import sys


def parse_arches(text):
    arches = []
    for item in re.split(r"[;,\s]+", text or ""):
        item = item.strip().replace("+PTX", "")
        if not item:
            continue
        match = re.fullmatch(r"(\d+)\.(\d+)", item)
        if match:
            arches.append(int(match.group(1)) * 10 + int(match.group(2)))
            continue
        match = re.fullmatch(r"(?:sm_|compute_)?(\d+)", item)
        if match:
            arches.append(int(match.group(1)))
    return arches


try:
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(major * 10 + minor)
        raise SystemExit(0)
except Exception:
    pass

arches = parse_arches(os.environ.get("TORCH_CUDA_ARCH_LIST", ""))

if not arches:
    try:
        import torch

        arches = parse_arches(";".join(torch.cuda.get_arch_list()))
    except Exception:
        pass

if not arches:
    sys.stderr.write(
        "ERROR: set TCNN_CUDA_ARCHITECTURES, or set TORCH_CUDA_ARCH_LIST, "
        "or run setup with a visible CUDA GPU.\n"
    )
    raise SystemExit(1)

print(max(set(arches)))
'@

    $Arch = & $PythonExe -c $ArchScript
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $env:TCNN_CUDA_ARCHITECTURES = $Arch.Trim()
}

Write-Host "  TCNN_CUDA_ARCHITECTURES: $env:TCNN_CUDA_ARCHITECTURES"

Push-Location $RootDir
if ($env:INSTALL_TCNN_WITH_UV -eq "1") {
    uv pip install --no-cache --no-build-isolation -r requirements_tinycudann.txt
} else {
    & $PythonExe -m pip install --no-cache-dir --no-build-isolation -r requirements_tinycudann.txt
}
$InstallExitCode = $LASTEXITCODE
Pop-Location
if ($InstallExitCode -ne 0) { exit $InstallExitCode }

& $PythonExe -c "import tinycudann; print('tiny-cuda-nn: ready')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
