<#
.SYNOPSIS
    Build the visualization stack (ANARI-SDK, VisRTX, GaussianViewer) via the
    CMake superbuild in this directory.

.DESCRIPTION
    Configures and builds the superbuild CMakeLists.txt that automatically
    downloads ANARI-SDK, VisRTX, and OptiX headers, then builds gaussian_viewer
    against them.

        visualization/
        ├── CMakeLists.txt  (superbuild)
        ├── src/            (C++ source)
        ├── gaussian_viewer/ (Python package)
        ├── build/          (superbuild build tree)
        └── install/        (shared CMAKE_INSTALL_PREFIX)

    Prerequisites: CMake 3.17+, CUDA 12+, a C++17 compiler (Visual Studio).
    OptiX SDK is downloaded automatically unless -OptiXDir is given.

.PARAMETER Root
    Base directory for the entire tree (default: this script's directory).

.PARAMETER OptiXDir
    Path to a local OptiX SDK.  When omitted the superbuild downloads the
    headers from https://github.com/NVIDIA/optix-dev automatically.

.PARAMETER BuildType
    CMake build type (default: Release).

.PARAMETER Generator
    CMake generator override. Leave empty to auto-select (Ninja if available,
    otherwise the default Visual Studio generator).

.PARAMETER Jobs
    Parallel build jobs (default: number of logical processors).

.PARAMETER Clean
    Remove existing build and install directories before building.

.EXAMPLE
    .\build_visualization.ps1
    .\build_visualization.ps1 -BuildType RelWithDebInfo
    .\build_visualization.ps1 -OptiXDir "D:\OptiX SDK 8.0.0" -Generator Ninja
#>

param (
    [string]$Root       = "",
    [string]$OptiXDir   = "",
    [string]$BuildType  = "Release",
    [string]$Generator  = "",
    [int]$Jobs          = 0,
    [switch]$Clean
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── helpers ──────────────────────────────────────────────────────────────────

function Write-Step  { param($msg) Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "   $msg"   -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "   $msg"   -ForegroundColor Yellow }

function Assert-ExitCode {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $StepName failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ── resolve paths ────────────────────────────────────────────────────────────

if ($Root -eq "") { $Root = $ScriptDir }
$SrcDir     = $Root
$BuildDir   = Join-Path $Root "build"
$InstallDir = Join-Path $Root "install"

if ($Jobs -le 0) { $Jobs = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum }

# ── locate CUDA ──────────────────────────────────────────────────────────────

if (-not $env:CUDA_HOME -or -not (Test-Path $env:CUDA_HOME)) {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $cudaDir = Get-ChildItem $cudaRoot -Directory |
                   Sort-Object Name -Descending |
                   Select-Object -First 1
        if ($cudaDir) { $env:CUDA_HOME = $cudaDir.FullName }
    }
}

if (-not $env:CUDA_HOME -or -not (Test-Path $env:CUDA_HOME)) {
    Write-Host "ERROR: CUDA toolkit not found. Set CUDA_HOME or install CUDA 12+." -ForegroundColor Red
    exit 1
}

# ── detect / initialize MSVC environment ─────────────────────────────────────

$CudaCompatibleVsIds = @("2017", "2019", "2022", "15", "16", "17")

function Get-VsVersionFromPath([string]$Path) {
    $parts = $Path.Replace("/", "\") -split "\\"
    for ($i = 0; $i -lt $parts.Length; $i++) {
        if ($parts[$i] -eq "Microsoft Visual Studio" -and ($i + 1) -lt $parts.Length) {
            return $parts[$i + 1]
        }
    }
    return ""
}

function Find-InVsInstall([string[]]$Patterns) {
    $compatible = $null
    $fallback = $null
    foreach ($pattern in $Patterns) {
        foreach ($p in (Resolve-Path $pattern -ErrorAction SilentlyContinue)) {
            if (-not $fallback) { $fallback = $p.Path }
            if (-not $compatible -and
                ($CudaCompatibleVsIds -contains (Get-VsVersionFromPath $p.Path))) {
                $compatible = $p.Path
            }
        }
    }
    if ($compatible) { return $compatible }
    return $fallback
}

$existingCl = Get-Command cl.exe -ErrorAction SilentlyContinue
$existingRc = Get-Command rc.exe -ErrorAction SilentlyContinue
$libIsX64   = $env:LIB -and ($env:LIB -match "\\x64|\\amd64")

$needVcvars = -not $existingCl -or -not $existingRc -or -not $libIsX64
if ($needVcvars) {
    $vcvarsall = Find-InVsInstall @(
        "C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat"
    )
    if (-not $vcvarsall) {
        Write-Host "ERROR: Visual Studio C++ compiler not found." -ForegroundColor Red
        Write-Host "  Install Visual Studio Build Tools with the 'Desktop development with C++' workload." -ForegroundColor Red
        exit 1
    }
    Write-Host "Initializing MSVC x64 environment via vcvarsall.bat..."
    foreach ($line in (cmd /c "`"$vcvarsall`" x64 >nul 2>&1 && set" 2>&1)) {
        if ($line -match "^([^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    $existingCl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $existingCl) {
        Write-Host "ERROR: vcvarsall.bat ran but cl.exe is still not on PATH." -ForegroundColor Red
        exit 1
    }

    if (Test-Path $BuildDir) {
        Get-ChildItem $BuildDir -Recurse -Filter "CMakeCache.txt" | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

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
            $env:Path = "$dir;$env:Path"
        } elseif ($tool.Required) {
            Write-Host "ERROR: $($tool.Name) not found. Install CMake or Visual Studio Build Tools." -ForegroundColor Red
            exit 1
        }
    }
}

# ── pick CMake generator ────────────────────────────────────────────────────

$generatorArgs = @()
if ($Generator -ne "") {
    $generatorArgs = @("-G", $Generator)
} else {
    $ninja = Get-Command ninja -ErrorAction SilentlyContinue
    if ($ninja) {
        $generatorArgs = @("-G", "Ninja")
    }
}

# ── summary ──────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Visualization Superbuild"               -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Root:        $Root"
Write-Host "  Source:      $SrcDir"
Write-Host "  Build:       $BuildDir"
Write-Host "  Install:     $InstallDir"
Write-Host "  BuildType:   $BuildType"
Write-Host "  Generator:   $(if ($generatorArgs.Count) { $generatorArgs[1] } else { '(default)' })"
Write-Host "  Jobs:        $Jobs"
Write-Host "  CUDA_HOME:   $env:CUDA_HOME"
Write-Host "  OptiX SDK:   $(if ($OptiXDir) { $OptiXDir } else { '(auto-detect / download)' })"
Write-Host ""

# ── optional clean ───────────────────────────────────────────────────────────

if ($Clean) {
    Write-Step "Cleaning previous build/install directories"
    if (Test-Path $BuildDir)   { Remove-Item $BuildDir   -Recurse -Force }
    if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
    Write-Ok "Clean complete"
}

# ═════════════════════════════════════════════════════════════════════════════
#  Configure superbuild
# ═════════════════════════════════════════════════════════════════════════════

Write-Step "Configuring superbuild"

$cmakeArgs = @()
$cmakeArgs += $generatorArgs
$cmakeArgs += @("-S", $SrcDir)
$cmakeArgs += @("-B", $BuildDir)
$cmakeArgs += "-DCMAKE_BUILD_TYPE=$BuildType"
$cmakeArgs += "-DCMAKE_INSTALL_PREFIX=$InstallDir"
$cmakeArgs += "-DBUILD_PYTHON_BINDINGS=ON"

if ($OptiXDir -ne "") {
    $cmakeArgs += "-DOPTIX_ROOT=$OptiXDir"
}

cmake @cmakeArgs
Assert-ExitCode "Superbuild configure"

# ═════════════════════════════════════════════════════════════════════════════
#  Build everything
# ═════════════════════════════════════════════════════════════════════════════

Write-Step "Building all targets - $BuildType, $Jobs jobs"
cmake --build $BuildDir --config $BuildType --parallel $Jobs
Assert-ExitCode "Superbuild build"

# ═════════════════════════════════════════════════════════════════════════════
#  Done
# ═════════════════════════════════════════════════════════════════════════════

$pythonDir = $Root
$standaloneBin = Join-Path $InstallDir "bin"
$standaloneLib = Join-Path $InstallDir "lib"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  BUILD COMPLETE"                          -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Install prefix:    $InstallDir"
Write-Host "  commandline_viewer: $(Join-Path $standaloneBin 'commandline_viewer.exe')"
Write-Host "  interactive_viewer: $(Join-Path $standaloneBin 'interactive_viewer.exe')"
Write-Host ""
Write-Host "  To use in your own CMake project:" -ForegroundColor Cyan
Write-Host "    cmake -DCMAKE_PREFIX_PATH=`"$InstallDir`" .."
Write-Host ""
Write-Host "  To use at runtime, add the bin/lib dirs to PATH:" -ForegroundColor Cyan
Write-Host "    `$env:PATH = `"$InstallDir\bin;$InstallDir\lib;`$env:PATH`""
Write-Host ""
Write-Host "  To use the Python bindings (both variables are required):" -ForegroundColor Cyan
Write-Host "    `$env:PATH = `"$standaloneBin;$standaloneLib;`$env:PATH`""
Write-Host "    `$env:PYTHONPATH = `"$pythonDir;$standaloneLib;`$env:PYTHONPATH`""
Write-Host ""
