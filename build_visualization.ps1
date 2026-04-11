<#
.SYNOPSIS
    Clone and build ANARI-SDK + VisRTX under a local visualization/ folder.

.DESCRIPTION
    Downloads, configures, builds, and installs ANARI-SDK and VisRTX (RTX device)
    into a self-contained visualization/ directory tree:

        visualization/
        ├── src/          (cloned repos)
        ├── build/        (out-of-source build trees)
        └── install/      (shared CMAKE_INSTALL_PREFIX)

    Prerequisites: CMake 3.17+, CUDA 12+, a C++17 compiler (Visual Studio),
    and the NVIDIA OptiX SDK headers.

.PARAMETER Root
    Base directory for the entire tree (default: visualization/ next to this script).

.PARAMETER OptiXDir
    Path to the OptiX SDK. When omitted the script searches the standard
    ProgramData location.

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

if ($Root -eq "") { $Root = Join-Path $ScriptDir "visualization" }
$SrcDir     = Join-Path $Root "src"
$BuildDir   = Join-Path $Root "build"
$InstallDir = Join-Path $Root "install"

if ($Jobs -le 0) { $Jobs = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum }

# ── locate OptiX SDK ────────────────────────────────────────────────────────

if ($OptiXDir -eq "") {
    $optixRoot = "C:\ProgramData\NVIDIA Corporation"
    if (Test-Path $optixRoot) {
        $found = Get-ChildItem $optixRoot -Directory |
                 Where-Object { $_.Name -like "OptiX SDK*" } |
                 Sort-Object Name -Descending |
                 Select-Object -First 1
        if ($found) { $OptiXDir = $found.FullName }
    }
}

if ($OptiXDir -eq "" -or -not (Test-Path $OptiXDir)) {
    Write-Host "ERROR: OptiX SDK not found. Specify -OptiXDir explicitly." -ForegroundColor Red
    exit 1
}
$optixInclude = Join-Path $OptiXDir "include"
if (-not (Test-Path $optixInclude)) {
    Write-Host "ERROR: OptiX include directory not found at $optixInclude" -ForegroundColor Red
    exit 1
}

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
Write-Host "  ANARI-SDK + VisRTX Build"               -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Root:        $Root"
Write-Host "  Install:     $InstallDir"
Write-Host "  BuildType:   $BuildType"
Write-Host "  Generator:   $(if ($generatorArgs.Count) { $generatorArgs[1] } else { '(default)' })"
Write-Host "  Jobs:        $Jobs"
Write-Host "  CUDA_HOME:   $env:CUDA_HOME"
Write-Host "  OptiX SDK:   $OptiXDir"
Write-Host ""

# ── optional clean ───────────────────────────────────────────────────────────

if ($Clean) {
    Write-Step "Cleaning previous build/install directories"
    if (Test-Path $BuildDir)   { Remove-Item $BuildDir   -Recurse -Force }
    if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
    Write-Ok "Clean complete"
}

# ── create directory tree ────────────────────────────────────────────────────

foreach ($d in @($SrcDir, $BuildDir, $InstallDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

# ═════════════════════════════════════════════════════════════════════════════
#  1. ANARI-SDK
# ═════════════════════════════════════════════════════════════════════════════

$anariSrc   = Join-Path $SrcDir   "ANARI-SDK"
$anariBuild = Join-Path $BuildDir "anari-sdk"

Write-Step "Cloning ANARI-SDK"
if (Test-Path $anariSrc) {
    Write-Warn "Source already exists - pulling latest"
    git -C $anariSrc pull --ff-only
} else {
    git clone https://github.com/KhronosGroup/ANARI-SDK.git $anariSrc
}
Assert-ExitCode "ANARI-SDK clone/pull"

Write-Step "Configuring ANARI-SDK"
if (-not (Test-Path $anariBuild)) { New-Item -ItemType Directory -Path $anariBuild -Force | Out-Null }

cmake @generatorArgs `
    -S $anariSrc `
    -B $anariBuild `
    "-DCMAKE_BUILD_TYPE=$BuildType" `
    "-DCMAKE_INSTALL_PREFIX=$InstallDir" `
    -DBUILD_SHARED_LIBS=ON `
    -DBUILD_HELIDE_DEVICE=ON `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_TESTING=OFF `
    -DBUILD_CTS=OFF `
    -DBUILD_VIEWER=OFF
Assert-ExitCode "ANARI-SDK configure"

Write-Step "Building ANARI-SDK - $BuildType, $Jobs jobs"
cmake --build $anariBuild --config $BuildType --parallel $Jobs
Assert-ExitCode "ANARI-SDK build"

Write-Step "Installing ANARI-SDK"
cmake --install $anariBuild --config $BuildType
Assert-ExitCode "ANARI-SDK install"
Write-Ok "ANARI-SDK installed to $InstallDir"

# ═════════════════════════════════════════════════════════════════════════════
#  2. VisRTX
# ═════════════════════════════════════════════════════════════════════════════

$visrtxSrc   = Join-Path $SrcDir   "VisRTX"
$visrtxBuild = Join-Path $BuildDir "visrtx"

Write-Step "Cloning VisRTX - next_release branch"
if (Test-Path $visrtxSrc) {
    Write-Warn "Source already exists - pulling latest"
    git -C $visrtxSrc pull --ff-only
} else {
    git clone --branch next_release https://github.com/NVIDIA/VisRTX.git $visrtxSrc
}
Assert-ExitCode "VisRTX clone/pull"

$prefixPath = "$InstallDir;$OptiXDir"

Write-Step "Configuring VisRTX"
if (-not (Test-Path $visrtxBuild)) { New-Item -ItemType Directory -Path $visrtxBuild -Force | Out-Null }

cmake @generatorArgs `
    -S $visrtxSrc `
    -B $visrtxBuild `
    "-DCMAKE_BUILD_TYPE=$BuildType" `
    "-DCMAKE_INSTALL_PREFIX=$InstallDir" `
    "-DCMAKE_PREFIX_PATH=$prefixPath" `
    "-DOptiX_ROOT=$OptiXDir" `
    -DVISRTX_BUILD_RTX_DEVICE=ON `
    -DVISRTX_BUILD_GL_DEVICE=OFF `
    -DVISRTX_BUILD_TESTS=ON
Assert-ExitCode "VisRTX configure"

Write-Step "Building VisRTX - $BuildType, $Jobs jobs"
cmake --build $visrtxBuild --config $BuildType --parallel $Jobs
Assert-ExitCode "VisRTX build"

Write-Step "Installing VisRTX"
cmake --install $visrtxBuild --config $BuildType
Assert-ExitCode "VisRTX install"
Write-Ok "VisRTX installed to $InstallDir"

# ═════════════════════════════════════════════════════════════════════════════
#  3. GaussianViewer (standalone sample app)
# ═════════════════════════════════════════════════════════════════════════════

$viewerSrc   = Join-Path $SrcDir   "gaussian_viewer"
$viewerBuild = Join-Path $BuildDir "gaussian_viewer"

Write-Step "Configuring GaussianViewer"
if (-not (Test-Path $viewerBuild)) { New-Item -ItemType Directory -Path $viewerBuild -Force | Out-Null }

cmake @generatorArgs `
    -S $viewerSrc `
    -B $viewerBuild `
    "-DCMAKE_BUILD_TYPE=$BuildType" `
    "-DCMAKE_PREFIX_PATH=$InstallDir"
Assert-ExitCode "GaussianViewer configure"

Write-Step "Building GaussianViewer - $BuildType, $Jobs jobs"
cmake --build $viewerBuild --config $BuildType --parallel $Jobs
Assert-ExitCode "GaussianViewer build"
Write-Ok "GaussianViewer built"

# ═════════════════════════════════════════════════════════════════════════════
#  Done
# ═════════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  BUILD COMPLETE"                          -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Install prefix:  $InstallDir"
Write-Host ""
Write-Host "  GaussianViewer:  $(Join-Path (Join-Path $viewerBuild $BuildType) 'GaussianViewer.exe')"
Write-Host ""
Write-Host "  To use in your own CMake project:" -ForegroundColor Cyan
Write-Host "    cmake -DCMAKE_PREFIX_PATH=`"$InstallDir`" .."
Write-Host ""
Write-Host "  To use at runtime, add the bin/lib dirs to PATH:" -ForegroundColor Cyan
Write-Host "    `$env:PATH = `"$InstallDir\bin;$InstallDir\lib;`$env:PATH`""
Write-Host ""
