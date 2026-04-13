#!/usr/bin/env bash
#
# Build the visualization stack (ANARI-SDK, VisRTX, GaussianViewer) via the
# CMake superbuild in this directory.
#
# The superbuild automatically downloads ANARI-SDK, VisRTX, and (optionally)
# OptiX headers, then builds gaussian_viewer against them.
#
#   visualization/
#   ├── CMakeLists.txt  (superbuild)
#   ├── src/            (C++ source)
#   ├── gaussian_viewer/ (Python package)
#   ├── build/          (superbuild build tree)
#   └── install/        (shared CMAKE_INSTALL_PREFIX)
#
# Prerequisites: CMake 3.17+, CUDA 12+, a C++17 compiler.
# OptiX SDK is downloaded automatically unless --optix-dir is given.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_visualization.sh [options]

Options:
  --root PATH          Base directory for visualization tree (default: this script's directory)
  --optix-dir PATH     Use local OptiX headers directory (must contain include/).
                       When omitted, headers are downloaded from GitHub automatically.
  --build-type TYPE    CMake build type (default: Release)
  --generator NAME     CMake generator (default: Ninja if available)
  --jobs N             Parallel build jobs (default: logical CPU count)
  --clean              Remove build/ and install/ before building
  -h, --help           Show this help message

Examples:
  ./build_visualization.sh
  ./build_visualization.sh --build-type RelWithDebInfo
  ./build_visualization.sh --optix-dir /usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
EOF
}

step() { printf '\n\033[1;36m>> %s\033[0m\n' "$1"; }
ok() { printf '   \033[1;32m%s\033[0m\n' "$1"; }
warn() { printf '   \033[1;33m%s\033[0m\n' "$1"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

run_checked() {
  local label="$1"
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    fail "${label} failed (exit code ${rc})"
  fi
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    fail "${flag} requires a value"
  fi
}

detect_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" ]] && [[ -d "${CUDA_HOME}" ]]; then
    return 0
  fi

  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    return 0
  fi

  if [[ -d "/usr/local/cuda" ]]; then
    CUDA_HOME="/usr/local/cuda"
    export CUDA_HOME
    return 0
  fi

  local discovered
  discovered="$(
    ls -d /usr/local/cuda-* 2>/dev/null \
      | sort -V \
      | tail -n 1 \
      || true
  )"
  if [[ -n "${discovered}" ]]; then
    CUDA_HOME="${discovered}"
    export CUDA_HOME
    return 0
  fi

  return 1
}

ROOT=""
OPTIX_DIR="${OPTIX_DIR:-}"
BUILD_TYPE="Release"
GENERATOR=""
JOBS=0
CLEAN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      require_value "$1" "${2:-}"
      ROOT="$2"
      shift 2
      ;;
    --root=*)
      ROOT="${1#*=}"
      shift
      ;;
    --optix-dir)
      require_value "$1" "${2:-}"
      OPTIX_DIR="$2"
      shift 2
      ;;
    --optix-dir=*)
      OPTIX_DIR="${1#*=}"
      shift
      ;;
    --build-type)
      require_value "$1" "${2:-}"
      BUILD_TYPE="$2"
      shift 2
      ;;
    --build-type=*)
      BUILD_TYPE="${1#*=}"
      shift
      ;;
    --generator)
      require_value "$1" "${2:-}"
      GENERATOR="$2"
      shift 2
      ;;
    --generator=*)
      GENERATOR="${1#*=}"
      shift
      ;;
    --jobs|-j)
      require_value "$1" "${2:-}"
      JOBS="$2"
      shift 2
      ;;
    --jobs=*|-j=*)
      JOBS="${1#*=}"
      shift
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${ROOT}" ]]; then
  ROOT="${SCRIPT_DIR}"
fi

if [[ "${JOBS}" -le 0 ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS="$(getconf _NPROCESSORS_ONLN)"
  fi
fi

if ! [[ "${JOBS}" =~ ^[0-9]+$ ]] || [[ "${JOBS}" -le 0 ]]; then
  fail "--jobs must be a positive integer"
fi

if ! detect_cuda_home; then
  fail "CUDA toolkit not found. Set CUDA_HOME or install CUDA 12+."
fi

SRC_DIR="${ROOT}"
BUILD_DIR="${ROOT}/build"
INSTALL_DIR="${ROOT}/install"

generator_args=()
if [[ -n "${GENERATOR}" ]]; then
  generator_args=(-G "${GENERATOR}")
elif command -v ninja >/dev/null 2>&1; then
  generator_args=(-G Ninja)
fi

# ── summary ──────────────────────────────────────────────────────────────────

printf '\n\033[1;36m=========================================\033[0m\n'
printf '\033[1;36m  Visualization Superbuild\033[0m\n'
printf '\033[1;36m=========================================\033[0m\n'
printf '  Root:        %s\n' "${ROOT}"
printf '  Source:      %s\n' "${SRC_DIR}"
printf '  Build:       %s\n' "${BUILD_DIR}"
printf '  Install:     %s\n' "${INSTALL_DIR}"
printf '  BuildType:   %s\n' "${BUILD_TYPE}"
if [[ ${#generator_args[@]} -gt 0 ]]; then
  printf '  Generator:   %s\n' "${generator_args[1]}"
else
  printf '  Generator:   (default)\n'
fi
printf '  Jobs:        %s\n' "${JOBS}"
printf '  CUDA_HOME:   %s\n' "${CUDA_HOME}"
if [[ -n "${OPTIX_DIR}" ]]; then
  printf '  OptiX SDK:   %s\n' "${OPTIX_DIR}"
else
  printf '  OptiX SDK:   (auto-detect / download)\n'
fi
printf '\n'

# ── optional clean ───────────────────────────────────────────────────────────

if [[ "${CLEAN}" == true ]]; then
  step "Cleaning previous build/install directories"
  rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
  ok "Clean complete"
fi

# ============================================================================
#  Configure superbuild
# ============================================================================

step "Configuring superbuild"

cmake_args=("${generator_args[@]}")
cmake_args+=(-S "${SRC_DIR}")
cmake_args+=(-B "${BUILD_DIR}")
cmake_args+=("-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
cmake_args+=("-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}")
cmake_args+=("-DBUILD_PYTHON_BINDINGS=ON")

if [[ -n "${OPTIX_DIR}" ]]; then
  cmake_args+=("-DOPTIX_ROOT=${OPTIX_DIR}")
fi

run_checked "Superbuild configure" cmake "${cmake_args[@]}"

# ============================================================================
#  Build everything
# ============================================================================

step "Building all targets - ${BUILD_TYPE}, ${JOBS} jobs"
run_checked "Superbuild build" cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${JOBS}"

# ============================================================================
#  Done
# ============================================================================

python_dir="${ROOT}"

# ExternalProject places gaussian_viewer output under this path.
viewer_build="${BUILD_DIR}/gaussian_viewer-prefix/src/gaussian_viewer-build"

viewer_bin="${viewer_build}/GaussianViewer"
interactive_bin="${viewer_build}/InteractiveViewer"
if [[ ! -x "${viewer_bin}" && -x "${viewer_build}/${BUILD_TYPE}/GaussianViewer" ]]; then
  viewer_bin="${viewer_build}/${BUILD_TYPE}/GaussianViewer"
fi
if [[ ! -x "${interactive_bin}" && -x "${viewer_build}/${BUILD_TYPE}/InteractiveViewer" ]]; then
  interactive_bin="${viewer_build}/${BUILD_TYPE}/InteractiveViewer"
fi

printf '\n\033[1;32m=========================================\033[0m\n'
printf '\033[1;32m  BUILD COMPLETE\033[0m\n'
printf '\033[1;32m=========================================\033[0m\n'
printf '\n'
printf '  Install prefix:    %s\n' "${INSTALL_DIR}"
printf '  GaussianViewer:    %s\n' "${viewer_bin}"
printf '  InteractiveViewer: %s\n' "${interactive_bin}"
printf '\n'
printf '  To use in your own CMake project:\n'
printf '    cmake -DCMAKE_PREFIX_PATH="%s" ..\n' "${INSTALL_DIR}"
printf '\n'
printf '  To use at runtime, add the bin/lib dirs to LD_LIBRARY_PATH:\n'
printf '    export LD_LIBRARY_PATH="%s/lib:%s/bin:${LD_LIBRARY_PATH:-}"\n' "${INSTALL_DIR}" "${INSTALL_DIR}"
printf '\n'
printf '  To use the Python bindings:\n'
printf '    export PYTHONPATH="%s:%s:${PYTHONPATH:-}"\n' "${viewer_build}" "${python_dir}"
printf '\n'
