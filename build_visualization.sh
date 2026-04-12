#!/usr/bin/env bash
#
# Clone and build ANARI-SDK + VisRTX under a local visualization/ folder.
#
# This script downloads, configures, builds, and installs ANARI-SDK and VisRTX
# (RTX device) into a self-contained visualization/ directory tree:
#
#   visualization/
#   ├── src/          (cloned repos)
#   ├── build/        (out-of-source build trees)
#   └── install/      (shared CMAKE_INSTALL_PREFIX)
#
# Prerequisites: CMake 3.17+, CUDA 12+, a C++17 compiler, and OptiX SDK headers.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_visualization.sh [options]

Options:
  --root PATH          Base directory for visualization tree (default: ./visualization)
  --optix-dir PATH     Use local OptiX headers directory (must contain include/)
  --visrtx-branch NAME VisRTX branch to build (default: next_release)
  --optix-branch NAME  OptiX branch (default: same as --visrtx-branch)
  --build-type TYPE    CMake build type (default: Release)
  --generator NAME     CMake generator (default: Ninja if available)
  --jobs N             Parallel build jobs (default: logical CPU count)
  --clean              Remove build/ and install/ before building
  -h, --help           Show this help message

Examples:
  ./build_visualization.sh
  ./build_visualization.sh --build-type RelWithDebInfo
  ./build_visualization.sh --visrtx-branch next_release
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

validate_optix_dir() {
  if [[ ! -d "${OPTIX_DIR}" ]]; then
    fail "OptiX path does not exist: ${OPTIX_DIR}"
  fi
  if [[ ! -d "${OPTIX_DIR}/include" ]]; then
    fail "OptiX include directory not found at ${OPTIX_DIR}/include"
  fi
}

remote_branch_exists() {
  local repo_url="$1"
  local branch="$2"
  git ls-remote --exit-code --heads "${repo_url}" "${branch}" >/dev/null 2>&1
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
OPTIX_IS_LOCAL=false
VISRTX_BRANCH="next_release"
OPTIX_BRANCH=""
OPTIX_BRANCH_EXPLICIT=false
OPTIX_REPO_URL="https://github.com/NVIDIA/optix-dev.git"
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
      OPTIX_IS_LOCAL=true
      shift 2
      ;;
    --optix-dir=*)
      OPTIX_DIR="${1#*=}"
      OPTIX_IS_LOCAL=true
      shift
      ;;
    --visrtx-branch)
      require_value "$1" "${2:-}"
      VISRTX_BRANCH="$2"
      shift 2
      ;;
    --visrtx-branch=*)
      VISRTX_BRANCH="${1#*=}"
      shift
      ;;
    --optix-branch)
      require_value "$1" "${2:-}"
      OPTIX_BRANCH="$2"
      OPTIX_BRANCH_EXPLICIT=true
      shift 2
      ;;
    --optix-branch=*)
      OPTIX_BRANCH="${1#*=}"
      OPTIX_BRANCH_EXPLICIT=true
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
  ROOT="${SCRIPT_DIR}/visualization"
fi

if [[ -n "${OPTIX_DIR}" ]]; then
  OPTIX_IS_LOCAL=true
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

if [[ -z "${OPTIX_BRANCH}" ]]; then
  OPTIX_BRANCH="${VISRTX_BRANCH}"
fi

if [[ "${OPTIX_IS_LOCAL}" == false ]]; then
  if ! remote_branch_exists "${OPTIX_REPO_URL}" "${OPTIX_BRANCH}"; then
    if [[ "${OPTIX_BRANCH_EXPLICIT}" == true ]]; then
      fail "OptiX branch '${OPTIX_BRANCH}' not found in ${OPTIX_REPO_URL}"
    fi
    warn "OptiX branch '${OPTIX_BRANCH}' not found in optix-dev, falling back to 'main'"
    OPTIX_BRANCH="main"
  fi
fi

if ! detect_cuda_home; then
  fail "CUDA toolkit not found. Set CUDA_HOME or install CUDA 12+."
fi

SRC_DIR="${ROOT}/src"
BUILD_DIR="${ROOT}/build"
INSTALL_DIR="${ROOT}/install"

generator_args=()
if [[ -n "${GENERATOR}" ]]; then
  generator_args=(-G "${GENERATOR}")
elif command -v ninja >/dev/null 2>&1; then
  generator_args=(-G Ninja)
fi

printf '\n\033[1;36m=========================================\033[0m\n'
printf '\033[1;36m  ANARI-SDK + VisRTX Build (Linux)\033[0m\n'
printf '\033[1;36m=========================================\033[0m\n'
printf '  Root:        %s\n' "${ROOT}"
printf '  Install:     %s\n' "${INSTALL_DIR}"
printf '  BuildType:   %s\n' "${BUILD_TYPE}"
if [[ ${#generator_args[@]} -gt 0 ]]; then
  printf '  Generator:   %s\n' "${generator_args[1]}"
else
  printf '  Generator:   (default)\n'
fi
printf '  Jobs:        %s\n' "${JOBS}"
printf '  CUDA_HOME:   %s\n' "${CUDA_HOME}"
printf '  VisRTX branch: %s\n' "${VISRTX_BRANCH}"
if [[ "${OPTIX_IS_LOCAL}" == true ]]; then
  printf '  OptiX source: local (%s)\n' "${OPTIX_DIR}"
else
  printf '  OptiX repo:  %s\n' "${OPTIX_REPO_URL}"
  printf '  OptiX branch: %s\n' "${OPTIX_BRANCH}"
fi
printf '\n'

if [[ "${CLEAN}" == true ]]; then
  step "Cleaning previous build/install directories"
  rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
  ok "Clean complete"
fi

mkdir -p "${SRC_DIR}" "${BUILD_DIR}" "${INSTALL_DIR}"

# ============================================================================
# 1) ANARI-SDK
# ============================================================================

anari_src="${SRC_DIR}/ANARI-SDK"
anari_build="${BUILD_DIR}/anari-sdk"

step "Cloning ANARI-SDK"
if [[ -d "${anari_src}/.git" ]]; then
  warn "Source already exists - pulling latest"
  run_checked "ANARI-SDK pull" git -C "${anari_src}" pull --ff-only
else
  run_checked "ANARI-SDK clone" git clone https://github.com/KhronosGroup/ANARI-SDK.git "${anari_src}"
fi

step "Configuring ANARI-SDK"
mkdir -p "${anari_build}"
run_checked "ANARI-SDK configure" cmake "${generator_args[@]}" \
  -S "${anari_src}" \
  -B "${anari_build}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_HELIDE_DEVICE=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_CTS=OFF \
  -DBUILD_VIEWER=OFF

step "Building ANARI-SDK - ${BUILD_TYPE}, ${JOBS} jobs"
run_checked "ANARI-SDK build" cmake --build "${anari_build}" --config "${BUILD_TYPE}" --parallel "${JOBS}"

step "Installing ANARI-SDK"
run_checked "ANARI-SDK install" cmake --install "${anari_build}" --config "${BUILD_TYPE}"
ok "ANARI-SDK installed to ${INSTALL_DIR}"

# ============================================================================
# 2) OptiX headers
# ============================================================================

if [[ "${OPTIX_IS_LOCAL}" == true ]]; then
  validate_optix_dir
  ok "Using local OptiX headers at ${OPTIX_DIR}"
else
  optix_src="${SRC_DIR}/optix-dev"
  step "Cloning OptiX headers - ${OPTIX_BRANCH} branch"
  if [[ -d "${optix_src}/.git" ]]; then
    warn "Source already exists - syncing requested branch"
    run_checked "OptiX fetch" git -C "${optix_src}" fetch --prune origin
    run_checked "OptiX checkout ${OPTIX_BRANCH}" git -C "${optix_src}" checkout -B "${OPTIX_BRANCH}" "origin/${OPTIX_BRANCH}"
    run_checked "OptiX pull ${OPTIX_BRANCH}" git -C "${optix_src}" pull --ff-only origin "${OPTIX_BRANCH}"
  else
    run_checked "OptiX clone" git clone --branch "${OPTIX_BRANCH}" --single-branch "${OPTIX_REPO_URL}" "${optix_src}"
  fi
  OPTIX_DIR="${optix_src}"
  validate_optix_dir
  ok "OptiX headers ready at ${OPTIX_DIR}"
fi

# ============================================================================
# 3) VisRTX
# ============================================================================

visrtx_src="${SRC_DIR}/VisRTX"
visrtx_build="${BUILD_DIR}/visrtx"
prefix_path="${INSTALL_DIR};${OPTIX_DIR}"

step "Cloning VisRTX - ${VISRTX_BRANCH} branch"
if [[ -d "${visrtx_src}/.git" ]]; then
  warn "Source already exists - syncing requested branch"
  run_checked "VisRTX fetch" git -C "${visrtx_src}" fetch --prune origin
  run_checked "VisRTX checkout ${VISRTX_BRANCH}" git -C "${visrtx_src}" checkout -B "${VISRTX_BRANCH}" "origin/${VISRTX_BRANCH}"
  run_checked "VisRTX pull ${VISRTX_BRANCH}" git -C "${visrtx_src}" pull --ff-only origin "${VISRTX_BRANCH}"
else
  run_checked "VisRTX clone" git clone --branch "${VISRTX_BRANCH}" --single-branch https://github.com/NVIDIA/VisRTX.git "${visrtx_src}"
fi

step "Configuring VisRTX"
mkdir -p "${visrtx_build}"
run_checked "VisRTX configure" cmake "${generator_args[@]}" \
  -S "${visrtx_src}" \
  -B "${visrtx_build}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}" \
  "-DCMAKE_PREFIX_PATH=${prefix_path}" \
  "-DOptiX_ROOT=${OPTIX_DIR}" \
  -DVISRTX_BUILD_RTX_DEVICE=ON \
  -DVISRTX_BUILD_GL_DEVICE=OFF \
  -DVISRTX_BUILD_TESTS=ON

step "Building VisRTX - ${BUILD_TYPE}, ${JOBS} jobs"
run_checked "VisRTX build" cmake --build "${visrtx_build}" --config "${BUILD_TYPE}" --parallel "${JOBS}"

step "Installing VisRTX"
run_checked "VisRTX install" cmake --install "${visrtx_build}" --config "${BUILD_TYPE}"
ok "VisRTX installed to ${INSTALL_DIR}"

# ============================================================================
# 4) GaussianViewer (standalone sample app)
# ============================================================================

viewer_src="${SRC_DIR}/gaussian_viewer"
viewer_build="${BUILD_DIR}/gaussian_viewer"

if [[ ! -f "${viewer_src}/CMakeLists.txt" ]]; then
  fail "Expected viewer source at ${viewer_src}. Make sure visualization/src/gaussian_viewer is present."
fi

step "Configuring GaussianViewer"
mkdir -p "${viewer_build}"
run_checked "GaussianViewer configure" cmake "${generator_args[@]}" \
  -S "${viewer_src}" \
  -B "${viewer_build}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  "-DCMAKE_PREFIX_PATH=${INSTALL_DIR}" \
  -DBUILD_PYTHON_BINDINGS=ON

step "Building GaussianViewer - ${BUILD_TYPE}, ${JOBS} jobs"
run_checked "GaussianViewer build" cmake --build "${viewer_build}" --config "${BUILD_TYPE}" --parallel "${JOBS}"
ok "GaussianViewer built"

viewer_bin="${viewer_build}/GaussianViewer"
interactive_bin="${viewer_build}/InteractiveViewer"
if [[ ! -x "${viewer_bin}" && -x "${viewer_build}/${BUILD_TYPE}/GaussianViewer" ]]; then
  viewer_bin="${viewer_build}/${BUILD_TYPE}/GaussianViewer"
fi
if [[ ! -x "${interactive_bin}" && -x "${viewer_build}/${BUILD_TYPE}/InteractiveViewer" ]]; then
  interactive_bin="${viewer_build}/${BUILD_TYPE}/InteractiveViewer"
fi

python_dir="${ROOT}/python/gaussian_renderer"

printf '\n\033[1;32m=========================================\033[0m\n'
printf '\033[1;32m  BUILD COMPLETE\033[0m\n'
printf '\033[1;32m=========================================\033[0m\n'
printf '\n'
printf '  Install prefix:  %s\n' "${INSTALL_DIR}"
printf '  OptiX headers:   %s\n' "${OPTIX_DIR}"
printf '  GaussianViewer:  %s\n' "${viewer_bin}"
printf '  InteractiveViewer: %s\n' "${interactive_bin}"
printf '\n'
printf '  To use in your own CMake project:\n'
printf '    cmake -DCMAKE_PREFIX_PATH="%s" ..\n' "${INSTALL_DIR}"
printf '\n'
printf '  To use at runtime, add the bin/lib dirs to LD_LIBRARY_PATH:\n'
printf '    export LD_LIBRARY_PATH="%s/lib:%s/bin:${LD_LIBRARY_PATH:-}"\n' "${INSTALL_DIR}" "${INSTALL_DIR}"
printf '\n'
printf '  To use the Python bindings:\n'
printf '    export PYTHONPATH="%s:%s:${PYTHONPATH:-}"\n' "${viewer_build}" "${ROOT}/python"
printf '\n'
