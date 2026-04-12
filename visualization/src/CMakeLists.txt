cmake_minimum_required(VERSION 3.17)
project(VisualizationSuperbuild LANGUAGES NONE)

include(ExternalProject)

# ── shared install prefix ────────────────────────────────────────────────────

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  get_filename_component(_default_prefix "${CMAKE_SOURCE_DIR}/../install" ABSOLUTE)
  set(CMAKE_INSTALL_PREFIX "${_default_prefix}" CACHE PATH
      "Shared install prefix for all dependencies" FORCE)
endif()

set(_ip "${CMAKE_INSTALL_PREFIX}")

# ── options ──────────────────────────────────────────────────────────────────

set(OPTIX_ROOT "" CACHE PATH
    "Path to a local NVIDIA OptiX SDK (must contain an include/ directory). "
    "When empty the headers are downloaded automatically from GitHub.")

option(BUILD_PYTHON_BINDINGS
    "Build pybind11 Python bindings for GaussianRendererCore" OFF)

# ── resolve OptiX SDK headers ────────────────────────────────────────────────
# Priority: explicit -DOPTIX_ROOT > Windows auto-detect > GitHub download.

if(NOT OPTIX_ROOT)
  if(WIN32)
    file(GLOB _optix_candidates "C:/ProgramData/NVIDIA Corporation/OptiX SDK*")
    if(_optix_candidates)
      list(SORT _optix_candidates)
      list(GET _optix_candidates -1 OPTIX_ROOT)
      message(STATUS "Auto-detected local OptiX SDK: ${OPTIX_ROOT}")
    endif()
  endif()
endif()

if(NOT OPTIX_ROOT OR NOT EXISTS "${OPTIX_ROOT}/include")
  message(STATUS "No local OptiX SDK found – downloading headers from GitHub …")
  include(FetchContent)
  FetchContent_Declare(optix_headers
    GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
    GIT_TAG        v9.1.0
    GIT_SHALLOW    ON
  )
  FetchContent_GetProperties(optix_headers)
  if(NOT optix_headers_POPULATED)
    FetchContent_Populate(optix_headers)
  endif()
  set(OPTIX_ROOT "${optix_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
endif()

# ── build type default ───────────────────────────────────────────────────────

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(_bt "${CMAKE_BUILD_TYPE}")

# ── summary ──────────────────────────────────────────────────────────────────

message(STATUS "Build type:      ${_bt}")
message(STATUS "Install prefix:  ${_ip}")
message(STATUS "OptiX SDK:       ${OPTIX_ROOT}")
message(STATUS "Python bindings: ${BUILD_PYTHON_BINDINGS}")

# ═════════════════════════════════════════════════════════════════════════════
#  1. ANARI-SDK
# ═════════════════════════════════════════════════════════════════════════════

ExternalProject_Add(anari_sdk
  GIT_REPOSITORY  https://github.com/KhronosGroup/ANARI-SDK.git
  GIT_SHALLOW     ON
  GIT_TAG         v0.15.0
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${_bt}
    -DCMAKE_INSTALL_PREFIX:PATH=${_ip}
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_HELIDE_DEVICE=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_TESTING=OFF
    -DBUILD_CTS=OFF
    -DBUILD_VIEWER=OFF
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${_bt} --parallel
  INSTALL_COMMAND
    ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${_bt}
)

# ═════════════════════════════════════════════════════════════════════════════
#  2. VisRTX  (depends on installed ANARI-SDK)
# ═════════════════════════════════════════════════════════════════════════════

ExternalProject_Add(visrtx
  GIT_REPOSITORY  https://github.com/NVIDIA/VisRTX.git
  GIT_TAG         v0.13.0
  GIT_SHALLOW     ON
  LIST_SEPARATOR  "|"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${_bt}
    -DCMAKE_INSTALL_PREFIX:PATH=${_ip}
    "-DCMAKE_PREFIX_PATH:STRING=${_ip}|${OPTIX_ROOT}"
    -DOptiX_ROOT:PATH=${OPTIX_ROOT}
    -DVISRTX_BUILD_RTX_DEVICE=ON
    -DVISRTX_BUILD_GL_DEVICE=OFF
    -DVISRTX_BUILD_TESTS=ON
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${_bt} --parallel
  INSTALL_COMMAND
    ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${_bt}
  DEPENDS anari_sdk
)

# ═════════════════════════════════════════════════════════════════════════════
#  3. gaussian_viewer  (local subfolder, depends on both)
# ═════════════════════════════════════════════════════════════════════════════

set(_gv_cmake_args
  -DCMAKE_BUILD_TYPE=${_bt}
  -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
  -DCMAKE_PREFIX_PATH:PATH=${_ip}
  -DBUILD_PYTHON_BINDINGS=${BUILD_PYTHON_BINDINGS}
)

# When driven by scikit-build-core these variables point to the wheel staging
# directories.  Forward them so the inner project's install() rules can place
# executables and Python modules in the right locations.
foreach(_var SKBUILD_PLATLIB_DIR SKBUILD_SCRIPTS_DIR)
  if(DEFINED ${_var})
    list(APPEND _gv_cmake_args "-D${_var}=${${_var}}")
  endif()
endforeach()

ExternalProject_Add(gaussian_viewer
  SOURCE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/gaussian_viewer
  DOWNLOAD_COMMAND  ""
  CMAKE_ARGS        ${_gv_cmake_args}
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${_bt} --parallel
  INSTALL_COMMAND
    ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${_bt}
  BUILD_ALWAYS      ON
  DEPENDS anari_sdk visrtx
)
