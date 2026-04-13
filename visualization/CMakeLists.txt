cmake_minimum_required(VERSION 3.17)
project(VisualizationSuperbuild LANGUAGES NONE)

include(ExternalProject)

# ── shared install prefix ────────────────────────────────────────────────────
# For wheel builds, install the full superbuild payload into the Python package
# directory so dependencies, plugins, binaries, and metadata are shipped
# together. For standalone builds, keep the local install/ fallback.

if(DEFINED SKBUILD_PLATLIB_DIR)
  set(_wheel_prefix "${SKBUILD_PLATLIB_DIR}/gaussian_viewer")
  set(CMAKE_INSTALL_PREFIX "${_wheel_prefix}" CACHE PATH
      "Shared install prefix for all dependencies" FORCE)
elseif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  get_filename_component(_default_prefix "${CMAKE_SOURCE_DIR}/install" ABSOLUTE)
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

option(BUILD_HELIDE "Build the ANARI helide reference device" OFF)
option(BUILD_VISRTX "Build the VisRTX RTX device (requires OptiX)" ON)
option(BUILD_VISGL  "Build the VisRTX OpenGL device" ON)

set(ANARI_VERSION  "v0.15.0" CACHE STRING "ANARI-SDK git tag")
set(VISRTX_VERSION "v0.13.0" CACHE STRING "VisRTX git tag")
set(OPTIX_VERSION  "v9.1.0"  CACHE STRING "OptiX SDK git tag to download")

if(BUILD_VISRTX OR BUILD_VISGL)
  set(_need_visrtx TRUE)
else()
  set(_need_visrtx FALSE)
endif()

# ── resolve OptiX SDK headers ────────────────────────────────────────────────

if(NOT OPTIX_ROOT OR NOT EXISTS "${OPTIX_ROOT}/include")
  include(FetchContent)
  FetchContent_Declare(optix_headers
    GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
    GIT_TAG        ${OPTIX_VERSION}
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
message(STATUS "ANARI version:   ${ANARI_VERSION}")
message(STATUS "VisRTX version:  ${VISRTX_VERSION}")
message(STATUS "OptiX version:   ${OPTIX_VERSION}")
message(STATUS "OptiX SDK:       ${OPTIX_ROOT}")
message(STATUS "Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Helide device:   ${BUILD_HELIDE}")
message(STATUS "VisRTX RTX dev:  ${BUILD_VISRTX}")
message(STATUS "VisRTX GL dev:   ${BUILD_VISGL}")

# ═════════════════════════════════════════════════════════════════════════════
#  1. ANARI-SDK
# ═════════════════════════════════════════════════════════════════════════════

ExternalProject_Add(anari_sdk
  GIT_REPOSITORY  https://github.com/KhronosGroup/ANARI-SDK.git
  GIT_SHALLOW     ON
  GIT_TAG         ${ANARI_VERSION}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${_bt}
    -DCMAKE_INSTALL_PREFIX:PATH=${_ip}
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_HELIDE_DEVICE=${BUILD_HELIDE}
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

if(_need_visrtx)
  ExternalProject_Add(visrtx
    GIT_REPOSITORY  https://github.com/NVIDIA/VisRTX.git
    GIT_TAG         ${VISRTX_VERSION}
    GIT_SHALLOW     ON
    LIST_SEPARATOR  "|"
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${_bt}
      -DCMAKE_INSTALL_PREFIX:PATH=${_ip}
      "-DCMAKE_PREFIX_PATH:STRING=${_ip}|${OPTIX_ROOT}"
      -DOptiX_ROOT:PATH=${OPTIX_ROOT}
      -DVISRTX_BUILD_RTX_DEVICE=${BUILD_VISRTX}
      -DVISRTX_BUILD_GL_DEVICE=${BUILD_VISGL}
      -DVISRTX_BUILD_TESTS=ON
    BUILD_COMMAND
      ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${_bt} --parallel
    INSTALL_COMMAND
      ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${_bt}
    DEPENDS anari_sdk
  )
endif()

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
# native binaries and Python modules under the package directory.
foreach(_var SKBUILD_PLATLIB_DIR)
  if(DEFINED ${_var})
    list(APPEND _gv_cmake_args "-D${_var}=${${_var}}")
  endif()
endforeach()

set(_gv_deps anari_sdk)
if(_need_visrtx)
  list(APPEND _gv_deps visrtx)
endif()

ExternalProject_Add(gaussian_viewer
  SOURCE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/src
  DOWNLOAD_COMMAND  ""
  CMAKE_ARGS        ${_gv_cmake_args}
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${_bt} --parallel
  INSTALL_COMMAND
    ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${_bt}
  BUILD_ALWAYS      ON
  DEPENDS ${_gv_deps}
)
