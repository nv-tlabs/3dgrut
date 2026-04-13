cmake_minimum_required(VERSION 3.17)
project(GaussianViewerSuite LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── dependencies (shared) ────────────────────────────────────────────────────

find_package(anari REQUIRED)
find_package(OpenGL REQUIRED)
find_package(VisRTX QUIET)
find_package(CUDAToolkit QUIET)

if(VisRTX_FOUND)
  message(STATUS "VisRTX found -- will link anari_library_visrtx for runtime discovery")
else()
  message(STATUS "VisRTX not found -- VisRTX device will not be bundled (can still be loaded at runtime)")
endif()

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA toolkit found -- enabling CUDA framebuffer support")
else()
  message(STATUS "CUDA toolkit not found -- CUDA framebuffer path disabled")
endif()

include(FetchContent)

FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_SHALLOW    ON
)

FetchContent_Declare(happly
  GIT_REPOSITORY https://github.com/nmwsharp/happly.git
  GIT_SHALLOW    ON
)

# ── dependencies (interactive viewer) ────────────────────────────────────────

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    ON
)

FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.8
  GIT_SHALLOW    ON
)

FetchContent_MakeAvailable(stb happly glfw imgui)

# ── glad (generated OpenGL 3.3 compatibility loader) ─────────────────────────
# Pre-generated via glad2 (https://glad.sh).  To regenerate:
#   pip install glad2
#   glad --api gl:compatibility=3.3 --out-path visualization/src/glad c

set(GLAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glad)
add_library(glad STATIC ${GLAD_DIR}/src/gl.c)
target_include_directories(glad PUBLIC ${GLAD_DIR}/include)

# ── Dear ImGui library (GLFW + OpenGL3 backend) ─────────────────────────────

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw glad OpenGL::GL)

# ── offline renderer (commandline_viewer) ────────────────────────────────────

add_library(gaussian_renderer_core STATIC renderer_core.cpp)
set_target_properties(gaussian_renderer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gaussian_renderer_core PUBLIC
  ${happly_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gaussian_renderer_core PUBLIC
  anari::anari
)
if(VisRTX_FOUND)
  target_link_libraries(gaussian_renderer_core PUBLIC VisRTX::anari_library_visrtx)
endif()
if(CUDAToolkit_FOUND)
  target_link_libraries(gaussian_renderer_core PUBLIC CUDA::cudart)
  target_compile_definitions(gaussian_renderer_core PUBLIC GRUT_HAS_CUDA)
endif()

add_executable(commandline_viewer commandline_viewer.cpp)

target_include_directories(commandline_viewer PRIVATE
  ${stb_SOURCE_DIR}
  ${happly_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(commandline_viewer PRIVATE
  gaussian_renderer_core
)

# ── interactive viewer ───────────────────────────────────────────────────────

add_executable(interactive_viewer interactive_viewer.cpp)

target_include_directories(interactive_viewer PRIVATE
  ${stb_SOURCE_DIR}
  ${happly_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(interactive_viewer PRIVATE
  gaussian_renderer_core
  imgui_lib
  glad
  glfw
  OpenGL::GL
)
if(CUDAToolkit_FOUND)
  target_link_libraries(interactive_viewer PRIVATE CUDA::cudart)
endif()

# ── Python bindings (optional) ───────────────────────────────────────────

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module for GaussianRendererCore" OFF)

if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

  # Make pybind11's legacy FindPythonInterp/FindPythonLibs honour the Python3
  # that was just discovered (e.g. a venv Python) instead of picking up a
  # different system-wide interpreter.
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
  set(PYTHON_INCLUDE_DIR "${Python3_INCLUDE_DIRS}" CACHE PATH "" FORCE)
  set(PYTHON_LIBRARY "${Python3_LIBRARIES}" CACHE FILEPATH "" FORCE)

  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6
      GIT_SHALLOW    ON
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(_gaussian_renderer_core python_bindings.cpp)
  target_link_libraries(_gaussian_renderer_core PRIVATE gaussian_renderer_core)
endif()

set(_all_targets commandline_viewer interactive_viewer)
if(BUILD_PYTHON_BINDINGS)
  list(APPEND _all_targets _gaussian_renderer_core)
endif()

# ── runtime library search paths (RPATH) ──────────────────────────────────
# Our executables and the pybind11 extension link against shared libraries
# (VisRTX, ANARI, …) that may live in different relative locations depending
# on how the project was packaged:
#
#   Layout A – pip wheel (`gaussian_viewer` Python package):
#     gaussian_viewer/
#       _gaussian_renderer_core.pyd   ← the pybind11 module
#       lib/  or  bin/                ← bundled .so/.dll copied by install()
#
#   Layout B – standalone CMake install:
#     bin/
#       commandline_viewer            ← the executables
#       interactive_viewer
#     lib/  or  lib64/                ← shared libraries
#
# The dynamic linker resolves $ORIGIN (Linux) / @loader_path (macOS) to the
# directory containing the binary itself.  We add both "same directory" and
# "sibling directory" patterns so that the loader finds the bundled shared
# libs in either layout without requiring LD_LIBRARY_PATH or install_name_tool
# fixups.
if(APPLE)
  set(_bundle_origin "@loader_path")
else()
  set(_bundle_origin "$ORIGIN")
endif()
set(_bundle_rpaths
  "${_bundle_origin}"            # same dir as the binary  (wheel flat layout)
  "${_bundle_origin}/lib"        # child lib/   (wheel with sub-directory)
  "${_bundle_origin}/lib64"      # child lib64/ (RPM-style 64-bit convention)
  "${_bundle_origin}/bin"        # child bin/   (Windows-style DLL location)
  "${_bundle_origin}/../lib"     # sibling lib/ (standalone: bin/../lib)
  "${_bundle_origin}/../lib64"   # sibling lib64/
  "${_bundle_origin}/../bin"     # sibling bin/
)
set_target_properties(${_all_targets} PROPERTIES
  BUILD_RPATH "${_bundle_rpaths}"
  INSTALL_RPATH "${_bundle_rpaths}"
)

# ── install rules ─────────────────────────────────────────────────────────
# When built via scikit-build-core, SKBUILD_PLATLIB_DIR is the wheel platlib
# root. Install native binaries and bundled shared libs under the Python
# package directory — not SKBUILD_SCRIPTS_DIR (scikit-build-core assumes that
# directory contains only text console scripts).

if(DEFINED SKBUILD_PLATLIB_DIR)
  set(_pkg_dest "${SKBUILD_PLATLIB_DIR}/gaussian_viewer")
  set(_runtime_dest "${_pkg_dest}")
  set(_library_dest "${_pkg_dest}")
else()
  # Conventional standalone install layout.
  set(_runtime_dest "bin")
  set(_library_dest "lib")
endif()

install(TARGETS commandline_viewer interactive_viewer RUNTIME DESTINATION "${_runtime_dest}")

if(BUILD_PYTHON_BINDINGS)
  install(TARGETS _gaussian_renderer_core
    LIBRARY DESTINATION "${_library_dest}"
    RUNTIME DESTINATION "${_library_dest}"
  )
endif()
