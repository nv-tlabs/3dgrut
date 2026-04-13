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

# ── runtime shared libraries ──────────────────────────────────────────────
# Glob all shared libraries from the dependency install prefix.  By the time
# this project configures (as an ExternalProject), ANARI-SDK and VisRTX are
# already built and installed to CMAKE_PREFIX_PATH.  The prefix is private
# to this superbuild, so every shared library in it is ours to bundle.

set(_runtime_libs "")
set(_runtime_dirs "${CMAKE_PREFIX_PATH}/bin" "${CMAKE_PREFIX_PATH}/lib" "${CMAKE_PREFIX_PATH}/lib64")
foreach(_dir IN LISTS _runtime_dirs)
  if(NOT IS_DIRECTORY "${_dir}")
    continue()
  endif()

  if(WIN32)
    file(GLOB _found "${_dir}/*${CMAKE_SHARED_LIBRARY_SUFFIX}")
  else()
    file(GLOB _found "${_dir}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
  endif()
  list(APPEND _runtime_libs ${_found})
endforeach()
list(REMOVE_DUPLICATES _runtime_libs)

set(_all_targets commandline_viewer interactive_viewer)
if(BUILD_PYTHON_BINDINGS)
  list(APPEND _all_targets _gaussian_renderer_core)
endif()

foreach(_tgt IN LISTS _all_targets)
  foreach(_lib IN LISTS _runtime_libs)
    add_custom_command(TARGET ${_tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_lib}" $<TARGET_FILE_DIR:${_tgt}>
    )
  endforeach()
endforeach()

# Load bundled .so/.dylib from the same directory as executables and the Python
# extension (wheel layout: all under site-packages/gaussian_viewer/).
if(APPLE)
  set(_bundle_rpath "@loader_path")
else()
  set(_bundle_rpath "$ORIGIN")
endif()
set_target_properties(${_all_targets} PROPERTIES
  BUILD_RPATH "${_bundle_rpath}"
  INSTALL_RPATH "${_bundle_rpath}"
)

# ── install rules ─────────────────────────────────────────────────────────
# When built via scikit-build-core, SKBUILD_PLATLIB_DIR is the wheel platlib
# root. Install native binaries and bundled shared libs under the Python
# package directory — not SKBUILD_SCRIPTS_DIR (scikit-build-core assumes that
# directory contains only text console scripts).

if(DEFINED SKBUILD_PLATLIB_DIR)
  set(_pkg_dest "${SKBUILD_PLATLIB_DIR}/gaussian_viewer")
else()
  set(_pkg_dest "lib")
endif()

install(TARGETS commandline_viewer interactive_viewer RUNTIME DESTINATION "${_pkg_dest}")

if(BUILD_PYTHON_BINDINGS)
  install(TARGETS _gaussian_renderer_core
    LIBRARY DESTINATION "${_pkg_dest}"
    RUNTIME DESTINATION "${_pkg_dest}"
  )
endif()

foreach(_lib IN LISTS _runtime_libs)
  install(FILES "${_lib}" DESTINATION "${_pkg_dest}")
endforeach()
