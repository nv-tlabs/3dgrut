cmake_minimum_required(VERSION 3.17)
project(GaussianViewerSuite LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── dependencies (shared) ────────────────────────────────────────────────────

find_package(anari REQUIRED)
find_package(VisRTX REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

# ── glad (generated OpenGL 3.3 core loader) ──────────────────────────────────

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

# ── offline renderer (GaussianViewer) ────────────────────────────────────────

add_library(gaussian_renderer_core STATIC renderer_core.cpp)
set_target_properties(gaussian_renderer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gaussian_renderer_core PUBLIC
  ${happly_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gaussian_renderer_core PUBLIC
  VisRTX::anari_library_visrtx
  anari::anari_static
  CUDA::cudart
)

add_executable(GaussianViewer GaussianViewer.cpp)

target_include_directories(GaussianViewer PRIVATE
  ${stb_SOURCE_DIR}
  ${happly_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(GaussianViewer PRIVATE
  gaussian_renderer_core
)

add_custom_command(TARGET GaussianViewer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:VisRTX::anari_library_visrtx>
    $<TARGET_FILE_DIR:GaussianViewer>
)

# ── interactive viewer ───────────────────────────────────────────────────────

add_executable(InteractiveViewer InteractiveViewer.cpp)

target_include_directories(InteractiveViewer PRIVATE
  ${stb_SOURCE_DIR}
  ${happly_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(InteractiveViewer PRIVATE
  gaussian_renderer_core
  imgui_lib
  glad
  glfw
  OpenGL::GL
  CUDA::cudart
)

add_custom_command(TARGET InteractiveViewer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:VisRTX::anari_library_visrtx>
    $<TARGET_FILE_DIR:InteractiveViewer>
)

# ── Python bindings (optional) ───────────────────────────────────────────

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module for GaussianRendererCore" OFF)

if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
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

  add_custom_command(TARGET _gaussian_renderer_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:VisRTX::anari_library_visrtx>
      $<TARGET_FILE_DIR:_gaussian_renderer_core>
  )
endif()
