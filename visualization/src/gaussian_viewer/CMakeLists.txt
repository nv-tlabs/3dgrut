cmake_minimum_required(VERSION 3.17)
project(GaussianViewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── dependencies ─────────────────────────────────────────────────────────────

find_package(anari REQUIRED)
find_package(VisRTX REQUIRED)

include(FetchContent)

FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_SHALLOW    ON
)

FetchContent_Declare(happly
  GIT_REPOSITORY https://github.com/nmwsharp/happly.git
  GIT_SHALLOW    ON
)

FetchContent_MakeAvailable(stb happly)

# ── executable ───────────────────────────────────────────────────────────────

add_executable(${PROJECT_NAME} GaussianViewer.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${stb_SOURCE_DIR}
  ${happly_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  VisRTX::anari_library_visrtx
  anari::anari_static
)

# Copy the VisRTX DLL next to the executable so it is found at runtime
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:VisRTX::anari_library_visrtx>
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)
