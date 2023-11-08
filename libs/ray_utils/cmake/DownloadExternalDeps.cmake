include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
  set(LIBIGL_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
  set(LIBIGL_EXTRA_OPTIONS "")
endif()

# Shortcut function
function(download_dep name)
  download_project(
    PROJ         ${name}
    SOURCE_DIR   ${EXTERNAL_DEP_DIR}/${name}
    DOWNLOAD_DIR ${EXTERNAL_DEP_DIR}/.cache/${name}
    QUIET
    ${ARGN}
  )
endfunction()
