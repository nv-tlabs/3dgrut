#!/bin/bash

set -euo pipefail

# Persist the following environment variables into the venv activation script
# so they are restored automatically on `source .venv/bin/activate`:
#
#   Compiler:  CC  CXX
#   CUDA:      CUDA_HOME  CUDA_VERSION  CUDA_FULL_VERSION
#              CUDA_MAJOR  CUDA_MAJOR_TARGET  CUDA_MINOR_TARGET
#   Build:     TORCH_CUDA_ARCH_LIST
#   UV:        UV_PROJECT_ENVIRONMENT  UV_PYTHON
#   Torch:     TORCH_INDEX_URL  TORCH_VERSION

BANNER="# --- 3DGRUT env vars -------"
FINISH="# --- end 3DGRUT env vars ---"

ACTIVATE=".venv/bin/activate"

if ! grep -q "$BANNER" "$ACTIVATE" 2>/dev/null; then

  cat >> "$ACTIVATE" <<ENVEOF

$BANNER

###############################################################
# Save any pre-existing values so deactivate can restore them #
###############################################################
record_var () {
  local _var="\$1" _val="\$2"
  printf -v "_3DGRUT_OLD_\${_var}" '%s' "\${!_var:-}"
  export "\$_var"="\$_val"
}

record_var CC                     "${CC}"
record_var CXX                    "${CXX}"
record_var CUDA_HOME              "${CUDA_HOME}"
record_var CUDA_VERSION           "${CUDA_VERSION}"
record_var CUDA_FULL_VERSION      "${CUDA_FULL_VERSION}"
record_var CUDA_MAJOR             "${CUDA_MAJOR}"
record_var CUDA_MAJOR_TARGET      "${CUDA_MAJOR_TARGET}"
record_var CUDA_MINOR_TARGET      "${CUDA_MINOR_TARGET}"
record_var TORCH_CUDA_ARCH_LIST   "${TORCH_CUDA_ARCH_LIST}"
record_var UV_PYTHON              "${UV_PYTHON}"
record_var TORCH_INDEX_URL        "${TORCH_INDEX_URL}"
record_var TORCH_VERSION          "${TORCH_VERSION}"

unset -f record_var

export _3DGRUT_OLD_PATH="\$PATH"
export _3DGRUT_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
if [ -n "\${CUDA_HOME:-}" ]; then
  export PATH="\${CUDA_HOME}/bin:\$PATH"
  export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
fi

###################################################################
# Wrap deactivate to undo 3DGRUT env vars                         #
###################################################################
restore_var () {
  local _var="\$1" _old="_3DGRUT_OLD_\${1}"
  if [ -n "\${!_old}" ]; then export "\$_var"="\${!_old}"; else unset "\$_var" 2>/dev/null; fi
  unset "\$_old"
}

if !  declare -f _threedgrut_orig_deactivate >/dev/null 2>&1; then
  eval "\$(echo '_threedgrut_orig_deactivate()'; declare -f deactivate | tail -n +2)"
  deactivate () {
    restore_var CC
    restore_var CXX
    restore_var CUDA_HOME
    restore_var CUDA_VERSION
    restore_var CUDA_FULL_VERSION
    restore_var CUDA_MAJOR
    restore_var CUDA_MAJOR_TARGET
    restore_var CUDA_MINOR_TARGET
    restore_var TORCH_CUDA_ARCH_LIST
    restore_var UV_PYTHON
    restore_var TORCH_INDEX_URL
    restore_var TORCH_VERSION
    unset -f restore_var
    export PATH="\${_3DGRUT_OLD_PATH}"
    export LD_LIBRARY_PATH="\${_3DGRUT_OLD_LD_LIBRARY_PATH}"
    unset _3DGRUT_OLD_PATH
    unset _3DGRUT_OLD_LD_LIBRARY_PATH
    _threedgrut_orig_deactivate "\$@"
  }
fi

${FINISH}

ENVEOF

  echo "  Persisted following environment variables in ${ACTIVATE}:"
  echo "                   CC  CXX  UV_PYTHON                      "
  echo "      CUDA_HOME      CUDA_VERSION    CUDA_FULL_VERSION     "
  echo "      CUDA_MAJOR  CUDA_MAJOR_TARGET  CUDA_MINOR_TARGET     "
  echo "     TORCH_CUDA_ARCH_LIST TORCH_INDEX_URL TORCH_VERSION    "
  echo ""

fi

# NOTE: We intentionally do not persist UV_LINK, UV_FIND_LINKS and UV_CONSTRAINT here to explicitly
# disable `uv run`, `uv lock`, and `uv sync` from being used. Because they will try to incorrectly
# refresh the virtual environment, leading to Undefined Symbols errors.
