#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
if [[ -n "${UV_PYTHON:-}" && -x "${UV_PYTHON}" ]]; then
    PYTHON_BIN="${UV_PYTHON}"
else
    PYTHON_BIN=python
fi

if [[ ! -d "${ROOT_DIR}/thirdparty/tiny-cuda-nn/bindings/torch" ]]; then
    echo "ERROR: thirdparty/tiny-cuda-nn is missing. Run git submodule update --init --recursive first." >&2
    exit 1
fi

if [[ -z "${TCNN_CUDA_ARCHITECTURES:-}" ]]; then
    export TCNN_CUDA_ARCHITECTURES="$("${PYTHON_BIN}" - <<'PY'
import os
import re
import sys


def parse_arches(text):
    arches = []
    for item in re.split(r"[;,\s]+", text or ""):
        item = item.strip().replace("+PTX", "")
        if not item:
            continue
        match = re.fullmatch(r"(\d+)\.(\d+)", item)
        if match:
            arches.append(int(match.group(1)) * 10 + int(match.group(2)))
            continue
        match = re.fullmatch(r"(?:sm_|compute_)?(\d+)", item)
        if match:
            arches.append(int(match.group(1)))
    return arches


try:
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(major * 10 + minor)
        raise SystemExit(0)
except Exception:
    pass

arches = parse_arches(os.environ.get("TORCH_CUDA_ARCH_LIST", ""))

if not arches:
    try:
        import torch

        arches = parse_arches(";".join(torch.cuda.get_arch_list()))
    except Exception:
        pass

if not arches:
    sys.stderr.write(
        "ERROR: set TCNN_CUDA_ARCHITECTURES, or set TORCH_CUDA_ARCH_LIST, "
        "or run setup with a visible CUDA GPU.\n"
    )
    raise SystemExit(1)

print(max(set(arches)))
PY
)"
fi

echo "  TCNN_CUDA_ARCHITECTURES: ${TCNN_CUDA_ARCHITECTURES}"

pushd "${ROOT_DIR}" > /dev/null
if [[ "${INSTALL_TCNN_WITH_UV:-0}" == "1" ]]; then
    uv pip install --no-cache --no-build-isolation -r requirements_tinycudann.txt
else
    "${PYTHON_BIN}" -m pip install --no-cache-dir --no-build-isolation -r requirements_tinycudann.txt
fi
popd > /dev/null

"${PYTHON_BIN}" -c "import tinycudann; print('tiny-cuda-nn: ready')"
