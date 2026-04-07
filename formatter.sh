#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

pushd "$SCRIPT_DIR"

# # Paths must match what `find .` prints: leading `./` (otherwise -prune never matches).
# find . \
#   \( -path './thirdparty/tiny-cuda-nn' -o -path './threedgrt_tracer/dependencies/optix-dev' -o -path './.venv' \) -prune -o \
#   -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) \
#   -exec clang-format -i {} \;

black . --target-version=py311 --line-length=120 --extend-exclude=thirdparty/tiny-cuda-nn
isort . --extend-skip=thirdparty/tiny-cuda-nn --profile=black

popd
