#!/bin/bash

set -euof pipefail

SCRIPT_DIR=$(dirname "$0")
CHECK_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_MODE=true; shift ;;
        *) echo "Usage: $0 [--check]"; exit 1 ;;
    esac
done

FAILED=0
BLACK_OPTS=()
ISORT_OPTS=()
if [ "$CHECK_MODE" = true ]; then
    BLACK_OPTS=(--check --diff)
    ISORT_OPTS=(--check --diff)
fi

pushd "$SCRIPT_DIR" &> /dev/null

if command -v clang-format &> /dev/null; then
    CLANG_FILES=$(find . \
        \( -path './thirdparty/tiny-cuda-nn' -o -path './thirdparty/kaolin' -o -path './threedgrt_tracer/dependencies/optix-dev' -o -path './.venv' \) -prune -o \
        -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) -print)

    if [ "$CHECK_MODE" = true ]; then
        echo "Checking C/C++/CUDA code with clang-format..."
        echo "$CLANG_FILES" | xargs -r clang-format --dry-run --Werror || FAILED=1
    else
        echo "Formatting C/C++/CUDA code with clang-format..."
        echo "$CLANG_FILES" | xargs -r clang-format -i
    fi
    echo ""
fi

echo "${CHECK_MODE:+Checking}${CHECK_MODE:-Formatting} Python code with black..."
black . "${BLACK_OPTS[@]}" --target-version=py311 --line-length=120 --extend-exclude=thirdparty/tiny-cuda-nn || FAILED=1
echo ""

echo "${CHECK_MODE:+Checking}${CHECK_MODE:-Formatting} Python code with isort..."
isort . "${ISORT_OPTS[@]}" --extend-skip=thirdparty/tiny-cuda-nn --extend-skip=thirdparty/kaolin --profile=black || FAILED=1
echo ""

popd &> /dev/null

if [ "$FAILED" -ne 0 ]; then
    echo "Formatting check failed. Run without --check to auto-format."
    exit 1
fi
