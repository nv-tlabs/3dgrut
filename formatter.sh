#!/bin/bash

set -euo pipefail

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
    ACTION="Checking"
    BLACK_OPTS=(--check --diff)
    ISORT_OPTS=(--check --diff)
else
    ACTION="Formatting"
fi

pushd "$SCRIPT_DIR" &> /dev/null

CLANG_FORMAT_REQUIRED_VERSION=18
if command -v clang-format &> /dev/null; then
    CLANG_FORMAT_VERSION=$(clang-format --version | grep -oP '\d+' | head -1)
    if [ "$CLANG_FORMAT_VERSION" -ne "$CLANG_FORMAT_REQUIRED_VERSION" ]; then
        echo "clang-format version $CLANG_FORMAT_VERSION detected, but version $CLANG_FORMAT_REQUIRED_VERSION is required."
        echo "Install with: pip3 install clang-format==$CLANG_FORMAT_REQUIRED_VERSION.*"
        FAILED=1
    else
        mapfile -d '' CLANG_FILES < <(
            git ls-files -z --cached --others --exclude-standard -- '*.cpp' '*.cuh' '*.cu' '*.h' |
                while IFS= read -r -d '' file; do
                    case "$file" in
                        thirdparty/tiny-cuda-nn/* | thirdparty/kaolin/* | threedgrt_tracer/dependencies/optix-dev/* | .venv/*) continue ;;
                    esac
                    printf '%s\0' "$file"
                done
        )

        if [ "$CHECK_MODE" = true ]; then
            echo "Checking C/C++/CUDA code with clang-format (version: $(clang-format --version))..."
            if [ "${#CLANG_FILES[@]}" -gt 0 ]; then
                clang-format --dry-run --Werror "${CLANG_FILES[@]}" || FAILED=1
            fi
        else
            echo "Formatting C/C++/CUDA code with clang-format (version: $(clang-format --version))..."
            if [ "${#CLANG_FILES[@]}" -gt 0 ]; then
                clang-format -i "${CLANG_FILES[@]}"
            fi
        fi
    fi
    echo ""
else
    echo "clang-format not found. Install with: pip3 install clang-format"
    echo ""
fi

echo "$ACTION Python code with black..."
black . "${BLACK_OPTS[@]}" --target-version=py311 --line-length=120 --extend-exclude=thirdparty/tiny-cuda-nn || FAILED=1
echo ""

echo "$ACTION Python code with isort..."
isort . "${ISORT_OPTS[@]}" --skip-gitignore --dont-follow-links --extend-skip=thirdparty/tiny-cuda-nn --profile=black || FAILED=1
echo ""

popd &> /dev/null

if [ "$FAILED" -ne 0 ]; then
    echo "Formatting check failed. Run without --check to auto-format."
    exit 1
fi
