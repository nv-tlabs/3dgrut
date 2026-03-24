#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

pushd "$SCRIPT_DIR"

black . --target-version=py311 --line-length=120 --exclude=thirdparty/tiny-cuda-nn
isort . --skip=thirdparty/tiny-cuda-nn --profile=black

popd
