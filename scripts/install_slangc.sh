#!/bin/bash

set -euo pipefail

# This files installs slangc from the official source into UV_PROJECT_ENVIRONMENT
echo "Installing slangc from the official source"
echo "  environment: $UV_PROJECT_ENVIRONMENT"

SLANGC_VERSION="2026.5.2"

error_with_color_and_exit() {
    echo -e "\033[31m${@}\033[0m" >&2
    exit 1
}

# Check if slangc is already installed
if [ -f "$UV_PROJECT_ENVIRONMENT/bin/slangc" ]; then
    echo "Slangc found in $UV_PROJECT_ENVIRONMENT/bin/slangc"
    echo "  Checking version..."
    # check if the version is correct
    actual_version=$(slangc -version 2>&1)
    if [ "$actual_version" != "$SLANGC_VERSION" ]; then
        error_with_color_and_exit "  ERROR: Slangc version is incorrect, expected $SLANGC_VERSION, got $actual_version"
    fi
    echo "  Slangc is already installed and matches the expected version"
    exit 0
fi

# Determine the download URL based on the platform
if [ "$(uname -m)" == "x86_64" ] || [ "$(uname -m)" == "amd64" ]; then
    DOWNLOAD_URL="https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-linux-x86_64.tar.gz"
elif [ "$(uname -m)" == "aarch64" ] || [ "$(uname -m)" == "arm64" ]; then
    DOWNLOAD_URL="https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-linux-aarch64.tar.gz"
else
    echo "Unsupported platform: $(uname -m)"
    exit 1
fi

# Install slangc
if [ ! -f /tmp/$(basename $DOWNLOAD_URL) ]; then
    wget -O /tmp/$(basename $DOWNLOAD_URL) $DOWNLOAD_URL
fi

# Unpack the tarball and install into the environment
if [ ! -d $UV_PROJECT_ENVIRONMENT/bin ]; then
    error_with_color_and_exit "  ERROR: $UV_PROJECT_ENVIRONMENT/bin does not exist, this indicates that the virtual environment has not been created correctly."
fi
tar -xzf /tmp/$(basename $DOWNLOAD_URL) -C $UV_PROJECT_ENVIRONMENT
echo "Slangc installed in $UV_PROJECT_ENVIRONMENT/bin/slangc"
