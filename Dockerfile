# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
ENV EGL_PLATFORM=surfaceless
ENV XDG_DATA_HOME="/usr/local/share/uv"
ENV XDG_BIN_HOME="/usr/local/bin"
ENV UV_INSTALL_DIR="/usr/local/bin"
ENV UV_SYSTEM_PYTHON=1
ENV FORCE_CUDA=1

RUN apt-get update && \
    apt-get install -y --allow-unauthenticated ca-certificates && \
    apt-get install -y -qq --no-install-recommends build-essential \
        wget git curl libgl1-mesa-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
COPY . .
RUN bash ./install_env_uv.sh

RUN echo "source $(pwd)/.venv/bin/activate" >> ~/.bashrc
