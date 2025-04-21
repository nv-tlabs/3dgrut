FROM ubuntu:24.04

ARG CUDA_VERSION=11.8.0
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    wget git \
    curl \
    build-essential \
    gcc-11 g++-11 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_25.1.1-2-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:$PATH
RUN conda init

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV FORCE_CUDA=1

# # Make sure TORCH_CUDA_ARCH_LIST matches the pytorch wheel setting.
# # Reference: https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh#L54
# #
# # (cuda11) $ python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
# # 11.8 ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_37', 'sm_90', 'compute_37']
# #
# # (cuda12) $ python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
# # 12.8 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
# #
# RUN if   [ "$CUDA_VERSION" = "11.8.0" ]; then                                                        \
#       echo 'export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"' >> /etc/profile.d/cuda_arch.sh;       \
#     elif [ "$CUDA_VERSION" = "12.8.1" ]; then                                                        \
#       echo 'export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0"' >> /etc/profile.d/cuda_arch.sh; \
#     fi

WORKDIR /workspace
COPY . .

RUN CUDA_VERSION=$CUDA_VERSION bash ./install_env.sh 3dgrut WITH_GCC11 
RUN echo "conda activate 3dgrut" >> ~/.bashrc
