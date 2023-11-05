FROM nvcr.io/nvidian/pytorch:22.12-py3
LABEL org.opencontainers.image.authors="rdelutio@nvidia.com"

RUN pip uninstall torch -y
RUN pip uninstall torchvision -y
RUN pip uninstall torchaudio -y

# Create workspace
USER root
RUN apt-get -y update

RUN chmod -R 777 .
RUN apt-get install -y build-essential git
RUN apt-get install -y libgl1-mesa-glx

# # Install conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /workspace
ADD requirements.txt /workspace/

ADD ray-tracing-gaussians/ /workspace/ray-tracing-gaussians
WORKDIR /workspace/ray-tracing-gaussians

# create environment and install dependencies
RUN conda create -n 3dgrt python=3.10 && \
    source activate 3dgrt && \
    pip install -r requirements.txt && \
    cd ./thirdparty/simple-knn && \
    python setup.py install

WORKDIR /workspace
# removing the code to avoid confusion with the code in the workspace
RUN rm -r ray-tracing-gaussians
