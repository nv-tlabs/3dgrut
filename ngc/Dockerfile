FROM nvcr.io/nvidian/pytorch:23.08.02-py3
LABEL org.opencontainers.image.authors="rdelutio@nvidia.com"

RUN pip uninstall torch -y
RUN pip uninstall torchvision -y
RUN pip uninstall torchaudio -y

# Create workspace
USER root
RUN apt-get update


RUN chmod -R 777 .
RUN apt-get install -y build-essential git
RUN apt install -y libgl1-mesa-glx

# Install conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Add repo from local to avoid passing around tokens
WORKDIR /workspace

# ADD ray-tracing-gaussians /workspace/ray-tracing-gaussians/
# WORKDIR /workspace/ray-tracing-gaussians
# ADD truck /workspace/truck

ADD requirements.txt /workspace/

# Create conda env
RUN conda create -n 3dgrt python=3.10

RUN echo "source /usr/local/etc/profile.d/conda.sh" >> ~/.bashrc
RUN source /opt/conda/etc/profile.d/conda.sh 
RUN source activate 3dgrt

# Install dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install plyfile
RUN pip install tensorboard
RUN pip install torchmetrics
RUN pip install tqdm
RUN pip install fire
RUN pip install toml
RUN pip install -r requirements.txt

# RUN pip install -r requirements.txt
