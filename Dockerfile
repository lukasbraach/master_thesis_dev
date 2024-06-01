# Use PyTorch base image with CUDA and cuDNN support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /workspace

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 htop nload git git-lfs nano curl wget iputils-ping traceroute \
    net-tools net-tools ifupdown iproute2 grep gawk sed coreutils rsync openssh-client tar gzip unzip bzip2 cron openssh-client tmux screen -y
RUN git lfs install

RUN curl \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=master_thesis_dev

# Create the conda environment and install dependencies
RUN conda init && \
    source /root/.bashrc && \
    conda create -n master_thesis_dev && \
    conda activate master_thesis_dev && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Copy the environment file and install dependencies
COPY environment.yml .

RUN conda env update && \
    conda clean -afy

# Activate the conda environment by default
RUN echo "conda activate master_thesis_dev" >> ~/.bashrc
ENV PATH "/root/miniconda3/envs/myenv/bin:${PATH}"

# Copy the rest of your application's code into the container
COPY . /workspace/code

# Command to run when starting the container
CMD ["/bin/bash"]