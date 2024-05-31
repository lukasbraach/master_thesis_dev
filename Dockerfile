# Use PyTorch base image with CUDA and cuDNN support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /workspace

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 htop nload git git-lfs nano curl wget iputils-ping traceroute \
    net-tools net-tools ifupdown iproute2 grep gawk sed coreutils rsync openssh-client tar gzip unzip bzip2 cron openssh-client tmux screen -y
RUN git lfs install

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . /workspace/code

# Command to run when starting the container
CMD ["/bin/bash"]