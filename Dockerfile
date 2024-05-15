# Use PyTorch base image with CUDA and cuDNN support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Lightning
RUN pip install --no-cache-dir pytorch-lightning

# Copy the rest of your application's code into the container
COPY . /app

# Command to run when starting the container
CMD ["python", "src/train.py"]