# Use an NVIDIA CUDA base image that includes the development toolkit,
# which is often required for vLLM to build its custom kernels.
# Ubuntu 22.04 is a stable and common choice.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set an argument to prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install prerequisites for adding PPAs, then add the deadsnakes PPA for modern Python versions.
RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install the ollama CLI tool
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create a non-root user 'vllama' for security, mirroring the vllama.install script.
# The user's home directory is set to /opt/vllama, which is where we'll run the app.
RUN useradd -r -g vllama -d /opt/vllama -s /bin/bash vllama || groupadd -r vllama && useradd -r -g vllama -d /opt/vllama -s /bin/bash vllama

# Set the working directory for subsequent commands.
WORKDIR /opt/vllama

# Copy all project files into the working directory.
COPY . .

# Create a Python virtual environment and install dependencies into it.
# This follows the same logic as the PKGBUILD, ensuring a clean, isolated environment.
RUN python3.12 -m venv venv312 && \
    ./venv312/bin/pip install --no-cache-dir -r requirements.txt

# Change the ownership of the application directory to the 'vllama' user.
RUN chown -R vllama:vllama /opt/vllama

# Switch to the non-root user.
USER vllama

# The command to run when the container starts.
# It executes the main script using the Python interpreter from the virtual environment,
# which guarantees the correct Python version (3.12) is used.
CMD ["/opt/vllama/venv312/bin/python", "vllama.py"]
