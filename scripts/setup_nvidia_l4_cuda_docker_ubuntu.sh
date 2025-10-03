#!/bin/bash

# This script installs NVIDIA drivers (branch 535 LTS for L4 GPU) and CUDA toolkit on a Google Cloud VM (assuming Ubuntu 22.04 or 24.04),
# then installs Docker with NVIDIA GPU support, and configures Docker to run without sudo for the current user.
# IMPORTANT NOTES:
# - Run this script as a regular user with sudo privileges.
# - The NVIDIA installation may automatically reboot the VM one or more times. If the script interrupts due to reboot,
#   rerun the entire script—it will resume where needed (the installer handles idempotency).
# - After completion, reboot the VM if prompted, then verify with 'nvidia-smi'.
# - To use Docker without sudo, log out and log back in after the script finishes, or run 'newgrp docker' in your current session.
# - Assumes at least 40 GB free on boot disk and Python 3 installed (default on Ubuntu).
# - If using Secure Boot, this script may not work; use manual pre-signed driver steps from GCP docs instead.

set -e

# Set installation parameters for NVIDIA L4 (LTS branch 535, binary mode required for LTS)
INSTALL_BRANCH="lts"
INSTALL_MODE="binary"
INSTALL_DIR="/opt/google/cuda-installer"

# Download NVIDIA installer if not already present
if [ ! -f "${INSTALL_DIR}/cuda_installer.pyz" ]; then
  sudo mkdir -p "${INSTALL_DIR}"
  cd "${INSTALL_DIR}"
  sudo curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz --output cuda_installer.pyz
fi

cd "${INSTALL_DIR}"

# Install NVIDIA driver (may reboot)
sudo python3 cuda_installer.pyz install_driver --installation-mode=${INSTALL_MODE} --installation-branch=${INSTALL_BRANCH}

# Install CUDA toolkit (may reboot)
sudo python3 cuda_installer.pyz install_cuda --installation-mode=${INSTALL_MODE} --installation-branch=${INSTALL_BRANCH}

# Install Docker prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Set up Docker repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

# Install Docker packages
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install NVIDIA Container Toolkit for GPU support in Docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon for NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Configure Docker to run without sudo (add current user to docker group)
sudo groupadd docker || true
sudo usermod -aG docker "${USER}"

# Final instructions
echo "Installation complete. If the VM was rebooted during installation, verify NVIDIA with: nvidia-smi"
echo "It should show your NVIDIA L4 GPU with driver version ~535 and CUDA ~12.2 or later."
echo "To test GPU in Docker: docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi"
echo "Log out and back in (or run 'newgrp docker') to use Docker without sudo."
