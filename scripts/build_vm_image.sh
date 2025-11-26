#!/bin/bash

# Based on: Standard_NC24ads_A100_v4
# With: Canonical:ubuntu-24_04-lts:server:latest
# Secure boot is off.

# This script is not designed to be run automatically.

set -ex

# Build essentials are required.
sudo apt-get clean
sudo apt-get update
sudo apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    python3-dev \
    python3-pip \
    python3-venv \
    graphviz \
    unzip \
    tmux \
    vim \
    git-lfs \
    nodejs \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

git lfs install

# VM with GPU needs to install drivers. Reference:
# https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot now

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit
sudo reboot now

# Add paths globally
sudo bash -c "cat > /etc/profile.d/cuda.sh" <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
sudo chmod +x /etc/profile.d/cuda.sh

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt -y update

# Install the Docker packages
sudo apt -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Create docker group only if it doesn't exist
# sudo groupadd docker

# Add current user to docker group if not already a member
sudo usermod -aG docker "$USER"
# A hack to add cloudtest user to docker group as well
sudo sed -i '/^docker:/ s/$/,cloudtest/' /etc/group
# This shouldn't be run on CI
# newgrp docker

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure the NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Install Azure CLI
curl -sLS https://packages.microsoft.com/keys/microsoft.asc |
  gpg --dearmor | sudo tee /etc/apt/keyrings/microsoft.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/microsoft.gpg
AZ_DIST=$(lsb_release -cs)
echo "Types: deb
URIs: https://packages.microsoft.com/repos/azure-cli/
Suites: ${AZ_DIST}
Components: main
Architectures: $(dpkg --print-architecture)
Signed-by: /etc/apt/keyrings/microsoft.gpg" | sudo tee /etc/apt/sources.list.d/azure-cli.sources
sudo apt-get update
sudo apt-get install -y azure-cli

# Disable the periodical apt-get upgrade.
# Sometimes, unattended upgrade blocks apt-get install
sudo sed -i -e "s/Update-Package-Lists \"1\"/Update-Package-Lists \"0\"/g" /etc/apt/apt.conf.d/10periodic
sudo sed -i -e "s/Update-Package-Lists \"1\"/Update-Package-Lists \"0\"/g" /etc/apt/apt.conf.d/20auto-upgrades
sudo sed -i -e "s/Unattended-Upgrade \"1\"/Unattended-Upgrade \"0\"/g" /etc/apt/apt.conf.d/20auto-upgrades
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer
sudo systemctl stop apt-daily.service apt-daily-upgrade.service
sudo systemctl mask apt-daily.timer apt-daily-upgrade.timer
sudo systemctl mask apt-daily.service apt-daily-upgrade.service

# Deprovision and prepare for generalized image
sudo waagent -deprovision+user
