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
    nodejs

git lfs install

# VM with GPU needs to install drivers. Reference:
# https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot now

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo reboot now

# Add paths globally
sudo bash -c "cat > /etc/profile.d/cuda.sh" <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
sudo chmod +x /etc/profile.d/cuda.sh

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
