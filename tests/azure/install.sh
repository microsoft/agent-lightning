#!/bin/bash

set -ex

# Build essentials are required.
# But clean first...
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
    git-lfs

sudo git lfs install

# Install azcli for Azure resources access and management.
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# VM with GPU needs to install drivers. Reference:
# https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
