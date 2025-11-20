#!/bin/bash

set -euo pipefail

# Add Docker's official GPG key
sudo apt update
sudo apt install ca-certificates curl
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

cat /etc/group

# Create the docker group if it does not exist
echo "[INFO] Ensuring 'docker' group and membership..."

# Create docker group only if it doesn't exist
if getent group docker >/dev/null 2>&1; then
  echo "[INFO] Group 'docker' already exists."
else
  echo "[INFO] Creating group 'docker'."
  sudo groupadd docker
fi

# Add current user to docker group if not already a member
if id -nG "$USER" | grep -qw docker; then
  echo "[INFO] User '$USER' is already in 'docker' group."
else
  echo "[INFO] Adding '$USER' to 'docker' group."
  sudo usermod -aG docker "$USER"
  echo "[INFO] You must log out and log back in for this to take effect."
fi
