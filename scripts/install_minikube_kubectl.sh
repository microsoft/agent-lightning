#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print messages in green
echo_green() {
  echo -e "\033[0;32m$1\033[0m"
}

# Update the package list
echo_green "Updating package list..."
sudo apt-get update

# Install required dependencies
echo_green "Installing required dependencies..."
sudo apt-get install -y apt-transport-https ca-certificates curl

# Install kubectl
echo_green "Installing kubectl..."
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

# Verify kubectl installation
echo_green "Verifying kubectl installation..."
kubectl version --client --output=yaml

# Install Minikube
echo_green "Installing Minikube..."
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

# Verify Minikube installation
echo_green "Verifying Minikube installation..."
minikube version

echo_green "Installation of Minikube and kubectl completed successfully!"