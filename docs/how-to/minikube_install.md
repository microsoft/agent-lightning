Install Minikube on a Linux VM
Follow these steps to set up Minikube using Docker as the driver. This guide assumes you are using an Ubuntu/Debian-based distribution.

1. Update System & Install Dependencies
Ensure your package list is current and install basic transport tools.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl apt-transport-https
```

2. Install Docker (The Driver)
Minikube needs a way to run the cluster. Docker is the most common choice for VMs.

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
# Allow your user to run Docker without sudo
sudo usermod -aG docker $USER && newgrp docker
```

3. Install kubectl
This is the command-line tool used to control Kubernetes.

```bash
# Download the latest stable binary
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
# Install it to your path
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
# Verify installation
kubectl version --client
```

4. Install Minikube
Now, download and install the Minikube binary.

```bash
# Download the Minikube binary
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
# Install the binary
sudo install minikube-linux-amd64 /usr/local/bin/minikube
# Verify installation
minikube version
```

5. Start the Cluster
Initialize Minikube using the Docker driver.

```bash
minikube start --driver=docker
```

Note: The first run will take a few minutes to download the VM images.

6. Verify & Test
Check if your node is online and healthy.

```bash
# Check Minikube status
minikube status
# See your cluster node
kubectl get nodes
```

------------------------------
Useful Commands

* Stop the cluster: `minikube stop`
* Delete the cluster: `minikube delete`
* Open Dashboard: `minikube dashboard`
* Pause Kubernetes: `minikube pause`

Do you need help exposing the dashboard or services so you can access them from your host machine's browser?

