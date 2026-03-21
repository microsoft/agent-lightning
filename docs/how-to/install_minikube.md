# Install Minikube on a Linux VM
Follow these steps to set up Minikube using Docker as the driver. This guide assumes you are using an Ubuntu/Debian-based distribution.

Before you begin, ensure you have the prerequisites (such as Docker) installed. If you haven't done so, install them first by following the [prerequisites guide](./install_prerequisites.md).

## kubectl
This is the command-line tool used to control Kubernetes.

```bash
# Download the latest stable binary
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
# Install it to your path
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
# Verify installation
kubectl version --client
```

## Minikube
Now, download and install the Minikube binary.

```bash
# Download the Minikube binary
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
# Install the binary
sudo install minikube-linux-amd64 /usr/local/bin/minikube
# Verify installation
minikube version
```

### Start the Cluster
Initialize Minikube using the Docker driver.

```bash
minikube start --driver=docker
```

Note: The first run will take a few minutes to download the VM images.

### Verify & Test
Check if your node is online and healthy.

```bash
# Check Minikube status
minikube status
# See your cluster node
kubectl get nodes
```
Some useful commands to manage your Minikube cluster:

```bash
# Check cluster status
minikube status

# Check cluster pods
kubectl get pods 
kubectl describe pod <pod-name>

# Pause and resume the cluster
minikube pause
minikube unpause

# Stop the cluster
minikube stop
```

Do you need help exposing the dashboard or services so you can access them from your host machine's browser?

