# Install Prerequisites

## Update System & Install Dependencies
Ensure your package list is current and install basic transport tools.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl apt-transport-https
```

## Docker
Docker is a platform that allows you to run applications in containers. It provides an easy way to create, deploy, and manage applications in a consistent environment.

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
# Allow your user to run Docker without sudo
sudo usermod -aG docker $USER && newgrp docker
```
You may need to log out and log back in for the group changes to take effect.

## uv
`uv` is a tool for managing Python environments and dependencies. It simplifies the process of creating isolated environments for your projects. You can install `uv` using pip or by downloading the binary.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
You may need to add it to your PATH. Follow the instructions provided by the installer.

## Node.js
Node.js is a JavaScript runtime that allows you to run JavaScript code outside of a browser. It is commonly used for building server-side applications and tools. You can install Node.js using a package manager like `nvm` (Node Version Manager) or by downloading the binary. Here’s how to install Node.js using `nvm`:

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
```

After installing `nvm`, you can install the latest version of Node.js:

```bash
nvm install node
```

`pnpm` is better than `npm` (the default package manager for Node.js) in terms of speed and efficiency. It uses a content-addressable filesystem to store all files from all module installations in a single place on the disk, which allows for faster installations and less disk space usage. `pnpm` is also compatible with the npm registry, so you can use it as a drop-in replacement for npm. You can install `pnpm` using npm:

```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

To install a package from the source code, you can use the following command:

```bash
cd path/to/your/package
pnpm run build # Build the package
pnpm link # Link the package globally
```