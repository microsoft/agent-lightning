FROM python:3.11-slim

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
RUN echo "tzdata tzdata/Areas select America" | debconf-set-selections && \
    echo "tzdata tzdata/Zones/America select Los_Angeles" | debconf-set-selections

# Install system dependencies
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y git curl wget build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create directories
RUN mkdir -p /testbed /root/.local/bin

# Set up PATH environment variables
ENV PATH="/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Install common Python libraries directly to system Python
RUN pip install --upgrade pip && \
    pip install numpy sympy scipy matplotlib mpmath ipython chardet

# Set working directory
WORKDIR /testbed

# Set environment
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore

CMD ["/bin/bash"]
