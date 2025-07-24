FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# create an app user for github actions runner
RUN groupadd --gid 1001 app && \
    useradd  --uid 1001 --gid app --create-home app

# install anything that still needs root
RUN apt-get update && apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    python3-pip \
    graphviz \
    unzip \
    tmux \
    vim \
    git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

USER app
WORKDIR /workspace
