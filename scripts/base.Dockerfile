FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    graphviz \
    unzip \
    tmux \
    vim \
    git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
