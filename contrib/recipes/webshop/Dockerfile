# Unified WebShop Training Image
#
# This Dockerfile creates a single image containing all components needed for
# the WebShop training pipeline:
#   - WebShop Flask server (Python + Java for pyserini)
#   - Agent Lightning coordinator (Python + optional VERL for GPU)
#   - Headless runner (Node.js + pnpm)
#
# Build context must be the repository root:
#   docker build -f examples/vercel_ai_webshop/Dockerfile -t webshop-agl .
#
# For GPU training:
#   docker build -f examples/vercel_ai_webshop/Dockerfile --build-arg INSTALL_GPU=true -t webshop-agl-gpu .
#
# Run modes:
#   - Full stack: docker run webshop-agl scripts/run_stack.sh qwen
#   - WebShop only: docker run webshop-agl python server/webshop_server.py
#   - Runner only:  docker run webshop-agl pnpm headless

# Base image with CUDA support for GPU training
FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:latest

# Build argument for GPU support
ARG INSTALL_GPU=false

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64 \
    PATH="/usr/lib/jvm/temurin-21-jdk-amd64/bin:${PATH}"

WORKDIR /app

# ==============================================================================
# System Dependencies
# ==============================================================================

# Install system packages:
# - Java 21 (Temurin) for pyserini search engine
# - Node.js 20 for headless runner
# - Git, curl, wget for general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget gnupg ca-certificates procps \
    # Add Adoptium (Temurin) repository for Java 21
    && mkdir -p /etc/apt/keyrings \
    && wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb bookworm main" > /etc/apt/sources.list.d/adoptium.list \
    # Add NodeSource repository for Node.js 20
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        temurin-21-jdk \
        nodejs \
    # Install pnpm globally
    && npm install -g pnpm \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Python Dependencies
# ==============================================================================

# Copy Agent Lightning package files from repo root
COPY pyproject.toml uv.lock README.md ./
COPY agentlightning/ ./agentlightning/

# Install Agent Lightning (with optional GPU extras)
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    if [ "$INSTALL_GPU" = "true" ]; then \
        pip install --no-cache-dir -e ".[verl]" || pip install --no-cache-dir -e .; \
    else \
        pip install --no-cache-dir -e .; \
    fi

# Copy and install WebShop server requirements
COPY examples/vercel_ai_webshop/server/requirements.txt ./server-requirements.txt
RUN pip install --no-cache-dir -r server-requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy and install AGL coordinator requirements
COPY examples/vercel_ai_webshop/agl/requirements.txt ./agl-requirements.txt
RUN pip install --no-cache-dir -r agl-requirements.txt

# ==============================================================================
# WebShop Setup
# ==============================================================================

# Clone WebShop repository
RUN git clone --depth 1 https://github.com/princeton-nlp/WebShop.git /app/webshop

# Install WebShop-specific dependencies
RUN cd /app/webshop && \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir flask gym beautifulsoup4 rank_bm25 thefuzz numpy pandas tqdm

ENV PYTHONPATH="/app/webshop:${PYTHONPATH}"

# ==============================================================================
# Node.js Dependencies
# ==============================================================================

# Copy package files and install dependencies
COPY examples/vercel_ai_webshop/package.json examples/vercel_ai_webshop/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile || pnpm install

# ==============================================================================
# Application Code
# ==============================================================================

# Copy example source code
COPY examples/vercel_ai_webshop/tsconfig.json ./
COPY examples/vercel_ai_webshop/src/ ./src/
COPY examples/vercel_ai_webshop/scripts/ ./scripts/
COPY examples/vercel_ai_webshop/agl/ ./agl/
COPY examples/vercel_ai_webshop/server/webshop_server.py ./server/webshop_server.py
COPY examples/vercel_ai_webshop/server/docker-entrypoint.sh ./server/docker-entrypoint.sh

# Make scripts executable
RUN chmod +x scripts/*.sh server/docker-entrypoint.sh

# ==============================================================================
# Runtime Configuration
# ==============================================================================

# Default environment variables
ENV AGENT_LIGHTNING_STORE_HOST=0.0.0.0 \
    AGENT_LIGHTNING_STORE_PORT=4747 \
    WEBSHOP_URL=http://127.0.0.1:3000 \
    N_RUNNERS=1

# Expose ports
# - 3000: WebShop server
# - 4747: Agent Lightning Store
EXPOSE 3000 4747

# Volume for WebShop dataset persistence
VOLUME /app/webshop/data

# Health check for the Store server
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4747/v1/agl/health || exit 1

# Default: run full stack with qwen config
CMD ["bash", "scripts/run_stack.sh", "qwen"]
