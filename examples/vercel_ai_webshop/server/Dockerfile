# WebShop Flask Server
FROM python:3.11-slim

# Install system dependencies and Java 21 (required for pyserini)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget gnupg ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb bookworm main" > /etc/apt/sources.list.d/adoptium.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends temurin-21-jdk \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Install Python dependencies (Cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown && \
    python -m spacy download en_core_web_sm

# Clone WebShop Code
RUN git clone --depth 1 https://github.com/princeton-nlp/WebShop.git /app/webshop

# Install WebShop dependencies
RUN cd /app/webshop && \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir flask gym beautifulsoup4 rank_bm25 thefuzz numpy pandas tqdm

ENV PYTHONPATH="/app/webshop:${PYTHONPATH}"

# Copy Server Code & Entrypoint
COPY webshop_server.py .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Expose port and volume for dataset persistence
EXPOSE 3000
VOLUME /app/webshop/data

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "webshop_server.py", "--host", "0.0.0.0", "--port", "3000"]
