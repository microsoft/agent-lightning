FROM python:3.12-slim

# Install uv for fast dependency resolution.
RUN pip install uv

WORKDIR /src
COPY . /src

# Install agl-lite with controller extras.
RUN uv pip install --system ".[controller]"

# Default: serve. Override CMD for controller.
CMD ["agl-lite", "serve", "--host", "0.0.0.0", "--port", "8080"]
