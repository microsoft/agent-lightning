# Agent image for calc_x example.
# Build context: examples/calc_x/
#   minikube image build -t calc-x-agent:dev -f Dockerfile examples/calc_x/

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    openai \
    httpx \
    sympy \
    "autogen-agentchat" \
    "autogen-ext[openai]" \
    "mcp>=1.10.0" \
    mcp-server-calculator

COPY calc_agent.py eval_utils.py /app/

WORKDIR /app

RUN python -m compileall -q "$(python -c 'import site; print(site.getsitepackages()[0])')" /app
RUN python -c "import mcp_server_calculator" \
    && timeout 3 python -m mcp_server_calculator; \
    status=$?; \
    if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then exit "$status"; fi