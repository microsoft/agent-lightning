set -ex

python -m pip install --upgrade --no-cache-dir pip

pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
# This has to be pinned for VLLM to work.
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir flash-attn --no-build-isolation
# This must match pytorch version.
pip install --no-cache-dir vllm==0.11.0
# Latest VERL release version.
pip install --no-cache-dir verl

pip install --no-cache-dir -e .[dev,agent]
# Upgrade agentops to the latest version
pip install --no-cache-dir -U agentops
