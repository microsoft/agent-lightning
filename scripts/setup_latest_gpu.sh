set -ex

python -m pip install --upgrade --no-cache-dir pip

pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
# This has to be pinned for VLLM to work.
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir flash-attn --no-build-isolation
pip install --no-cache-dir vllm

git clone https://github.com/volcengine/verl
cd verl
pip install --no-cache-dir -e .
cd ..

pip install --no-cache-dir -e .[dev,agent]
# Upgrade agentops to the latest version
pip install --no-cache-dir -U agentops
