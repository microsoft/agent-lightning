set -ex

python -m pip install --upgrade --no-cache-dir pip

pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir flash-attn --no-build-isolation
pip install --no-cache-dir vllm

git clone https://github.com/volcengine/verl
cd verl
pip install -e --no-cache-dir .
cd ..

pip install -e --no-cache-dir .[dev,agent]
# Upgrade agentops to the latest version
pip install --no-cache-dir -U agentops
