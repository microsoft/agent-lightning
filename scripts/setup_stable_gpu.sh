set -ex

python -m pip install --upgrade pip

pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir --no-deps trl unsloth  # For type checking, not for running examples
pip install --no-cache-dir transformers==4.53.3
pip install --no-cache-dir flash-attn==2.8.1 --no-build-isolation
pip install --no-cache-dir vllm==0.9.2
pip install --no-cache-dir verl==0.5.0

pip install --no-cache-dir -e .[dev,agent,apo]
