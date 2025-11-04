set -ex
python -m pip install --upgrade --no-cache-dir pip

# CPU version full installation

pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
pip install --no-cache-dir vllm  # pytorch auto installed when installing vllm
pip install --no-cache-dir --no-deps trl unsloth
pip install --no-cache-dir verl==0.5.0

pip install --no-cache-dir -e .[dev,agent,apo]
