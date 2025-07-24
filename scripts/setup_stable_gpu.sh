set -ex
sudo apt update

python -m pip install --upgrade pip

pip install packaging ninja numpy pandas ipython ipykernel gdown
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.1 --no-build-isolation
pip install vllm==0.9.2

git clone https://github.com/volcengine/verl
cd verl
pip install -e .
cd ..

pip install -e .[dev,agent]
