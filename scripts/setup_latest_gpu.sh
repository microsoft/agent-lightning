set -ex
sudo apt update
sudo apt install -y wget curl unzip vim

python -m pip install --upgrade pip

pip install packaging ninja numpy pandas ipython ipykernel gdown
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install vllm

git clone https://github.com/volcengine/verl
cd verl
pip install -e .
cd ..

pip install -e .[dev,agent]
