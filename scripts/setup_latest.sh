set -ex
python -m pip install --upgrade pip
pip install -e .[dev,agent]
pip install -U agentops
