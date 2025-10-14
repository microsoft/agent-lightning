set -ex
python -m pip install --upgrade --no-cache-dir pip

# CPU version full installation
pip install --no-cache-dir -e .[dev,agent,apo,verl]
