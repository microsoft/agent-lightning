set -ex
python -m pip install --upgrade --no-cache-dir pip
pip install --no-cache-dir -e .[dev,agent,apo]
