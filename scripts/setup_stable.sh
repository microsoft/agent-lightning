set -ex
python -m pip install --upgrade --no-cache-dir pip
pip install -e --no-cache-dir .[dev,agent]
