set -ex
python -m pip install --upgrade --no-cache-dir pip
pip install --no-cache-dir -e .[dev,agent,apo]
# Upgrade agentops to the latest version
pip install --no-cache-dir -U agentops
