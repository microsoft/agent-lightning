# This setup is based on CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.56.1
pip install wandb
pip install vllm==0.10.2
pip install verl==0.5.0
pip install click==8.2.1
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder flash_attn==2.8.3+pt2.8.0cu126
pip install -e .[dev]

pip install 'openai-agents[litellm]'==0.2.9
pip install -U "autogen-agentchat" "autogen-ext[openai]"
