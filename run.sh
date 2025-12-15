export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="sk-44019fa179c244b182f1872177bdcf74"
export OPENAI_MODEL="qwen3-4b"

export DASHSCOPE_API_KEY="$OPENAI_API_KEY"
export VERL_SPIDER_DATA_DIR="examples/spider/data"

python examples/spider/sql_agent.py