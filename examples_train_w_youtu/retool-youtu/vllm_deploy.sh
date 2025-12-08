MODEL_PATH=$BASE_MODEL
PORT=8033
HOST="0.0.0.0"
TP_SIZE=1
GPU_MEMORY_UTIL=0.9
MAX_MODEL_LEN=16384
export VLLM_LOGGING_LEVEL=DEBUG
python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --disable-log-requests \
  --served-model-name "qwen"


