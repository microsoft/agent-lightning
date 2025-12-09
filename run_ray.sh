#!/usr/bin/env bash

TRAIN_SCRIPT=${1-"examples/search_r1_youtu/trainer32b_utu_onpolicy.sh"}
MASTER_PORT=${2-6379}

SESSION=deploy
window_name=store

# 1. 如果同名 session 不存在就新建一个 detached 的
tmux has-session -t "$SESSION" 2>//null || \
    tmux new-session -d -s "$SESSION"

ray stop
ray stop
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"

export VLLM_USE_V1=1
export VLLM_USE_V1=1
export RAY_DEBUG=legacy 
export HYDRA_FULL_ERROR=1

source ~/miniconda3/bin/activate
conda activate agent-lightning

echo $PWD
NODE_LIST=${NODE_LIST}
GPUS_PER_NODE=${GPU_NUM_PER_NODE}
NNODES=${NODE_NUM}
NODE_RANK=${INDEX}

if [ "$GPUS_PER_NODE" = "" ]; then
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi

if [ "$NNODES" = "" ]; then
    NNODES=1
fi

if [ "$NODE_RANK" = "" ]; then
    NODE_RANK=0
fi

echo "GPUS_PER_NODE=${GPUS_PER_NODE}, NNODES=${NNODES}, NODE_RANK=${NODE_RANK}"

MASTER_ADDR=${MASTER_ADDR}
if [ "${MASTER_ADDR}" = "" ]; then
    export MASTER_ADDR="127.0.0.1"
fi

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# launch the master node of ray in container
echo "Now, running on node index $NODE_RANK"
# 设置内部字段分隔符为逗号
IFS=','

# 将字符串分割成数组
if [ "${NNODES}" = 1 ]; then
    NODE_SUBADDR_IP="127.0.0.1"
    echo "CURRENT IP ADDRESS=${NODE_SUBADDR_IP}"

else
    read -ra NODE_SUBLIST <<< "${NODE_LIST}"
    NODE_SUBADDR=${NODE_SUBLIST[${NODE_RANK}]}
    NODE_SUBADDR_IP="${NODE_SUBADDR%:*}"
    echo "CURRENT IP ADDRESS=${NODE_SUBADDR_IP}"

fi

SUBMIT_MASTER_PORT="8265"
export RAY_ADDRESS="http://127.0.0.1:${SUBMIT_MASTER_PORT}"


# —— 以下代码已经在 tmux session 里 ——
# tmux new-window -d -t "$SESSION" -n "$window_name"
# tmux send-keys -t "$SESSION:$window_name" 'source ~/miniconda3/bin/activate' C-m
# tmux send-keys -t "$SESSION:$window_name" 'conda activate agent-lightning' C-m
# tmux send-keys -t "$SESSION:$window_name" 'agl store --port 9999' C-m
# sleep 5
# echo "[$window_name] 窗口已启动，脚本继续..."


if [ "${NODE_RANK}" != "0" ]; then
    # if you want to launch ray on more nodes, use
    echo "Start NODE RANK $NODE_RANK"
    ray start --address=${MASTER_ADDR}:${MASTER_PORT} --node-ip-address=${NODE_SUBADDR_IP} --num-gpus=${GPUS_PER_NODE}
    sleep 5
else
    echo "Start MASTER NODE RANK $NODE_RANK"
    ray start --head --node-ip-address=${MASTER_ADDR} --port=${MASTER_PORT} --dashboard-host=0.0.0.0 --dashboard-port=${SUBMIT_MASTER_PORT} --num-gpus=${GPUS_PER_NODE}
    sleep 5
fi

if [ "$NNODES" = "1" ]; then
    echo "Start single-node ray submit"
    ray job submit --address="http://127.0.0.1:${SUBMIT_MASTER_PORT}" -- /bin/bash ${TRAIN_SCRIPT}
    #  ${MASTER_ADDR} ${MASTER_PORT} ${NODE_RANK}

else
    echo "Start multi-node ray submit"
    if [ "${NODE_RANK}" = "0" ]; then
        echo "only submit multi-node training from the master"
        ray job submit --address="http://127.0.0.1:${SUBMIT_MASTER_PORT}" -- /bin/bash ${TRAIN_SCRIPT}
        #  ${MASTER_ADDR} ${MASTER_PORT} ${NODE_RANK}
    else
        echo "other nodes waiting"
        echo "START GPUS LOADING"
    fi
fi

sleep 365d
