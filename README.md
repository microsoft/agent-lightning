
# Training Youtu-agent with AgentLightning

Modifications are made to the original **[AgentLightning@v0.2.2](https://github.com/microsoft/agent-lightning/releases/tag/v0.2.2)** for:
* scaling up training with more agent runners;
* fixing bugs when bridged with [Youtu-agent](https://github.com/TencentCloudADP/youtu-agent/tree/rl/agl);
* correcting GRPO advantage estimation for multi-turn trajectories;
* stabilizing RL training with tricks (e.g., filtering).

## Quick Start

Clone the project and install verl 0.5.0 and agentlightning:
```
pip install verl==0.5.0
cd agent-lightning
pip install -e ./
```


## Experimental Settings
We provide two examples that show how to train your LLM under the [**Youtu-Agent**]([Youtu-agent](https://github.com/TencentCloudADP/youtu-agent/tree/rl/agl)) with [**Agent Lightning**](https://github.com/microsoft/agent-lightning/tree/contrib/youtu-agent-lightning).


### ReTool

ReTool challenges LLMs with tool-integrated reasoning where a Python code interpretor is provided to solve mathematic problems [https://github.com/ReTool-RL/ReTool].

* Training Dataset 🤗 [https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k]
* Testing Dataset 🤗 [https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024]
* Sandbox Service ⌨️ [https://github.com/bytedance/SandboxFusion]

### ASearcher 

ASearcher challenges LLMs with web/database information retrieval capabilities where a local wiki retrieval/search API tool is provided to solve complicated, multi-hop questions [https://github.com/inclusionAI/ASearcher].


* Training Dataset 🤗 [https://huggingface.co/datasets/inclusionAI/ASearcher-train-data]
* Testing Dataset 🤗 [https://huggingface.co/datasets/inclusionAI/ASearcher-test-data]
* Retrieval Service 🔍 [https://github.com/inclusionAI/ASearcher/blob/main/scripts/launch_local_server.sh]


## Dataset Preparation

### ReTool

* Download all the necessary datasets used in ReTool from the huggingface websites.


### SearchR1

* Download all the necessary datasets used in ASearcher from the huggingface websites.
* Run the following script:
```
bash examples_train_w_youtu/search_r1_youtu/data_preprocess/run_preprocess.sh
```

## Training and Validation

### ReTool
For detailed testing, please refer to this directory `examples_train_w_youtu/retool-youtu`.
* For 7B model, we recommend at least 32 GPUs with 96GB memory.

#### Debugging and Testing

* Deployment of vLLM service

**Prerequisites:** Before starting the agent, please ensure that you have installed youtu-agent, and that the `retool` directory from `examples_train_w_youtu/retool-youtu/retool` is placed in `youtu-agent/configs/agents/retool`.

```bash
# launch vLLM backend server
export BASE_MODEL="YOUR_MODEL_PATH"
bash vllm_deploy.sh

# run the agent code
# You must launch the sandbox server first! (SandBoxFusion)
export CODESNIP_SERVER_URL="YOUR_SANDBOX_URL"
python calc_sandbox_agent_youtu.py
```

* Deployment of Store and Runner service
1. Store

```bash
agl store --port 9999
```

2. Runner

```bash
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner python train_calc_sandbox_agent.py --external-store-address http://localhost:9999 --n-runners 10
```

#### Training

* For single node test:
```
bash scripts/restart_ray.sh
bash examples_train_w_youtu/retool-youtu/run_qwen2.5_7b_single_node.sh
```

* For multi-node (e.g., 4 nodes with 32 GPUs)
```
bash run_ray.sh examples_train_w_youtu/retool-youtu/run_qwen2.5_7b.sh
```

### SearchR1
* For 3B model, we recommend at least 2 GPUs with 96GB memory.
* For 32B model, we recommend at least 32 GPUs with 96GB memory.
* Please modify the number of nodes and number of GPUs in the `examples_train_w_youtu/search_r1-youtu/train_search_agent.py`.
* Make sure all the environment variables mentioned below are properly set.
* It is noted that for reward score, we use both rule-based (exact-match) and llm-based (llm-as-a-judge) scoring techniques. Therefore, a llm service (openai-compatible) URL should be prepared in advance.


```
# 3B model
bash run_ray.sh examples_train_w_youtu/search_r1_youtu/trainer3b_utu_onpolicy.sh

# 32B model
bash run_ray.sh examples_train_w_youtu/search_r1_youtu/trainer32b_utu_onpolicy.sh
```


## Training Dynamics Curve

We provide the training dynamics for reference.

### ReTool

https://api.wandb.ai/links/1275747829-fudan-university/yboqje53

### SearchR1

https://api.wandb.ai/links/yuleiqin-tencent/0e2hs7io


# Acknowledgement

We sincerely appreciate the efforts from the following projects:

* Youtu-agent
```
@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/youtu-agent}},
}
```

* AgentLightning
```
@misc{luo2025agentlightningtrainai,
      title={Agent Lightning: Train ANY AI Agents with Reinforcement Learning},
      author={Xufang Luo and Yuge Zhang and Zhiyuan He and Zilong Wang and Siyun Zhao and Dongsheng Li and Luna K. Qiu and Yuqing Yang},
      year={2025},
      eprint={2508.03680},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.03680},
}
```

* VeRL
```
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

