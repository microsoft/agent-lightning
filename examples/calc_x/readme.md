# 1. 先测试mcp环境是否正常可以使用
```bash
python diagnose_mcp.py
```
 2. 测试agent是否正常启用
部署vllm
```bash
bash vllm_deploy.sh
```
禁用代理端口监听
```bash
export http_proxy="http://172.31.255.10:8888"
export https_proxy="http://172.31.255.10:8888"

# 不通过代理的地址列表（注意要写上本地端口和常见内网段）
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export no_proxy="$NO_PROXY"
```
测试rollout
```bash
python cal_agent.py
```
结果
```bash
(verl) root@nb-1431158932796753920-avu5ya6a25mo:/cfs_turbo/zhijianzhou/agent-lightning/examples/calc_x# python calc_agent.py 
Processing request of type ListToolsRequest
Processing request of type ListToolsRequest
Processing request of type CallToolRequest
answer: 180 ground_truth: 180 reward: 1.0
Processing request of type ListToolsRequest
Processing request of type ListToolsRequest
Processing request of type CallToolRequest
answer: 16 ground_truth: 16 reward: 1.0
```
3.测试训练端
```bash
python train_calc_agent.py --train-file /cfs_turbo/zhijianzhou/VERL/agent-lightning/calx_data/train.parquet \
--val-file /cfs_turbo/zhijianzhou/VERL/agent-lightning/calx_data/test.parquet \
--model /cfs_turbo/zhijianzhou/Model/Qwen/Qwen2.5-1.5B-Instruct 
```

# 4. 监控资源使用
查看当前 CPU 和内存占用
```bash
python check_resources.py
```
实时监控训练过程
```bash
# 实时监控（每秒更新）
python monitor_resources.py

# 保存监控数据到 CSV
python monitor_resources.py --output resources.csv
```
详细说明见 [RESOURCE_MONITORING.md](RESOURCE_MONITORING.md)



