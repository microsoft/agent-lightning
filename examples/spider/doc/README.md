## 训练与日志系统变更说明

本文件总结了近期对 Spider 训练脚本与日志的所有改动，便于复现和日常使用。

### 1. 主要改动概览
- **外部配置支持**：`train_sql_agent.py` 支持 `--config-file <path>` 读取 JSON/YAML 配置，默认的 0.5B 配置迁移到 `examples/spider/configs/local_qwen05.json`。位置参数 `config` 仍需提供（argparse 限制），但实际超参以配置文件为准。
- **每次训练生成独立运行目录**：`examples/spider/log/<timestamp>_config_<label>/`
  - `config.json`：最终训练配置（包含注入的日志路径）
  - `progress.txt`：带时间戳的训练/验证进度、ETA（窗口/累计步时）、吞吐、reward 等
  - `sequence_lengths.csv`：prompt/response 长度 mean/p50/p95/max
  - `hardware.txt`：硬件快照（GPU/CPU/CUDA 等）
  - `objectives.txt`：目标/约束占位（可用环境变量覆盖）
  - `gpu_monitor.csv` / `cpu_monitor.csv`：启动后约 3 分钟的采样
  - `error.txt`：异常信息
- **进度/验证日志改进**：
  - 训练行带时间戳、窗口/累计步时、平滑 ETA。
  - 验证触发时写入 `[val]` 行，优先记录 `val-core/*` 指标。
  - prompt/response 长度分位数写入 `sequence_lengths.csv`。
- **本地启动脚本**：`run_train_local.sh`
  - 支持 nohup 后台运行，日志重定向到 `examples/spider/log/train_<config_stem>_<timestamp>.log`，终端关闭不影响训练。
  - 硬编码 `CONFIG_PATH`（默认 `/home/lthpc/student/LiTengfei/project/myfork/agent-lightning/examples/spider/configs/singleGPU_qwen05b.json`），直接修改这一行即可切换配置。
  - 启动时记录环境信息（时间、主机、用户、CWD、CUDA_VISIBLE_DEVICES、Python 版本、Git 分支/提交）。

### 2. 文件与路径
- 训练脚本：`examples/spider/train_sql_agent.py`
- 默认配置文件：`examples/spider/configs/local_qwen05.json`
- 示例配置（自定义）：`examples/spider/configs/singleGPU_qwen05b.json` 等
- 启动脚本：`run_train_local.sh`
- 运行产物：`examples/spider/log/<timestamp>_config_<label>/`

### 3. 使用方法
#### 命令行直接运行
```bash
# 使用自定义配置文件（推荐）
python train_sql_agent.py local_qwen05 --config-file examples/spider/configs/local_qwen05.json
```
- 位置参数 `local_qwen05` 仅作为标签/默认命名；如未提供 `--config-file`，会调用内置配置函数。
- 其他预设：`fast`/`qwen`/`llama`/`npu` 仍可用。

#### 使用脚本（后台运行，日志重定向）
```bash
./run_train_local.sh
```
- 在脚本顶部修改 `CONFIG_PATH` 即可切换配置。
- 运行后输出 PID、配置路径、日志路径；日志文件位于 `examples/spider/log/train_<config_stem>_<timestamp>.log`。

### 4. 环境变量
- `LOCAL_QWEN_MODEL_PATH`：覆盖模型权重路径（优先级高于配置文件）。
- `LOCAL_QWEN05_CONFIG_FILE`：替换默认 0.5B 配置文件路径（当使用 `config_train_local_qwen05` 时）。
- `RUN_OBJECTIVE` / `OOM_POLICY`：写入 `objectives.txt` 说明目标与 OOM 容忍度。
- `CUDA_VISIBLE_DEVICES`：可在脚本或运行命令中设置 GPU。

### 5. 日志格式示例
- 训练行：
```
[2025-12-14 20:16:22] [train] step 32/7000 | eta 21:59:51 | step_win 3.20s | step_avg 3.25s | toks/s 12.3k (12.3k/gpu) | reward 0.12 | lr 1e-06
```
- 验证行：
```
[2025-12-14 20:16:22] [val] step 32/7000 | val-core/<metric>=<value>
```

### 6. 注意事项
- `config` 位置参数仍必填（argparse choices），但实际超参以 `--config-file` 或默认外部配置文件为准。
- 监控采样默认约 3 分钟，1 秒间隔；如需调整可修改 `start_system_monitors`。
- 新增配置文件建议放在 `examples/spider/configs/`，并在脚本或命令中指向它。

### 7. 主要修改文件列表
- `examples/spider/train_sql_agent.py`
- `examples/spider/configs/local_qwen05.json`
- `agentlightning/verl/trainer.py`
- `run_train_local.sh`
