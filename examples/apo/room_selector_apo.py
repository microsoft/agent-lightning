# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to use an existing APO algorithm to tune the prompts."""

import logging
import os
from typing import Tuple, cast

from openai import AsyncOpenAI
from room_selector import RoomSelectionTask, load_room_tasks, prompt_template_baseline, room_selector

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset


def load_train_val_dataset() -> Tuple[Dataset[RoomSelectionTask], Dataset[RoomSelectionTask]]:
    """
    加载房间选择任务数据集，并将其分割为训练集和验证集
    
    返回:
        Tuple[Dataset[RoomSelectionTask], Dataset[RoomSelectionTask]]: 
            (训练数据集, 验证数据集)的元组
    """
    # 从room_tasks.jsonl文件加载完整数据集
    dataset_full = load_room_tasks()
    
    # 将数据集对半分割：前半部分作为训练集，后半部分作为验证集
    train_split = len(dataset_full) // 2
    dataset_train = [dataset_full[i] for i in range(train_split)]
    dataset_val = [dataset_full[i] for i in range(train_split, len(dataset_full))]
    
    return cast(Dataset[RoomSelectionTask], dataset_train), cast(Dataset[RoomSelectionTask], dataset_val)


def setup_apo_logger(file_path: str = "apo.log") -> None:
    """
    为APO算法设置日志记录器，将所有日志输出到指定文件
    
    参数:
        file_path: 日志文件路径，默认为"apo.log"
    """
    # 创建文件处理器，将日志写入文件
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式：时间戳、日志级别、进程ID、日志器名称、消息内容
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    
    # 为APO算法的日志器添加文件处理器
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def main() -> None:
    """
    主函数：配置并启动APO训练流程
    
    训练流程包括：
    1. 初始化OpenAI客户端
    2. 配置APO算法参数（beam search、批次大小等）
    3. 创建Trainer并设置初始资源（基线提示词模板）
    4. 加载训练和验证数据集
    5. 开始训练，优化提示词模板
    """
    # 设置Agent-Lightning的基础日志配置
    setup_logging()
    # 为APO算法单独设置日志记录器
    setup_apo_logger()

    # 初始化OpenAI异步客户端，用于APO算法中的LLM调用
    openai_client = AsyncOpenAI()

    # 创建APO（Asynchronous Prompt Optimization）算法实例
    # APO算法通过beam search和梯度优化来改进提示词模板
    algo = APO[RoomSelectionTask](
        openai_client,              # OpenAI客户端，用于LLM推理
        val_batch_size=10,          # 验证批次大小：每次验证时评估10个样本
        gradient_batch_size=4,      # 梯度批次大小：每次计算梯度时使用4个样本
        beam_width=2,               # Beam search宽度：每轮保留2个最佳提示词候选
        branch_factor=2,            # 分支因子：每个提示词生成2个变体
        beam_rounds=2,              # Beam search轮数：进行2轮搜索优化
        _poml_trace=True,           # 启用POML（Prompt Optimization Markup Language）追踪
    )
    
    # 创建训练器，管理整个训练流程
    trainer = Trainer(
        algorithm=algo,             # 使用上面配置的APO算法
        # 增加runner数量可以并行运行更多rollout，加快训练速度
        n_runners=8,                # 并行运行的runner数量：8个worker并行执行rollout
        
        # APO算法需要一个基线提示词模板作为起始点
        # 可以在这里设置，也可以在算法中设置
        initial_resources={
            # 资源键名可以是任意字符串，只要与agent函数中的参数名匹配即可
            "prompt_template": prompt_template_baseline()  # 基线提示词模板
        },
        
        # APO算法需要一个适配器来处理rollout产生的traces（追踪信息）
        # TraceToMessages适配器将spans（追踪片段）转换为消息格式，供算法分析
        adapter=TraceToMessages(),
    )
    
    # 加载训练和验证数据集
    dataset_train, dataset_val = load_train_val_dataset()
    
    # 开始训练
    # agent: 要训练的智能体函数（room_selector）
    # train_dataset: 训练数据集，用于优化提示词
    # val_dataset: 验证数据集，用于评估优化后的提示词性能
    trainer.fit(agent=room_selector, train_dataset=dataset_train, val_dataset=dataset_val)


if __name__ == "__main__":
    main()
