# Copyright (c) Microsoft. All rights reserved.
"""
会议室选择系统 - 使用Agent Lightning框架实现的智能会议室选择代理
该系统可以根据用户需求（日期、时间、人数、设备需求等）选择最合适的会议室
"""

import asyncio
import json
import os
import traceback
from typing import List, Optional, Tuple, TypedDict, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field
from rich.console import Console

from agentlightning.adapter import TraceToMessages
from agentlightning.litagent import rollout
from agentlightning.reward import find_final_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import Dataset, PromptTemplate

console = Console()


class JudgeResponse(BaseModel):
    """评分响应模型 - 用于评估代理选择的会议室是否正确"""
    reason: str = Field(description="The reason for the score. No more than 100 characters.")
    score: float = Field(description="The score for the match on a 0-1 scale. Be critical.")


class Room(TypedDict):
    """会议室基础信息模型"""
    id: str  # 会议室ID
    capacity: int  # 容纳人数
    equipment: List[str]  # 设备列表（如：投影仪、白板等）
    accessible: bool  # 是否无障碍
    distance_m: int  # 距离（米）
    booked: List[Tuple[str, str, int]]  # 已预订时段列表：(日期, 时间, 持续时间分钟)


class RoomStatus(Room):
    """会议室状态模型 - 包含是否空闲的信息"""
    free: bool  # 当前时段是否空闲


class AvailableRooms(TypedDict):
    """可用会议室列表模型"""
    rooms: List[RoomStatus]  # 会议室状态列表


class RoomRequirement(TypedDict):
    """会议室需求模型 - 用户提出的会议室选择要求"""
    date: str  # 日期 YYYY-MM-DD
    time: str  # 时间 HH:MM (24小时制)
    duration_min: int  # 会议持续时间（分钟）
    attendees: int  # 参会人数
    needs: List[str]  # 需要的设备列表
    accessible_required: bool  # 是否需要无障碍设施


class RoomSelectionTask(TypedDict):
    """会议室选择任务模型 - 包含任务输入和期望结果"""
    id: str  # 任务ID
    task_input: RoomRequirement  # 任务输入（用户需求）
    expected_choice: str  # 期望选择的会议室ID（用于评估）


# 工具定义 - 定义代理可以调用的工具（函数）
# 代理可以通过这些工具查询会议室信息
TOOL_DEFINITIONS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_rooms_and_availability",  # 获取会议室及其可用性
            "description": "Return meeting rooms with capacity, equipment, accessibility, distance, and booked time slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM 24h local"},
                    "duration_min": {"type": "integer", "description": "Meeting duration minutes"},
                },
                "required": ["date", "time", "duration_min"],  # 必需参数
            },
        },
    },
]


def prompt_template_baseline() -> PromptTemplate:
    """返回基础提示模板
    这个模板可以通过Agent Lightning的APO算法进行优化
    """
    return PromptTemplate(
        template="Find a room on {date} at {time} for {duration_min} minutes, {attendees} attendees. Needs: {needs}. Accessible required: {accessible_required}",
        engine="f-string",  # 使用f-string引擎进行模板格式化
    )


def room_selection_grader(client: OpenAI, final_message: Optional[str], expected_choice: str) -> float:
    """评分函数 - 评估代理选择的会议室是否正确
    
    Args:
        client: OpenAI客户端实例
        final_message: 代理的最终输出消息（包含选择的会议室）
        expected_choice: 期望选择的会议室ID
    
    Returns:
        评分分数 (0-1之间，1表示完全正确)
    """
    # 构建评分提示词
    judge_prompt = (
        f"You are a strict grader of exact room choice."
        f"Task output:\n{final_message}\n\n"
        f"Task expected answer:\n{expected_choice}\n\n"
        f"Score the match on a 0-1 scale. Be critical.\n"
        f"Bear in mind that the score can be partially correct (between 0 and 1)."
    )
    # 使用LLM作为评分者，评估代理的答案
    judge = client.chat.completions.parse(
        model="deepseek-chat",  # 使用DeepSeek模型
        messages=[
            {"role": "user", "content": judge_prompt},
        ],
        response_format=JudgeResponse,  # 使用结构化输出格式
        temperature=0.0,  # 设置为0以确保结果一致性
    )

    judge_result = judge.choices[0].message.content
    console.print(f"[bold yellow]=== Judge ===[/bold yellow]")
    console.print(judge_result)

    # 解析评分结果
    judge_result_parsed = JudgeResponse.model_validate_json(judge_result)  # type: ignore

    console.print(f"[bold yellow]=== Judge Score ===[/bold yellow]")
    console.print(judge_result_parsed.score)
    return judge_result_parsed.score


@rollout
def room_selector(task: RoomSelectionTask, prompt_template: PromptTemplate) -> float:
    """会议室选择代理 - 根据给定需求选择最合适的会议室
    
    这是使用@rollout装饰器标记的主代理函数，Agent Lightning框架会追踪其执行过程。
    
    工作流程：
    1. 根据任务输入格式化用户提示
    2. 调用LLM进行初始响应
    3. 如果LLM调用工具，执行工具调用并获取会议室信息
    4. 将工具结果返回给LLM，让LLM做出最终选择
    5. 使用评分函数评估选择的正确性
    
    Oracle System Prompt (适用于gpt-5 mini低推理模式，准确率100%):
        系统提示：你是一个调度助手。
        硬约束：时间段空闲、容量>=参会人数、包含所有必需设备、如需无障碍则accessible==True
        平局评分（越小越好）：
        1) capacity_slack = capacity - attendees (最小化)
        2) extra_equipment = 提供设备数 - 必需设备数 (最小化)
        3) distance_m (最小化)
        4) 当天预订时段总数 (最小化)
        如果没有满足约束的会议室，返回"No Room"
        严格返回格式：
        final_choice: <ROOM_ID>
        reason: <一行说明决定性标准>

    当前实现大大简化了oracle提示，提示模板由参数提供。
    提示模板应该通过Agent Lightning的APO算法进行优化。
    也应该能在非常小的模型（如gpt-4.1-nano）上工作。
    
    Args:
        task: 会议室选择任务（包含用户需求和期望答案）
        prompt_template: 提示模板（用于生成用户消息）
    
    Returns:
        评分分数 (0-1之间)
    """
    # 注意：代码中存在一些问题，需要修复：
    # 1. AsyncOpenAI未导入但被使用
    # 2. openai_client定义但后面使用client
    # 3. model变量未定义
    openai_client = AsyncOpenAI(
        base_url="http://127.0.0.1:30003/v1",
        api_key=""
    )

    # 使用提示模板格式化用户消息，填入任务输入的具体值
    user_message = prompt_template.format(**task["task_input"])

    # 构建初始消息列表
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a scheduling assistant."},  # 系统提示
        {
            "role": "user",
            "content": user_message,  # 用户需求
        },
    ]

    console.print(f"[bold yellow]=== User Message ===[/bold yellow]")
    console.print(user_message)

    # 第一次LLM调用 - 代理可能会决定调用工具
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_DEFINITIONS,  # 提供可用的工具定义
        tool_choice="auto",  # 让模型自动决定是否调用工具
        # Minimize the randomness
        temperature=0.0,  # 温度设为0以减少随机性
        # Uncomment for gpt-5
        # reasoning_effort="low",
    )

    console.print(f"[bold yellow]=== Assistant Message ===[/bold yellow]")
    console.print(resp.choices[0].message)

    # 解析和处理工具调用
    tool_calls = resp.choices[0].message.tool_calls
    if tool_calls:
        # 如果代理决定调用工具（查询会议室信息）

        tool_call_params: List[ChatCompletionMessageFunctionToolCallParam] = []
        tool_results: List[ChatCompletionToolMessageParam] = []
        for tc in tool_calls:
            # 验证工具调用类型
            if tc.type != "function":
                raise ValueError(f"Tool call is not a function: {tc}")
            if tc.function.name != "get_rooms_and_availability":
                raise ValueError(f"Tool call is not get_rooms_and_availability: {tc}")
            # 保存工具调用参数（用于消息历史）
            tool_call_params.append(
                ChatCompletionMessageFunctionToolCallParam(
                    id=tc.id,
                    type="function",
                    function={"name": tc.function.name, "arguments": tc.function.arguments},
                )
            )
            # 解析工具调用参数并执行工具函数
            args = json.loads(tc.function.arguments)
            try:
                tool_output = get_rooms_and_availability(args["date"], args["time"], args["duration_min"])
            except Exception as e:
                # 如果工具执行出错，返回错误信息
                tool_output = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            console.print(f"[bold yellow]=== Tool Message ===[/bold yellow]")
            console.print(tool_output)
            # 构建工具响应消息
            tool_results.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tc.id,  # 关联到对应的工具调用
                    content=json.dumps(tool_output),
                )
            )

        # 更新消息列表，添加助手的工具调用和工具响应
        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=resp.choices[0].message.content,
                tool_calls=tool_call_params,
            )
        )
        messages.extend(tool_results)  # 添加工具执行结果

        # 第二次LLM调用 - 基于工具结果做出最终选择
        next_resp = client.chat.completions.create(
            model=model,
            messages=messages,
            # Minimize the randomness
            temperature=0.0,
        )
        console.print(f"[bold yellow]=== Final Assistant Message ===[/bold yellow]")
        console.print(next_resp.choices[0].message.content)
        final_message = next_resp.choices[0].message.content

    else:
        # 如果代理没有调用工具，直接使用第一次响应
        final_message = resp.choices[0].message.content

    # 使用评分函数评估代理的选择是否正确
    return room_selection_grader(client, final_message, task["expected_choice"])


# 本地会议室数据库 - 存储所有会议室的信息
# 可能有多个符合条件的会议室
ROOMS: List[Room] = [
    {
        "id": "Orion",
        "capacity": 4,
        "equipment": ["tv", "whiteboard"],
        "accessible": True,
        "distance_m": 12,
        "booked": [("2025-10-13", "10:00", 60), ("2025-10-13", "15:00", 30)],
    },
    {
        "id": "Lyra",
        "capacity": 10,
        "equipment": ["projector", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 30,
        "booked": [("2025-10-13", "09:30", 30), ("2025-10-13", "11:00", 60)],
    },
    {
        "id": "Vega",
        "capacity": 6,
        "equipment": ["tv"],
        "accessible": False,
        "distance_m": 22,
        "booked": [("2025-10-13", "14:00", 60)],
    },
    {
        "id": "Nova",
        "capacity": 12,
        "equipment": ["ledwall", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 45,
        "booked": [],
    },
    {
        "id": "Quark",
        "capacity": 8,
        "equipment": ["projector", "whiteboard"],
        "accessible": False,
        "distance_m": 18,
        "booked": [("2025-10-13", "10:30", 30)],
    },
    # 添加两个额外房间以创建更难的选择（平局情况）
    {
        "id": "Atlas",
        "capacity": 6,
        "equipment": ["projector", "whiteboard"],
        "accessible": True,
        "distance_m": 10,
        "booked": [("2025-10-13", "09:00", 30), ("2025-10-13", "13:30", 30)],
    },
    {
        "id": "Pulse",
        "capacity": 8,
        "equipment": ["tv", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 8,
        "booked": [("2025-10-13", "16:30", 30)],
    },
]


def overlaps(start: str, dur: int, other_start: str, other_dur: int) -> bool:
    """检查两个时间段是否重叠
    
    Args:
        start: 第一个时间段的开始时间 (HH:MM格式)
        dur: 第一个时间段的持续时间（分钟）
        other_start: 第二个时间段的开始时间 (HH:MM格式)
        other_dur: 第二个时间段的持续时间（分钟）
    
    Returns:
        如果两个时间段重叠返回True，否则返回False
    """
    def tmin(t: str):
        """将时间字符串(HH:MM)转换为分钟数"""
        return int(t[:2]) * 60 + int(t[3:])

    # 将两个时间段转换为分钟数范围
    a0, a1 = tmin(start), tmin(start) + dur
    b0, b1 = tmin(other_start), tmin(other_start) + other_dur
    # 判断是否有重叠：两个区间有重叠当且仅当 max(a0, b0) < min(a1, b1)
    return max(a0, b0) < min(a1, b1)


def get_rooms_and_availability(date: str, time_str: str, duration_min: int) -> AvailableRooms:
    """获取所有会议室及其在指定时段的可用性
    
    这是代理可以调用的工具函数，用于查询会议室信息。
    
    Args:
        date: 查询日期 (YYYY-MM-DD格式)
        time_str: 查询开始时间 (HH:MM格式)
        duration_min: 会议持续时间（分钟）
    
    Returns:
        包含所有会议室及其可用性状态的字典
    """
    avail: List[RoomStatus] = []
    for r in ROOMS:
        # 检查会议室在指定时段是否空闲
        # 如果所有已预订时段都与查询时段不重叠，则空闲
        free = all(
            not (b_date == date and overlaps(time_str, duration_min, b_time, b_dur))
            for (b_date, b_time, b_dur) in r["booked"]
        )
        # 创建会议室状态对象，包含原始信息和空闲状态
        item: RoomStatus = {
            **r,  # 复制原始会议室信息
            "free": free,  # 添加空闲状态
        }
        avail.append(item)
    return {"rooms": avail}


def load_room_tasks() -> Dataset[RoomSelectionTask]:
    """从JSONL文件加载会议室选择任务数据集
    
    Returns:
        包含多个会议室选择任务的数据集
    """
    tasks: List[RoomSelectionTask] = []
    # 读取JSONL文件（每行一个JSON对象）
    for line in open("room_tasks.jsonl"):
        task = json.loads(line)
        tasks.append(RoomSelectionTask(**task))
    return cast(Dataset[RoomSelectionTask], tasks)


async def debug_room_selector(limit: int = 1):
    """调试函数 - 运行会议室选择代理并展示详细执行过程
    
    Args:
        limit: 任务数量限制（当前未使用）
    """
    # 准备运行代理所需的所有组件
    runner = LitAgentRunner[RoomSelectionTask](AgentOpsTracer())  # 创建代理运行器，使用AgentOps追踪器
    store = InMemoryLightningStore()  # 创建内存存储，用于保存执行轨迹
    prompt_template = prompt_template_baseline()  # 获取基础提示模板
    tasks = load_room_tasks()  # 加载任务数据集
    # 在运行上下文中执行代理
    with runner.run_context(agent=room_selector, store=store):
        for task in tasks:
            console.print("[bold green]=== Task ===[/bold green]", task, sep="\n")
            # 运行代理 - 执行一个任务步骤
            rollout = await runner.step(task, resources={"main_prompt": prompt_template})
            # 获取执行轨迹并转换为消息格式
            # 这对于调试和分析非常有用
            spans = await store.query_spans(rollout.rollout_id)  # 查询执行跨度（spans）
            adapter = TraceToMessages()  # 创建适配器，将轨迹转换为消息
            messages = adapter.adapt(spans)  # 转换
            # 打印所有消息（用于事后分析）
            for message_idx, message in enumerate(messages):
                console.print(f"[bold purple]=== Postmortem Message #{message_idx} ===[/bold purple]")
                console.print(json.dumps(message))
            # 查找最终奖励（评分）
            reward = find_final_reward(spans)
            console.print("[bold purple]=== Postmortem Reward ===[/bold purple]", reward, sep="\n")


if __name__ == "__main__":
    # 主入口 - 运行调试函数
    asyncio.run(debug_room_selector())
