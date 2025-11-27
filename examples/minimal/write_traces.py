# Copyright (c) Microsoft. All rights reserved.

"""示例：通过 OpenTelemetry 或 AgentOpsTracer 将 traces 写入 LightningStore。

该示例可以在使用或不使用 Lightning Store 服务器的情况下运行。
当运行服务器时，traces 将通过 OTLP 端点写入服务器。

在使用 `--use-client` 标志运行此示例之前，请先启动启用了 OTLP 的 LightningStore 服务器：

```bash
agl store --port 45993 --log-level DEBUG
```
"""

import argparse
import asyncio
import time
from typing import Sequence

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import AgentOpsTracer, LightningStoreClient, OtelTracer, Span, emit_reward, setup_logging
from agentlightning.store import InMemoryLightningStore

# 用于美化输出的控制台对象
console = Console()


async def send_traces_via_otel(use_client: bool = False):
    """
    使用 OpenTelemetry Tracer 发送 traces 到 LightningStore。
    1. 创建 OpenTelemetry tracer
    2. 选择使用内存存储 或 远程客户端
    3. 创建 rollout 和 trace context
    4. 创建嵌套的 spans
    5. 发送奖励信号
    6. 验证 traces 的正确性
    
    Args:
        use_client: 如果为 True，使用远程 LightningStoreClient；否则使用内存存储
    """
    # 创建 OpenTelemetry tracer 实例
    tracer = OtelTracer()
    
    # 根据参数选择存储方式：内存存储或远程客户端
    if not use_client:
        store = InMemoryLightningStore()  # 内存存储，用于本地测试
    else:
        store = LightningStoreClient("http://localhost:45993")  # 远程客户端，连接到服务器
    
    # 启动一个新的 rollout（一次完整的运行尝试）
    rollout = await store.start_rollout(input={"origin": "write_traces_example"})

    # tracer.lifespan 管理 tracer 的生命周期，确保资源正确清理
    with tracer.lifespan(store):
        # 初始化单个 rollout 的单个 trace 捕获
        # trace_context 创建一个新的 trace 上下文，用于组织相关的 spans
        async with tracer.trace_context(
            "trace-manual", store=store, rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ) as tracer:
            # 创建第一个 span（根 span）
            with tracer.start_as_current_span("grpc-span-1"):
                time.sleep(0.01)

                # 创建嵌套的 span（grpc-span-2 是 grpc-span-1 的子 span）
                with tracer.start_as_current_span("grpc-span-2"):
                    time.sleep(0.01)

            # 创建另一个与 grpc-span-1 同级的 span
            with tracer.start_as_current_span("grpc-span-3"):
                time.sleep(0.01)

            # 发送奖励信号，这会创建一个 reward span
            emit_reward(1.0)

    # 查询并打印所有 traces
    traces = await store.query_spans(rollout_id=rollout.rollout_id)
    console.print(traces)

    # 快速验证 traces 的正确性
    assert len(traces) == 4  # 应该包含 3 个 grpc spans 和 1 个 reward span
    span_names = [span.name for span in traces]
    assert "grpc-span-1" in span_names
    assert "grpc-span-2" in span_names
    assert "grpc-span-3" in span_names
    assert "agentlightning.reward" in span_names

    # 验证最后一个 span 是 reward span，且奖励值为 1.0
    last_span = traces[-1]
    assert last_span.name == "agentlightning.reward"
    # 注意：尽量不要依赖此属性，它可能在将来发生变化
    # 使用 agentlightning.emitter 中的工具来获取奖励值
    assert last_span.attributes["reward"] == 1.0

    # 如果使用客户端模式，验证资源属性中包含了 rollout_id 和 attempt_id
    if use_client:
        # 使用客户端时，资源应该设置了 rollout_id 和 attempt_id
        for span in traces:
            assert "agentlightning.rollout_id" in span.resource.attributes
            assert "agentlightning.attempt_id" in span.resource.attributes

    # 如果是客户端连接，需要关闭连接
    if isinstance(store, LightningStoreClient):
        await store.close()


async def send_traces_via_agentops(use_client: bool = False):
    """
    使用 AgentOpsTracer 发送 traces 到 LightningStore。
    
    该函数演示了如何：
    1. 使用 AgentOpsTracer 自动捕获 OpenAI API 调用
    2. 创建 rollout 和 trace context
    3. 在 trace context 中调用 OpenAI API（会自动被捕获为 span）
    4. 验证捕获的 traces
    
    Args:
        use_client: 如果为 True，使用远程 LightningStoreClient；否则使用内存存储
    """
    # 创建 AgentOpsTracer 实例（会自动捕获 OpenAI 等 API 调用）
    tracer = AgentOpsTracer()
    
    # 根据参数选择存储方式
    if not use_client:
        store = InMemoryLightningStore()
    else:
        store = LightningStoreClient("http://localhost:45993")
    
    # 启动一个新的 rollout
    rollout = await store.start_rollout(input={"origin": "write_traces_example"})

    # 初始化 tracer 的生命周期
    # 一个生命周期可以包含多个 traces
    with tracer.lifespan(store):
        # 初始化单个 rollout 的单个 trace 捕获
        async with tracer.trace_context(
            "trace-1", rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ):
            # 在 trace context 中调用 OpenAI API
            # AgentOpsTracer 会自动捕获这个调用并创建相应的 span
            openai_client = AsyncOpenAI()
            response = await openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, what's your name?"},
                ],
            )
            console.print(response)
            # 验证响应内容
            assert response.choices[0].message.content is not None
            assert "chatgpt" in response.choices[0].message.content.lower()

    # 查询并打印所有 traces
    traces = await store.query_spans(rollout_id=rollout.rollout_id)
    console.print(traces)
    
    # 验证 AgentOps traces 的正确性
    await _verify_agentops_traces(traces, use_client=use_client)
    
    # 如果是客户端连接，需要关闭连接
    if isinstance(store, LightningStoreClient):
        await store.close()


async def _verify_agentops_traces(spans: Sequence[Span], use_client: bool = False):
    """
    验证 AgentOps traces 的正确性。
    
    该函数检查捕获的 spans 是否包含预期的属性和结构。
    预期的 traces 应该类似于：
    
    ```python
    Span(
        rollout_id='ro-ef9ff8a429d1',
        attempt_id='at-37cc5f24',
        sequence_id=1,
        trace_id='b3a16b603f7805934215d467e717c9e7',
        span_id='2782d5d750f49b2d',
        parent_id='2fb97c818363bce3',
        name='openai.chat.completion',
        status=TraceStatus(status_code='OK', description=None),
        attributes={
            'gen_ai.request.type': 'chat',
            'gen_ai.system': 'OpenAI',
            'gen_ai.request.model': 'gpt-4.1-mini',
            'gen_ai.request.streaming': False,
            'gen_ai.prompt.0.role': 'system',
            'gen_ai.prompt.0.content': 'You are a helpful assistant.',
            'gen_ai.prompt.1.role': 'user',
            'gen_ai.prompt.1.content': "Hello, what's your name?",
            'gen_ai.response.id': 'chatcmpl-Cc1osPWiArOwCS8nUkp0kZuZPkpY4',
            'gen_ai.response.model': 'gpt-4.1-mini-2025-04-14',
            'gen_ai.completion.0.role': 'assistant',
            'gen_ai.completion.0.content': "Hello! I'm ChatGPT, your AI assistant. How can I help you today?",
        },
        resource=OtelResource(
            attributes={
                'agentops.project.id': 'temporary',
                'agentlightning.rollout_id': 'ro-ef9ff8a429d1',
                'agentlightning.attempt_id': 'at-37cc5f24'
            },
            schema_url=''
        )
    )
    ```
    
    Args:
        spans: 要验证的 span 序列
        use_client: 如果为 True，验证客户端模式下的额外属性
    """
    # 应该包含 2 个 spans：一个 session span 和一个 OpenAI completion span
    assert len(spans) == 2
    
    for span in spans:
        if span.name == "openai.chat.completion":
            # 验证 OpenAI completion span 的属性
            assert span.attributes["gen_ai.request.model"] == "gpt-4.1-mini"
            assert span.attributes["gen_ai.request.streaming"] == False
            assert span.attributes["gen_ai.prompt.0.role"] == "system"
            assert span.attributes["gen_ai.prompt.0.content"] == "You are a helpful assistant."
            assert span.attributes["gen_ai.prompt.1.role"] == "user"
            assert span.attributes["gen_ai.prompt.1.content"] == "Hello, what's your name?"
            # 验证响应内容中包含 "chatgpt"
            assert "chatgpt" in span.attributes["gen_ai.completion.0.content"].lower()  # type: ignore
            
            # 如果使用客户端，验证资源属性
            if use_client:
                assert "agentlightning.rollout_id" in span.resource.attributes
                assert "agentlightning.attempt_id" in span.resource.attributes
        else:
            # 验证 session span（trace context 创建的根 span）
            assert "trace-1" in span.name
            assert span.attributes["agentops.span.kind"] == "session"


def main():
    """
    主函数：解析命令行参数并运行相应的示例。
    
    使用方式：
        python write_traces.py otel              # 使用 OpenTelemetry tracer，内存存储
        python write_traces.py agentops          # 使用 AgentOpsTracer，需要 OpenAI API key
        python write_traces.py otel --use-client  # 使用 OpenTelemetry tracer，远程客户端
        python write_traces.py agentops --use-client  # 使用 AgentOpsTracer，远程客户端
    """
    # 设置日志级别为 DEBUG，便于查看详细的运行信息
    setup_logging("DEBUG")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["otel", "agentops"])
    # --use-client 标志：如果设置，使用远程客户端而不是内存存储
    parser.add_argument("--use-client", action="store_true")
    args = parser.parse_args()

    if args.mode == "otel":
        asyncio.run(send_traces_via_otel(use_client=args.use_client))
    elif args.mode == "agentops":
        asyncio.run(send_traces_via_agentops(use_client=args.use_client))
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
