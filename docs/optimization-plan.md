# Agent Lightning 优化计划

## 目标

本轮优化按 major version 处理，完全不考虑向后兼容。目标不是新增能力，而是把核心训练闭环收敛到一套职责清晰、状态一致、易读易维护的模型。

必须同时达成以下结果：

1. 所有模块职责划分清晰，层级和依赖方向清楚。
2. 删除历史兼容层，不再为了兼容旧 runtime、旧协议或旧 span 格式做折中。
3. 删除所有 deprecated 项，并把调用点迁移到新实现。
4. 修正正确性和状态一致性问题，尤其是 rollout、attempt 和 span 写入语义。
5. 完成必要的性能、生产化和工程清理。

本计划的执行原则：

- 先定义新模型，再删除旧代码，最后做生产化清理。
- 公共 Interface 必须表达完整语义，包括状态、错误模式、排序规则、资源归属和写入确认。
- 不保留 fallback、dual mode、deprecated warning 或 hidden compatibility branch。
- 新代码优先保证阅读顺序自然、角色边界明确、测试入口稳定。

## 新核心模型

开始修改实现前，先把核心模块职责和依赖方向固定下来。后续所有删除、重构和测试都以这一层为准。

| 模块 | 职责 | 允许依赖 | 禁止依赖 |
| --- | --- | --- | --- |
| `Trainer` | 组合系统并驱动训练生命周期。 | `Execution`、`Algorithm`、`Runner`、`Store`、`Adapter`、`LLMProxy`、datasets、resources。 | 旧 client/server 协议、legacy runner。 |
| `Execution` | 决定各角色以什么拓扑和并发方式运行。 | `Trainer` 提供的角色实例、local/remote store facade。 | 训练算法细节、agent 业务逻辑。 |
| `Algorithm` | 决定训练循环如何推进。 | `AlgorithmContext`、`Store` Interface、`Adapter`、`LLMProxy`、datasets。 | `Trainer`、`Runner`、`AgentLightningClient`、`AgentLightningServer`。 |
| `Runner` | 领取 rollout，执行 attempt，写入状态和结果。 | `Store` Interface、`Tracer`、`LitAgent`、hooks、resources。 | `Algorithm`、legacy client/server。 |
| `LitAgent` | 定义一次 rollout 中 agent 如何完成任务。 | 输入、resources、业务依赖。 | `Trainer`、`Runner`、`Tracer`、`Store`。 |
| `Tracer` | 把执行过程转成标准 spans。 | `SpanWriter`、OpenTelemetry/AgentOps adapter。 | `Runner` 内部状态、legacy sync trace 入口。 |
| `Emitter` | 主动发出 reward、annotation、operation 等标准训练信号。 | 当前 trace context、semconv。 | legacy reward payload。 |
| `Store` | 解耦 `Algorithm` 和 `Runner`，保存 rollout 状态、attempts、resources、spans。 | collection adapter、remote client adapter、threading adapter。 | HTTP server lifecycle。 |
| `Adapter` | 把标准 spans 转成训练样本。 | 新 semconv、当前 instrumentation 输出。 | 历史 attribute key、旧 reward 格式。 |

### 必须先落定的 Interface

| Interface | 决策 | 验收点 |
| --- | --- | --- |
| `Trainer.algorithm` | `Trainer` 始终持有具体 `Algorithm`；未显式提供时统一使用 `Baseline`。`None` 不再隐式表达 external algorithm 或 manual queue 生命周期。 | `fit()`/`dev()` 共用同一算法实例；默认 Baseline 缺少 dataset 时明确失败；外部 Store/Runner 拓扑由调用者通过 `Execution` 和 Store adapters 显式组合。 |
| `AlgorithmContext` | 由 `Trainer` 构造并传给 `Algorithm.run(context)`；包含 `store`、`adapter`、`llm_proxy`、`initial_resources`、`train_dataset`、`val_dataset`、`event` 和运行配置。 | `Algorithm` 不再保存 trainer weakref，不再提供 `get_client()`，也不 import legacy client；协作式停止只通过 `context.event` 表达。 |
| `@algo` decorator | `@algo` 是把函数适配到 `Algorithm` Interface 的 Adapter；新函数只接受 `context: AlgorithmContext`，不再按参数名隐式注入 `store`、`adapter`、dataset 等依赖。 | decorator 代码不再 inspect 任意参数名；示例算法从 `def f(*, store, train_dataset)` 改成 `def f(context)`。 |
| Rollout/Attempt 状态机 | `RolloutStatus = queuing | preparing | running | succeeded | failed | cancelled | requeuing`；`AttemptStatus = preparing | running | succeeded | failed | cancelled | timeout | unresponsive`。 | `timeout` 和 `unresponsive` 只属于 attempt；`requeuing` 只属于 rollout；不存在 `queueing` 拼写。 |
| Agent rollout 返回值 | `RolloutResult = None | float | list[AgentSpanPayload]`；`AgentSpanPayload` 只包含 span 业务字段，不包含 `rollout_id`、`attempt_id`、`sequence_id`。 | 删除 OTEL `ReadableSpan`、已归属 store `Span` 和隐式 bool/int reward 自动转换路径；rollout/attempt/sequence 归属只由 `Runner` 和 `Tracer` 写入。 |
| Span ownership | `Runner`/`Tracer` 拥有 `rollout_id`、`attempt_id`、`sequence_id`、`created_at` 等上下文字段；agent 不直接决定归属。 | 不存在跨 rollout 污染、重复写入和 sequence 冲突路径。 |
| `SpanWriter` | 统一同步 callback 到 async store 的桥接；`LightningSpanProcessor` 和 `LLMProxy` exporter 都使用同一个写入 Interface。 | 写入 timeout、确认、失败日志和 shutdown 行为只有一处实现。 |
| Store 批量写入 | 使用显式 `SpanWriteResult`，返回 `inserted`、`duplicates`、`failed`；不要求跨后端 all-or-nothing 事务。 | 正常路径保持 bulk 写入，异常后读回确认部分成功项；Runner/OTLP 等调用方必须处理 `failed`，不能静默继续。 |
| Store 角色模型 | `LightningStore` 是业务 Interface；local collection store、remote client、threaded wrapper 是 adapters；`LightningStoreServer` 是 has-a store 的 HTTP/lifecycle container。 | server 不再继承 `LightningStore`；`ClientServerExecutionStrategy` 可以负责启动和关闭 store server，但传给 `algorithm_bundle`、`runner_bundle`、proxy 的必须是纯 `LightningStore` facade，例如 `LightningStoreClient`，不能是 `LightningStoreServer` runtime 对象。 |

### 状态转移表

状态迁移必须集中在 Store/Runner 的状态机中实现，不能在各模块手写分支。`requeuing` 只表示 rollout 等待下一次 attempt，attempt 自身永远不会进入 `queuing` 或 `requeuing`。

| 事件 | Attempt 状态 | Rollout 状态 | 规则 |
| --- | --- | --- | --- |
| algorithm enqueue | 无 attempt | `queuing` | 只创建 rollout，不创建 attempt。 |
| runner dequeue/start | `preparing` -> `running` | `preparing` -> `running` | dequeue 创建新 attempt；runner 开始执行后进入 `running`。 |
| agent 正常完成 | `succeeded` | `succeeded` | 终态；不会再 retry。 |
| agent 抛异常 | `failed` | `requeuing` 或 `failed` | attempts 未耗尽且 retry policy 允许 `failed` 时 requeue，否则 failed。 |
| 资源获取失败 | `failed` | `requeuing` 或 `failed` | 资源缺失发生在 attempt start 之后，必须给 attempt 明确失败终态。 |
| direct async step cancel | `cancelled` | `cancelled` | 仅取消可安全中止的 async rollout；抛出 `CancelledError`。 |
| runner stop / algorithm complete | 保持真实终态 | 保持真实终态 | 停止领取新 rollout，已经 dequeue 的 attempt drain 到完成。 |
| sync rollout stop request | 保持真实终态 | 保持真实终态 | Python 无法安全终止已经执行的同步工作；等待完成后按真实结果写入。 |
| attempt timeout | `timeout` | `requeuing` 或 `failed` | timeout 只属于 attempt；rollout 根据 retry policy 决定是否 requeue。 |
| worker unresponsive | `unresponsive` | `requeuing` 或 `failed` | watchdog 只结束当前 attempt；是否创建下一次 attempt 由 retry policy 决定。 |
| no work | 无变化 | 无变化 | runner 没领取到 rollout 时不得写入状态。 |

## 优先级

| 优先级 | 类型 | 标准 |
| --- | --- | --- |
| P0 | 正确性和状态一致性 | 可能造成错误终态、重复/错误 span、返回值不可信、取消语义错误的问题。 |
| P1 | 架构边界和兼容层删除 | deprecated、legacy 或 v0/v1 双轨逻辑影响核心设计，应直接删除或替换为新实现。 |
| P2 | 性能、生产化和工程清理 | 不阻塞正确性，但影响吞吐、维护成本、生产可用性或文档清晰度。 |

## P0：先修正确性和状态一致性

| 模块 | 项目 | 现状 | 影响 | 建议动作 |
| --- | --- | --- | --- | --- |
| Types/Store | 状态模型不一致 | `AttemptStatus` 没有 `cancelled`，文档和实现对 attempt 是否能 `cancelled`/`requeuing` 表述不一致，且存在 `queueing`/`queuing` 拼写差异。 | Runner、Store、Algorithm 会对同一终态做不同解释。 | 按状态转移表重写枚举和文档；attempt 支持 `cancelled`，但不支持 `queuing`/`requeuing`；拼写统一为 `queuing`/`requeuing`。 |
| Runner | attempt 终态 | `LitAgentRunner._step_impl()` 在资源获取失败时会提前返回，绕过后续 attempt 状态更新。 | 已经 dequeue/start 的 attempt 可能没有明确终态，算法侧等待或扫描时语义不稳定。 | 把资源获取纳入统一状态机；失败时标记明确终态或把 rollout 显式 requeue。 |
| Runner | 取消语义 | `asyncio.CancelledError` 不被 `except Exception` 捕获，但 `finally` 仍执行，`has_exception=False` 时可能把 attempt 标记为 `succeeded`。 | 中断、超时或 runner shutdown 可能留下错误成功记录。 | 单独捕获取消并重新抛出，同时把 attempt 映射到明确的取消或失败终态。 |
| Store | 批量 span 写入 | `CollectionBasedLightningStore._insert_spans_with_fallback()` 批量插入失败时，部分 span 可能已经写入，但返回值无法准确表达实际成功项。 | 调用方可能误判 span 写入结果，重复插入和追踪完整性变得不可预测。 | collection 层和 client/server 层统一返回 `SpanWriteResult(inserted, duplicates, failed)`；调用方按结果记录和重试。 |
| Runner/Types | Agent 返回 `Span` 校验 | `_post_process_rollout_result()` 对 `list[Span]` 直接写入 store，没有校验 `rollout_id`、`attempt_id`、`sequence_id` 是否属于当前 attempt。 | 错误 span 可能污染其他 rollout，或破坏同一 attempt 内的 sequence 顺序。 | 删除 agent 返回已归属 `Span` 的能力；只允许 `AgentSpanPayload`，由 Runner 重写上下文。 |
| Runner/Tracer | `ReadableSpan` 重复写入 | 当 tracer 已是 `OtelTracer` 时，代码只 warning，但仍把 agent 返回的 `ReadableSpan` 再写入 store。 | warning 与行为不一致，可能产生重复 span。 | 删除 agent 直接返回 `ReadableSpan` 的路径；OpenTelemetry span 只能由 tracer 写入。 |
| Runner | `run_context(worker_id=...)` | `Runner.run_context()` 总是用 `worker_id=0` 调用 `init_worker()`，但 teardown 使用传入的 `worker_id`。 | 初始化和清理不对称，debug/测试上下文可能清理错误 worker。 | 用同一个 resolved worker id 初始化和清理，并补测试覆盖非 0 worker。 |
| Tracer/LLMProxy | Trace 写入同步桥 | `LightningSpanProcessor` 和 `LLMProxy.LightningSpanExporter` 都需要从同步 callback 写入 async store，但当前逻辑分散。 | timeout、事件循环、失败确认和 shutdown 语义重复且难测。 | 抽出共享 `SpanWriter`/`AsyncStoreSpanWriter`，统一写入确认、超时、错误日志和关闭。 |

## P1：删除兼容层并重划职责

| 模块 | 项目 | 现状 | 建议动作 |
| --- | --- | --- | --- |
| Public API | deprecated 导出 | `agentlightning.__init__` 仍导出 `AgentLightningClient`、`AgentLightningServer`、`configure_logger` 等 deprecated API。 | 只导出新核心 API；删除 deprecated 导出和相关文档入口。 |
| Trainer | legacy 继承和旧参数 | `Trainer` 仍继承 `TrainerLegacy`，保留 `fit_v0`、`n_workers`、`max_tasks`、`daemon`、`triplet_exporter`、`dev=True` 等旧入口或折中参数。 | 删除 `TrainerLegacy` 和 `fit_v0`；构造参数只保留新 runtime 需要的依赖；旧运行方式迁移到 `Execution`。 |
| Algorithm | 反向引用 `Trainer` | `Algorithm` 保存 `_trainer_ref`，提供 `set_trainer()`/`get_trainer()`。 | 删除 trainer weakref；改为 `Trainer` 调用 `Algorithm.run(context)`。 |
| Algorithm | legacy client 反向引用 | `Algorithm` 和 `VERL` import `AgentLightningClient`，并保留 deprecated `get_client()`。 | 删除 `get_client()` 和所有 algorithm 到 legacy client 的 import；algorithm 只通过 `Store` 和 context 工作。 |
| Algorithm | decorator 隐式注入 | `@algo` 通过 inspect 函数参数名注入 `store`、`adapter`、`llm_proxy`、datasets 等依赖。 | 删除参数名注入；`@algo` 只把 `Callable[[AlgorithmContext], None | Awaitable[None]]` 适配成 `Algorithm`。 |
| Runner | `LegacyAgentRunner` | 仍依赖旧 `AgentLightningClient`、`RolloutLegacy`、`RolloutRawResultLegacy`、旧 lifecycle methods、`is_v0_1_rollout_api()` 和 tracer 私有 `_trace_context_sync()`。 | 删除 `agentlightning/runner/legacy.py`、相关 tests/examples/CI；`runner.__init__` 不再导出 legacy runner。 |
| Runner | `Runner.run()` | 基类声明 deprecated 并抛错，legacy runner 又覆盖回旧同步行为。 | 从核心基类删除同步入口；保留 `iter()` 和 `step()` 作为唯一 runner 运行 Interface。 |
| LitAgent | `trained_agents` | 构造参数已 deprecated，但仍保存到实例。 | 删除参数和字段；训练目标匹配只由 adapter 的 `agent_match` 表达。 |
| LitAgent | trainer/runner/tracer 反向引用 | `LitAgent` 可通过 runner/trainer fallback 获取 tracer。 | 删除 `set_trainer/get_trainer/trainer`、`set_runner/get_runner/runner`、`get_tracer/tracer`；agent 只表达 rollout 行为。 |
| LitAgent | `on_rollout_start/end()` | 方法已标注 deprecated，但 legacy runner 仍调用。 | 删除旧 lifecycle methods；统一使用明确的 Hook Interface。 |
| Types | legacy 类型传播 | `RolloutLegacy`、`Task`、`TaskIfAny`、`RolloutRawResultLegacy`、`ReadableSpan` result path 仍被 runtime 或 VERL 引用。 | 删除 legacy 类型或移动到非核心迁移文档；runtime 类型只保留新 rollout/attempt/span/resource 模型和 `RolloutResult = None | float | list[AgentSpanPayload]`。 |
| Tracer | `trace_context(store=...)` | `OtelTracer` 和 `AgentOpsTracer` 仍接受 deprecated `store` 参数。 | 删除参数；store 只通过 `init_worker(worker_id, store)` 或 `SpanWriter` 注入。 |
| Tracer | `_trace_context_sync()` | 私有同步入口仅为兼容保留，legacy runner 直接调用。 | 删除私有同步入口；如需要同步 tracing，设计正式 `SyncTracer`，否则不提供。 |
| Tracer | `trace_run()` / `trace_run_async()` | convenience wrapper 已 deprecated，推荐定制 Runner。 | 删除 wrapper；trace 生命周期只能由 Runner 或显式 `trace_context` 管理。 |
| Store | `query_rollouts(status=..., rollout_ids=...)` | 旧参数仍在抽象类、collection store、client wrapper、thread-safe wrapper 和测试中存在。 | 删除旧参数；只保留 `status_in` 和 `rollout_id_in`。 |
| Store | store 类型层级混乱 | `LightningStoreServer` 和 Store adapters 混在同一层级，导致 HTTP lifecycle container 被当成业务 Store 使用。 | 保留 `LightningStore` 作为业务 Interface；local store、remote client、threaded wrapper 是合法 Store adapters；server 改为 has-a store 且不实现 `LightningStore`。 |
| Execution/Store | server runtime 泄漏到 bundle | `ClientServerExecutionStrategy` 在 `managed_store=True` 时把 `LightningStoreServer` wrapper 传给 algorithm，导致 lifecycle runtime 对象被当成业务 Store 使用。 | `ClientServerExecutionStrategy` 可以负责创建、启动和关闭 `LightningStoreServer`；但传给 `algorithm_bundle`、`runner_bundle` 和 proxy 的参数必须是纯 `LightningStore` facade，server 模式下统一使用 `LightningStoreClient`。 |
| Client/Server | legacy HTTP stack | `agentlightning/client.py` 和 `agentlightning/server.py` 仍保留原始 Agent Lightning protocol。 | 删除 legacy client/server；远程运行统一走 `LightningStoreServer` + `LightningStoreClient`。 |
| Reward/Emitter | legacy reward 解析 | `agentlightning/reward.py` 是 deprecated re-export；旧解析主要在 `agentlightning/emitter/reward.py`，仍兼容 AgentOps/v0.2 reward payload。 | reward 统一为 `AGL_ANNOTATION` + `agentlightning.reward.*` attributes；删除旧格式解析、deprecated reward decorator 和 re-export。 |
| Adapter | legacy span 格式兼容 | `TracerTraceToTriplet`、`LlmProxyTraceToTriplet` 中保留多种历史 attribute key。 | 先定义“当前 instrumentation key”清单；adapter 只消费新 semconv 和该清单，删除历史兼容分支并补 fixture。 |
| VERL | v0/v1 双模式 | `VERL` 和 `AgentModeDaemon` 同时支持无 store 的 v0 mode 和 store/proxy/adapter 的 v1 mode。 | 删除 v0 mode、legacy server、legacy rollout 转换和 `get_client()`；VERL 只接受 store + llm_proxy + adapter 的新执行模型。 |
| Execution | 未实现 execution | `agentlightning/execution/inter_process.py` 只有 TODO。 | 要么实现完整 inter-process strategy，要么删除占位和导出。 |
| SQLite | 未实现 store | `agentlightning/store/sqlite.py` 只有 TODO。 | major 重构中二选一：实现完整 SQLite store，或删除文件、导出和文档入口。 |

## P2：性能、生产化和工程清理

| 模块 | 项目 | 现状 | 建议动作 |
| --- | --- | --- | --- |
| Runner | 同步 rollout 执行策略 | async runner 直接调用同步 `training_rollout()`/`validation_rollout()`，会阻塞 event loop、心跳和取消响应。 | 引入 rollout execution policy：默认直接执行，阻塞型 agent 可选择 `asyncio.to_thread`、进程池或外部 worker。 |
| Runner | `step(..., event=...)` | 接口接受 `ExecutionEvent`，但实现未使用。 | 接入取消/超时机制；如果不支持，删除该参数。 |
| Runner | hook 失败策略 | `_trigger_hooks()` 吞掉所有 hook 异常。 | 明确 hook 类型：观测型 best-effort，控制型可失败；测试固定行为。 |
| Runner | heartbeat 后台模型 | thread heartbeat 创建独立 event loop 并跨线程调用 async store/client。 | 抽成 worker heartbeat service，明确默认模式、线程安全要求和 shutdown 语义。 |
| Store | latest attempt 查询 | `_unlocked_many_rollouts_to_attempted_rollouts()` 对 rollouts 顺序查询 latest attempt。 | 在 collection 层提供按 `rollout_id_in` 批量查询 latest attempt，避免 N+1。 |
| Store | `LightningStoreClient.close()` | 跨 event loop session 关闭遇到 `RuntimeError` 后 best-effort 忽略。 | 引入 per-loop session 管理和集中 shutdown hook。 |
| Store | server exception middleware | `/v1/agl` 全局 exception middleware 注释标明仅应开发模式启用。 | 增加 debug/dev 配置开关，生产环境返回稳定错误响应。 |
| Tracer | AgentOps teardown | teardown 不移除 `LightningSpanProcessor`，因为 AgentOps 全局状态无法稳定恢复。 | 把 AgentOps 全局状态隔离到独立生命周期模块；必要时要求独立进程运行。 |
| VERL | trajectory aggregation TODO | 仍有 mismatch diagnostics、outdated import、flawed stats grouping、canonical logger 等 TODO/FIXME。 | 在 v1-only 模式下修复这些 TODO，并把日志、mismatch dump、stats schema 文档化。 |
| Resources | resource registry TODO | `types/resources.py` 标注迁移到 registry。 | 放到兼容层删除和状态模型稳定之后；要么建立 registry，要么删除 TODO 和半成品入口。 |

## 推荐执行顺序

1. 写下新核心 Interface：`AlgorithmContext`、`@algo` Adapter、状态转移表、`RolloutResult`、`AgentSpanPayload`、span ownership、`SpanWriter`、Store 角色模型、`SpanWriteResult`。
2. 修 P0 状态一致性：attempt cancellation/resource failure、span 写入校验、batch insert 结果、`run_context` worker id、`queueing`/`queuing` 拼写。
3. 删除 Trainer/Algorithm legacy 反向依赖：移除 `TrainerLegacy`、`fit_v0`、deprecated trainer 参数、`Algorithm.get_client()`、`VERL.get_client()`。
4. 删除 legacy runtime：移除 legacy runner、legacy client/server、legacy types、v0.1/v0.2 examples/tests/CI。
5. 收敛 `LitAgent`：删除 `trained_agents`、trainer/runner/tracer 反向引用和旧 lifecycle methods。
6. 收敛 `Tracer` 和 `LLMProxy`：删除 `store=`、`_trace_context_sync()`、`trace_run*`，统一使用 `SpanWriter`。
7. 重构 `Store` 层级：server 不再继承 store；client/threaded/local store 作为 adapters；execution strategy 统一传 client facade。
8. 清理 Store API：删除 `query_rollouts` 旧参数，补 latest attempt 批量查询和 `SpanWriteResult` 写入确认语义。
9. 收敛 reward/emitter/adapter：删除 legacy reward 格式，adapter 只支持新 semconv 和当前 instrumentation key。
10. 收敛 VERL：删除 v0 mode，修复 daemon/trainer 中的 legacy server、legacy rollout 转换和 trajectory aggregation TODO。
11. 处理未完成模块：实现或删除 SQLite store、inter-process execution、resource registry 占位。
12. 更新 docs/examples/tests：所有示例使用同一套新接口、同一套角色语言、同一套生命周期模型。
13. 做 P2 生产化清理：heartbeat、sync rollout policy、client session lifecycle、server error response、AgentOps lifecycle。

## Verification

每一阶段都必须用测试或静态检查验证，而不是只靠 symbol 删除。

| 阶段 | 必须新增或更新的验证 |
| --- | --- |
| 状态机 | Store contract tests 覆盖 enqueue、dequeue/start、success、agent exception、resource failure、cancel/shutdown、timeout、unresponsive、retry exhausted。 |
| Runner 正确性 | Runner integration tests 覆盖 `CancelledError` 不会写成 `succeeded`、资源获取失败有终态、`run_context(worker_id)` 初始化和清理一致。 |
| Span 写入 | Store contract tests 覆盖 `SpanWriteResult.inserted`、`duplicates`、`failed`；Runner tests 覆盖 agent 只能返回 `AgentSpanPayload`。 |
| Tracer/LLMProxy | Unit tests 覆盖 `LightningSpanProcessor` 和 `LightningSpanExporter` 共用 `SpanWriter`，包括 timeout、失败日志、shutdown。 |
| Algorithm | Tests 覆盖 `Algorithm.run(context)`、`@algo` 只接受 `AlgorithmContext`、`context.event` 能触发协作式停止。 |
| Store 分层 | Type/static tests 或 unit tests 确认 `LightningStoreServer` 不满足 `LightningStore`，`LightningStoreClient` 和 `LightningStoreThreaded` 满足 `LightningStore`；`ClientServerExecutionStrategy` 启动 server 后传给 `algorithm_bundle` 的是 `LightningStoreClient` facade。 |
| VERL | v1-only smoke tests 覆盖 store/proxy/adapter 路径；确认无 store 的 v0 fallback 不存在。 |
| 文档和示例 | examples 和 docs 全部使用 `AlgorithmContext`、`RolloutResult`、新 Store facade 和新状态命名。 |

最终检查命令：

- `uv run --no-sync pyright`
- `uv run --no-sync pytest -v`
- `uv run --no-sync pre-commit run --all-files --show-diff-on-failure`
- `uv run --no-sync mkdocs build --strict`

## 验收标准

- `Algorithm` 不再 import `AgentLightningClient`，不再保存 trainer weakref，不再提供 `get_client()`；训练依赖只来自 `AlgorithmContext`。
- `AlgorithmContext` 包含 `store`、`adapter`、`llm_proxy`、`initial_resources`、datasets、`event` 和运行配置；algorithm 协作式停止只通过 `context.event`。
- `@algo` 不再按参数名注入依赖，只把 `Callable[[AlgorithmContext], None | Awaitable[None]]` 适配成 `Algorithm`。
- `Trainer` 不再继承 `TrainerLegacy`，不再提供 `fit_v0` 或 deprecated 构造参数。
- 核心包不再导出 `AgentLightningClient`、`AgentLightningServer`、`LegacyAgentRunner`、`RolloutLegacy`、`RolloutRawResultLegacy`、`Task` 等 legacy runtime symbols。
- `Runner` 不直接调用 `Algorithm`，`Algorithm` 不直接调用 `Runner`；二者只通过 `Store` 和明确上下文协作。
- `LitAgent` 不再持有 trainer/runner/tracer/store 反向引用，只负责 rollout 行为。
- 每个 dequeue/start 出来的 attempt 在成功、失败、取消、超时、资源缺失、runner shutdown 时都按状态转移表写入明确终态或明确 requeue 行为。
- Rollout/attempt 状态枚举、文档和 Store 查询参数使用同一套命名；`timeout`/`unresponsive` 只属于 attempt，`requeuing` 只属于 rollout，不存在 `queueing`/`queuing` 混用。
- Span 写入路径对 rollout/attempt/sequence 有统一规则，重复写入和跨 rollout 污染有测试覆盖。
- `RolloutResult` 精确定义为 `None | float | list[AgentSpanPayload]`；agent 不能返回 OTEL `ReadableSpan` 或已归属 store `Span`；OpenTelemetry span 只由 tracer 写入。
- `LightningSpanProcessor` 和 `LLMProxy.LightningSpanExporter` 共用 `SpanWriter`，写入超时、失败和 shutdown 行为有测试覆盖。
- Store batch insert 返回显式 `SpanWriteResult(inserted, duplicates, failed)`；调用方不再猜测部分成功，Store Interface 文档不再承诺跨后端 all-or-nothing。
- `LightningStoreServer` 不再继承或实现 `LightningStore`；`LightningStoreClient` 和 `LightningStoreThreaded` 是满足 `LightningStore` 的 adapters；`ClientServerExecutionStrategy` 可以管理 server lifecycle，但传给 algorithm、runner、proxy 的必须是 `LightningStoreClient` 或其他纯 `LightningStore` facade，不能是 `LightningStoreServer` runtime 对象。
- Reward 只使用 `AGL_ANNOTATION` + `agentlightning.reward.*` 新格式，不再解析 legacy reward payload。
- Adapter 只消费新 semconv 和当前 instrumentation 输出，trajectory 构建规则有 fixture 覆盖。
- VERL 只支持 store/proxy/adapter 的新执行模型，不再保留 v0 mode、legacy server 或 `get_client()`。
- `rg` 验收按 symbol 和核心 runtime 路径执行，避免把第三方术语或迁移说明误判为失败；docs 可保留明确标记的 migration note，但核心实现不能依赖 deprecated/legacy symbols。
- 文档、examples、tests 使用同一套新接口命名和生命周期模型。
