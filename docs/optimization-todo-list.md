# Agent Lightning Major Version TODO List

本 TODO 按 `optimization-plan.md` 的推荐执行顺序排列。每个实现任务必须同步更新对应测试，不能把验证集中留到最后。

| 序号 | 任务名称 | 优先级 | 分类 | 状态 | 对应测试状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 落定并冻结核心接口：`AlgorithmContext` 模型定义与 context 透传边界 | P0 | 架构收敛 | 已完成 | 通过：`py_compile` + `test_apo`/`test_baseline`/`test_decorator`/`test_trainer_dev` 基线 | 明确运行时上下文字段（store/event/adapter/llm_proxy/dataset/config） |
| 1.2 | 落定并冻结核心接口：`@algo` Adapter 与上下文签名统一为 `AlgorithmContext` | P0 | 架构收敛 | 已完成 | 通过：`tests/algorithm/test_decorator.py` | 移除旧参数名注入，统一接收 context，返回 `None` 或 `Awaitable[None]` |
| 1.3 | 落定并冻结核心接口：`Algorithm.run` 与核心算法执行路径切换到 context 调用 | P0 | 架构收敛 | 已完成 | 通过：`tests/algorithm/test_baseline.py`, `tests/algorithm/test_apo.py`, `tests/trainer/test_trainer_dev.py` | Trainer/legacy 兼容路径已同步注入 context |
| 1.4 | 落定并冻结核心接口：核心类型约束（`RolloutResult`/`AgentSpanPayload`/`SpanWriteResult`/`SpanWriter`） | P0 | 架构收敛 | 已完成 | 已验证：`tests/types/test_core.py` | `RolloutResult` 已固定为 `Union[None, float, list[AgentSpanPayload]]`；待 `Runner` 与 `Tracing` 契约迁移 |
| 1.5 | 落定并冻结核心接口：`状态转移` 与 `Store` 角色模型冻结 | P0 | 架构收敛 | 已完成 | 已验证：`tests/store/test_utils.py::test_rollout_status_from_attempt_direct_statuses` + `tests/store/test_core.py::test_update_attempt_sets_end_time_for_cancelled_status` + `tests/store/test_core.py::test_update_attempt_cancelled_marks_worker_idle` | 已冻结 `Store` 对状态机和 worker 身份映射的职责：`AttemptStatus` 新增 `cancelled`，`rollout_status_from_attempt`/`update_attempt`/`_unlocked_sync_worker_with_attempt` 与文档一致，终态更新会带动 `rollout end_time` 与 `worker` idle/unknown 切换。
| 1.6 | 落定并冻结核心接口：span ownership、已归属 span 写入边界定义与清理基线 | P0 | 架构收敛 | 已完成 | 已验证：`tests/runner/test_agent_runner.py::test_post_process_agent_span_payloads` + `tests/runner/test_agent_runner.py::test_post_process_agent_span_payloads_rewrite_ownership` + `tests/runner/test_agent_runner.py::test_post_process_rejects_readable_spans` + `tests/runner/test_agent_runner.py::test_step_rejects_readable_span_list` | Runner 不再直接接受 `ReadableSpan`/已归属 `Span`/`SpanCoreFields`；仅接受 `AgentSpanPayload`，并在 Runner 侧重写 `rollout_id`/`attempt_id` 与 `sequence_id`。任务 7/8 将继续细化类型与 tracer 路径边界。 |
| 2 | 统一状态枚举和文档中的命名（`queuing`、`requeuing`、attempt/rollout 状态） | P0 | 状态模型 | 已完成 | 通过：`tests/store/test_core.py`（状态流与回退命名路径） | 已统一 `queueing` 的历史拼写为 `queuing`，并同步相关存储/文档注释文本 |
| 3 | 修复 Attempt 终态模型（支持 `cancelled`，移除 `queuing`/`requeuing`） | P0 | 运行状态 | 已完成 | 已验证：`tests/store/test_utils.py::test_rollout_status_from_attempt_direct_statuses` + `tests/store/test_utils.py::test_attempt_status_literal_rejects_queued_transitions` | `timeout`、`unresponsive` 只属于 attempt；`requeuing` 只属于 rollout |
| 4 | 修复 `LitAgentRunner._step_impl()` 资源获取失败分支写入 | P0 | Runner | 已完成 | 已验证：`tests/runner/test_agent_runner.py::test_step_impl_returns_rollout_id_on_resource_failure_and_marks_failed` | 资源获取失败必须走状态机并给出明确 attempt/rollout 终态 |
| 5 | 修复 `CancelledError` 处理，避免误判为 `succeeded` | P0 | Runner | 已完成 | 已验证：`tests/runner/test_agent_runner.py::test_step_impl_cancelled_marks_cancelled_and_raises` | 单独捕获取消并映射到 `cancelled` 或失败终态 |
| 6 | 统一批量 span 写入返回值为 `SpanWriteResult(inserted/duplicates/failed)` | P0 | Store | 已完成 | 已验证：`tests/store/test_core.py::test_add_many_spans_handles_mixed_rollouts_and_attempts`, `tests/store/test_client_server.py` 的 batch insert 断言, `tests/store/test_threading.py::test_threaded_store_delegates_all_methods` | 替换当前不可区分的 fallback 写入结果；不再承诺跨后端 all-or-nothing |
| 7 | 约束 agent 返回值与 span 归属，移除已归属 `Span` 写入路径和隐式 `bool`/`int` reward 转换 | P0 | Runner/Types | 已完成 | 通过：`tests/runner/test_agent_runner.py::test_step_rejects_bool_result` + `tests/runner/test_agent_runner.py::test_step_rejects_int_result` + `tests/runner/test_agent_runner.py::test_post_process_rejects_readable_spans` + `tests/runner/test_agent_runner.py::test_step_rejects_readable_span_list` | `Runner`/`LitAgent` 约定统一为 `RolloutResult = None | float | list[AgentSpanPayload]`，并在 `_post_process_rollout_result` 中删除 bool/int 隐式转换与已归属 Span 路径。
| 8 | 删除 `ReadableSpan` 直接写入路径，避免重复 span | P0 | Tracer/Runner | 已完成 | 已验证：`tests/runner/test_agent_runner.py::test_step_records_spans_for_none_result` + `tests/runner/test_agent_runner.py::test_step_rejects_readable_span_list` + `tests/runner/test_agent_runner.py::test_post_process_rejects_readable_spans` | OTEL span 只由 Tracer 侧写入 |
| 9 | 修复 `Runner.run_context(worker_id=...)` 初始化/teardown worker_id 不一致 | P0 | Runner | 已完成 | 已验证：`tests/runner/test_runner_context.py::test_run_context_uses_passed_worker_id` | 初始化和清理使用同一 resolved worker id |
| 10 | 抽取并共享 `SpanWriter`，统一 `LightningSpanProcessor` 与 `LLMProxy` 导出行为 | P0 | Tracer/LLMProxy | 已完成 | 已验证：`tests/tracer/test_otel.py::test_store_write_timeout` + `tests/llm_proxy/test_llm_proxy_cpu.py::test_exporter_export_handles_store_failures` + `tests/llm_proxy/test_llm_proxy_cpu.py::test_exporter_shutdown_is_idempotent` | 写入确认、超时和关闭语义只能有一处实现 |
| 11 | 清理 `agentlightning.__init__` 的 deprecated 导出 | P1 | API | 已完成 | 已验证：`tests/test_init_exports.py::test_top_level_exports_do_not_include_deprecated_symbols` | 移除 `AgentLightningClient`、`AgentLightningServer`、`configure_logger` 等旧符号 |
| 12 | 去除 `Trainer` 对 `TrainerLegacy` 的继承与 `fit_v0` | P1 | Trainer | 已完成 | 已验证：`tests/trainer/test_trainer_init.py::test_trainer_no_longer_has_fit_v0` + `tests/trainer/test_trainer_init.py::test_trainer_fit_rejects_string_dataset` | 统一新 runtime 流程，不再通过 legacy 路径运行 |
| 13 | 移除 `Trainer` 旧入口参数（`n_workers`、`max_tasks`、`daemon`、`triplet_exporter`、`dev=True` 等） | P1 | Trainer | 已完成 | 已验证：`tests/trainer/test_trainer_init.py::test_trainer_rejects_legacy_constructor_args` + `tests/trainer/test_trainer_init.py::test_trainer_fit_rejects_string_dataset` | 旧运行方式迁移到 `Execution` |
| 14 | 删除 `Algorithm` 对 `Trainer` 的弱引用（`_trainer_ref`） | P1 | Algorithm | 已完成 | 已验证：`tests/algorithm/test_decorator.py::test_algorithm_no_longer_exposes_trainer_accessors` | 仅保留 `AlgorithmContext` 传递，弱引用 Trainer 依赖清理 |
| 15 | 移除 `Algorithm.set_trainer/get_trainer` 与 `get_client` | P1 | Algorithm | 已完成 | 通过：`tests/algorithm/test_decorator.py::test_algorithm_no_longer_exposes_trainer_accessors` + `tests/algorithm/test_decorator.py::test_algorithm_no_longer_exposes_get_client` | 禁止反向依赖 Trainer 和 legacy client |
| 16 | 删除 `@algo` 参数名隐式注入逻辑 | P1 | Algorithm | 已完成 | 通过：`tests/algorithm/test_decorator.py::test_algorithm_preserves_signature` + `tests/algorithm/test_decorator.py::test_algo_rejects_non_context_parameter_name` | 仅适配接收 `AlgorithmContext` 并返回 `None` 或 `Awaitable[None]` 的 callable |
| 17 | 删除 legacy runner 体系（`runner/legacy.py`） | P1 | Runner | 已完成 | 通过：`tests/runner/test_runner_imports.py::test_runner_exports_do_not_include_legacy_runner` | 移除旧生命周期、`RolloutLegacy` 分支和 tracer 私有同步入口依赖 |
| 18 | 删除 `Runner.run()` 同步接口与 legacy 回退 | P1 | Runner | 已完成 | 已验证：`tests/runner/test_runner_imports.py::test_runner_sync_run_interface_is_removed` | 主 Runner 接口统一为 `iter()` 和 `step()` |
| 19 | 去掉 `LitAgent` 反向引用（trainer/runner/tracer/store） | P1 | LitAgent | 已完成 | 已验证：`tests/litagent/test_litagent.py::test_litagent_no_reverse_references_to_runner_trainer_or_tracer` | Agent 只关心 rollout 行为 |
| 20 | 移除 `LitAgent` 的 `trained_agents` 参数与持久字段 | P1 | LitAgent | 已完成 | 已验证：`tests/litagent/test_litagent.py::test_litagent_does_not_store_trained_agents_marker` | 训练目标匹配只由 adapter 的 `agent_match` 表达 |
| 21 | 删除 `on_rollout_start/end` deprecated 生命周期 | P1 | LitAgent | 已完成 | 已验证：`tests/litagent/test_litagent.py::test_litagent_no_legacy_rollout_lifecycle_methods` | 不保留 legacy lifecycle fallback |
| 22 | 清理 legacy 类型传播（`RolloutLegacy`、`Task`、`TaskIfAny`、`RolloutRawResultLegacy`、`ReadableSpan` result path） | P1 | Types | 已完成 | 已验证：`tests/types/test_core.py::test_core_does_not_expose_legacy_result_contracts` | 运行时仅保留新 rollout/attempt/span/resource 模型；legacy 类型集中在 `agentlightning/types/legacy.py` |
| 23 | 删除 Tracer `trace_context(store=...)` 参数 | P1 | Tracer | 已完成 | 已验证：`tests/tracer/test_agentops.py::test_agentops_trace_with_store_disable` + `tests/tracer/test_agentops.py::test_agentops_trace_with_store_enable` | `store` 通过 `init_worker(worker_id, store)` 或 `SpanWriter` 注入 |
| 24 | 删除 `OtelTracer` 私有 `_trace_context_sync()` | P1 | Tracer | 待开始 | 同步更新：同步追踪行为测试或移除扫描 | legacy runner 删除后不再依赖私有同步入口 |
| 25 | 删除 `trace_run` / `trace_run_async` deprecated wrappers | P1 | Tracer | 待开始 | 同步更新：API/文档检查 | 生命周期由 Runner 或显式 `trace_context` 管理 |
| 26 | 统一 `Store.query_rollouts` 参数为 `status_in`/`rollout_id_in` | P1 | Store | 待开始 | 同步更新：Store API 契约测试 | 删除 `status=`、`rollout_ids=` 旧参数分支 |
| 27 | 重构 Store 层级：`LightningStoreServer` 不再实现业务 `Store` | P1 | Store | 待开始 | 同步新增：类型/职责断言测试 | server 为 has-a lifecycle container；client/threaded/local 为 Store adapters |
| 28 | 修复 `ClientServerExecutionStrategy` 传递纯 `LightningStore` facade | P1 | Execution/Store | 待开始 | 同步新增：bundle 传参测试 | server 模式下传给 algorithm/runner/proxy 的必须是 `LightningStoreClient` facade |
| 29 | 删除 legacy client/server 协议栈 | P1 | 网络 | 待开始 | 同步更新：导出、API 与 runtime 路径扫描 | 远程运行统一走 `LightningStoreServer` + `LightningStoreClient` |
| 30 | 删除 legacy reward 解析与 deprecated reward re-export | P1 | Reward/Emitter | 待开始 | 同步新增：reward payload 测试 | 仅支持 `AGL_ANNOTATION` 与 `agentlightning.reward.*` 新 attributes |
| 31 | 精简 adapter 兼容键列表，移除历史 attribute key | P1 | Adapter | 待开始 | 同步新增：adapter fixture 测试 | 先定义当前 instrumentation key 清单，再清理 `TracerTraceToTriplet`、`LlmProxyTraceToTriplet` |
| 32 | 删除 VERL 的 v0/v1 双轨执行逻辑 | P1 | VERL | 待开始 | 同步新增：v1-only smoke tests | 仅保留 store/proxy/adapter 新模型；禁止 v0 fallback 和 `get_client()` |
| 33 | 处理 `inter_process.py` TODO：实现或删除 | P1 | Execution | 待开始 | 同步新增或更新：execution strategy 测试 | Major 中不能保留未实现占位逻辑 |
| 34 | 处理 `store/sqlite.py` TODO：实现或删除 | P1 | Store | 待开始 | 同步新增或更新：sqlite entry 测试 | 与整体 Store 架构一致后再决策 |
| 35 | 更新 docs/examples/tests 使用新接口与新生命周期模型 | P2 | 文档/示例 | 待开始 | 同步验证：example smoke、docs 链接、`mkdocs build --strict` | 放在 P2 生产化清理前，避免旧接口在示例和文档中滞留 |
| 36 | 引入 rollout execution policy，默认/阻塞/线程/进程可配 | P2 | Runner | 待开始 | 同步新增：阻塞策略切换测试 | 解决 event loop 被同步 rollout 卡住问题 |
| 37 | 完善 `step(..., event=...)` 取消与超时协同 | P2 | Runner | 待开始 | 同步新增：step 取消/超时测试 | 如果不支持该参数，应删除 API |
| 38 | 明确 hook 失败语义（观测型/控制型） | P2 | Runner | 待开始 | 同步新增：hook 异常传播测试 | 现在的吞掉异常行为需改为可预期策略 |
| 39 | 重构 heartbeat 后台服务模型 | P2 | Runner | 待开始 | 同步新增：heartbeat 生命周期测试 | 分离线程事件循环与 store 调用，固定 shutdown 语义 |
| 40 | 优化 latest attempt 查询，批量避免 N+1 | P2 | Store | 待开始 | 同步新增：批量查询正确性与性能测试 | 在 collection 层新增按 `rollout_id_in` 查询 latest attempt 的能力 |
| 41 | 重构 `LightningStoreClient.close()` 会话与 shutdown | P2 | Store | 待开始 | 同步新增：跨 event loop close 测试 | 引入 per-loop session 管理和集中 shutdown hook |
| 42 | 增加 server 异常中间件生产模式开关 | P2 | Server | 待开始 | 同步新增：错误响应快照测试 | `/v1/agl` 仅在 debug/dev 启用详细异常行为 |
| 43 | 处理 AgentOps teardown 全局状态与生命周期隔离 | P2 | Tracer | 待开始 | 同步新增：teardown 集成测试 | 必要时文档化独立进程建议 |
| 44 | 修复 VERL trajectory aggregation TODO 与统计链路 | P2 | VERL | 待开始 | 同步新增：trajectory 聚合测试 | 清理 outdated import、mismatch diagnostics、stats schema 与 logger TODO |
| 45 | 处理 `types/resources.py` 的 registry TODO（实现或移除） | P2 | 资源 | 待开始 | 同步新增或更新：资源注册与查询测试 | 在兼容层清理和状态模型稳定之后执行 |
| 46 | 执行最终检查命令（`pyright`、`pytest -v`、`pre-commit`、`mkdocs build --strict`） | P0 | 验收 | 待开始 | 全量执行并记录结果 | 每个阶段完成后也要运行相关子集，最终再跑全量 |
