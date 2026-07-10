# Copyright (c) Microsoft. All rights reserved.

import functools
import logging
from collections.abc import Sequence
from typing import Any, TypeVar, Union, cast

from agentlightning.adapter import TraceAdapter, TracerTraceToTriplet
from agentlightning.algorithm import Algorithm, Baseline, FastAlgorithm
from agentlightning.execution.base import ExecutionStrategy
from agentlightning.execution.client_server import ClientServerExecutionStrategy
from agentlightning.execution.events import ExecutionEvent
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy
from agentlightning.runner import LitAgentRunner, Runner
from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import Tracer
from agentlightning.types import AlgorithmContext, Dataset, Hook, NamedResources

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


async def _run_runner_bundle(
    store: LightningStore,
    worker_id: int,
    event: ExecutionEvent,
    *,
    runner: Runner[Any],
    agent: LitAgent[Any],
    hooks: Sequence[Hook],
) -> None:
    """Run a runner without capturing the whole Trainer instance.

    Client/server execution may pickle runner bundles for multiprocessing
    ``spawn``. Capturing ``Trainer`` would also capture its algorithm, which is
    unnecessary for runner workers and can make scripts run as ``__main__``
    unpicklable on macOS.
    """
    runner_initialized = False
    worker_initialized = False
    try:
        runner.init(agent=agent, hooks=hooks)
        runner_initialized = True
        runner.init_worker(worker_id, store)
        worker_initialized = True
        await runner.iter(event=event)
    except Exception:
        logger.exception("Runner bundle encountered an error (worker_id=%s).", worker_id)
        raise
    finally:
        if worker_initialized:
            try:
                runner.teardown_worker(worker_id)
            except Exception:
                logger.exception("Error during runner worker teardown (worker_id=%s).", worker_id)
        if runner_initialized:
            try:
                runner.teardown()
            except Exception:
                logger.exception("Error during runner teardown (worker_id=%s).", worker_id)


class Trainer:
    """High-level orchestration layer that wires Algorithm <-> Runner <-> Store.

    A [`Trainer`][agentlightning.Trainer] packages the moving parts of Agent-Lightning's
    training loop into a single entry point:

    * **Algorithm lifecycle:** Instantiates or accepts an [`Algorithm`][agentlightning.Algorithm],
      attaches the current [`LightningStore`][agentlightning.LightningStore], adapter, and
      initial resources, then executes the algorithm role inside the configured execution strategy.
    * **Runner fleet:** Spawns one or more [`Runner`][agentlightning.Runner] instances (defaulting
      to [`LitAgentRunner`][agentlightning.LitAgentRunner]) that hydrate a [`LitAgent`][agentlightning.LitAgent],
      claim rollouts, stream spans, and respect graceful termination signals from the execution strategy.
    * **Execution strategy:** Delegates process management to an
      [`ExecutionStrategy`][agentlightning.ExecutionStrategy] (shared memory, client/server, etc.),
      so advanced users can swap orchestration backends without changing trainer code.
    * **Telemetry plumbing:** Ensures tracers, adapters, and optional [`LLMProxy`][agentlightning.LLMProxy]
      are wired into both algorithm and runners so telemetry flows back into the store.

    The trainer exposes two convenience entry points:
    [`fit()`][agentlightning.Trainer.fit] for full training and
    [`dev()`][agentlightning.Trainer.dev] for fast, reproducible dry-runs. See the
    [Train the First Agent](../how-to/train-first-agent.md) and
    [Write the First Algorithm](../how-to/write-first-algorithm.md) tutorials for the broader context.
    """

    algorithm: Algorithm
    """An instance of [`Algorithm`][agentlightning.Algorithm] to use for training."""

    store: LightningStore
    """An instance of [`LightningStore`][agentlightning.LightningStore] to use for storing tasks and traces."""

    runner: Runner[Any]
    """An instance of [`Runner`][agentlightning.Runner] to use for running the agent."""

    initial_resources: NamedResources | None
    """An instance of [`NamedResources`][agentlightning.NamedResources] to use for bootstrapping the fit/dev process.

    The resources will be handed over to the algorithm. Note that not all algorithms support seeding resources.
    """

    n_runners: int
    """Number of agent runners to run in parallel."""

    max_rollouts: int | None
    """Maximum number of rollouts to process per runner. If None, workers run until no more rollouts are available."""

    strategy: ExecutionStrategy
    """An instance of [`ExecutionStrategy`][agentlightning.ExecutionStrategy] to use for spawning the algorithm and runners."""

    tracer: Tracer
    """An instance of [`Tracer`][agentlightning.Tracer].
    If None, a default [`AgentOpsTracer`][agentlightning.AgentOpsTracer] will be created with the current settings."""

    hooks: Sequence[Hook]
    """A sequence of [`Hook`][agentlightning.Hook] instances to be called at various lifecycle stages (e.g., `on_trace_start`,
    `on_trace_end`, `on_rollout_start`, `on_rollout_end`)."""

    adapter: TraceAdapter[Any]
    """An instance of [`TraceAdapter`][agentlightning.TraceAdapter] to export data consumble by algorithms from traces."""

    llm_proxy: LLMProxy | None
    """An instance of [`LLMProxy`][agentlightning.LLMProxy] to use for intercepting the LLM calls.
    If not provided, algorithm may create one on its own."""

    def __init__(
        self,
        *,
        n_runners: int | None = None,
        max_rollouts: int | None = None,
        initial_resources: NamedResources | None = None,
        tracer: Tracer | None = None,
        adapter: TraceAdapter[Any] | None = None,
        store: LightningStore | None = None,
        runner: Runner[Any] | None = None,
        strategy: ExecutionStrategy | None = None,
        algorithm: Algorithm | None = None,
        llm_proxy: LLMProxy | None = None,
        hooks: Union[Hook, Sequence[Hook]] | None = None,
    ):
        """Configure the trainer and resolve user-provided component instances.

        Component keywords accept concrete instances. Build configurable components
        before constructing the trainer.

        Configure strategy-specific options, such as client/server ports, on the
        strategy instance before constructing the trainer.
        """
        resolved_n_runners = 1 if n_runners is None else n_runners

        self.max_rollouts = max_rollouts

        self.tracer = self._make_tracer(tracer)
        self.adapter = self._make_adapter(adapter)

        self.algorithm = self._make_algorithm(algorithm)

        # We might be able to support a list of resources in future.
        self.initial_resources = initial_resources

        self.strategy = self._make_strategy(
            strategy,
            n_runners=resolved_n_runners,
        )

        strategy_n_runners = getattr(self.strategy, "n_runners", None)
        if isinstance(strategy_n_runners, int):
            if n_runners is not None and strategy is not None and n_runners != strategy_n_runners:
                raise ValueError(
                    "n_runners is configured on both Trainer and the execution strategy with different values: "
                    f"Trainer={n_runners}, strategy={strategy_n_runners}. Configure it in one place."
                )
            self.n_runners = strategy_n_runners
        else:
            self.n_runners = resolved_n_runners

        # The active store for the current execution context
        self.store = self._make_store(store, self.strategy)
        self.runner = self._make_runner(runner)

        self.llm_proxy = self._make_llm_proxy(llm_proxy)

        self.hooks = self._normalize_hooks(hooks)

    @staticmethod
    def _make_tracer(tracer: Tracer | None) -> Tracer:
        """Resolve the tracer component from user input, falling back to AgentOpsTracer."""
        if tracer is not None:
            if not isinstance(cast("object", tracer), Tracer):
                raise TypeError(f"tracer must be an instance of Tracer, got {type(tracer).__name__}.")
            return tracer
        return AgentOpsTracer(
            agentops_managed=True,
            instrument_managed=True,
            daemon=True,
        )

    @staticmethod
    def _make_algorithm(algorithm: Algorithm | None) -> Algorithm:
        """Resolve the algorithm used by both fit and dev, defaulting to Baseline."""
        if algorithm is None:
            return Baseline()
        if not isinstance(cast("object", algorithm), Algorithm):
            raise TypeError(f"algorithm must be an instance of Algorithm, got {type(algorithm).__name__}.")
        return algorithm

    @staticmethod
    def _make_adapter(adapter: TraceAdapter[Any] | None) -> TraceAdapter[Any]:
        """Resolve the adapter used to transform spans into algorithm-ready payloads."""
        if adapter is None:
            default_adapter: TraceAdapter[Any] = TracerTraceToTriplet()
            return default_adapter
        if not isinstance(cast("object", adapter), TraceAdapter):
            raise TypeError(f"adapter must be an instance of TraceAdapter, got {type(adapter).__name__}.")
        return adapter

    @staticmethod
    def _make_store(
        store: LightningStore | None,
        strategy: ExecutionStrategy,
    ) -> LightningStore:
        """Resolve the store implementation backing rollouts, attempts, spans, and resources.

        By default, it's always a in-memory store. If using a client/server execution strategy,
        the in-memory store will be initialized in a thread-safe manner.
        """
        if store is not None:
            if not isinstance(cast("object", store), LightningStore):
                raise TypeError(f"store must be an instance of LightningStore, got {type(store).__name__}.")
            return store
        is_client_server = isinstance(strategy, ClientServerExecutionStrategy)
        return InMemoryLightningStore(thread_safe=is_client_server)

    @staticmethod
    def _make_strategy(
        strategy: ExecutionStrategy | None,
        *,
        n_runners: int,
    ) -> ExecutionStrategy:
        """Resolve the execution strategy and seed defaults such as `n_runners`."""
        if strategy is not None:
            if not isinstance(cast("object", strategy), ExecutionStrategy):
                raise TypeError(f"strategy must be an instance of ExecutionStrategy, got {type(strategy).__name__}.")
            return strategy
        return ClientServerExecutionStrategy(n_runners=n_runners)

    @staticmethod
    def _make_llm_proxy(llm_proxy: LLMProxy | None) -> LLMProxy | None:
        """Resolve an optional LLM proxy instance."""
        if llm_proxy is None:
            return None
        if not isinstance(cast("object", llm_proxy), LLMProxy):
            raise TypeError(f"llm_proxy must be an instance of LLMProxy, got {type(llm_proxy).__name__}.")
        return llm_proxy

    def _make_runner(
        self,
        runner: Runner[Any] | None,
    ) -> Runner[Any]:
        """Resolve the runner responsible for executing the agent inside each worker."""
        if runner is not None:
            if not isinstance(cast("object", runner), Runner):
                raise TypeError(f"runner must be an instance of Runner, got {type(runner).__name__}.")
            return runner
        if self.max_rollouts is not None:
            return LitAgentRunner[Any](tracer=self.tracer, max_rollouts=self.max_rollouts)
        return LitAgentRunner[Any](tracer=self.tracer)

    @staticmethod
    def _normalize_hooks(hooks: Union[Hook, Sequence[Hook]] | None) -> Sequence[Hook]:
        """Coerce hook inputs into an immutable sequence for runner initialization."""
        if hooks is None:
            return ()
        if isinstance(hooks, Hook):
            return (hooks,)
        return tuple(hooks)

    def fit(
        self,
        agent: LitAgent[T_co],
        train_dataset: Dataset[T_co] | None = None,
        *,
        val_dataset: Dataset[T_co] | None = None,
    ) -> None:
        """Execute the full algorithm/runner training loop.

        [`Trainer.fit`][agentlightning.Trainer.fit] packages the algorithm and runner bundles,
        then hands them to the active [`ExecutionStrategy`][agentlightning.ExecutionStrategy].
        The strategy rarely returns until:

        * The algorithm exhausts the dataset(s) and stops enqueuing rollouts.
        * `max_rollouts` causes individual runners to exit.
        * An exception or interrupt cancels the shared [`ExecutionEvent`][agentlightning.ExecutionEvent].

        Args:
            agent: [`LitAgent`][agentlightning.LitAgent] implementation executed by runners.
            train_dataset: Optional iterable of rollout inputs consumed by the algorithm.
            val_dataset: Optional iterable consumed by validation passes.
        """
        if isinstance(train_dataset, str):
            raise TypeError(
                "Trainer.fit no longer accepts string datasets. Use the latest execution-based API "
                "with `ExecutionStrategy` and store-backed datasets."
            )
        algorithm_bundle = functools.partial(
            self._algorithm_bundle,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            algorithm=self.algorithm,
        )
        runner_bundle = functools.partial(_run_runner_bundle, agent=agent, runner=self.runner, hooks=self.hooks)

        self.strategy.execute(algorithm_bundle, runner_bundle, self.store)

    def dev(
        self,
        agent: LitAgent[T_co],
        train_dataset: Dataset[T_co] | None = None,
        *,
        val_dataset: Dataset[T_co] | None = None,
    ) -> None:
        """Exercise the infrastructure using a fast, synchronous algorithm.

        [`Trainer.dev`][agentlightning.Trainer.dev] mirrors [`fit()`][agentlightning.Trainer.fit] but
        insists on an [`Algorithm`][agentlightning.Algorithm] subtype that also derives from
        [`FastAlgorithm`][agentlightning.FastAlgorithm]. This keeps the loop responsive for
        debugging while still touching the same store, runners, hooks, and tracer plumbing.

        Args:
            agent: [`LitAgent`][agentlightning.LitAgent] implementation to execute.
            train_dataset: Optional iterable passed to the algorithm.
            val_dataset: Optional iterable passed to the algorithm.

        Raises:
            TypeError: If the configured algorithm does not inherit from `FastAlgorithm`.
        """
        if not isinstance(self.algorithm, FastAlgorithm):
            raise TypeError(
                "Trainer.dev() requires an algorithm that inherits from FastAlgorithm. "
                f"Received {type(self.algorithm).__name__}."
            )

        algorithm_bundle = functools.partial(
            self._algorithm_bundle,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            algorithm=self.algorithm,
        )
        runner_bundle = functools.partial(_run_runner_bundle, agent=agent, runner=self.runner, hooks=self.hooks)
        self.strategy.execute(algorithm_bundle, runner_bundle, self.store)

    async def _algorithm_bundle(
        self,
        store: LightningStore,
        event: ExecutionEvent,
        train_dataset: Dataset[T_co] | None,
        val_dataset: Dataset[T_co] | None,
        algorithm: Algorithm,
    ) -> None:
        """Internal entry point executed by the strategy for the algorithm role.

        This coroutine is scheduled inside the strategy's process/thread. It packages the
        runtime dependencies into one immutable [`AlgorithmContext`][agentlightning.AlgorithmContext]
        and passes that context to [`Algorithm.run`][agentlightning.Algorithm.run].
        """
        if self.llm_proxy is not None:
            self.llm_proxy.set_store(store)

        context = AlgorithmContext(
            store=store,
            event=event,
            adapter=self.adapter,
            llm_proxy=self.llm_proxy,
            initial_resources=self.initial_resources,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        try:
            result = algorithm.run(context)
            if result is not None:
                await result
        except Exception:
            logger.exception("Algorithm bundle encountered an error.")
            raise
