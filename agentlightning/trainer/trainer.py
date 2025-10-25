# Copyright (c) Microsoft. All rights reserved.

import asyncio
import functools
import logging
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar, Union

from agentlightning.adapter import TraceAdapter, TracerTraceToTriplet
from agentlightning.algorithm import Algorithm, Baseline, FastAlgorithm
from agentlightning.client import AgentLightningClient
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
from agentlightning.types import Dataset, Hook, NamedResources

from .init_utils import build_component, instantiate_component
from .legacy import TrainerLegacy
from .registry import ExecutionStrategyRegistry

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

ComponentSpec = Union[T, type[T], Callable[[], T], str, Dict[str, Any], None]


class Trainer(TrainerLegacy):
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

    algorithm: Optional[Algorithm]
    """An instance of [`Algorithm`][agentlightning.Algorithm] to use for training."""

    store: LightningStore
    """An instance of [`LightningStore`][agentlightning.LightningStore] to use for storing tasks and traces."""

    runner: Runner[Any]
    """An instance of [`Runner`][agentlightning.Runner] to use for running the agent."""

    initial_resources: Optional[NamedResources]
    """An instance of [`NamedResources`][agentlightning.NamedResources] to use for bootstrapping the fit/dev process.

    The resources will be handed over to the algorithm. Note that not all algorithms support seeding resources.
    """

    n_runners: int
    """Number of agent runners to run in parallel."""

    max_rollouts: Optional[int]
    """Maximum number of rollouts to process per runner. If None, workers run until no more rollouts are available."""

    strategy: ExecutionStrategy
    """An instance of [`ExecutionStrategy`][agentlightning.ExecutionStrategy] to use for spawning the algorithm and runners."""

    tracer: Tracer
    """A tracer instance, or a string pointing to the class full name or a dictionary with a 'type' key
    that specifies the class full name and other initialization parameters.
    If None, a default [`AgentOpsTracer`][agentlightning.AgentOpsTracer] will be created with the current settings."""

    hooks: Sequence[Hook]
    """A sequence of [`Hook`][agentlightning.Hook] instances to be called at various lifecycle stages (e.g., `on_trace_start`,
    `on_trace_end`, `on_rollout_start`, `on_rollout_end`)."""

    adapter: TraceAdapter[Any]
    """An instance of [`TraceAdapter`][agentlightning.TraceAdapter] to export data consumble by algorithms from traces."""

    llm_proxy: Optional[LLMProxy]
    """An instance of [`LLMProxy`][agentlightning.LLMProxy] to use for intercepting the LLM calls.
    If not provided, algorithm may create one on its own."""

    n_workers: int
    """Number of agent workers to run in parallel. Deprecated in favor of `n_runners`."""

    max_tasks: Optional[int]
    """Maximum number of tasks to process per runner. Deprecated in favor of `max_rollouts`."""

    daemon: bool
    """Whether worker processes should be daemons. Daemon processes
    are terminated automatically when the main process exits. Deprecated.
    Only have effect with `fit_v0`."""

    triplet_exporter: TraceAdapter[Any]
    """An instance of [`TracerTraceToTriplet`][agentlightning.TracerTraceToTriplet] to export triplets from traces,
    or a dictionary with the initialization parameters for the exporter.
    Deprecated. Use [`adapter`][agentlightning.Trainer.adapter] instead."""

    port: Optional[int]
    """Port forwarded to [`ClientServerExecutionStrategy`][agentlightning.ClientServerExecutionStrategy]."""

    def __init__(
        self,
        *,
        dev: bool = False,
        n_runners: Optional[int] = None,
        max_rollouts: Optional[int] = None,
        initial_resources: Optional[NamedResources] = None,
        tracer: ComponentSpec[Tracer] = None,
        adapter: ComponentSpec[TraceAdapter[Any]] = None,
        store: ComponentSpec[LightningStore] = None,
        runner: ComponentSpec[Runner[Any]] = None,
        strategy: ComponentSpec[ExecutionStrategy] = None,
        port: Optional[int] = None,
        algorithm: ComponentSpec[Algorithm] = None,
        llm_proxy: ComponentSpec[LLMProxy] = None,
        n_workers: Optional[int] = None,
        max_tasks: Optional[int] = None,
        daemon: bool = True,
        triplet_exporter: ComponentSpec[TracerTraceToTriplet] = None,
        hooks: Optional[Union[Hook, Sequence[Hook]]] = None,
    ):
        """Configure the trainer and resolve user-provided component specifications.

        Each keyword accepts either a concrete instance, a class, a callable factory, a
        registry string, or a lightweight configuration dictionary (see
        [`build_component()`][agentlightning.trainer.init_utils.build_component]).

        When ``port`` is provided it is forwarded to
        [`ClientServerExecutionStrategy`][agentlightning.ClientServerExecutionStrategy]
        instances constructed (or supplied) for the trainer.
        """
        # Do not call super().__init__() here.
        # super().__init__() will call TrainerLegacy's initialization, which is not intended.
        self.worker_id: Optional[int] = None

        self._dev = dev
        self.daemon = daemon
        self._client: AgentLightningClient | None = None  # Will be initialized in fit or fit_v0

        if n_workers is not None:
            warnings.warn(
                "`n_workers` is deprecated. Please use `n_runners`.",
                DeprecationWarning,
                stacklevel=2,
            )

        if n_runners is None:
            n_runners = n_workers if n_workers is not None else 1
        else:
            if n_workers is not None and n_workers != n_runners:
                warnings.warn(
                    "`n_workers` is ignored when `n_runners` is provided.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self.n_runners = n_runners
        self.n_workers = n_runners  # Backwards compatibility for fit_v0

        if max_tasks is not None:
            warnings.warn(
                "`max_tasks` is deprecated. Please use `max_rollouts`.",
                DeprecationWarning,
                stacklevel=2,
            )

        if max_rollouts is None:
            max_rollouts = max_tasks
        elif max_tasks is not None and max_tasks != max_rollouts:
            warnings.warn(
                "`max_tasks` is ignored when `max_rollouts` is provided.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.max_rollouts = max_rollouts
        self.max_tasks = max_tasks if max_tasks is not None else max_rollouts

        self.tracer = self._make_tracer(tracer)

        if adapter is not None and triplet_exporter is not None:
            warnings.warn(
                "`triplet_exporter` is deprecated and ignored because `adapter` is provided.",
                DeprecationWarning,
                stacklevel=2,
            )

        adapter_spec = adapter if adapter is not None else triplet_exporter
        self.adapter = self._make_adapter(adapter_spec)
        self.triplet_exporter = self.adapter  # Backwards compatibility

        self.algorithm = self._make_algorithm(algorithm)

        # We might be able to support a list of resources in future.
        self.initial_resources = initial_resources

        # The active store for the current execution context
        self.store = self._make_store(store)
        self.runner = self._make_runner(runner)

        self.port = port

        self.strategy = self._make_strategy(
            strategy,
            n_runners=self.n_runners,
            port=port,
        )
        if hasattr(self.strategy, "n_runners"):
            strategy_runners = getattr(self.strategy, "n_runners")
            if isinstance(strategy_runners, int) and strategy_runners > 0:
                self.n_runners = strategy_runners
                self.n_workers = strategy_runners

        self.llm_proxy = self._make_llm_proxy(llm_proxy, store=self.store)

        self.hooks = self._normalize_hooks(hooks)

        if not self.daemon:
            logger.warning(
                "daemon=False. Worker processes are non-daemonic. "
                "The worker processes will NOT be terminated when the main process exits. "
                "The cleanup must be handled manually."
            )

    def _make_tracer(self, tracer: ComponentSpec[Tracer]) -> Tracer:
        """Resolve the tracer component from user input, falling back to AgentOpsTracer."""
        default_factory = lambda: AgentOpsTracer(
            agentops_managed=True,
            instrument_managed=True,
            daemon=self.daemon,
        )
        return build_component(
            tracer,
            expected_type=Tracer,
            spec_name="tracer",
            default_factory=default_factory,
            dict_requires_type=True,
            invalid_spec_error_fmt="Invalid tracer type: {actual_type}. Expected Tracer, str, dict, or None.",
            type_error_fmt="Tracer factory returned {type_name}, which is not a Tracer subclass.",
        )

    def _make_algorithm(self, algorithm: ComponentSpec[Algorithm]) -> Optional[Algorithm]:
        """Resolve the algorithm component, allowing `None` for dev-mode dry runs."""
        return build_component(
            algorithm,
            expected_type=Algorithm,
            spec_name="algorithm",
            allow_none=True,
            invalid_spec_error_fmt="Invalid algorithm type: {actual_type}. Expected Algorithm, str, dict, or None.",
            type_error_fmt="Algorithm factory returned {type_name}, which is not a Algorithm subclass.",
        )

    def _make_adapter(self, adapter: ComponentSpec[TraceAdapter[Any]]) -> TraceAdapter[Any]:
        """Resolve the adapter used to transform spans into algorithm-ready payloads."""
        return build_component(
            adapter,
            expected_type=TraceAdapter,
            spec_name="adapter",
            default_factory=TracerTraceToTriplet,
            dict_requires_type=False,
            dict_default_cls=TracerTraceToTriplet,
            invalid_spec_error_fmt="Invalid adapter type: {actual_type}. Expected TraceAdapter, dict, or None.",
            type_error_fmt="Adapter factory returned {type_name}, which is not a TraceAdapter subclass.",
        )

    def _make_store(self, store: ComponentSpec[LightningStore]) -> LightningStore:
        """Resolve the store implementation backing rollouts, attempts, spans, and resources."""
        return build_component(
            store,
            expected_type=LightningStore,
            spec_name="store",
            default_factory=InMemoryLightningStore,
            invalid_spec_error_fmt="Invalid store type: {actual_type}. Expected LightningStore, str, dict, or None.",
            type_error_fmt="Store factory returned {type_name}, which is not a LightningStore subclass.",
        )

    def _make_strategy(
        self,
        strategy: ComponentSpec[ExecutionStrategy],
        *,
        n_runners: int,
        port: Optional[int] = None,
    ) -> ExecutionStrategy:
        """Resolve the execution strategy and seed defaults such as `n_runners`."""
        if isinstance(strategy, ExecutionStrategy):
            if port is not None and isinstance(strategy, ClientServerExecutionStrategy):
                strategy.server_port = port
            return strategy
        optional_defaults: Dict[str, Callable[[], Any]] = {"n_runners": lambda: n_runners}
        if port is not None:
            optional_defaults["server_port"] = lambda: port

        def default_factory() -> ExecutionStrategy:
            if port is not None:
                return ClientServerExecutionStrategy(n_runners=n_runners, server_port=port)
            return ClientServerExecutionStrategy(n_runners=n_runners)

        return build_component(
            strategy,
            expected_type=ExecutionStrategy,
            spec_name="strategy",
            default_factory=default_factory,
            optional_defaults=optional_defaults,
            invalid_spec_error_fmt="Invalid strategy type: {actual_type}. Expected ExecutionStrategy, str, dict, or None.",
            type_error_fmt="Strategy factory returned {type_name}, which is not an ExecutionStrategy subclass.",
            registry=ExecutionStrategyRegistry,
        )

    def _make_llm_proxy(
        self,
        llm_proxy: ComponentSpec[LLMProxy],
        *,
        store: LightningStore,
    ) -> Optional[LLMProxy]:
        """Resolve an optional LLM proxy and ensure it shares the trainer's store instance."""
        if isinstance(llm_proxy, LLMProxy):
            return llm_proxy

        optional_defaults: Dict[str, Callable[[], Any]] = {"store": lambda: store}
        if isinstance(llm_proxy, dict):
            llm_proxy = {**llm_proxy}
            llm_proxy.setdefault("store", store)

        return build_component(
            llm_proxy,
            expected_type=LLMProxy,
            spec_name="llm_proxy",
            allow_none=True,
            optional_defaults=optional_defaults,
            invalid_spec_error_fmt="Invalid llm_proxy type: {actual_type}. Expected LLMProxy, dict, str, or None.",
            type_error_fmt="llm_proxy factory returned {type_name}, which is not an LLMProxy subclass.",
        )

    def _make_runner(self, runner: ComponentSpec[Runner[Any]]) -> Runner[Any]:
        """Resolve the runner responsible for executing the agent inside each worker."""
        optional_defaults: Dict[str, Callable[[], Any]] = {"tracer": lambda: self.tracer}
        if self.max_rollouts is not None:
            optional_defaults["max_rollouts"] = lambda: self.max_rollouts

        def default_runner_factory() -> Runner[Any]:
            return instantiate_component(LitAgentRunner, optional_defaults=optional_defaults)

        return build_component(
            runner,
            expected_type=Runner,
            spec_name="runner",
            default_factory=default_runner_factory,
            optional_defaults=optional_defaults,
            invalid_spec_error_fmt="Invalid runner type: {actual_type}. Expected Runner, callable, str, dict, or None.",
            type_error_fmt="Runner factory returned {type_name}, which is not a Runner subclass.",
        )

    def _normalize_hooks(self, hooks: Optional[Union[Hook, Sequence[Hook]]]) -> Sequence[Hook]:
        """Coerce hook inputs into an immutable sequence for runner initialization."""
        if hooks is None:
            return ()
        if isinstance(hooks, Hook):
            return (hooks,)
        return tuple(hooks)

    def fit(
        self,
        agent: LitAgent[T_co],
        train_dataset: Optional[Dataset[T_co]] = None,
        *,
        val_dataset: Optional[Dataset[T_co]] = None,
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
            logger.warning(
                "Trainer.fit will no longer accepts a string URL in future version. "
                "To continue using a string URL, please use Trainer.fit_v0 instead. "
                "See documentation for how to migrate to latest version: https://microsoft.github.io/agent-lightning/stable/"
            )
            return self.fit_v0(  # type: ignore
                agent,
                train_dataset,
                val_dataset,  # type: ignore
            )

        agent.set_trainer(self)

        algorithm_bundle = functools.partial(
            self._algorithm_bundle,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            algorithm=self.algorithm,
        )
        runner_bundle = functools.partial(self._runner_bundle, agent=agent)

        self.strategy.execute(algorithm_bundle, runner_bundle, self.store)

    def dev(
        self,
        agent: LitAgent[T_co],
        train_dataset: Optional[Dataset[T_co]] = None,
        *,
        val_dataset: Optional[Dataset[T_co]] = None,
    ) -> None:
        """Exercise the infrastructure using a fast, synchronous algorithm.

        [`Trainer.dev`][agentlightning.Trainer.dev] mirrors [`fit()`][agentlightning.Trainer.fit] but
        insists on an [`Algorithm`][agentlightning.Algorithm] subtype that also derives from
        [`FastAlgorithm`][agentlightning.FastAlgorithm]. This keeps the loop responsive for
        debugging while still touching the same store, runners, hooks, and tracer plumbing.

        If no algorithm is provided, a default [`Baseline`][agentlightning.Baseline] algorithm will be used.

        Args:
            agent: [`LitAgent`][agentlightning.LitAgent] implementation to execute.
            train_dataset: Optional iterable passed to the algorithm.
            val_dataset: Optional iterable passed to the algorithm.

        Raises:
            TypeError: If the configured algorithm does not inherit from `FastAlgorithm`.
        """
        agent.set_trainer(self)

        # Sanity check
        if self.algorithm is None:
            algorithm = Baseline()
        else:
            algorithm = self.algorithm

        if not isinstance(algorithm, FastAlgorithm):
            raise TypeError(
                "Trainer.dev() requires an algorithm that inherits from FastAlgorithm. "
                f"Received {type(algorithm).__name__}."
            )

        algorithm_bundle = functools.partial(
            self._algorithm_bundle,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            algorithm=algorithm,
        )
        runner_bundle = functools.partial(self._runner_bundle, agent=agent)
        self.strategy.execute(algorithm_bundle, runner_bundle, self.store)

    async def _algorithm_bundle(
        self,
        store: LightningStore,
        event: ExecutionEvent,
        train_dataset: Optional[Dataset[T_co]],
        val_dataset: Optional[Dataset[T_co]],
        algorithm: Optional[Algorithm],
    ) -> None:
        """Internal entry point executed by the strategy for the algorithm role.

        This coroutine is scheduled inside the strategy's process/thread and is responsible
        for binding algorithm dependencies (store, adapter, initial resources, proxy) before
        invoking [`Algorithm.run`][agentlightning.Algorithm.run].
        When `algorithm` is `None` the bundle simply waits for the
        shared `event` to signal shutdown so runners can still execute (useful for manual queue
        seeding or external algorithms).
        """
        if algorithm is not None:
            algorithm.set_trainer(self)
            algorithm.set_store(store)
            algorithm.set_adapter(self.adapter)
            if self.initial_resources is not None:
                algorithm.set_initial_resources(self.initial_resources)
            if self.llm_proxy is not None:
                self.llm_proxy.set_store(store)
                algorithm.set_llm_proxy(self.llm_proxy)

        if algorithm is None:
            while not event.is_set():
                await asyncio.sleep(0.1)
            return
        try:
            if algorithm.is_async():
                await algorithm.run(  # type: ignore
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                )
            else:
                # This will block the event loop to maximize the debugging experience
                # It's the responsibility of the execution strategy to enable async execution
                algorithm.run(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                )
        except Exception:
            logger.exception("Algorithm bundle encountered an error.")
            raise

    async def _runner_bundle(
        self, store: LightningStore, worker_id: int, event: ExecutionEvent, agent: LitAgent[T_co]
    ) -> None:
        """Internal entry point executed by the strategy for each runner role.

        The bundle materializes the configured runner, binds the agent and hooks, associates
        the worker with the shared store, and then drives the runner's [`iter`][agentlightning.Runner.iter]
        loop until the execution event is set or an exception occurs. Cleanup mirrors the initialization
        sequence to keep tracer state, hooks, and agent resources consistent across restarts.
        """
        runner_instance: Runner[Any] | None = None
        runner_initialized = False
        worker_initialized = False
        try:
            # If not using shm execution strategy, we are already in the forked process
            runner_instance = self.runner
            runner_instance.init(agent=agent, hooks=self.hooks)
            runner_initialized = True
            runner_instance.init_worker(worker_id, store)
            worker_initialized = True
            await runner_instance.iter(event=event)
        except Exception:
            logger.exception("Runner bundle encountered an error (worker_id=%s).", worker_id)
            raise
        finally:
            if runner_instance is not None:
                if worker_initialized:
                    try:
                        runner_instance.teardown_worker(worker_id)
                    except Exception:
                        logger.exception("Error during runner worker teardown (worker_id=%s).", worker_id)
                if runner_initialized:
                    try:
                        runner_instance.teardown()
                    except Exception:
                        logger.exception("Error during runner teardown (worker_id=%s).", worker_id)
