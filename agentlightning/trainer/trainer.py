# Copyright (c) Microsoft. All rights reserved.

import asyncio
import functools
import inspect
import logging
import multiprocessing
import signal
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

from agentlightning.adapter import TraceAdapter, TraceTripletAdapter
from agentlightning.algorithm.base import BaseAlgorithm
from agentlightning.algorithm.mock import MockAlgorithm
from agentlightning.client import AgentLightningClient
from agentlightning.execution.base import ExecutionStrategy
from agentlightning.execution.client_server import ClientServerExecutionStrategy
from agentlightning.execution.events import Event
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy
from agentlightning.runner import AgentRunner, AgentRunnerV2, BaseRunner
from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import BaseTracer
from agentlightning.types import Dataset, Hook, NamedResources, ParallelWorkerBase

from .init_utils import build_component, instantiate_component
from .registry import ExecutionStrategyRegistry

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

ComponentSpec = Union[T, type[T], Callable[[], T], str, Dict[str, Any], None]


class Trainer(ParallelWorkerBase):
    """Orchestrates the distributed execution of agent rollouts.

    The Trainer is responsible for launching one or more worker processes
    that run the agent's execution loop. It manages multiprocessing,
    handles graceful shutdown, and serves as the main entry point for
    running a client-side agent fleet.

    Attributes:
        algorithm: An instance of `BaseAlgorithm` to use for training.
        store: An instance of `LightningStore` to use for storing tasks and traces.
        runner: An instance of `BaseRunner` to use for running the agent.
        initial_resources: An instance of `Resources` to use for bootstrapping the fit/dev process.
            The resources will be handed over to the algorithm.
            Note that not all algorithms support seeding resources.
        n_runners: Number of agent runners to run in parallel.
        max_rollouts: Maximum number of rollouts to process per runner. If None,
                      workers run until no more rollouts are available.
        strategy: An instance of `ExecutionStrategy` to use for spawning the algorithm and runners.
        tracer: A tracer instance, or a string pointing to the class full name or a dictionary with a 'type' key
                that specifies the class full name and other initialization parameters.
                If None, a default `AgentOpsTracer` will be created with the current settings.
        hooks: A sequence of `Hook` instances to be called at various lifecycle stages (e.g., on_trace_start,
               on_trace_end, on_rollout_start, on_rollout_end).
        adapter: An instance of `TraceTripletAdapter` to export data consumble by algorithms from traces.
        llm_proxy: An instance of `LLMProxy` to use for intercepting the LLM calls.
                   If not provided, algorithm will create one on its own.
        n_workers: Number of agent workers to run in parallel. Deprecated in favor of `n_runners`.
        max_tasks: Maximum number of tasks to process per runner. Deprecated in favor of `max_rollouts`.
        daemon: Whether worker processes should be daemons. Daemon processes
                are terminated automatically when the main process exits. Deprecated.
                Only have effect with `fit_v0`.
        triplet_exporter: An instance of `TraceTripletAdapter` to export triplets from traces,
                          or a dictionary with the initialization parameters for the exporter.
                          Deprecated. Use `adapter` instead.
        dev: If True, rollouts are run against the dev endpoint provided in `fit`.
             Deprecated in favor of `dev()` method.
    """

    def __init__(
        self,
        *,
        dev: bool = False,
        n_runners: Optional[int] = None,
        max_rollouts: Optional[int] = None,
        initial_resources: Optional[NamedResources] = None,
        tracer: ComponentSpec[BaseTracer] = None,
        adapter: ComponentSpec[TraceAdapter[Any]] = None,
        store: ComponentSpec[LightningStore] = None,
        runner: ComponentSpec[BaseRunner[Any]] = None,
        strategy: ComponentSpec[ExecutionStrategy] = None,
        algorithm: ComponentSpec[BaseAlgorithm] = None,
        llm_proxy: ComponentSpec[LLMProxy] = None,
        n_workers: Optional[int] = None,
        max_tasks: Optional[int] = None,
        daemon: bool = True,
        triplet_exporter: ComponentSpec[TraceTripletAdapter] = None,
        hooks: Optional[Union[Hook, Sequence[Hook]]] = None,
    ):
        super().__init__()
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

        self.strategy = self._make_strategy(strategy, n_runners=self.n_runners)
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

    def _make_tracer(self, tracer: ComponentSpec[BaseTracer]) -> BaseTracer:
        """Creates a tracer instance based on the provided configuration."""
        default_factory = lambda: AgentOpsTracer(
            agentops_managed=True,
            instrument_managed=True,
            daemon=self.daemon,
        )
        return build_component(
            tracer,
            expected_type=BaseTracer,
            spec_name="tracer",
            default_factory=default_factory,
            dict_requires_type=True,
            invalid_spec_error_fmt="Invalid tracer type: {actual_type}. Expected BaseTracer, str, dict, or None.",
            type_error_fmt="Tracer factory returned {type_name}, which is not a BaseTracer subclass.",
        )

    def _make_algorithm(self, algorithm: ComponentSpec[BaseAlgorithm]) -> Optional[BaseAlgorithm]:
        """Creates an algorithm instance based on the provided configuration."""
        return build_component(
            algorithm,
            expected_type=BaseAlgorithm,
            spec_name="algorithm",
            allow_none=True,
            invalid_spec_error_fmt="Invalid algorithm type: {actual_type}. Expected BaseAlgorithm, str, dict, or None.",
            type_error_fmt="Algorithm factory returned {type_name}, which is not a BaseAlgorithm subclass.",
        )

    def _make_adapter(self, adapter: ComponentSpec[TraceAdapter[Any]]) -> TraceAdapter[Any]:
        return build_component(
            adapter,
            expected_type=TraceAdapter,
            spec_name="adapter",
            default_factory=TraceTripletAdapter,
            dict_requires_type=False,
            dict_default_cls=TraceTripletAdapter,
            invalid_spec_error_fmt="Invalid adapter type: {actual_type}. Expected TraceAdapter, dict, or None.",
            type_error_fmt="Adapter factory returned {type_name}, which is not a TraceAdapter subclass.",
        )

    def _make_store(self, store: ComponentSpec[LightningStore]) -> LightningStore:
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
    ) -> ExecutionStrategy:
        if isinstance(strategy, ExecutionStrategy):
            return strategy
        optional_defaults: Dict[str, Callable[[], Any]] = {"n_runners": lambda: n_runners}

        def default_factory() -> ExecutionStrategy:
            return ClientServerExecutionStrategy(n_runners=n_runners, role="both")

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

    def _make_runner(self, runner: ComponentSpec[BaseRunner[Any]]) -> BaseRunner[Any]:
        optional_defaults: Dict[str, Callable[[], Any]] = {"tracer": lambda: self.tracer}
        if self.max_rollouts is not None:
            optional_defaults["max_rollouts"] = lambda: self.max_rollouts

        def default_runner_factory() -> BaseRunner[Any]:
            return instantiate_component(AgentRunnerV2, optional_defaults=optional_defaults)

        return build_component(
            runner,
            expected_type=BaseRunner,
            spec_name="runner",
            default_factory=default_runner_factory,
            optional_defaults=optional_defaults,
            invalid_spec_error_fmt="Invalid runner type: {actual_type}. Expected BaseRunner, callable, str, dict, or None.",
            type_error_fmt="Runner factory returned {type_name}, which is not a BaseRunner subclass.",
        )

    def _normalize_hooks(self, hooks: Optional[Union[Hook, Sequence[Hook]]]) -> Sequence[Hook]:
        if hooks is None:
            return ()
        if isinstance(hooks, Hook):
            return (hooks,)
        return tuple(hooks)

    def fit_v2(
        self,
        agent: LitAgent[T_co],
        train_dataset: Optional[Dataset[T_co]] = None,
        *,
        val_dataset: Optional[Dataset[T_co]] = None,
    ) -> None:
        """Run the training loop using the configured strategy, store, and runner.

        Args:
            agent: The LitAgent instance to be trained on.
            train_dataset: The dataset to train on.
            val_dataset: The dataset to validate on.
        """
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
        """Dry run the training loop with a FastAlgorithm and the real runner.

        Args:
            agent: The LitAgent instance to be trained on.
            train_dataset: The dataset to train on.
            val_dataset: The dataset to validate on.
        """
        agent.set_trainer(self)

        # Sanity check
        if self.algorithm is None:
            algorithm = MockAlgorithm()
        else:
            algorithm = self.algorithm

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
        event: Event,
        train_dataset: Optional[Dataset[T_co]],
        val_dataset: Optional[Dataset[T_co]],
        algorithm: Optional[BaseAlgorithm],
    ) -> None:
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
            if inspect.iscoroutinefunction(algorithm.run):
                await algorithm.run(
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

    async def _runner_bundle(self, store: LightningStore, worker_id: int, event: Event, agent: LitAgent[T_co]) -> None:
        runner_instance: BaseRunner[Any] | None = None
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

    def _extract_client_from_data(
        self, data: Union[str, AgentLightningClient, Dataset[Any]]
    ) -> Optional[AgentLightningClient]:
        """Extract client from data if it's a string URL or AgentLightningClient."""
        if isinstance(data, str):
            if not data.startswith("http://") and not data.startswith("https://"):
                raise ValueError("String data must be a valid URL starting with http:// or https://")
            return AgentLightningClient(endpoint=data)
        elif isinstance(data, AgentLightningClient):
            return data
        return None

    def _extract_dataset_from_data(
        self, data: Union[str, AgentLightningClient, Dataset[Any]]
    ) -> Optional[Dataset[Any]]:
        """Extract dataset from data if it's a Dataset."""
        if isinstance(data, str) or isinstance(data, AgentLightningClient):
            return None
        return data

    def _determine_backend(
        self,
        train_data: Union[str, AgentLightningClient, Dataset[Any]],
        dev_data: Union[str, AgentLightningClient, Dataset[Any], None] = None,
    ) -> Union[str, AgentLightningClient]:
        """Determine which backend to use for initialization."""
        if self._dev:
            if dev_data is None:
                raise ValueError("dev_data must be provided when dev=True.")
            client = self._extract_client_from_data(dev_data)
            if client is None:
                raise ValueError("dev_data must be a string URL or AgentLightningClient when dev=True.")
            return client
        else:
            client = self._extract_client_from_data(train_data)
            if client is None and self.algorithm is None:
                raise ValueError(
                    "train_data must be a string URL or AgentLightningClient when no algorithm is provided."
                )
            elif client is None and self.algorithm is not None:
                # Algorithm will be responsible for creating the client
                client = self.algorithm.get_client()
                logger.info(f"Algorithm created client: {client}")
                return client
            if client is None:
                raise ValueError(
                    "train_data must be a string URL or AgentLightningClient when no algorithm is provided."
                )
            return client

    def init(self, backend: Union[str, AgentLightningClient]) -> None:
        logger.info(f"Initializing Trainer...")

        self._init_client(backend)

        self.tracer.init()

        logger.info(f"Trainer main initialization complete.")

    def teardown(self) -> None:
        logger.info(f"Cleaning up Trainer...")
        self.tracer.teardown()

        self._client = None
        logger.info(f"Trainer main cleanup complete.")

    def client(self) -> AgentLightningClient:
        """Returns the AgentLightningClient instance."""
        if self._client is None:
            raise RuntimeError("AgentLightningClient has not been initialized. Call `init` first.")
        return self._client

    def _init_client(self, backend: Union[str, AgentLightningClient]) -> AgentLightningClient:
        if self._client is None:
            if isinstance(backend, AgentLightningClient):
                logger.info("Using provided AgentLightningClient instance.")
                self._client = backend
            else:
                logger.info(f"Initializing AgentLightningClient with endpoint: {backend}")
                if not isinstance(backend, str):  # type: ignore
                    raise ValueError("backend must be a string URL or an AgentLightningClient instance.")
                if not backend.startswith("http://") and not backend.startswith("https://"):
                    raise ValueError("backend must be a valid URL starting with http:// or https://")
                # Initialize the client with the provided backend URL
                self._client = AgentLightningClient(endpoint=backend)
        else:
            logger.warning("AgentLightningClient already initialized. Returning existing instance.")
        return self._client

    def _worker_main_loop(self, agent: LitAgent[Any], worker_id: int, is_async: bool):
        """The main function for each worker process.

        This function initializes the client and the loop, then starts the
        execution. It also configures process-specific settings like the
        process title and signal handling.

        Args:
            agent: The `LitAgent` instance to run.
            worker_id: The unique ID for this worker.
            is_async: A boolean indicating if the async loop should be run.
        """
        if self.n_workers > 1:
            import setproctitle

            # Ignore Ctrl+C in worker processes; the main process handles it
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            setproctitle.setproctitle(multiprocessing.current_process().name)

        # Now we are in child processes, so we can safely set up the environment.
        agent.set_trainer(self)
        if not isinstance(self.triplet_exporter, TraceTripletAdapter):
            raise ValueError("triplet_exporter must be a TraceTripletAdapter for the legacy trainer.")
        # TODO: this should be set elsewhere
        if agent.trained_agents:
            self.triplet_exporter.agent_match = agent.trained_agents
        self._initialize_worker_env(worker_id)

        mode = "Async" if is_async else "Sync"
        logger.info(f"[Worker {worker_id}] {mode} worker process started.")

        num_processed = 0

        try:
            client = self.client()
            loop = AgentRunner(
                agent=agent,
                client=client,
                tracer=self.tracer,
                triplet_exporter=self.triplet_exporter,
                max_tasks=self.max_tasks,
                worker_id=worker_id,
            )
            loop.init_worker(worker_id)  # type: ignore
            if is_async:
                num_processed = asyncio.run(loop.iter_async())
            else:
                num_processed = loop.iter()
        except Exception:
            logger.exception(f"[Worker {worker_id}] Unhandled exception in worker loop.")
        finally:
            self._teardown_worker_env(worker_id)

        return num_processed

    def _initialize_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Setting up trainer environment...")  # worker_id included in process name
        self.tracer.init_worker(worker_id)

    def _teardown_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Cleaning up trainer environment...")
        self.tracer.teardown_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Environment cleanup complete.")

    @staticmethod
    def kill_orphaned_processes() -> None:
        """
        Kill any orphaned processes that may have been left behind by previous runs.
        This is useful for cleaning up after crashes or unexpected exits.
        """
        import psutil

        for proc in psutil.process_iter():  # type: ignore
            # check whether the process name matches
            if proc.name().startswith("AgentLightning-"):
                proc.kill()

    def _terminate_processes(self, processes: List[multiprocessing.Process]) -> None:
        if self.n_workers > 1 and len(processes) > 0:
            for i, p in enumerate(processes):
                if p.is_alive():
                    logger.info(f"Terminating worker {i} (name: {p.name}, PID: {p.pid})...")
                    p.terminate()
                else:
                    logger.info(f"Worker {i} (name: {p.name}, PID: {p.pid}) is not alive or has already terminated.")
            for i, p in enumerate(processes):
                if p.is_alive():
                    p.join(timeout=10)  # Give some time to terminate
                if p.is_alive():  # If still alive, kill
                    logger.warning(
                        f"Worker {i} (name: {p.name}, PID: {p.pid}) did not terminate gracefully, killing..."
                    )
                    p.kill()
                    p.join(timeout=10)  # Ensure it's reaped

    def fit(
        self,
        agent: LitAgent[T_co],
        train_data: Union[str, AgentLightningClient, Dataset[T_co]],
        *,
        val_data: Union[str, AgentLightningClient, Dataset[T_co], None] = None,
        dev_data: Union[str, AgentLightningClient, Dataset[T_co], None] = None,
        dev_backend: Union[str, AgentLightningClient, None] = None,
    ):
        """Train the agent using the provided data.

        Each data argument can be a string URL connecting to a agent-lightning server,
        or an AgentLightningClient instance connecting to a server (or mock server), or a dataset.
        If no algorithm is provided when instantiating the trainer, the data must be
        provided to connecting a server. Otherwise, dataset is also allowed and will be
        passed to the algorithm.

        If the algorithm is instantiated and there is no URL/client provided,
        the algorithm will be responsible for creating a client that will connect to itself.
        It can also create a mock client if the algorithm does not require a server.
        """

        if dev_backend is not None:
            warnings.warn("dev_backend is deprecated. Use dev_data instead.")
            if dev_data is not None:
                raise ValueError("dev_data and dev_backend cannot be provided at the same time.")
            dev_data = dev_backend

        # Extract datasets for algorithm if available
        train_dataset = self._extract_dataset_from_data(train_data)
        val_dataset = self._extract_dataset_from_data(val_data) if val_data else None

        # Initialize the algorithm with trainer if provided
        if self.algorithm is not None:
            self.algorithm.set_trainer(self)
            # DO NOT RUN TRAINING HERE. Need to spawn the worker first.

        # Determine the backend to use for client-server mode
        backend = self._determine_backend(train_data, dev_data)

        if self._dev:
            logger.warning(f"Running in dev mode. Using dev backend: {backend}")
        else:
            logger.debug(f"Running in non-dev mode. Using backend: {backend}")

        self.init(backend)

        processes: List[multiprocessing.Process] = []

        # Determine if the agent is asynchronous

        mode = "asynchronous" if agent.is_async() else "synchronous"

        try:
            if self.n_workers == 1:
                logger.info(f"Running with n_workers=1 ({mode} in main process).")

                # Warn if algorithm is set with single worker mode
                if self.algorithm is not None:
                    logger.warning(
                        "Algorithm is set but using single worker mode. Algorithm will never get the chance to run."
                    )
                    # Ideally the single worker should be run in a separate thread or process.

                num_tasks = self._worker_main_loop(agent, 0, agent.is_async())
                logger.info(f"Single worker mode finished. Tasks processed: {num_tasks}")

                # If algorithm is provided and we have datasets, run algorithm after worker completes
                if self.algorithm is not None and train_dataset is not None:
                    logger.info("Running algorithm training after worker completion.")
                    self.algorithm.run(
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                    )
            else:
                logger.info(f"Running with n_workers={self.n_workers} ({mode} multiprocessing).")
                for i in range(self.n_workers):
                    process_name = f"AgentLightning-Worker-{i}"
                    p = multiprocessing.Process(
                        target=self._worker_main_loop,
                        args=(agent, i, agent.is_async()),
                        daemon=self.daemon,
                        name=process_name,
                    )
                    processes.append(p)
                    logger.info(f"Starting worker process {i} (name: {process_name})...")
                    p.start()

                if self.daemon:
                    # If algorithm is provided and we have datasets, pass them to the algorithm
                    if self.algorithm is not None:
                        logger.info("All workers have been spawned. Running algorithm training with provided datasets.")
                        self.algorithm.run(
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                        )
                        logger.info("Algorithm exits. Killing the workers.")
                        self._terminate_processes(processes)

                    for i, p in enumerate(processes):
                        p.join()  # Wait for the process to complete
                        logger.info(
                            f"Worker process {i} (name: {p.name}, PID: {p.pid}) joined with exit code {p.exitcode}."
                        )
                        if p.exitcode != 0:
                            logger.warning(
                                f"Worker process {i} (name: {p.name}, PID: {p.pid}) exited with non-zero code: {p.exitcode}."
                            )

                    logger.info(f"All {self.n_workers} worker processes have completed.")
                else:
                    logger.info("All worker processes started. Main process will not wait.")

                    # A hack to stop the main process from waiting for child processes to finish.
                    time.sleep(1)  # Give workers time to start
                    import multiprocessing.process as multiprocessing_process

                    multiprocessing_process._children.clear()  # type: ignore

                    if self.algorithm is not None:
                        logger.info("Main process continues to run algorithm.")
                        self.algorithm.run(
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                        )
                        logger.info("Algorithm exits. Killing the workers.")
                        self._terminate_processes(processes)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Killing the workers.")
            self._terminate_processes(processes)
            logger.info(f"Workers terminated or single worker interrupted.")
            raise
        except Exception:
            logger.exception(f"Unhandled exception in fit method.")
            self._terminate_processes(processes)
            logger.info(f"Workers terminated or single worker interrupted.")
            raise
        finally:
            if self.daemon:
                self.teardown()
            else:
                logger.info("Main process exiting. Please use Trainer.kill_orphaned_processes() for cleanup.")
