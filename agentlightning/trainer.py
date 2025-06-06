from __future__ import annotations

import logging
import os
import multiprocessing
import signal
import time
from typing import Any, TYPE_CHECKING, List, Optional

import agentops
from .client import SamplingParameters, VerlAgentClient
from .instrumentation import instrument_all
from .instrumentation.agentops import AgentOpsServerManager
from .trace import LightningSpanProcessor, lightning_span_processor

if TYPE_CHECKING:
    from agentops.integration.callbacks.langchain import LangchainCallbackHandler

logger = logging.getLogger(__name__)


class LitAgent:
    """Base class for the combination of training and validation logic of an agent.
    It will be sent to the trainer to perform those steps.
    """

    _trainer: Trainer | None = None

    def __init__(self, *, trained_agents: Optional[str] = None) -> None:  # FIXME: str | None won't work for cli
        """
        Initialize the LitAgent.

        :param trained_agents: Optional string representing the trained agents.
            This can be used to track which agents have been trained by this instance.
        """
        self.trained_agents = trained_agents

    def set_trainer(self, trainer: Trainer) -> None:
        """
        Set the trainer for this agent.

        :param trainer: The Trainer instance that will handle training and validation.
        """
        self._trainer = trainer

    @property
    def trainer(self) -> Trainer:
        """
        Get the trainer for this agent.

        :return: The Trainer instance associated with this agent.
        :raises ValueError: If the trainer has not been set.
        """
        if self._trainer is None:
            raise ValueError("Trainer has not been set for this agent.")
        return self._trainer

    def training_rollout(
        self,
        sample: Any,
        *,
        sampling_parameters: SamplingParameters | None = None,
        rollout_id: str | None = None,
    ) -> Any:
        """
        Perform a single training rollout on the provided sample.

        :param sample: The input data for the training rollout.
        :param sampling_parameters: Optional parameters for LLM sampling, which the agent is expected to respect when calling the LLMs.
        :param rollout_id: Optional unique identifier for the current task, useful for tracking.
        :return: The result of the training rollout, typically the final reward.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def validation_rollout(
        self,
        sample: Any,
        *,
        sampling_parameters: SamplingParameters | None = None,
        rollout_id: str | None = None,
    ) -> Any:
        """
        Perform a single validation rollout on the provided sample.
        Redirect to `training_rollout` by default.

        :param sample: The input data for the validation rollout.
        :param sampling_parameters: Optional parameters for LLM sampling, which the agent is expected to respect when calling the LLMs.
        :param rollout_id: Optional unique identifier for the current task, useful for tracking.
        :return: The result of the validation rollout, typically the final reward.
        """
        return self.training_rollout(sample, sampling_parameters=sampling_parameters, rollout_id=rollout_id)


class Trainer:
    def __init__(
        self,
        *,
        max_tasks: int | None = None,
        agentops_managed: bool = True,
        instrument_managed: bool = True,
        n_workers: int = 1,
        daemon: bool = True,
    ) -> None:
        """
        Initialize the Trainer.

        :param max_tasks: Optional maximum number of tasks to process. If None, will run indefinitely.
        :param agentops_managed: Whether to automatically manage `agentops`.
            When set to true, trainer calls `agentops.init()` automatically and launching an agentops endpoint locally,
            in which case the agentops will not behave normally.
            If not, you are responsible for calling and using it before using the trainer.
        :param instrument_managed: Whether to automatically manage instrumentation.
            When set to false, you will manage the instrumentation yourself and the trainer might not work as expected.
        :param n_workers: Number of agent workers (processes) running in parallel.
            Not recommended to set > 1 when using `agentops_managed=False` as it might cause issues with tracing.
        :param daemon: Whether the worker processes should be daemon processes.
            When set to true, the worker process will be terminated when the main process exits.
            Therefore the main process will wait for the worker processes to finish before exiting.
            Otherwise the worker processes will continue running in the background even after the main process exits,
            in which case the cleanup method will not be called automatically.
        """
        self.max_tasks = max_tasks
        self.agentops_managed = agentops_managed
        self.instrument_managed = instrument_managed
        self.n_workers = n_workers
        self.daemon = daemon

        self._agentops_server_manager: AgentOpsServerManager | None = None
        self._agentops_server_port_val: int | None = None  # Stores the picklable port number

        self._verl_client: VerlAgentClient | None = None  # Will be initialized in fit method

        if self.agentops_managed:
            self._agentops_server_manager = AgentOpsServerManager(self.daemon)

        if not self.agentops_managed and self.n_workers > 1:
            logger.warning(
                "Using n_workers > 1 with agentops_managed=False. Ensure manual AgentOps setup is process-safe."
            )
        if not self.instrument_managed:
            logger.warning("instrument_managed=False. You are responsible for all instrumentation.")
        if not self.daemon:
            logger.warning(
                "daemon=False. Worker processes are non-daemonic. "
                "The worker processes will NOT be terminated when the main process exits. "
                "The cleanup must be handled manually."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_agentops_server_manager"] = None  # Exclude the unpicklable server manager
        # _agentops_server_port_val (int) is inherently picklable and will be included.
        logger.debug(f"Getting state for pickling Trainer (PID {os.getpid()}). _agentops_server_manager excluded.")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # In child process, self._agentops_server_manager will be None.
        logger.debug(f"Setting state for unpickled Trainer (PID {os.getpid()}). _agentops_server_manager is None.")

    def init(self, endpoint: str) -> None:
        logger.info(f"Initializing Trainer...")
        self._verl_client = VerlAgentClient(endpoint=endpoint)

        if self.agentops_managed and self._agentops_server_manager:
            self._agentops_server_manager.start()
            self._agentops_server_port_val = self._agentops_server_manager.get_port()
            if self._agentops_server_port_val is None:
                if (
                    self._agentops_server_manager.server_process is not None
                    and self._agentops_server_manager.server_process.is_alive()
                ):
                    raise RuntimeError("AgentOps server started but port is None. Check server manager logic.")
                elif (
                    self._agentops_server_port_val is None and self._agentops_server_manager.server_process is None
                ):  # Server failed to start
                    raise RuntimeError("AgentOps server manager indicates server is not running and port is None.")
        logger.info(f"Trainer main initialization complete.")

    def cleanup(self) -> None:
        logger.info(f"Cleaning up Trainer...")
        if self.agentops_managed and self._agentops_server_manager:
            self._agentops_server_manager.stop()
        self._verl_client = None
        logger.info(f"Trainer main cleanup complete.")

    def _initialize_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Setting up environment...")  # worker_id included in process name
        if self.agentops_managed:
            if self._agentops_server_port_val:  # Use the stored, picklable port value
                base_url = f"http://localhost:{self._agentops_server_port_val}"
                env_vars_to_set = {
                    "AGENTOPS_API_KEY": "dummy",
                    "AGENTOPS_API_ENDPOINT": base_url,
                    "AGENTOPS_APP_URL": f"{base_url}/notavailable",
                    "AGENTOPS_EXPORTER_ENDPOINT": f"{base_url}/traces",
                }
                for key, value in env_vars_to_set.items():
                    os.environ[key] = value
                    logger.info(f"[Worker {worker_id}] Env var set: {key}={value}")
            else:
                logger.warning(
                    f"[Worker {worker_id}] AgentOps managed, but local server port is not available. Client may not connect as expected."
                )

            if not agentops.get_client().initialized:
                agentops.init()
                logger.info(f"[Worker {worker_id}] AgentOps client initialized.")
            else:
                logger.warning(f"[Worker {worker_id}] AgentOps client was already initialized.")

        if self.instrument_managed:
            instrument_all()
            logger.info(f"[Worker {worker_id}] Instrumentation applied.")

    def _teardown_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Cleaning up environment...")
        # Do nothing for now.
        logger.info(f"[Worker {worker_id}] Environment cleanup complete.")

    def _execute_rollout_loop(self, agent: LitAgent, worker_id: int) -> int:
        client = self.get_verl_client()
        processor = lightning_span_processor()
        num_tasks_processed = 0

        logger.info(
            f"[Worker {worker_id}] Started rollouts. Polling for tasks "
            f"(max: {self.max_tasks if self.max_tasks is not None else 'unlimited'})."
        )
        loop_count = 0
        while True:
            if self.max_tasks is not None and loop_count >= self.max_tasks:
                logger.info(f"[Worker {worker_id}] Max tasks ({self.max_tasks}) for this worker reached. Exiting loop.")
                break

            sample = client.poll_next_task()
            if sample is None:
                logger.info(f"[Worker {worker_id}] No more tasks available from endpoint. Exiting loop.")
                break

            rollout_id = sample["rollout_id"]
            sampling_parameters = client.poll_sampling_parameters()
            logger.info(
                f"[Worker {worker_id}] (Rollout {rollout_id}) "
                f"Sampling parameters: {sampling_parameters}  "
                f"Sample: {sample} "
            )

            try:
                with processor:
                    start_time = time.time()
                    if sample["is_train"]:
                        reward = agent.training_rollout(
                            sample, sampling_parameters=sampling_parameters, rollout_id=rollout_id
                        )
                    else:
                        reward = agent.validation_rollout(
                            sample, sampling_parameters=sampling_parameters, rollout_id=rollout_id
                        )
                    end_time = time.time()
                    logger.info(
                        f"[Worker {worker_id}] (Rollout {rollout_id}) Completed in {end_time - start_time:.2f} "
                        f"seconds with reward: {reward}"
                    )

                last_trace = processor.last_trace()
                last_trajectory = last_trace.to_trajectory(agent_match=agent.trained_agents, final_reward=reward)
                if len(last_trajectory) > 0:
                    client.post_trajectory(rollout_id, last_trajectory)
                else:
                    logger.warning(
                        f"[Worker {worker_id}] (Rollout {rollout_id}) Empty trace found. Skipping post_trajectory."
                    )
            except Exception as e:
                logger.exception(
                    f"[Worker {worker_id}] (Rollout {rollout_id}) Exception during rollout Continuing to next task."
                )

            num_tasks_processed += 1
            loop_count += 1
            if loop_count % 10 == 0 or loop_count == 1:
                logger.info(
                    f"[Worker {worker_id}] Completed task {rollout_id}. Progress: {loop_count}/{self.max_tasks if self.max_tasks is not None else 'unlimited'}"
                )

        logger.info(f"[Worker {worker_id}] Finished rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

    def _worker_main_loop(
        self, agent: LitAgent, worker_id: int
    ) -> int:
        """Orchestrates worker setup, execution, and teardown. Target for multiprocessing or direct call."""
        if self.n_workers > 1:
            import setproctitle
            signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in worker processes
            setproctitle.setproctitle(multiprocessing.current_process().name)
        logger.info(
            f"[Worker {worker_id}] Worker main loop started."
        )  # worker_id included in process name via logger format
        self._initialize_worker_env(worker_id)
        num_processed = 0
        try:
            num_processed = self._execute_rollout_loop(agent, worker_id)
        except Exception as e:
            logger.exception(f"[Worker {worker_id}] Unhandled exception in worker main loop execution phase.")
        finally:
            self._teardown_worker_env(worker_id)
        return num_processed

    def get_verl_client(self) -> VerlAgentClient:
        if self._verl_client is None:
            raise RuntimeError("VerlAgentClient has not been initialized. Call fit() first.")
        return self._verl_client

    def get_openai_endpoint(self) -> str:
        """
        Get an OpenAI compatible endpoint.
        """
        return self.get_verl_client().openai_endpoint

    def get_lightning_span_processor(self) -> LightningSpanProcessor:
        return lightning_span_processor()

    def get_langchain_callback_handler(self, tags: list[str] | None = None) -> LangchainCallbackHandler:
        """
        Get the Langchain callback handler for integrating with Langchain.

        :param tags: Optional list of tags to apply to the Langchain callback handler.
        :return: An instance of the Langchain callback handler.
        """
        from agentops.integration.callbacks.langchain import LangchainCallbackHandler

        tags = tags or []
        client_instance = agentops.get_client()
        api_key = None
        if client_instance.initialized:
            api_key = client_instance.config.api_key
        else:
            logger.warning(
                "AgentOps client not initialized when creating LangchainCallbackHandler. API key may be missing."
            )
        return LangchainCallbackHandler(api_key=api_key, tags=tags)

    @staticmethod
    def kill_orphaned_processes() -> None:
        """
        Kill any orphaned processes that may have been left behind by previous runs.
        This is useful for cleaning up after crashes or unexpected exits.
        """
        import psutil

        for proc in psutil.process_iter():
            # check whether the process name matches
            if proc.name().startswith('AgentLightning-'):
                proc.kill()

    def fit(self, agent: LitAgent, endpoint: str) -> None:
        agent.set_trainer(self)
        self.init(endpoint)
        processes: List[multiprocessing.Process] = []

        try:
            if self.n_workers == 1:
                logger.info(f"Running with n_workers=1 (synchronous in main process).")
                # For n_workers=1, worker_id is 0, process name will be 'MainProcess' or similar
                num_tasks = self._worker_main_loop(agent, 0)
                logger.info(f"Single worker mode finished. Tasks processed: {num_tasks}")
            else:
                logger.info(f"Running with n_workers={self.n_workers} (multiprocessing).")

                for i in range(self.n_workers):
                    process_name = f"AgentLightning-Worker-{i}"
                    # 'self' (Trainer instance) is pickled here.
                    # __getstate__ ensures _agentops_server_manager (and its live process) is not pickled.
                    # Child process gets a Trainer copy with _agentops_server_manager=None, but _agentops_server_port_val is present.
                    p = multiprocessing.Process(
                        target=self._worker_main_loop,  # Instance method as target
                        args=(agent, i),
                        daemon=self.daemon,
                        name=process_name,
                    )
                    processes.append(p)
                    logger.info(f"Starting worker process {i} (name: {process_name})...")
                    p.start()

                if self.daemon:
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
                    multiprocessing_process._children.clear()

        except KeyboardInterrupt:
            if self.n_workers > 1 and len(processes) > 0:
                logger.info(f"KeyboardInterrupt received. Terminating workers...")
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
            logger.info(f"Workers terminated or single worker interrupted.")
        except Exception as e:
            logger.exception(f"Unhandled exception in fit method.")
        finally:
            if self.daemon:
                self.cleanup()
            else:
                logger.info("Main process exiting. Please use Trainer.kill_orphaned_processes() for cleanup.")
