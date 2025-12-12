# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from pydantic import validate_call
from weave.trace_server import trace_server_interface as tsi
from weave.trace_server.ids import generate_id
from weave.trace_server_bindings.client_interface import TraceServerClientInterface
from weave.trace_server_bindings.models import ServerInfoRes

from agentlightning.instrumentation.weave import instrument_weave, uninstrument_weave
from agentlightning.store.base import LightningStore
from agentlightning.types.tracer import OtelResource, Span, SpanContext, TraceStatus

from .base import Tracer

if TYPE_CHECKING:
    from weave.trace.call import Call

JSONPrimitive = Union[str, int, float, bool, None]

logger = logging.getLogger(__name__)


def generate_span_id() -> str:
    return "sp-" + hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]


class InMemoryTraceServer(TraceServerClientInterface):
    """A minimal in-memory implementation of the TraceServerInterface.

    It stores calls and objects in local dictionaries and returns valid Pydantic
    responses to satisfy the Weave client and FullTraceServerInterface protocol.
    """

    def __init__(self):
        # Minimal storage to allow basic querying in tests
        self.calls: Dict[str, tsi.CallSchema] = {}
        self.objs: Dict[str, Any] = {}
        self.files: Dict[str, bytes] = {}
        self.feedback: List[tsi.FeedbackCreateReq] = []

    @classmethod
    def from_env(cls, *args: Any, **kwargs: Any) -> "InMemoryTraceServer":
        return cls()

    def server_info(self) -> ServerInfoRes:
        return ServerInfoRes(min_required_weave_python_version="0.52.22")

    def ensure_project_exists(self, entity: str, project: str) -> tsi.EnsureProjectExistsRes:
        return tsi.EnsureProjectExistsRes(project_name=project)

    # --- Call API ---

    @validate_call
    def call_start(self, req: tsi.CallStartReq) -> tsi.CallStartRes:
        request_content = req.start.model_dump(exclude_none=True)
        call_id = request_content.get("id") or generate_id()
        trace_id = request_content.get("trace_id") or generate_id()
        request_content["id"] = call_id
        request_content["trace_id"] = trace_id

        self.calls[call_id] = tsi.CallSchema(**request_content)
        return tsi.CallStartRes(id=call_id, trace_id=trace_id)

    @validate_call
    def call_end(self, req: tsi.CallEndReq) -> tsi.CallEndRes:
        if req.end.id in self.calls:
            request_content = req.end.model_dump(exclude_none=True)
            self.calls[req.end.id] = self.calls[req.end.id].model_copy(update=request_content)
        return tsi.CallEndRes()

    @validate_call
    def call_start_batch(self, req: tsi.CallCreateBatchReq) -> tsi.CallCreateBatchRes:
        for item in req.batch:
            if isinstance(item, tsi.CallStartReq):
                self.call_start(item)
            elif isinstance(item, tsi.CallEndReq):
                self.call_end(item)
        return tsi.CallCreateBatchRes(res=[])

    @validate_call
    def call_read(self, req: tsi.CallReadReq) -> tsi.CallReadRes:
        call_data = self.calls.get(req.id)
        return tsi.CallReadRes(call=call_data)

    @validate_call
    def calls_query(self, req: tsi.CallsQueryReq) -> tsi.CallsQueryRes:
        return tsi.CallsQueryRes(calls=list(self.calls_query_stream(req)))

    @validate_call
    def calls_query_stream(self, req: tsi.CallsQueryReq) -> Iterator[tsi.CallSchema]:
        yield from self.calls.values()

    @validate_call
    def calls_delete(self, req: tsi.CallsDeleteReq) -> tsi.CallsDeleteRes:
        num_deleted = 0
        for call_id in req.call_ids:
            if call_id in self.calls:
                del self.calls[call_id]
                num_deleted += 1
        return tsi.CallsDeleteRes(num_deleted=num_deleted)

    @validate_call
    def call_update(self, req: tsi.CallUpdateReq) -> tsi.CallUpdateRes:
        return tsi.CallUpdateRes()

    @validate_call
    def calls_query_stats(self, req: tsi.CallsQueryStatsReq) -> tsi.CallsQueryStatsRes:
        return tsi.CallsQueryStatsRes(count=len(self.calls))

    # --- Cost API ---

    @validate_call
    def cost_create(self, req: tsi.CostCreateReq) -> tsi.CostCreateRes:
        return tsi.CostCreateRes(ids=[(generate_id(), generate_id()) for _ in req.costs])

    @validate_call
    def cost_query(self, req: tsi.CostQueryReq) -> tsi.CostQueryRes:
        return tsi.CostQueryRes(results=[])

    @validate_call
    def cost_purge(self, req: tsi.CostPurgeReq) -> tsi.CostPurgeRes:
        return tsi.CostPurgeRes()

    # --- Object API (Legacy V1) ---

    @validate_call
    def obj_create(self, req: tsi.ObjCreateReq) -> tsi.ObjCreateRes:
        digest = generate_id()
        self.objs[digest] = req.obj
        return tsi.ObjCreateRes(digest=digest)

    @validate_call
    def obj_read(self, req: tsi.ObjReadReq) -> tsi.ObjReadRes:
        return tsi.ObjReadRes(obj=self.objs.get(req.digest, {}))

    @validate_call
    def objs_query(self, req: tsi.ObjQueryReq) -> tsi.ObjQueryRes:
        return tsi.ObjQueryRes(objs=[])

    @validate_call
    def obj_delete(self, req: tsi.ObjDeleteReq) -> tsi.ObjDeleteRes:
        return tsi.ObjDeleteRes(num_deleted=0)

    # --- Table API ---

    @validate_call
    def table_create(self, req: tsi.TableCreateReq) -> tsi.TableCreateRes:
        return tsi.TableCreateRes(digest=generate_id(), row_digests=[])

    @validate_call
    def table_create_from_digests(self, req: tsi.TableCreateFromDigestsReq) -> tsi.TableCreateFromDigestsRes:
        return tsi.TableCreateFromDigestsRes(digest=generate_id())

    @validate_call
    def table_update(self, req: tsi.TableUpdateReq) -> tsi.TableUpdateRes:
        return tsi.TableUpdateRes(digest=generate_id(), updated_row_digests=[])

    @validate_call
    def table_query(self, req: tsi.TableQueryReq) -> tsi.TableQueryRes:
        return tsi.TableQueryRes(rows=[])

    @validate_call
    def table_query_stream(self, req: tsi.TableQueryReq) -> Iterator[tsi.TableRowSchema]:
        yield from []

    @validate_call
    def table_query_stats(self, req: tsi.TableQueryStatsReq) -> tsi.TableQueryStatsRes:
        return tsi.TableQueryStatsRes(count=0)

    @validate_call
    def table_query_stats_batch(self, req: tsi.TableQueryStatsBatchReq) -> tsi.TableQueryStatsBatchRes:
        return tsi.TableQueryStatsBatchRes(tables=[])

    # --- Ref API ---

    @validate_call
    def refs_read_batch(self, req: tsi.RefsReadBatchReq) -> tsi.RefsReadBatchRes:
        return tsi.RefsReadBatchRes(vals=[])

    # --- File API ---

    def file_create(self, req: tsi.FileCreateReq) -> tsi.FileCreateRes:
        self.files[req.name] = req.content
        return tsi.FileCreateRes(digest=generate_id())

    def file_content_read(self, req: tsi.FileContentReadReq) -> tsi.FileContentReadRes:
        return tsi.FileContentReadRes(content=self.files.get(req.digest, b"dummy_content"))

    def files_stats(self, req: tsi.FilesStatsReq) -> tsi.FilesStatsRes:
        total_size = sum(len(c) for c in self.files.values())
        return tsi.FilesStatsRes(total_size_bytes=total_size)

    # --- Feedback API ---

    @validate_call
    def feedback_create(self, req: tsi.FeedbackCreateReq) -> tsi.FeedbackCreateRes:
        req.id = req.id or generate_id()
        self.feedback.append(req)
        return tsi.FeedbackCreateRes(
            id=req.id,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            wb_user_id="dummy_user",
            payload=req.payload,
        )

    def feedback_create_batch(self, req: tsi.FeedbackCreateBatchReq) -> tsi.FeedbackCreateBatchRes:
        results: List[tsi.FeedbackCreateRes] = []
        for item in req.batch:
            res = self.feedback_create(item)
            results.append(res)
        return tsi.FeedbackCreateBatchRes(res=results)

    @validate_call
    def feedback_query(self, req: tsi.FeedbackQueryReq) -> tsi.FeedbackQueryRes:
        return tsi.FeedbackQueryRes(result=[])

    @validate_call
    def feedback_purge(self, req: tsi.FeedbackPurgeReq) -> tsi.FeedbackPurgeRes:
        self.feedback.clear()
        return tsi.FeedbackPurgeRes()

    @validate_call
    def feedback_replace(self, req: tsi.FeedbackReplaceReq) -> tsi.FeedbackReplaceRes:
        return tsi.FeedbackReplaceRes(
            id=req.id or generate_id(),
            created_at=datetime.datetime.now(datetime.timezone.utc),
            wb_user_id="dummy",
            payload={},
        )

    # --- Action API ---

    @validate_call
    def actions_execute_batch(self, req: tsi.ActionsExecuteBatchReq) -> tsi.ActionsExecuteBatchRes:
        return tsi.ActionsExecuteBatchRes()

    # --- Execute LLM API ---

    @validate_call
    def completions_create(self, req: tsi.CompletionsCreateReq) -> tsi.CompletionsCreateRes:
        return tsi.CompletionsCreateRes(response={"choices": [{"text": "dummy completion"}]})

    @validate_call
    def completions_create_stream(self, req: tsi.CompletionsCreateReq) -> Iterator[dict[str, Any]]:
        yield {"choices": [{"text": "dummy "}]}
        yield {"choices": [{"text": "stream"}]}

    # --- Execute Image Generation API ---

    @validate_call
    def image_create(self, req: tsi.ImageGenerationCreateReq) -> tsi.ImageGenerationCreateRes:
        return tsi.ImageGenerationCreateRes(response={})

    # --- Project Statistics API ---

    @validate_call
    def project_stats(self, req: tsi.ProjectStatsReq) -> tsi.ProjectStatsRes:
        return tsi.ProjectStatsRes(
            trace_storage_size_bytes=0,
            objects_storage_size_bytes=0,
            tables_storage_size_bytes=0,
            files_storage_size_bytes=0,
        )

    # --- Thread API ---

    @validate_call
    def threads_query_stream(self, req: tsi.ThreadsQueryReq) -> Iterator[tsi.ThreadSchema]:
        yield from []

    # --- Evaluation API (V1) ---

    @validate_call
    def evaluate_model(self, req: tsi.EvaluateModelReq) -> tsi.EvaluateModelRes:
        return tsi.EvaluateModelRes(call_id=generate_id())

    @validate_call
    def evaluation_status(self, req: tsi.EvaluationStatusReq) -> tsi.EvaluationStatusRes:
        return tsi.EvaluationStatusRes(status=tsi.EvaluationStatusNotFound())

    # --- OTEL API ---

    def otel_export(self, req: tsi.OtelExportReq) -> tsi.OtelExportRes:
        return tsi.OtelExportRes()

    # ==========================================
    # Object Interface (V2 APIs)
    # ==========================================

    # --- Ops ---
    def op_create(self, req: tsi.OpCreateReq) -> tsi.OpCreateRes:
        return tsi.OpCreateRes(digest=generate_id(), object_id=generate_id(), version_index=0)

    def op_read(self, req: tsi.OpReadReq) -> tsi.OpReadRes:
        return tsi.OpReadRes(op=None)  # type: ignore

    def op_list(self, req: tsi.OpListReq) -> Iterator[tsi.OpReadRes]:
        yield from []

    def op_delete(self, req: tsi.OpDeleteReq) -> tsi.OpDeleteRes:
        return tsi.OpDeleteRes(num_deleted=0)

    # --- Datasets ---
    def dataset_create(self, req: tsi.DatasetCreateReq) -> tsi.DatasetCreateRes:
        return tsi.DatasetCreateRes(digest=generate_id(), object_id=generate_id(), version_index=0)

    def dataset_read(self, req: tsi.DatasetReadReq) -> tsi.DatasetReadRes:
        return tsi.DatasetReadRes(dataset=None)  # type: ignore

    def dataset_list(self, req: tsi.DatasetListReq) -> Iterator[tsi.DatasetReadRes]:
        yield from []

    def dataset_delete(self, req: tsi.DatasetDeleteReq) -> tsi.DatasetDeleteRes:
        return tsi.DatasetDeleteRes(num_deleted=0)

    # --- Scorers ---
    def scorer_create(self, req: tsi.ScorerCreateReq) -> tsi.ScorerCreateRes:
        return tsi.ScorerCreateRes(digest=generate_id(), object_id=generate_id(), version_index=0, scorer=generate_id())

    def scorer_read(self, req: tsi.ScorerReadReq) -> tsi.ScorerReadRes:
        return tsi.ScorerReadRes(scorer=None)  # type: ignore

    def scorer_list(self, req: tsi.ScorerListReq) -> Iterator[tsi.ScorerReadRes]:
        yield from []

    def scorer_delete(self, req: tsi.ScorerDeleteReq) -> tsi.ScorerDeleteRes:
        return tsi.ScorerDeleteRes(num_deleted=0)

    # --- Evaluations (V2) ---
    def evaluation_create(self, req: tsi.EvaluationCreateReq) -> tsi.EvaluationCreateRes:
        return tsi.EvaluationCreateRes(
            digest=generate_id(), object_id=generate_id(), version_index=0, evaluation_ref=generate_id()
        )

    def evaluation_read(self, req: tsi.EvaluationReadReq) -> tsi.EvaluationReadRes:
        return tsi.EvaluationReadRes(evaluation=None)  # type: ignore

    def evaluation_list(self, req: tsi.EvaluationListReq) -> Iterator[tsi.EvaluationReadRes]:
        yield from []

    def evaluation_delete(self, req: tsi.EvaluationDeleteReq) -> tsi.EvaluationDeleteRes:
        return tsi.EvaluationDeleteRes(num_deleted=0)

    # --- Models ---
    def model_create(self, req: tsi.ModelCreateReq) -> tsi.ModelCreateRes:
        return tsi.ModelCreateRes(
            digest=generate_id(), object_id=generate_id(), version_index=0, model_ref=generate_id()
        )

    def model_read(self, req: tsi.ModelReadReq) -> tsi.ModelReadRes:
        return tsi.ModelReadRes(model=None)  # type: ignore

    def model_list(self, req: tsi.ModelListReq) -> Iterator[tsi.ModelReadRes]:
        yield from []

    def model_delete(self, req: tsi.ModelDeleteReq) -> tsi.ModelDeleteRes:
        return tsi.ModelDeleteRes(num_deleted=0)

    # --- Evaluation Runs ---
    def evaluation_run_create(self, req: tsi.EvaluationRunCreateReq) -> tsi.EvaluationRunCreateRes:
        return tsi.EvaluationRunCreateRes(evaluation_run_id=generate_id())

    def evaluation_run_read(self, req: tsi.EvaluationRunReadReq) -> tsi.EvaluationRunReadRes:
        return tsi.EvaluationRunReadRes(evaluation_run=None)  # type: ignore

    def evaluation_run_list(self, req: tsi.EvaluationRunListReq) -> Iterator[tsi.EvaluationRunReadRes]:
        yield from []

    def evaluation_run_delete(self, req: tsi.EvaluationRunDeleteReq) -> tsi.EvaluationRunDeleteRes:
        return tsi.EvaluationRunDeleteRes(num_deleted=0)

    def evaluation_run_finish(self, req: tsi.EvaluationRunFinishReq) -> tsi.EvaluationRunFinishRes:
        return tsi.EvaluationRunFinishRes(success=True)

    # --- Predictions ---
    def prediction_create(self, req: tsi.PredictionCreateReq) -> tsi.PredictionCreateRes:
        return tsi.PredictionCreateRes(prediction_id=generate_id())

    def prediction_read(self, req: tsi.PredictionReadReq) -> tsi.PredictionReadRes:
        return tsi.PredictionReadRes(prediction=None)  # type: ignore

    def prediction_list(self, req: tsi.PredictionListReq) -> Iterator[tsi.PredictionReadRes]:
        yield from []

    def prediction_delete(self, req: tsi.PredictionDeleteReq) -> tsi.PredictionDeleteRes:
        return tsi.PredictionDeleteRes(num_deleted=0)

    def prediction_finish(self, req: tsi.PredictionFinishReq) -> tsi.PredictionFinishRes:
        return tsi.PredictionFinishRes(success=True)

    # --- Scores ---
    def score_create(self, req: tsi.ScoreCreateReq) -> tsi.ScoreCreateRes:
        return tsi.ScoreCreateRes(score_id=generate_id())

    def score_read(self, req: tsi.ScoreReadReq) -> tsi.ScoreReadRes:
        return tsi.ScoreReadRes(score=None)  # type: ignore

    def score_list(self, req: tsi.ScoreListReq) -> Iterator[tsi.ScoreReadRes]:
        yield from []

    def score_delete(self, req: tsi.ScoreDeleteReq) -> tsi.ScoreDeleteRes:
        return tsi.ScoreDeleteRes(num_deleted=0)


class WeaveTracer(Tracer):
    """Tracer implementation using Weave for telemetry and trace logging.

    This replaces AgentOpsTracer with a Weave-based manual trace context. It tracks:

    - Function/method calls
    - Input/Output data
    - Exceptions

    and logs them to Weave Cloud (W&B backend) or optionally bypasses the network for testing.

    Attributes:
        project_name: Name of the Weave project. Used to initialize the Weave client.
        instrument_managed: Whether to patch the Weave/W&B integration to bypass actual network calls for testing.
    """

    def __init__(
        self, *, project_name: str | None = None, wandb_api_key: str | None = None, instrument_managed: bool = True
    ):
        """
        Initialize a WeaveTracer instance.

        Args:
            project_name: Optional project name for Weave; defaults to the current module name.
            wandb_api_key: Optional W&B API key; sets environment variable if provided.
            instrument_managed: Whether to patch the Weave/W&B integration to bypass actual network calls for testing.
        """
        super().__init__()
        self.project_name = project_name or __name__
        self.sequence_id = 0
        self._store: Optional[LightningStore] = None
        self.instrument_managed = instrument_managed
        self._server = InMemoryTraceServer()

        self._default_sequence_counter: int = 0
        self._spans: List[Span] = []
        self._rollout_id: Optional[str] = None
        self._attempt_id: Optional[str] = None

        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

    def instrument(self, worker_id: int):
        instrument_weave()

    def uninstrument(self, worker_id: int):
        uninstrument_weave()

    def init_worker(self, worker_id: int, store: Optional[LightningStore] = None):
        """
        Initialize the tracer for a worker thread/process.

        Args:
            worker_id: Identifier of the worker.
            store: Optional LightningStore for storing spans.
        """
        super().init_worker(worker_id, store)
        logger.info(f"[Worker {worker_id}] Setting up Weave tracer...")
        self._store = store

        try:
            import weave
        except ImportError as exc:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.") from exc

        # Optionally patch network calls to bypass real Weave/W&B endpoints
        # if self.instrument_managed:
        #     self.instrument(worker_id)

        import weave.trace.weave_init

        def init_weave_get_server(*args, **kwargs):
            return self._server

        weave.trace.weave_init.init_weave_get_server = init_weave_get_server

        # Initialize the Weave client if not already initialized
        if weave.get_client() is None:  # type: ignore
            try:
                weave.init(project_name=self.project_name)  # type: ignore
                logger.info(f"[Worker {worker_id}] Weave client initialized.")
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Weave for project '{self.project_name}'") from exc
        else:
            # FIXME
            pass

    def teardown_worker(self, worker_id: int):
        """
        Clean up tracer resources for the worker.

        Args:
            worker_id: Identifier of the worker.
        """
        super().teardown_worker(worker_id)

        # if self.instrument_managed:
        #     self.uninstrument(worker_id)
        #     logger.info(f"[Worker {worker_id}] Instrumentation removed.")

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Synchronous implementation of the tracing context.

        Args:
            name: Optional operation name.
            store: Optional LightningStore instance.
            rollout_id: Optional rollout ID.
            attempt_id: Optional attempt ID.

        Raises:
            ValueError: If store, rollout_id, and attempt_id are inconsistently provided.
            RuntimeError: If Weave is not installed or client is uninitialized.
        """
        arg_op = name or self.project_name
        arg_inputs: dict[str, str] | None = {"rollout_id": rollout_id or "", "attempt_id": attempt_id or ""}

        self._spans.clear()

        try:
            import weave
        except ImportError as exc:
            raise RuntimeError("Weave is not installed. Install it to use WeaveTracer.") from exc

        weave_client = weave.get_client()
        if not weave_client:
            raise RuntimeError("Weave client is not initialized. Call init_worker() first.")

        if rollout_id is not None and attempt_id is not None:
            self._rollout_id = rollout_id
            self._attempt_id = attempt_id
        elif rollout_id is None and attempt_id is None:
            logger.warning("No rollout_id or attempt_id provided. Skipping writing to store.")
        else:
            raise ValueError("rollout_id and attempt_id must be either both provided or both None")

        # Create a new trace call object in Weave
        trace_call = weave_client.create_call(op=arg_op, inputs=arg_inputs)  # pyright: ignore[reportUnknownMemberType]
        trace_call.started_at = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            yield trace_call
        except Exception as e:
            # Finish trace and log any exception
            weave_client.finish_call(trace_call, exception=e)  # pyright: ignore[reportUnknownMemberType]
            logger.error(f"Trace failed for rollout_id={rollout_id}, attempt_id={attempt_id}, error={e}")
        finally:
            # Finish trace even if no exception
            weave_client.finish_call(trace_call)  # pyright: ignore[reportUnknownMemberType]
            await self._on_finish_handler(trace_call)

            self._rollout_id = None
            self._attempt_id = None

    def get_last_trace(self) -> List[Span]:
        return self._spans

    async def _on_finish_handler(self, call: Call, *args: Any, **kwargs: Any) -> None:
        """
        Handler called when a Weave Call finishes.

        Converts the call (including nested children) into spans and stores them in LightningStore.
        """
        spans = await self.convert_call_to_spans(call, self._rollout_id, self._attempt_id)
        self._spans.extend(spans)

        if self._store and self._rollout_id and self._attempt_id:
            try:
                await self._store.add_many_spans(spans)
            except Exception as e:
                logger.exception(f"Error adding span to store: {e}")

    async def _sequence_generator(self, request_count: int) -> Sequence[int]:
        if self._rollout_id and self._attempt_id and self._store:
            return await self._store.get_many_span_sequence_ids(
                [(self._rollout_id, self._attempt_id) for _ in range(request_count)]
            )
        else:
            ret = [self._default_sequence_counter + i for i in range(request_count)]
            self._default_sequence_counter += request_count
            return ret

    async def convert_call_to_spans(
        self,
        call: Call,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> List[Span]:
        """
        Recursively convert a Weave Call (with nested children) into a flat list of Agent Lightning Spans.

        Args:
            call: The Weave Call object.
            rollout_id: Optional rollout ID to attach to spans.
            attempt_id: Optional attempt ID to attach to spans.
            sequence_generator: Callable to generate sequence IDs.

        Returns:
            List of converted spans.
        """
        spans: List[Span] = []
        sequence_id = await self._sequence_generator(1)

        # print(f"call: {call}")

        rollout_id = rollout_id or ""
        attempt_id = attempt_id or ""

        start_dt = call.started_at
        start_ts: Optional[float] = start_dt.timestamp() if start_dt else None

        end_dt = call.ended_at
        end_ts: Optional[float] = end_dt.timestamp() if end_dt else None

        trace_id = call.trace_id
        span_id = call.id or generate_span_id()
        parent_id = call.parent_id

        exception = getattr(call, "exception", None)
        status_code = "ERROR" if exception else "OK"

        def sanitize(
            inputs: Dict[str, Any],
            output: Dict[str, Any],
        ) -> Dict[str, str | JSONPrimitive]:
            stack: List[Tuple[Any, str]] = [
                (inputs or {}, "input"),
                (output or {}, "output"),
            ]

            attributes: Dict[str, str | JSONPrimitive] = {}

            while stack:
                value, key = stack.pop()

                if isinstance(value, dict):
                    for k, v in cast(Dict[str, Any], value).items():
                        stack.append((v, f"{key}.{k}"))
                elif isinstance(value, (list, tuple)):
                    for i, v in enumerate(cast(List[Any], value)):
                        stack.append((v, f"{key}.{i}"))
                else:
                    if value is None:
                        attributes[key] = "None"
                    elif isinstance(value, (str, int, float, bool)):
                        attributes[key] = value
                    else:
                        try:
                            attributes[key] = str(value)
                        except Exception:
                            attributes[key] = "None"

            return attributes

        inputs = call.inputs
        output = call.output
        attributes = sanitize(inputs, output)

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_state={},
        )

        parent_context = (
            SpanContext(
                trace_id=trace_id,
                span_id=parent_id,
                is_remote=False,
                trace_state={},
            )
            if parent_id
            else None
        )

        # Build the Span object
        span = Span(
            rollout_id=rollout_id or "",
            attempt_id=attempt_id or "",
            sequence_id=sequence_id[0],
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=call.func_name,
            status=TraceStatus(status_code=status_code),
            attributes=attributes,  # type: ignore
            events=[],  # Weave calls do not generate events
            links=[],  # Weave calls do not generate links
            start_time=start_ts,
            end_time=end_ts,
            context=context,
            parent=parent_context,
            resource=OtelResource(attributes={}, schema_url=""),
        )

        spans.append(span)

        children: List[Call] = call._children  # pyright: ignore[reportPrivateUsage]
        # Recursively process child calls
        for child in children:
            child_spans = await self.convert_call_to_spans(
                child,
                rollout_id=rollout_id,
                attempt_id=attempt_id,
            )
            spans.extend(child_spans)

        return spans
