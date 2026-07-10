# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import threading
import warnings
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Callable, Dict, Iterator, List, Type, TypeVar

from pydantic import validate_call

logger = logging.getLogger(__name__)

T_response = TypeVar("T_response")

weave_trace_init: Any = import_module("weave.trace.weave_init")
tsi: Any = import_module("weave.trace_server.trace_server_interface")
generate_id: Callable[[], str] = getattr(import_module("weave.trace_server.ids"), "generate_id")
TraceServerClientInterface: Type[Any] = getattr(
    import_module("weave.trace_server_bindings.client_interface"),
    "TraceServerClientInterface",
)
ServerInfoRes: Type[Any] = getattr(import_module("weave.trace_server_bindings.models"), "ServerInfoRes")

__all__ = [
    "instrument_weave",
    "uninstrument_weave",
    "InMemoryWeaveTraceServer",
]


def _make_response(response_cls: type[T_response], **kwargs: Any) -> T_response:
    return response_cls(**kwargs)


class InMemoryWeaveTraceServer(TraceServerClientInterface):
    """A minimal in-memory implementation of the TraceServerInterface.

    It stores calls and objects in local dictionaries and returns valid Pydantic
    responses to satisfy the Weave client and FullTraceServerInterface protocol.
    """

    def __init__(self):
        # Minimal storage to allow basic querying in tests
        self.calls: Dict[str, Any] = {}
        self.partial_calls: Dict[str, Dict[str, Any]] = {}
        self.objs: Dict[str, Any] = {}
        self.files: Dict[str, bytes] = {}
        self.feedback: List[Any] = []

        self._call_threading_lock = threading.Lock()

    @classmethod
    def from_env(cls, *args: Any, **kwargs: Any) -> InMemoryWeaveTraceServer:
        return cls()

    def server_info(self) -> Any:
        return ServerInfoRes(min_required_weave_python_version="0.52.22")

    def ensure_project_exists(self, entity: str, project: str) -> Any:
        return tsi.EnsureProjectExistsRes(project_name=project)

    # --- Call API ---

    @validate_call
    def call_start(self, req: Any) -> Any:
        # NOTE: It's not necessary that call_end must be called after call_start.
        request_content = req.start.model_dump(exclude_none=True)

        # If id needs to be generated here, it's very likely we won't be able to find the call later.
        # This is just to make the type checker happy.
        call_id = request_content.get("id") or generate_id()
        trace_id = request_content.get("trace_id") or generate_id()
        request_content["id"] = call_id
        request_content["trace_id"] = trace_id

        with self._call_threading_lock:
            if call_id in self.partial_calls:
                # call_end has already been called for this call.
                kwargs = {**request_content, **self.partial_calls[call_id]}
                self.calls[call_id] = tsi.CallSchema(**kwargs)
                del self.partial_calls[call_id]
            else:
                self.partial_calls[call_id] = request_content

        return tsi.CallStartRes(id=call_id, trace_id=trace_id)

    @validate_call
    def call_end(self, req: Any) -> Any:
        request_content = req.end.model_dump(exclude_none=True)
        call_id = req.end.id

        with self._call_threading_lock:
            if call_id in self.partial_calls:
                # End request always override the start request content.
                kwargs = {**self.partial_calls[call_id], **request_content}
                self.calls[call_id] = tsi.CallSchema(**kwargs)
                del self.partial_calls[call_id]
            else:
                self.partial_calls[call_id] = request_content
        return tsi.CallEndRes()

    @validate_call
    def call_start_batch(self, req: Any) -> Any:
        for item in req.batch:
            if isinstance(item, tsi.CallStartReq):
                self.call_start(item)
            elif isinstance(item, tsi.CallEndReq):
                self.call_end(item)
        return tsi.CallCreateBatchRes(res=[])

    @validate_call
    def call_read(self, req: Any) -> Any:
        call_data = self.calls.get(req.id)
        return tsi.CallReadRes(call=call_data)

    @validate_call
    def calls_query(self, req: Any) -> Any:
        return tsi.CallsQueryRes(calls=list(self.calls_query_stream(req)))

    @validate_call
    def calls_query_stream(self, req: Any) -> Iterator[Any]:
        yield from self.calls.values()

    @validate_call
    def calls_delete(self, req: Any) -> Any:
        num_deleted = 0
        for call_id in req.call_ids:
            if call_id in self.calls:
                del self.calls[call_id]
                num_deleted += 1
        return tsi.CallsDeleteRes(num_deleted=num_deleted)

    @validate_call
    def call_update(self, req: Any) -> Any:
        return tsi.CallUpdateRes()

    @validate_call
    def calls_query_stats(self, req: Any) -> Any:
        return tsi.CallsQueryStatsRes(count=len(self.calls))

    # --- Cost API ---

    @validate_call
    def cost_create(self, req: Any) -> Any:
        return tsi.CostCreateRes(ids=[(generate_id(), generate_id()) for _ in req.costs])

    @validate_call
    def cost_query(self, req: Any) -> Any:
        return tsi.CostQueryRes(results=[])

    @validate_call
    def cost_purge(self, req: Any) -> Any:
        return tsi.CostPurgeRes()

    # --- Object API (Legacy V1) ---

    @validate_call
    def obj_create(self, req: Any) -> Any:
        digest = generate_id()
        self.objs[digest] = req.obj
        return tsi.ObjCreateRes(digest=digest)

    @validate_call
    def obj_read(self, req: Any) -> Any:
        return tsi.ObjReadRes(obj=self.objs.get(req.digest, {}))

    @validate_call
    def objs_query(self, req: Any) -> Any:
        return tsi.ObjQueryRes(objs=[])

    @validate_call
    def obj_delete(self, req: Any) -> Any:
        return tsi.ObjDeleteRes(num_deleted=0)

    # --- Table API ---

    @validate_call
    def table_create(self, req: Any) -> Any:
        return tsi.TableCreateRes(digest=generate_id(), row_digests=[])

    @validate_call
    def table_create_from_digests(self, req: Any) -> Any:
        return tsi.TableCreateFromDigestsRes(digest=generate_id())

    @validate_call
    def table_update(self, req: Any) -> Any:
        return tsi.TableUpdateRes(digest=generate_id(), updated_row_digests=[])

    @validate_call
    def table_query(self, req: Any) -> Any:
        return tsi.TableQueryRes(rows=[])

    @validate_call
    def table_query_stream(self, req: Any) -> Iterator[Any]:
        yield from []

    @validate_call
    def table_query_stats(self, req: Any) -> Any:
        return tsi.TableQueryStatsRes(count=0)

    @validate_call
    def table_query_stats_batch(self, req: Any) -> Any:
        return tsi.TableQueryStatsBatchRes(tables=[])

    # --- Ref API ---

    @validate_call
    def refs_read_batch(self, req: Any) -> Any:
        return tsi.RefsReadBatchRes(vals=[])

    # --- File API ---

    def file_create(self, req: Any) -> Any:
        self.files[req.name] = req.content
        return tsi.FileCreateRes(digest=generate_id())

    def file_content_read(self, req: Any) -> Any:
        return tsi.FileContentReadRes(content=self.files.get(req.digest, b"dummy_content"))

    def files_stats(self, req: Any) -> Any:
        total_size = sum(len(c) for c in self.files.values())
        return tsi.FilesStatsRes(total_size_bytes=total_size)

    # --- Feedback API ---

    @validate_call
    def feedback_create(self, req: Any) -> Any:
        req.id = req.id or generate_id()
        self.feedback.append(req)
        return tsi.FeedbackCreateRes(
            id=req.id,
            created_at=datetime.now(timezone.utc),
            wb_user_id="dummy_user",
            payload=req.payload,
        )

    def feedback_create_batch(self, req: Any) -> Any:
        results: List[Any] = []
        for item in req.batch:
            res = self.feedback_create(item)
            results.append(res)
        return tsi.FeedbackCreateBatchRes(res=results)

    @validate_call
    def feedback_query(self, req: Any) -> Any:
        return tsi.FeedbackQueryRes(result=[])

    @validate_call
    def feedback_purge(self, req: Any) -> Any:
        self.feedback.clear()
        return tsi.FeedbackPurgeRes()

    @validate_call
    def feedback_replace(self, req: Any) -> Any:
        return tsi.FeedbackReplaceRes(
            id=req.id or generate_id(),
            created_at=datetime.now(timezone.utc),
            wb_user_id="dummy",
            payload={},
        )

    # --- Action API ---

    @validate_call
    def actions_execute_batch(self, req: Any) -> Any:
        return tsi.ActionsExecuteBatchRes()

    # --- Execute LLM API ---

    @validate_call
    def completions_create(self, req: Any) -> Any:
        return tsi.CompletionsCreateRes(response={"choices": [{"text": "dummy completion"}]})

    @validate_call
    def completions_create_stream(self, req: Any) -> Iterator[dict[str, Any]]:
        yield {"choices": [{"text": "dummy "}]}
        yield {"choices": [{"text": "stream"}]}

    # --- Execute Image Generation API ---

    @validate_call
    def image_create(self, req: Any) -> Any:
        return tsi.ImageGenerationCreateRes(response={})

    # --- Project Statistics API ---

    @validate_call
    def project_stats(self, req: Any) -> Any:
        return tsi.ProjectStatsRes(
            trace_storage_size_bytes=0,
            objects_storage_size_bytes=0,
            tables_storage_size_bytes=0,
            files_storage_size_bytes=0,
        )

    # --- Thread API ---

    @validate_call
    def threads_query_stream(self, req: Any) -> Iterator[Any]:
        yield from []

    # --- Evaluation API (V1) ---

    @validate_call
    def evaluate_model(self, req: Any) -> Any:
        return tsi.EvaluateModelRes(call_id=generate_id())

    @validate_call
    def evaluation_status(self, req: Any) -> Any:
        return tsi.EvaluationStatusRes(status=tsi.EvaluationStatusNotFound())

    # --- OTEL API ---

    def otel_export(self, req: Any) -> Any:
        return tsi.OtelExportRes()

    # ==========================================
    # Object Interface (V2 APIs)
    # ==========================================

    # --- Ops ---
    def op_create(self, req: Any) -> Any:
        return tsi.OpCreateRes(digest=generate_id(), object_id=generate_id(), version_index=0)

    def op_read(self, req: Any) -> Any:
        return _make_response(tsi.OpReadRes, op=None)

    def op_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def op_delete(self, req: Any) -> Any:
        return tsi.OpDeleteRes(num_deleted=0)

    # --- Datasets ---
    def dataset_create(self, req: Any) -> Any:
        return tsi.DatasetCreateRes(digest=generate_id(), object_id=generate_id(), version_index=0)

    def dataset_read(self, req: Any) -> Any:
        return _make_response(tsi.DatasetReadRes, dataset=None)

    def dataset_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def dataset_delete(self, req: Any) -> Any:
        return tsi.DatasetDeleteRes(num_deleted=0)

    # --- Scorers ---
    def scorer_create(self, req: Any) -> Any:
        return tsi.ScorerCreateRes(digest=generate_id(), object_id=generate_id(), version_index=0, scorer=generate_id())

    def scorer_read(self, req: Any) -> Any:
        return _make_response(tsi.ScorerReadRes, scorer=None)

    def scorer_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def scorer_delete(self, req: Any) -> Any:
        return tsi.ScorerDeleteRes(num_deleted=0)

    # --- Evaluations (V2) ---
    def evaluation_create(self, req: Any) -> Any:
        return tsi.EvaluationCreateRes(
            digest=generate_id(), object_id=generate_id(), version_index=0, evaluation_ref=generate_id()
        )

    def evaluation_read(self, req: Any) -> Any:
        return _make_response(tsi.EvaluationReadRes, evaluation=None)

    def evaluation_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def evaluation_delete(self, req: Any) -> Any:
        return tsi.EvaluationDeleteRes(num_deleted=0)

    # --- Models ---
    def model_create(self, req: Any) -> Any:
        return tsi.ModelCreateRes(
            digest=generate_id(), object_id=generate_id(), version_index=0, model_ref=generate_id()
        )

    def model_read(self, req: Any) -> Any:
        return _make_response(tsi.ModelReadRes, model=None)

    def model_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def model_delete(self, req: Any) -> Any:
        return tsi.ModelDeleteRes(num_deleted=0)

    # --- Evaluation Runs ---
    def evaluation_run_create(self, req: Any) -> Any:
        return tsi.EvaluationRunCreateRes(evaluation_run_id=generate_id())

    def evaluation_run_read(self, req: Any) -> Any:
        return _make_response(tsi.EvaluationRunReadRes, evaluation_run=None)

    def evaluation_run_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def evaluation_run_delete(self, req: Any) -> Any:
        return tsi.EvaluationRunDeleteRes(num_deleted=0)

    def evaluation_run_finish(self, req: Any) -> Any:
        return tsi.EvaluationRunFinishRes(success=True)

    # --- Predictions ---
    def prediction_create(self, req: Any) -> Any:
        return tsi.PredictionCreateRes(prediction_id=generate_id())

    def prediction_read(self, req: Any) -> Any:
        return _make_response(tsi.PredictionReadRes, prediction=None)

    def prediction_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def prediction_delete(self, req: Any) -> Any:
        return tsi.PredictionDeleteRes(num_deleted=0)

    def prediction_finish(self, req: Any) -> Any:
        return tsi.PredictionFinishRes(success=True)

    # --- Scores ---
    def score_create(self, req: Any) -> Any:
        return tsi.ScoreCreateRes(score_id=generate_id())

    def score_read(self, req: Any) -> Any:
        return _make_response(tsi.ScoreReadRes, score=None)

    def score_list(self, req: Any) -> Iterator[Any]:
        yield from []

    def score_delete(self, req: Any) -> Any:
        return tsi.ScoreDeleteRes(num_deleted=0)

    # Experimental unstable APIs
    # We don't support these APIs yet.
    def annotation_queue_create(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def annotation_queues_query_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def annotation_queue_read(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def annotation_queue_add_calls(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def annotation_queues_stats(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def annotation_queue_items_query(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def annotator_queue_items_progress_update(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def calls_complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def call_start_v2(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def call_end_v2(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def call_stats(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def trace_usage(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def calls_usage(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


# Module-level storage for originals
_original_init_weave_get_server: Callable[..., Any] | None = None
_original_get_entity_project_from_project_name: Callable[..., Any] | None = None
_original_get_username: Callable[..., Any] | None = None


def init_weave_get_server_factory(server: InMemoryWeaveTraceServer) -> Callable[..., Any]:
    # Bypass the usage of Weave remote server
    def init_weave_get_server(*args: Any, **kwargs: Any) -> InMemoryWeaveTraceServer:
        return server

    return init_weave_get_server


def get_entity_project_from_project_name_factory(entity_name: str) -> tuple[str, str]:
    # Bypass the usage of API
    try:
        assert _original_get_entity_project_from_project_name is not None
        if _original_get_entity_project_from_project_name is not get_entity_project_from_project_name_factory:
            return _original_get_entity_project_from_project_name(entity_name)
        else:
            warnings.warn("W&B integration might have been repeatedly/recursively instrumented.")
            return "agl", "weave"
    except weave_trace_init.WeaveWandbAuthenticationException:
        # In case API is not available.
        return "agl", "weave"


def get_username() -> str:
    # Bypass the usage of API
    try:
        assert _original_get_username is not None
        return _original_get_username()
    except RuntimeError:
        return "agl"
    except Exception as exc:
        warnings.warn(f"Unexpected error in get_username. Using default username. Error: {exc}")
        return "agl"


def instrument_weave(server: InMemoryWeaveTraceServer):
    """Patch the Weave/W&B integration to bypass actual network calls for testing."""

    global _original_init_weave_get_server, _original_get_entity_project_from_project_name, _original_get_username
    _original_init_weave_get_server = weave_trace_init.init_weave_get_server
    _original_get_entity_project_from_project_name = weave_trace_init.get_entity_project_from_project_name
    _original_get_username = weave_trace_init.get_username
    weave_trace_init.init_weave_get_server = init_weave_get_server_factory(server)
    weave_trace_init.get_entity_project_from_project_name = get_entity_project_from_project_name_factory
    weave_trace_init.get_username = get_username


def uninstrument_weave():
    """Restore the original Weave/W&B integration methods and HTTP requests."""
    global _original_init_weave_get_server, _original_get_entity_project_from_project_name, _original_get_username

    if _original_init_weave_get_server is not None:
        weave_trace_init.init_weave_get_server = _original_init_weave_get_server
        _original_init_weave_get_server = None
    else:
        raise RuntimeError("Weave/W&B integration was not instrumented.")

    if _original_get_entity_project_from_project_name is not None:
        weave_trace_init.get_entity_project_from_project_name = _original_get_entity_project_from_project_name
        _original_get_entity_project_from_project_name = None
    else:
        raise RuntimeError("Weave/W&B integration was not instrumented.")

    if _original_get_username is not None:
        weave_trace_init.get_username = _original_get_username
        _original_get_username = None
    else:
        raise RuntimeError("Weave/W&B integration was not instrumented.")
