# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List

import weave.trace.weave_init
from pydantic import validate_call
from weave.trace_server import trace_server_interface as tsi
from weave.trace_server.ids import generate_id
from weave.trace_server_bindings.client_interface import TraceServerClientInterface
from weave.trace_server_bindings.models import ServerInfoRes

logger = logging.getLogger(__name__)

__all__ = [
    "instrument_weave",
    "uninstrument_weave",
    "InMemoryWeaveTraceServer",
]


class InMemoryWeaveTraceServer(TraceServerClientInterface):
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
    def from_env(cls, *args: Any, **kwargs: Any) -> InMemoryWeaveTraceServer:
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
        request_content = req.end.model_dump(exclude_none=True)
        if req.end.id in self.calls:
            self.calls[req.end.id] = self.calls[req.end.id].model_copy(update=request_content)
        else:
            self.calls[req.end.id] = tsi.CallSchema(**request_content)
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
            created_at=datetime.now(timezone.utc),
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
            created_at=datetime.now(timezone.utc),
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


# Module-level storage for originals
_original_init_weave_get_server: Callable[..., Any] | None = None
_original_get_entity_project_from_project_name: Callable[..., Any] | None = None


def init_weave_get_server_factory(server: InMemoryWeaveTraceServer) -> Callable[..., Any]:
    def init_weave_get_server(*args: Any, **kwargs: Any) -> InMemoryWeaveTraceServer:
        return server

    return init_weave_get_server


def get_entity_project_from_project_name_factory(entity_name: str) -> tuple[str, str]:
    try:
        assert _original_get_entity_project_from_project_name is not None
        return _original_get_entity_project_from_project_name(entity_name)
    except weave.trace.weave_init.WeaveWandbAuthenticationException:
        return "agl", "weave"


def instrument_weave(server: InMemoryWeaveTraceServer):
    """Patch the Weave/W&B integration to bypass actual network calls for testing."""

    global _original_init_weave_get_server, _original_get_entity_project_from_project_name
    _original_init_weave_get_server = weave.trace.weave_init.init_weave_get_server
    _original_get_entity_project_from_project_name = weave.trace.weave_init.get_entity_project_from_project_name
    weave.trace.weave_init.init_weave_get_server = init_weave_get_server_factory(server)
    weave.trace.weave_init.get_entity_project_from_project_name = get_entity_project_from_project_name_factory


def uninstrument_weave():
    """Restore the original Weave/W&B integration methods and HTTP requests."""
    global _original_init_weave_get_server, _original_get_entity_project_from_project_name

    if _original_init_weave_get_server is not None:
        weave.trace.weave_init.init_weave_get_server = _original_init_weave_get_server
        _original_init_weave_get_server = None
    else:
        raise RuntimeError("Weave/W&B integration was not instrumented.")

    if _original_get_entity_project_from_project_name is not None:
        weave.trace.weave_init.get_entity_project_from_project_name = _original_get_entity_project_from_project_name
        _original_get_entity_project_from_project_name = None
    else:
        raise RuntimeError("Weave/W&B integration was not instrumented.")
