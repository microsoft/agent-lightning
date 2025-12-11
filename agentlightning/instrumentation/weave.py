# Copyright (c) Microsoft. All rights reserved.

import logging
import os
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "instrument_weave",
    "uninstrument_weave",
]

# Module-level storage for originals
_original_default_entity_name_getter: Callable[..., Any] | None = None
_original_upsert_project_getter: Callable[..., Any] | None = None
_original_weave_get = False
_original_weave_post = False


def instrument_weave():
    """
    Patch the Weave/W&B integration to bypass actual network calls for testing.

    - Mocks HTTP POST/GET requests
    - Patches wandb.Api methods
    - Silences Weave logging
    - Sets dummy WANDB_API_KEY if not provided
    """
    try:
        import weave
        from weave.compat import wandb  # type: ignore
    except ImportError:
        logger.warning("Weave or wandb not installed; cannot uninstrument.")
        return

    _weave_tracer_entity_name = "weave_tracer_entity"

    def default_entity_name_getter(_self) -> str:  # type: ignore
        return _weave_tracer_entity_name

    def upsert_project_getter(
        _self, project: str, description: Optional[str] = None, entity: Optional[str] = None  # type: ignore
    ) -> dict[str, Any]:
        return {
            "upsertModel": {
                "model": {
                    "name": project,
                    "description": description or "",
                    "entity": entity or _weave_tracer_entity_name,
                }
            },
            "project": "weave_tracer_project",
        }

    # Mock network requests to avoid real HTTP calls
    def post(url: str, *args: Any, **kwargs: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"digest": "mocked_digest"}'
        return response

    def get(url: str, *args: Any, **kwargs: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"min_required_weave_python_version": "0.52.14"}'
        return response

    # Patch API methods and HTTP requests
    global _original_default_entity_name_getter
    global _original_upsert_project_getter
    global _original_weave_post
    global _original_weave_get
    _original_default_entity_name_getter = wandb.Api.default_entity_name  # type: ignore
    _original_upsert_project_getter = wandb.Api.upsert_project  # type: ignore
    _original_weave_post = weave.utils.http_requests.post  # type: ignore
    _original_weave_get = weave.utils.http_requests.get  # type: ignore

    # Patch API methods and HTTP requests
    wandb.Api.default_entity_name = default_entity_name_getter  # type: ignore
    wandb.Api.upsert_project = upsert_project_getter  # type: ignore
    weave.utils.http_requests.post = post  # type: ignore
    weave.utils.http_requests.get = get  # type: ignore

    # Silence Weave logging
    for name in logging.root.manager.loggerDict:
        if name.startswith("weave"):
            logging.getLogger(name).disabled = True

    # Set dummy API key if missing
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = "dumped_api_key_for_weave_tracer"

    # if needed in future tests, enable this and replace WF_TRACE_SERVER_URL to local server
    # full_url = f"http://127.0.0.1:{_port}"
    # os.environ["WF_TRACE_SERVER_URL"] = full_url


def uninstrument_weave():
    """
    Restore the original Weave/W&B integration methods and HTTP requests.
    """
    try:
        import weave
        from weave.compat import wandb  # type: ignore
    except ImportError:
        logger.warning("Weave or wandb not installed; cannot uninstrument.")
        return

    global _original_default_entity_name_getter
    if _original_default_entity_name_getter is not None:
        wandb.Api.default_entity_name = _original_default_entity_name_getter  # type: ignore
        _original_default_entity_name_getter = None
        logger.info("restored wandb.Api.default_entity_name")

    global _original_upsert_project_getter
    if _original_upsert_project_getter is not None:
        wandb.Api.upsert_project = _original_upsert_project_getter  # type: ignore
        _original_upsert_project_getter = None
        logger.info("restored wandb.Api.upsert_project")

    global _original_weave_post
    if _original_weave_post is not None:
        weave.utils.http_requests.post = _original_weave_post  # type: ignore
        _original_weave_post = None
        logger.info("restored weave.utils.http_requests.session.post")

    global _original_weave_get
    if _original_weave_get is not None:
        weave.utils.http_requests.get = _original_weave_get  # type: ignore
        _original_weave_get = None
        logger.info("restored weave.utils.http_requests.session.get")

    # Restore Weave logging
    for name in logging.root.manager.loggerDict:
        if name.startswith("weave"):
            logging.getLogger(name).disabled = False
