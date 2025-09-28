from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI

# LiteLLM custom callback base
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.config import ProxyConfig

# LiteLLM proxy + config (programmatic)
from litellm.proxy.proxy_server import ProxyConfig

# ---- import your types -------------------------------------------------------
# (from the code you gave me; not repeated here)
# - LLM, LightningStore, LLMProxyCache, ResourcesUpdate, RolloutV2, Span, etc.

SamplingParameters = Dict[str, Any]  # simple alias to keep things flexible


class _LightningCallback(CustomLogger):
    """
    Proxy callbacks for logging/tracing.

    Works both for SDK hooks (log_* methods) and proxy-only hooks:
      - async_pre_call_hook(data, user_api_key_dict)
      - async_post_call_success_hook(data, user_api_key_dict, response)
    Docs on hook names & kwargs: https://docs.litellm.ai/docs/observability/custom_callback
    """

    def __init__(self, *, store: Optional[LightningStore] = None, cache: Optional[LLMProxyCache] = None) -> None:
        super().__init__()
        self._store = store
        self._cache = cache
        self._log = logging.getLogger("LLMProxy.Callbacks")

    # ----- SDK-style hooks (for completeness when not running via proxy)
    def log_pre_api_call(self, model, messages, kwargs):
        self._log.debug("pre_api_call model=%s meta=%s", model, kwargs.get("metadata"))

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        self._log.debug(
            "post_api_call model=%s duration=%.3fs", kwargs.get("model"), (end_time - start_time).total_seconds()
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._log.info("success model=%s cost=%s", kwargs.get("model"), kwargs.get("response_cost"))

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._log.warning("failure model=%s error=%s", kwargs.get("model"), getattr(response_obj, "error", None))

    # ----- Proxy-only async hooks
    async def async_pre_call_hook(self, data: Dict[str, Any], user_api_key_dict: Dict[str, Any]):
        """
        Called by the LiteLLM Proxy before it hits a provider.

        'data' is the incoming OpenAI-format request body. We thread rollout path
        metadata into litellm_params.metadata so downstream callbacks can see it.
        """
        metadata = (data.get("litellm_params") or {}).get("metadata") or {}
        rollout_id = metadata.get("rollout_id") or data.get("metadata", {}).get("rollout_id")
        attempt_id = metadata.get("attempt_id") or data.get("metadata", {}).get("attempt_id")

        data.setdefault("litellm_params", {}).setdefault("metadata", {}).update(
            {"x_lineage_path": f"/rollout/{rollout_id or 'unknown'}/attempt/{attempt_id or 'unknown'}"}
        )

        # App-level, opt-in caching (very basic; you can swap in Redis etc.)
        # We only *log* cache hits here; to actually short-circuit responses you’d
        # write a lightweight Proxy plugin, but logging is usually enough.
        if self._cache:
            key = (data.get("model"), tuple((m.get("role"), m.get("content")) for m in data.get("messages", [])))
            try:
                cached = self._cache.get(key)
                if cached is not None:
                    self._log.info("app-cache: HIT model=%s", data.get("model"))
            except Exception as e:
                self._log.debug("app-cache get error: %s", e)

    async def async_post_call_success_hook(self, data: Dict[str, Any], user_api_key_dict: Dict[str, Any], response):
        """
        Called by the Proxy after provider returns successfully. Great place to
        forward lightweight telemetry to your LightningStore.
        """
        try:
            meta = (data.get("litellm_params") or {}).get("metadata") or {}
            lineage = meta.get("x_lineage_path", "")
            rollout_id = meta.get("rollout_id")
            attempt_id = meta.get("attempt_id")

            # Example: push a minimal 'Span-like' event to your store (if provided).
            if self._store and rollout_id and attempt_id:
                # You can evolve this to store full Events or map into your Span model.
                await self._store.add_span(
                    Span(
                        rollout_id=rollout_id,
                        attempt_id=attempt_id,
                        trace_id="",
                        span_id="",
                        parent_id=None,
                        name=f"llm_proxy:{data.get('model')}",
                        status={"status_code": "OK"},  # TraceStatus compatible
                        attributes={
                            "model": data.get("model"),
                            "route_lineage": lineage,
                            "cache_hit": getattr(response, "_hidden_params", {}).get("cache_hit", False),
                            "cost_usd": getattr(response, "_hidden_params", {}).get("response_cost", 0.0),
                        },
                        events=[],
                        links=[],
                        start_time=None,
                        end_time=None,
                        context=None,
                        parent=None,
                        resource={"attributes": {}, "schema_url": ""},
                    )
                )
        except Exception as e:
            self._log.debug("async_post_call_success_hook store error: %s", e)


class LLMProxy:
    """
    Proxy for LLM requests built on LiteLLM's Proxy server + Router.

    - Launches a local OpenAI-compatible gateway.
    - Registers custom callbacks for request/response logging.
    - Optionally wires into your LightningStore for span-like audit events.
    - Returns an `LLM` Resource pointing at this proxy via `as_resource()`.

    Path lineage tag: /rollout/<rollout_id>/attempt/<attempt_id>
    (Added to litellm_params.metadata so any callback / provider can see it.)
    """

    def __init__(
        self,
        host: str,
        port: int,
        servers: List[Dict[str, Any]],  # each entry is a 'litellm_params' dict (model/api_base/api_key/etc.)
        *,
        cache: Optional[LLMProxyCache] = None,
        store: Optional[LightningStore] = None,
        master_key: Optional[str] = None,
        litellm_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            servers: list of backend definitions, each directly usable as `litellm_params`
                     (e.g. {'model': 'openai/gpt-4o', 'api_key': '...', 'api_base': '...'})
                     This mirrors LiteLLM's 'model_list.litellm_params'. :contentReference[oaicite:0]{index=0}
            master_key: optional admin key for the proxy (auth, admin UI). :contentReference[oaicite:1]{index=1}
            general_settings/router_settings/model_settings: pass-through to ProxyConfig
                     (e.g. caching, budgets, prompt caching, etc.). :contentReference[oaicite:2]{index=2}
        """
        self.host = host
        self.port = port
        self._cache = cache
        self._store = store
        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn: Optional[uvicorn.Server] = None
        self._log = logging.getLogger("LLMProxy")

        # Build the LiteLLM proxy config (programmatic form).
        # Each backend becomes a model group (`model_name`) with a 'litellm_params' block.
        # You can use your own names here; clients must pass this name in `model=...`.
        model_list = []
        for i, params in enumerate(servers):
            model_list.append(
                {
                    "model_name": params.get("model_name") or f"backend-{i+1}",
                    "litellm_params": dict(params),  # e.g. {model, api_key, api_base, ...}
                }
            )

        # Server settings (bind address, etc.) live under `server_settings` on ProxyConfig (programmatic).
        server_settings = {"host": host, "port": port, "environment": "production"}

        # Install our custom callback handler so the proxy logs requests.
        self._callbacks = _LightningCallback(store=store, cache=cache)

        self._proxy_config = ProxyConfig(
            model_list=model_list,
            server_settings=server_settings,
            general_settings={**(general_settings or {}), **({"master_key": master_key} if master_key else {})},
            router_settings=router_settings or {},
            model_settings=model_settings or {},
            # You can also set litellm_settings.success_callback / failure_callback here,
            # but since we have a concrete handler instance, we attach it directly on the server.
        )

        # Build FastAPI app + mount routes from ProxyServer (supported programmatically). :contentReference[oaicite:3]{index=3}
        self._app = FastAPI(title="LiteLLM Proxy", version="1.0")
        self._proxy = ProxyServer(config=self._proxy_config, callbacks=[self._callbacks])
        self._proxy.add_routes(self._app)

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        """
        Start the LiteLLM proxy in-process (Uvicorn) on host:port.
        """
        if self._uvicorn and self._uvicorn.started:
            return  # already running

        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="info")
        self._uvicorn = uvicorn.Server(config)

        def _run():
            self._log.info("Starting LiteLLM Proxy on http://%s:%d", self.host, self.port)
            # run() blocks; we isolate it in a thread
            self._uvicorn.run()

        self._server_thread = threading.Thread(target=_run, name="LLMProxy(Uvicorn)", daemon=True)
        self._server_thread.start()

        # small wait so the socket binds before returning
        time.sleep(0.1)

    def stop(self) -> None:
        """
        Stop the local Uvicorn server if it's running.
        """
        if self._uvicorn:
            self._uvicorn.should_exit = True
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=3.0)
        self._uvicorn = None
        self._server_thread = None
        self._log.info("LiteLLM Proxy stopped")

    # --------------- resource export ---------------

    def as_resource(self, sampling_parameters: SamplingParameters) -> LLM:
        """
        Return an `LLM` Resource pointing at this proxy (OpenAI-compatible /v1).

        The `model` you pass at inference-time should be one of the model_name
        values from the constructed `model_list`. (e.g., 'backend-1', 'gpt-4o', ...).
        Clients will hit `endpoint/v1/...` like the OpenAI API. :contentReference[oaicite:4]{index=4}
        """
        return LLM(
            endpoint=f"http://{self.host}:{self.port}/v1",
            model="backend-1",  # default logical name; callers can override per-request
            sampling_parameters=dict(sampling_parameters or {}),
        )
