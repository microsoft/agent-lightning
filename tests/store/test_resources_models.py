"""Tests for InMemoryStore — resources and model server operations."""

import pytest

from agl_lite.schemas.errors import NotFoundError
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


class TestResources:
    def test_add_and_get(self, store: InMemoryStore):
        res = store.add_resources({"system_prompt": "Be helpful", "eval_config": {"metric": "pass@1"}})
        assert res.resources_id
        assert res.resources["system_prompt"] == "Be helpful"
        fetched = store.get_resources(res.resources_id)
        assert fetched.resources_id == res.resources_id

    def test_get_latest(self, store: InMemoryStore):
        store.add_resources({"version": 1})
        r2 = store.add_resources({"version": 2})
        latest = store.get_latest_resources()
        assert latest is not None
        assert latest.resources_id == r2.resources_id
        assert latest.resources["version"] == 2

    def test_get_latest_empty(self, store: InMemoryStore):
        assert store.get_latest_resources() is None

    def test_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.get_resources("nonexistent")

    def test_immutable_snapshots(self, store: InMemoryStore):
        """Each add creates a new snapshot — no mutation."""
        r1 = store.add_resources({"prompt": "v1"})
        r2 = store.add_resources({"prompt": "v2"})
        assert r1.resources_id != r2.resources_id
        assert store.get_resources(r1.resources_id).resources["prompt"] == "v1"
        assert store.get_resources(r2.resources_id).resources["prompt"] == "v2"

    def test_validates_job_defaults(self, store: InMemoryStore):
        """job_defaults key is validated as JobDefaults schema."""
        res = store.add_resources(
            {
                "job_defaults": {"timeout": 300, "overrides": {"dnsPolicy": "ClusterFirst"}},
                "prompt": "hello",
            }
        )
        assert res.resources["job_defaults"]["timeout"] == 300

    def test_rejects_invalid_job_defaults(self, store: InMemoryStore):
        from pydantic import ValidationError

        # timeout must be int, not a string
        with pytest.raises(ValidationError):
            store.add_resources({"job_defaults": {"timeout": "not-a-number"}})


class TestModelServers:
    def test_register_and_list(self, store: InMemoryStore):
        m = store.register_model("qwen-7b", "http://vllm:8000/v1", version=42)
        assert m.model == "qwen-7b"
        assert m.endpoint == "http://vllm:8000/v1"
        assert m.version == 42
        models = store.list_models()
        assert len(models) == 1
        assert models[0].endpoint == m.endpoint

    def test_register_multiple_servers_same_model(self, store: InMemoryStore):
        store.register_model("qwen-7b", "http://vllm-0:8000/v1", version=10)
        store.register_model("qwen-7b", "http://vllm-1:8000/v1", version=10)
        assert len(store.list_models()) == 2
        pool = store.get_model_pool("qwen-7b")
        assert len(pool) == 2

    def test_register_different_models(self, store: InMemoryStore):
        store.register_model("qwen-7b", "http://vllm-0:8000/v1")
        store.register_model("reward-model", "http://reward-0:8000/v1")
        assert len(store.list_models()) == 2
        assert len(store.get_model_pool("qwen-7b")) == 1
        assert len(store.get_model_pool("reward-model")) == 1

    def test_upsert_same_model_endpoint(self, store: InMemoryStore):
        """Re-registering the same (model, endpoint) updates version (upsert)."""
        store.register_model("qwen-7b", "http://vllm:8000/v1", version=1)
        store.register_model("qwen-7b", "http://vllm:8000/v1", version=2)
        pool = store.get_model_pool("qwen-7b")
        assert len(pool) == 1
        assert pool[0].version == 2

    def test_online_rl_rolling_update(self, store: InMemoryStore):
        """Online RL: update one server while others keep old version."""
        store.register_model("qwen-7b", "http://vllm-0:8000/v1", version=3)
        store.register_model("qwen-7b", "http://vllm-1:8000/v1", version=3)
        # Rolling update: vllm-0 gets new weights
        store.register_model("qwen-7b", "http://vllm-0:8000/v1", version=4)
        pool = store.get_model_pool("qwen-7b")
        versions = {s.endpoint: s.version for s in pool}
        assert versions["http://vllm-0:8000/v1"] == 4
        assert versions["http://vllm-1:8000/v1"] == 3

    def test_get_model_pool_not_found(self, store: InMemoryStore):
        assert store.get_model_pool("nonexistent") == []

    def test_remove_entire_model(self, store: InMemoryStore):
        store.register_model("qwen-7b", "http://vllm-0:8000/v1")
        store.register_model("qwen-7b", "http://vllm-1:8000/v1")
        store.remove_model_servers("qwen-7b")
        assert store.get_model_pool("qwen-7b") == []
        assert store.list_models() == []

    def test_remove_specific_endpoints(self, store: InMemoryStore):
        store.register_model("qwen-7b", "http://vllm-0:8000/v1")
        store.register_model("qwen-7b", "http://vllm-1:8000/v1")
        store.remove_model_servers("qwen-7b", endpoints=["http://vllm-0:8000/v1"])
        pool = store.get_model_pool("qwen-7b")
        assert len(pool) == 1
        assert pool[0].endpoint == "http://vllm-1:8000/v1"

    def test_remove_last_server_auto_deletes_pool(self, store: InMemoryStore):
        store.register_model("qwen-7b", "http://vllm-0:8000/v1")
        store.remove_model_servers("qwen-7b", endpoints=["http://vllm-0:8000/v1"])
        assert store.get_model_pool("qwen-7b") == []
        assert "qwen-7b" not in store._models

    def test_remove_model_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.remove_model_servers("nonexistent")

    def test_remove_nonexistent_endpoint_silent(self, store: InMemoryStore):
        """Removing an endpoint that doesn't exist in the pool is silently ignored."""
        store.register_model("qwen-7b", "http://vllm-0:8000/v1")
        store.remove_model_servers("qwen-7b", endpoints=["http://nonexistent:8000/v1"])
        assert len(store.get_model_pool("qwen-7b")) == 1  # unchanged

    def test_remove_all(self, store: InMemoryStore):
        store.register_model("qwen-7b", "http://vllm-0:8000/v1")
        store.register_model("reward-model", "http://reward-0:8000/v1")
        store.remove_all_models()
        assert store.list_models() == []

    def test_remove_all_empty(self, store: InMemoryStore):
        store.remove_all_models()  # should not raise
        assert store.list_models() == []

    def test_default_version(self, store: InMemoryStore):
        m = store.register_model("qwen-7b", "http://vllm:8000/v1")
        assert m.version == 0

    def test_token_stored(self, store: InMemoryStore):
        m = store.register_model("qwen-7b", "http://vllm:8000/v1", token="sk-secret")
        assert m.token == "sk-secret"

    def test_token_default_none(self, store: InMemoryStore):
        m = store.register_model("qwen-7b", "http://vllm:8000/v1")
        assert m.token is None
