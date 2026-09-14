
import pytest
from agentlightning.utils.cache import LRUCache

def test_lru_cache_capacity():
    cache = LRUCache(capacity=2)
    cache["a"] = 1
    cache["b"] = 2
    assert len(cache) == 2
    
    cache["c"] = 3
    assert len(cache) == 2
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache

def test_lru_cache_access_order():
    cache = LRUCache(capacity=2)
    cache["a"] = 1
    cache["b"] = 2
    
    # Access 'a' to make it most recently used
    _ = cache["a"]
    
    # Add 'c', should evict 'b' (least recently used)
    cache["c"] = 3
    
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache

def test_integration_client_import():
    try:
        from agentlightning.client import AgentLightningClient
        client = AgentLightningClient(endpoint="http://localhost:8000")
        assert isinstance(client._resource_cache, LRUCache)
        assert client._resource_cache.capacity == 100
    except ImportError:
        pytest.fail("Could not import AgentLightningClient or verify LRUCache integration")

def test_integration_server_import():
    try:
        from agentlightning.server import ServerDataStore
        store = ServerDataStore()
        assert isinstance(store._resource_versions, LRUCache)
        assert store._resource_versions.capacity == 100
    except ImportError:
        pytest.fail("Could not import ServerDataStore or verify LRUCache integration")
