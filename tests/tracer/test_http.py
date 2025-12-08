# Copyright (c) Microsoft. All rights reserved.

"""
Functional tests for HttpTracer with real HTTP requests.

These tests use actual HTTP requests to httpbingo.org to validate
that the HttpTracer correctly captures HTTP traffic in both
normal and subprocess modes.
"""

import aiohttp
import pytest
import requests

from agentlightning.tracer.http import HttpTracer


def sync_http_function():
    """A simple synchronous function that makes HTTP requests."""
    s = requests.Session()
    s.headers.update({"Accept-Encoding": "gzip, deflate"})  # no zstd
    response = s.get("https://httpbingo.org/get")
    response2 = s.post("https://httpbingo.org/post", json={"test": "data"})
    return {
        "get_status": response.status_code,
        "post_status": response2.status_code,
        "get_data": response.json(),
        "post_data": response2.json(),
    }


async def async_http_function():
    """A simple asynchronous function that makes HTTP requests."""
    async with aiohttp.ClientSession(
        auto_decompress=True, headers={"Accept-Encoding": "gzip, deflate"}  # exclude zstd/brotli
    ) as session:
        async with session.get("https://httpbingo.org/get") as response:
            get_data = await response.json()
            get_status = response.status

        async with session.post("https://httpbingo.org/post", json={"async": "test"}) as response:
            post_data = await response.json()
            post_status = response.status

    return {"get_status": get_status, "post_status": post_status, "get_data": get_data, "post_data": post_data}


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_normal_mode_sync_requests():
    """Test HttpTracer in normal mode with synchronous requests."""
    tracer = HttpTracer(include_headers=True, include_body=True, subprocess_mode=False)

    result = tracer.trace_run(sync_http_function)
    spans = tracer.get_last_trace()

    # Verify function result
    assert result["get_status"] == 200
    assert result["post_status"] == 200
    assert "get_data" in result
    assert "post_data" in result
    assert result["get_data"]["url"] == "https://httpbingo.org/get"
    assert result["post_data"]["json"]["test"] == "data"

    # Verify spans
    assert len(spans) == 2
    span_names = [span.name for span in spans]
    assert "HTTP GET https://httpbingo.org/get" in span_names
    assert "HTTP POST https://httpbingo.org/post" in span_names

    # Verify span attributes
    for span in spans:
        assert span.attributes is not None
        assert "http.method" in span.attributes
        assert "http.url" in span.attributes
        assert "http.status_code" in span.attributes
        assert span.attributes["http.status_code"] == 200


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_subprocess_mode_sync_requests():
    """Test HttpTracer in subprocess mode with synchronous requests."""
    tracer = HttpTracer(include_headers=True, include_body=True, subprocess_mode=True)

    result = tracer.trace_run(sync_http_function)
    spans = tracer.get_last_trace()

    # Verify function result
    assert result["get_status"] == 200
    assert result["post_status"] == 200
    assert "get_data" in result
    assert "post_data" in result
    assert result["get_data"]["url"] == "https://httpbingo.org/get"
    assert result["post_data"]["json"]["test"] == "data"

    # Verify spans
    assert len(spans) == 2
    span_names = [span.name for span in spans]
    assert "HTTP GET https://httpbingo.org/get" in span_names
    assert "HTTP POST https://httpbingo.org/post" in span_names


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.asyncio
async def test_subprocess_mode_async_requests():
    """Test HttpTracer in subprocess mode with asynchronous requests."""
    tracer = HttpTracer(include_headers=True, include_body=True, subprocess_mode=True)

    result = await tracer.trace_run_async(async_http_function)
    spans = tracer.get_last_trace()

    # Verify function result
    assert result["get_status"] == 200
    assert result["post_status"] == 200
    assert "get_data" in result
    assert "post_data" in result
    assert result["get_data"]["url"] == "https://httpbingo.org/get"
    assert result["post_data"]["json"]["async"] == "test"

    # Verify spans
    assert len(spans) == 2
    span_names = [span.name for span in spans]
    assert "HTTP GET https://httpbingo.org/get" in span_names
    assert "HTTP POST https://httpbingo.org/post" in span_names


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.asyncio
async def test_normal_mode_async_requests():
    """Test HttpTracer in normal mode with asynchronous requests."""
    tracer = HttpTracer(include_headers=True, include_body=True, subprocess_mode=False)

    result = await tracer.trace_run_async(async_http_function)
    spans = tracer.get_last_trace()

    # Verify function result
    assert result["get_status"] == 200
    assert result["post_status"] == 200
    assert "get_data" in result
    assert "post_data" in result

    # Verify spans
    assert len(spans) == 2
    span_names = [span.name for span in spans]
    assert "HTTP GET https://httpbingo.org/get" in span_names
    assert "HTTP POST https://httpbingo.org/post" in span_names


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_span_attributes_detailed():
    """Test that spans contain expected attributes when headers and body are included."""
    tracer = HttpTracer(include_headers=True, include_body=True, subprocess_mode=False)

    tracer.trace_run(sync_http_function)
    spans = tracer.get_last_trace()

    assert len(spans) == 2

    for span in spans:
        # Basic HTTP attributes
        assert span.attributes is not None
        assert "http.method" in span.attributes
        assert "http.url" in span.attributes
        assert "http.target" in span.attributes
        assert "http.host" in span.attributes
        assert "http.status_code" in span.attributes

        # Headers should be included
        has_request_headers = any(key.startswith("http.request.header.") for key in span.attributes.keys())
        assert has_request_headers

        # Check for specific expected headers
        assert "http.request.header.user-agent" in span.attributes
        assert "http.request.header.accept" in span.attributes


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_span_attributes_minimal():
    """Test that spans contain minimal attributes when headers and body are excluded."""
    tracer = HttpTracer(include_headers=False, include_body=False, subprocess_mode=False)

    tracer.trace_run(sync_http_function)
    spans = tracer.get_last_trace()

    assert len(spans) == 2

    for span in spans:
        # Basic HTTP attributes should still be present
        assert span.attributes is not None
        assert "http.method" in span.attributes
        assert "http.url" in span.attributes
        assert "http.status_code" in span.attributes

        # Headers should not be included
        has_request_headers = any(key.startswith("http.request.header.") for key in span.attributes.keys())
        assert not has_request_headers

        # Body should not be included
        assert "http.request.body" not in span.attributes
        assert "http.response.body" not in span.attributes


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_error_handling_in_subprocess():
    """Test that errors in subprocess mode are properly propagated."""

    def failing_function():
        requests.get("https://httpbingo.org/get")
        raise ValueError("Test error")

    tracer = HttpTracer(subprocess_mode=True)

    with pytest.raises(ValueError, match="Test error"):
        tracer.trace_run(failing_function)

    # Should still capture the successful request before the error
    spans = tracer.get_last_trace()
    assert len(spans) == 1
    assert "HTTP GET https://httpbingo.org/get" in spans[0].name
