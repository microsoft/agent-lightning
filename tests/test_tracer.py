import pytest
import requests
import json
import aiohttp
import pickle

from agentlightning.tracer.http import HttpTracer
from agentlightning.tracer.agentops import AgentOpsTracer


@pytest.fixture
def tracer():
    return HttpTracer(include_headers=True, include_body=True, include_agentlightning_requests=True)


def test_basic_http_trace(tracer):
    with tracer.trace_context():
        response = requests.get("https://httpbin.org/get")
        assert response.status_code == 200

    spans = tracer.get_last_trace()
    assert len(spans) >= 1
    span = next((s for s in spans if "httpbin.org/get" in s.name), None)
    assert span is not None
    assert span.attributes["http.method"] == "GET"
    assert span.attributes["http.status_code"] == 200


def test_include_headers(tracer):
    with tracer.trace_context():
        response = requests.get("https://httpbin.org/headers", headers={"X-Test-Header": "pytest"})
        assert response.status_code == 200

    spans = tracer.get_last_trace()
    span = next((s for s in spans if "httpbin.org/headers" in s.name), None)
    assert span is not None
    # Check that the custom header is present in the span attributes
    assert "http.request.header.x-test-header" in span.attributes


def test_include_body(tracer):
    with tracer.trace_context():
        response = requests.post("https://httpbin.org/post", data="pytest-body")
        assert response.status_code == 200

    spans = tracer.get_last_trace()
    span = next((s for s in spans if "httpbin.org/post" in s.name), None)
    assert span is not None
    # Check that the request body is present in the span attributes
    assert "http.request.body" in span.attributes
    assert b"pytest-body" in span.attributes["http.request.body"]


def test_agentlightning_request_filtering():
    tracer = HttpTracer(include_agentlightning_requests=False)
    with tracer.trace_context():
        # Simulate a request with the AgentLightning header
        response = requests.get("https://httpbin.org/get", headers={"x-agentlightning-client": "true"})
        assert response.status_code == 200

    spans = tracer.get_last_trace()
    if not spans:
        spans = []
    # Should be empty because the request should be filtered out
    assert all("x-agentlightning-client" not in (attr or "") for span in spans for attr in (span.attributes or {}))


@pytest.mark.asyncio
async def test_aiohttp_basic_trace(tracer):
    async with aiohttp.ClientSession() as session:
        with tracer.trace_context():
            async with session.get("https://httpbin.org/get") as response:
                assert response.status == 200
                await response.text()
    spans = tracer.get_last_trace()
    span = next((s for s in spans if "httpbin.org/get" in s.name), None)
    assert span is not None
    assert span.attributes["http.method"] == "GET"
    assert span.attributes["http.status_code"] == 200


@pytest.mark.asyncio
async def test_aiohttp_json_request_response(tracer):
    json_data = {"foo": "bar"}
    async with aiohttp.ClientSession() as session:
        with tracer.trace_context():
            async with session.post("https://httpbin.org/post", json=json_data) as response:
                assert response.status == 200
                resp_json = await response.json()
                assert resp_json["json"] == json_data
    spans = tracer.get_last_trace()
    span = next((s for s in spans if "httpbin.org/post" in s.name), None)
    assert span is not None
    # Check that the request body contains the JSON
    assert "http.request.body" in span.attributes
    body_bytes = span.attributes["http.request.body"]
    assert b'"foo": "bar"' in body_bytes or b'"foo":"bar"' in body_bytes
    # Parse and check JSON
    parsed_body = json.loads(body_bytes.decode())
    assert parsed_body == json_data
    # Check that the response body contains the JSON
    assert "http.response.body" in span.attributes
    resp_body_bytes = span.attributes["http.response.body"]
    assert b'"foo": "bar"' in resp_body_bytes or b'"foo":"bar"' in resp_body_bytes
    # Parse and check JSON
    parsed_resp_body = json.loads(resp_body_bytes.decode())
    # httpbin returns a JSON object with a 'json' field
    assert parsed_resp_body["json"] == json_data


def test_requests_json_request_response(tracer):
    json_data = {"hello": "world"}
    with tracer.trace_context():
        response = requests.post("https://httpbin.org/post", json=json_data)
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["json"] == json_data
    spans = tracer.get_last_trace()
    span = next((s for s in spans if "httpbin.org/post" in s.name), None)
    assert span is not None
    # Check that the request body contains the JSON
    assert "http.request.body" in span.attributes
    body_bytes = span.attributes["http.request.body"]
    assert b'"hello": "world"' in body_bytes or b'"hello":"world"' in body_bytes
    # Parse and check JSON
    parsed_body = json.loads(body_bytes.decode())
    assert parsed_body == json_data
    # Check that the response body contains the JSON
    assert "http.response.body" in span.attributes
    resp_body_bytes = span.attributes["http.response.body"]
    assert b'"hello": "world"' in resp_body_bytes or b'"hello":"world"' in resp_body_bytes
    # Parse and check JSON
    parsed_resp_body = json.loads(resp_body_bytes.decode())
    assert parsed_resp_body["json"] == json_data


def test_agentops_tracer_picklable():
    tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)
    pickled = pickle.dumps(tracer)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, AgentOpsTracer)
    # Should be able to call trace_context (will raise NotImplementedError if not implemented)
    with pytest.raises(RuntimeError):
        with unpickled.trace_context():
            pass


def test_http_tracer_picklable():
    tracer = HttpTracer()
    pickled = pickle.dumps(tracer)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, HttpTracer)
    # Should be able to call trace_context (will not raise, but will not record anything)
    with unpickled.trace_context():
        pass
