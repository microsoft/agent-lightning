"""Unit tests for streaming response assemblers."""

from __future__ import annotations

import pytest

from agl_lite.gateway.assemblers import select_assembler
from agl_lite.gateway.assemblers.anthropic import assemble_anthropic_message
from agl_lite.gateway.assemblers.chat_completion import assemble_chat_completion
from agl_lite.gateway.assemblers.completion import assemble_completion


# ---------------------------------------------------------------------------
# Registry: select_assembler
# ---------------------------------------------------------------------------

class TestSelectAssembler:
    def test_chat_completions(self):
        assert select_assembler("/v1/chat/completions") is assemble_chat_completion

    def test_chat_completions_trailing_slash(self):
        assert select_assembler("/v1/chat/completions/") is assemble_chat_completion

    def test_completions(self):
        assert select_assembler("/v1/completions") is assemble_completion

    def test_messages(self):
        assert select_assembler("/v1/messages") is assemble_anthropic_message

    def test_unknown_path(self):
        assert select_assembler("/v1/some/other/endpoint") is None

    def test_chat_before_completions_ordering(self):
        """'chat/completions' must match before 'completions'."""
        assert select_assembler("/v1/chat/completions") is assemble_chat_completion
        assert select_assembler("/v1/completions") is assemble_completion


# ---------------------------------------------------------------------------
# Chat completion assembler
# ---------------------------------------------------------------------------

class TestAssembleChatCompletion:
    def test_empty(self):
        assert assemble_chat_completion([]) == {}

    def test_basic_assembly(self):
        chunks = [
            {"id": "c1", "created": 100, "model": "m", "choices": [{"delta": {"role": "assistant"}}]},
            {"id": "c1", "created": 100, "model": "m", "choices": [{"delta": {"content": "Hi"}}]},
            {"id": "c1", "created": 100, "model": "m", "choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
        ]
        result = assemble_chat_completion(chunks)
        assert result["id"] == "c1"
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_preserves_token_ids(self):
        chunks = [
            {"id": "c1", "choices": [{"delta": {"content": "a"}, "token_ids": [10]}],
             "prompt_token_ids": [1, 2]},
            {"id": "c1", "choices": [{"delta": {"content": "b"}, "token_ids": [20]}]},
        ]
        result = assemble_chat_completion(chunks)
        assert result["prompt_token_ids"] == [1, 2]
        assert result["choices"][0]["token_ids"] == [10, 20]

    def test_multiple_choices(self):
        chunks = [
            {"id": "c1", "choices": [
                {"index": 0, "delta": {"content": "A"}},
                {"index": 1, "delta": {"content": "B"}},
            ]},
        ]
        result = assemble_chat_completion(chunks)
        assert len(result["choices"]) == 2
        assert result["choices"][0]["message"]["content"] == "A"
        assert result["choices"][1]["message"]["content"] == "B"

    def test_no_choices_fallback(self):
        chunks = [{"id": "c1", "model": "m"}]  # no choices key
        result = assemble_chat_completion(chunks)
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == ""


# ---------------------------------------------------------------------------
# Legacy completion assembler
# ---------------------------------------------------------------------------

class TestAssembleCompletion:
    def test_empty(self):
        assert assemble_completion([]) == {}

    def test_basic_assembly(self):
        chunks = [
            {"id": "cmpl-1", "created": 100, "model": "m",
             "choices": [{"text": "Hello", "index": 0, "finish_reason": None}]},
            {"id": "cmpl-1", "created": 100, "model": "m",
             "choices": [{"text": " world", "index": 0, "finish_reason": "stop"}]},
        ]
        result = assemble_completion(chunks)
        assert result["id"] == "cmpl-1"
        assert result["object"] == "text_completion"
        assert result["choices"][0]["text"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_preserves_token_ids(self):
        chunks = [
            {"id": "cmpl-1", "choices": [{"text": "a", "token_ids": [10]}],
             "prompt_token_ids": [1, 2]},
            {"id": "cmpl-1", "choices": [{"text": "b", "token_ids": [20]}]},
        ]
        result = assemble_completion(chunks)
        assert result["prompt_token_ids"] == [1, 2]
        assert result["choices"][0]["token_ids"] == [10, 20]

    def test_no_choices_fallback(self):
        chunks = [{"id": "cmpl-1", "model": "m"}]
        result = assemble_completion(chunks)
        assert len(result["choices"]) == 1
        assert result["choices"][0]["text"] == ""


# ---------------------------------------------------------------------------
# Anthropic message assembler
# ---------------------------------------------------------------------------

class TestAssembleAnthropicMessage:
    def test_empty(self):
        assert assemble_anthropic_message([]) == {}

    def test_basic_assembly(self):
        chunks = [
            {"type": "message_start", "message": {
                "id": "msg_1", "model": "claude-3", "role": "assistant",
                "usage": {"input_tokens": 10},
            }},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": " world"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        result = assemble_anthropic_message(chunks)
        assert result["id"] == "msg_1"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3"
        assert result["content"] == [{"type": "text", "text": "Hello world"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_multiple_content_blocks(self):
        chunks = [
            {"type": "message_start", "message": {"id": "msg_2", "model": "claude-3", "role": "assistant"}},
            {"type": "content_block_start", "index": 0},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "First"}},
            {"type": "content_block_start", "index": 1},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "text_delta", "text": "Second"}},
            {"type": "message_stop"},
        ]
        result = assemble_anthropic_message(chunks)
        assert len(result["content"]) == 2
        assert result["content"][0]["text"] == "First"
        assert result["content"][1]["text"] == "Second"

    def test_no_content_blocks(self):
        chunks = [
            {"type": "message_start", "message": {"id": "msg_3", "model": "claude-3", "role": "assistant"}},
            {"type": "message_stop"},
        ]
        result = assemble_anthropic_message(chunks)
        assert result["content"] == [{"type": "text", "text": ""}]
