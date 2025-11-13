# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Constants for evals module."""

SUPPORTED_PREDEFINED_METRICS = frozenset(
    {
        "general_quality_v1",
        "text_quality_v1",
        "instruction_following_v1",
        "grounding_v1",
        "safety_v1",
        "multi_turn_general_quality_v1",
        "multi_turn_text_quality_v1",
        "final_response_match_v2",
        "final_response_reference_free_v1",
        "final_response_quality_v1",
        "hallucination_v1",
        "tool_use_quality_v1",
        "gecko_text2image_v1",
        "gecko_text2video_v1",
    }
)

SUPPORTED_VERTEX_MAAS_MODEL_PREFIXES = frozenset(
    {
        "meta/",  # Meta/Llama
        "deepseek-ai/",  # DeepSeek AI
        "qwen/",  # Qwen
        "openai/",  # OpenAI (GPT-OSS)
        "claude-",  # Anthropic (Claude)
        "mistral-",  # Mistral AI
        "jamba-",  # AI21 (Jamba)
    }
)
INTERMEDIATE_EVENTS = "intermediate_events"
RESPONSE = "response"
PROMPT = "prompt"
REFERENCE = "reference"
SESSION_INPUT = "session_inputs"
CONTEXT = "context"

COMMON_DATASET_COLUMNS = frozenset(
    {
        INTERMEDIATE_EVENTS,
        PROMPT,
        REFERENCE,
        SESSION_INPUT,
        CONTEXT,
    }
)
