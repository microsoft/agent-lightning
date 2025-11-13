# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
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
"""Constants for evaluation."""
import dataclasses

# The number of concurrent workers to use for making model inference and
# evaluation requests.
MAX_WORKERS = 100
RESPONSE_ERROR = "Error"


@dataclasses.dataclass(frozen=True)
class Metric:
    """Namespace for Metrics."""

    # Model-based Pointwise Metrics.
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    SAFETY = "safety"
    GROUNDEDNESS = "groundedness"
    INSTRUCTION_FOLLOWING = "instruction_following"
    VERBOSITY = "verbosity"
    TEXT_QUALITY = "text_quality"
    SUMMARIZATION_QUALITY = "summarization_quality"
    QUESTION_ANSWERING_QUALITY = "question_answering_quality"
    MULTI_TURN_CHAT_QUALITY = "multi_turn_chat_quality"
    MULTI_TURN_SAFETY = "multi_turn_safety"

    # Model-based Pairwise Metrics.
    PAIRWISE_COHERENCE = "pairwise_coherence"
    PAIRWISE_FLUENCY = "pairwise_fluency"
    PAIRWISE_SAFETY = "pairwise_safety"
    PAIRWISE_GROUNDEDNESS = "pairwise_groundedness"
    PAIRWISE_INSTRUCTION_FOLLOWING = "pairwise_instruction_following"
    PAIRWISE_VERBOSITY = "pairwise_verbosity"
    PAIRWISE_TEXT_QUALITY = "pairwise_text_quality"
    PAIRWISE_SUMMARIZATION_QUALITY = "pairwise_summarization_quality"
    PAIRWISE_QUESTION_ANSWERING_QUALITY = "pairwise_question_answering_quality"
    PAIRWISE_MULTI_TURN_CHAT_QUALITY = "pairwise_multi_turn_chat_quality"
    PAIRWISE_MULTI_TURN_SAFETY = "pairwise_multi_turn_safety"

    POINTWISE_METRIC = "pointwise_metric"
    PAIRWISE_METRIC = "pairwise_metric"

    # Model-based translation Metrics.
    COMET = "comet"
    METRICX = "metricx"

    # Automatic Metrics.
    EXACT_MATCH = "exact_match"
    BLEU = "bleu"
    ROUGE = "rouge"
    ROUGE_1 = "rouge_1"
    ROUGE_2 = "rouge_2"
    ROUGE_L = "rouge_l"
    ROUGE_L_SUM = "rouge_l_sum"
    TOOL_CALL_VALID = "tool_call_valid"
    TOOL_NAME_MATCH = "tool_name_match"
    TOOL_PARAMETER_KEY_MATCH = "tool_parameter_key_match"
    TOOL_PARAMETER_KV_MATCH = "tool_parameter_kv_match"

    AUTOMATIC_METRIC_LIST = (
        EXACT_MATCH,
        BLEU,
        ROUGE,
        ROUGE_1,
        ROUGE_2,
        ROUGE_L,
        ROUGE_L_SUM,
        TOOL_CALL_VALID,
        TOOL_NAME_MATCH,
        TOOL_PARAMETER_KEY_MATCH,
        TOOL_PARAMETER_KV_MATCH,
    )

    POINTWISE_METRIC_PROMPT_TEMPLATE_EXAMPLE_LIST = (
        COHERENCE,
        FLUENCY,
        SAFETY,
        GROUNDEDNESS,
        INSTRUCTION_FOLLOWING,
        VERBOSITY,
        TEXT_QUALITY,
        SUMMARIZATION_QUALITY,
        QUESTION_ANSWERING_QUALITY,
        MULTI_TURN_CHAT_QUALITY,
        MULTI_TURN_SAFETY,
    )

    PAIRWISE_METRIC_PROMPT_TEMPLATE_EXAMPLE_LIST = (
        PAIRWISE_COHERENCE,
        PAIRWISE_FLUENCY,
        PAIRWISE_SAFETY,
        PAIRWISE_GROUNDEDNESS,
        PAIRWISE_INSTRUCTION_FOLLOWING,
        PAIRWISE_VERBOSITY,
        PAIRWISE_TEXT_QUALITY,
        PAIRWISE_SUMMARIZATION_QUALITY,
        PAIRWISE_QUESTION_ANSWERING_QUALITY,
        PAIRWISE_MULTI_TURN_CHAT_QUALITY,
        PAIRWISE_MULTI_TURN_SAFETY,
    )


@dataclasses.dataclass(frozen=True)
class MetricResult:
    """Namespace for Metric Results."""

    ROW_COUNT_KEY = "row_count"
    SCORE_KEY = "score"
    EXPLANATION_KEY = "explanation"
    PAIRWISE_CHOICE_KEY = "pairwise_choice"

    # Automatic Metrics.
    EXACT_MATCH_RESULTS = "exact_match_results"
    BLEU_RESULTS = "bleu_results"
    ROUGE_RESULTS = "rouge_results"
    TOOL_CALL_VALID_RESULTS = "tool_call_valid_results"
    TOOL_NAME_MATCH_RESULTS = "tool_name_match_results"
    TOOL_PARAMETER_KEY_MATCH_RESULTS = "tool_parameter_key_match_results"
    TOOL_PARAMETER_KV_MATCH_RESULTS = "tool_parameter_kv_match_results"

    POINTWISE_METRIC_RESULT = "pointwise_metric_result"
    PAIRWISE_METRIC_RESULT = "pairwise_metric_result"

    COMET_RESULT = "comet_result"
    METRICX_RESULT = "metricx_result"

    AUTOMATIC_METRIC_RESULTS_LIST = (
        EXACT_MATCH_RESULTS,
        BLEU_RESULTS,
        ROUGE_RESULTS,
        TOOL_CALL_VALID_RESULTS,
        TOOL_NAME_MATCH_RESULTS,
        TOOL_PARAMETER_KEY_MATCH_RESULTS,
        TOOL_PARAMETER_KV_MATCH_RESULTS,
    )


@dataclasses.dataclass(frozen=True)
class Dataset:
    # Default evaluation dataset schema column names.
    MODEL_RESPONSE_COLUMN = "response"
    BASELINE_MODEL_RESPONSE_COLUMN = "baseline_model_response"
    PROMPT_COLUMN = "prompt"
    REFERENCE_COLUMN = "reference"
    SOURCE_COLUMN = "source"


@dataclasses.dataclass(frozen=True)
class QuotaLimit:
    """Generative AI on Vertex AI quota limits."""

    # Default Evaluation Service QPS limit.
    EVAL_SERVICE_QPS = 10
