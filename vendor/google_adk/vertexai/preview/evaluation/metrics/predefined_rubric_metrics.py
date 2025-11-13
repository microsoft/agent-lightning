# -*- coding: utf-8 -*-

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

from google.cloud.aiplatform_v1beta1.types import (
    evaluation_service as gapic_eval_service_types,
)
from vertexai.preview.evaluation import utils
from vertexai.preview.evaluation.metrics import (
    _base as metrics_base,
)
from vertexai.preview.evaluation.metrics import (
    _default_templates,
)
from vertexai.preview.evaluation.metrics import (
    custom_output_config,
)
from vertexai.preview.evaluation.metrics import pairwise_metric
from vertexai.preview.evaluation.metrics import pointwise_metric
from vertexai.preview.evaluation.metrics import (
    rubric_based_metric,
)


AutoraterConfig = gapic_eval_service_types.AutoraterConfig

_POINTWISE_OUTPUT_CONFIG = custom_output_config.CustomOutputConfig(
    return_raw_output=True,
    parsing_fn=utils.parse_pointwise_rubric_result,
)

_PAIRWISE_OUTPUT_CONFIG = custom_output_config.CustomOutputConfig(
    return_raw_output=True,
    parsing_fn=utils.parse_pairwise_rubric_result,
)
_PAIRWISE_AUTORATER_CONFIG = AutoraterConfig(
    sampling_count=1,
)


class PredefinedRubricMetrics:
    """Predefined rubric-based metrics."""

    class Pointwise:
        """Pointwise rubric-based metrics."""

        INSTRUCTION_FOLLOWING = rubric_based_metric.RubricBasedMetric(
            generation_config=metrics_base.RubricGenerationConfig(
                prompt_template=_default_templates.INSTRUCTION_FOLLOWING_RUBRIC_GENERATION_PROMPT_TEMPLATE,
            ),
            critique_metric=pointwise_metric.PointwiseMetric(
                metric="rb_instruction_following",
                metric_prompt_template=_default_templates.INSTRUCTION_FOLLOWING_RUBRIC_CRITIQUE_TEMPLATE,
                custom_output_config=_POINTWISE_OUTPUT_CONFIG,
            ),
        )
        MULTIMODAL_UNDERSTANDING = rubric_based_metric.RubricBasedMetric(
            generation_config=metrics_base.RubricGenerationConfig(
                prompt_template=_default_templates.MULTIMODAL_UNDERSTANDING_RUBRIC_GENERATION_PROMPT_TEMPLATE
            ),
            critique_metric=pointwise_metric.PointwiseMetric(
                metric="rb_multimodal_understanding",
                metric_prompt_template=_default_templates.MULTIMODAL_UNDERSTANDING_RUBRIC_CRITIQUE_TEMPLATE,
                custom_output_config=_POINTWISE_OUTPUT_CONFIG,
            ),
        )
        TEXT_QUALITY = rubric_based_metric.RubricBasedMetric(
            generation_config=metrics_base.RubricGenerationConfig(
                prompt_template=_default_templates.TEXT_QUALITY_RUBRIC_GENERATION_PROMPT_TEMPLATE
            ),
            critique_metric=pointwise_metric.PointwiseMetric(
                metric="rb_text_quality",
                metric_prompt_template=_default_templates.TEXT_QUALITY_RUBRIC_CRITIQUE_TEMPLATE,
                custom_output_config=_POINTWISE_OUTPUT_CONFIG,
            ),
        )

    class Pairwise:
        """Pairwise rubric-based metrics."""

        INSTRUCTION_FOLLOWING = rubric_based_metric.RubricBasedMetric(
            generation_config=metrics_base.RubricGenerationConfig(
                prompt_template=_default_templates.INSTRUCTION_FOLLOWING_RUBRIC_GENERATION_PROMPT_TEMPLATE,
            ),
            critique_metric=pairwise_metric.PairwiseMetric(
                metric="pairwise_rb_instruction_following",
                metric_prompt_template=_default_templates.PAIRWISE_INSTRUCTION_FOLLOWING_RUBRIC_CRITIQUE_TEMPLATE,
                custom_output_config=_PAIRWISE_OUTPUT_CONFIG,
                autorater_config=_PAIRWISE_AUTORATER_CONFIG,
            ),
        )
        MULTIMODAL_UNDERSTANDING = rubric_based_metric.RubricBasedMetric(
            generation_config=metrics_base.RubricGenerationConfig(
                prompt_template=_default_templates.MULTIMODAL_UNDERSTANDING_RUBRIC_GENERATION_PROMPT_TEMPLATE
            ),
            critique_metric=pairwise_metric.PairwiseMetric(
                metric="pairwise_rb_multimodal_understanding",
                metric_prompt_template=_default_templates.PAIRWISE_MULTIMODAL_UNDERSTANDING_RUBRIC_CRITIQUE_TEMPLATE,
                custom_output_config=_PAIRWISE_OUTPUT_CONFIG,
                autorater_config=_PAIRWISE_AUTORATER_CONFIG,
            ),
        )
        TEXT_QUALITY = rubric_based_metric.RubricBasedMetric(
            generation_config=metrics_base.RubricGenerationConfig(
                prompt_template=_default_templates.TEXT_QUALITY_RUBRIC_GENERATION_PROMPT_TEMPLATE
            ),
            critique_metric=pairwise_metric.PairwiseMetric(
                metric="pairwise_rb_text_quality",
                metric_prompt_template=_default_templates.PAIRWISE_TEXT_QUALITY_RUBRIC_CRITIQUE_TEMPLATE,
                custom_output_config=_PAIRWISE_OUTPUT_CONFIG,
                autorater_config=_PAIRWISE_AUTORATER_CONFIG,
            ),
        )
