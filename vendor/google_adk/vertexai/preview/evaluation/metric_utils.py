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
"""Utility functions for metrics."""

import datetime
import io
from typing import Any, Dict, List, Optional, Union, Callable

from google.cloud.aiplatform import initializer
from google.cloud.aiplatform_v1beta1.types import (
    evaluation_service as gapic_eval_service_types,
)
from vertexai.preview.evaluation import metrics
from vertexai.preview.evaluation import prompt_template
from vertexai.preview.evaluation import utils
from vertexai.preview.evaluation.metrics import _schema
from vertexai.preview.evaluation.metrics import (
    custom_output_config,
)
from vertexai import generative_models
from vertexai.generative_models import _generative_models
import jsonschema
import ruamel.yaml
from ruamel.yaml import scalarstring


AutoraterConfig = gapic_eval_service_types.AutoraterConfig
GenerativeModel = generative_models.GenerativeModel
CustomOutputConfig = custom_output_config.CustomOutputConfig
PairwiseMetric = metrics.PairwiseMetric
PointwiseMetric = metrics.PointwiseMetric
RubricBasedMetric = metrics.RubricBasedMetric
RubricGenerationConfig = metrics.RubricGenerationConfig
PromptTemplate = prompt_template.PromptTemplate

# Initialize schema validator.
_schema = ruamel.yaml.YAML(typ="safe").load(_schema.AUTORATER_METRIC_SCHEMA)
_schema_validator = jsonschema.Draft202012Validator(schema=_schema)


def dump(
    metric: Union[PointwiseMetric, PairwiseMetric, RubricBasedMetric],
    file_path: str,
    version: Optional[str] = None,
):
    """Dumps a metric object to a YAML file.

    Args:
      metric: The metric to be dumped to a file.
      file_path: The path to the file. Local and GCS files are supported.
      version: Optional. The version of the metric. Defaults to the timestamp
        when the metric file is created.
    """
    yaml_data = dumps(metric, version)
    if file_path.startswith(utils._GCS_PREFIX):
        utils._upload_string_to_gcs(file_path, yaml_data)
    else:
        with open(file_path, "w") as f:
            f.write(yaml_data)


def dumps(
    metric: Union[PointwiseMetric, PairwiseMetric, RubricBasedMetric],
    version: Optional[str] = None,
) -> str:
    """Dumps a metric object to YAML data.

    Args:
      metric: The metric to be dumped to YAML data.
      version: Optional. The version of the metric. Defaults to the timestamp
        when the metric file is created.

    Returns:
      The YAML data of the metric.
    """
    steps = []
    metric_name = None
    if isinstance(metric, PointwiseMetric) or isinstance(metric, PairwiseMetric):
        metric_name = metric.metric_name
        steps.append(_dump_metric(metric))
    elif isinstance(metric, RubricBasedMetric):
        metric_name = metric.critique_metric.metric_name
        steps.append(_dump_rubric(metric.generation_config))
        steps.append(_dump_metric(metric.critique_metric))
    metadata = {
        "name": metric_name,
        "version": (
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if version is None
            else version
        ),
        "required_inputs": _parse_required_inputs(metric),
    }
    metric_config = {
        "metadata": metadata,
        "steps": steps,
    }
    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=4, offset=2)
    with io.StringIO() as s:
        yaml.dump(metric_config, s)
        return s.getvalue()


def _dump_metric(metric: Union[PointwiseMetric, PairwiseMetric]) -> Dict[str, Any]:
    """Dumps a metric object to autorater metric schema."""
    output_type = None
    if metric.custom_output_config and metric.custom_output_config.return_raw_output:
        output_type = "raw"
    step = {
        "type": (
            "pairwise_metric"
            if isinstance(metric, PairwiseMetric)
            else "pointwise_metric"
        ),
        "prompt": {
            "template": scalarstring.preserve_literal(metric.metric_prompt_template),
        },
    }
    if metric.system_instruction:
        step["prompt"]["system_instruction"] = metric.system_instruction
    if output_type:
        step["output"] = {
            "type": output_type,
        }
    if metric.autorater_config:
        step["model"] = {
            "model_name_or_endpoint": (metric.autorater_config.autorater_model),
        }
        options = {}
        if metric.autorater_config.flip_enabled:
            options["flip_enabled"] = metric.autorater_config.flip_enabled
        if metric.autorater_config.sampling_count:
            options["sample_count"] = metric.autorater_config.sampling_count
        if options:
            step["options"] = options
    return step


def _dump_rubric(generation_config: RubricGenerationConfig) -> Dict[str, Any]:
    """Dumps a rubric generation config to autorater metric schema."""
    # TODO: b/396217889 - add support for custom output.
    step = {
        "type": "rubric",
        "prompt": {
            "template": scalarstring.preserve_literal(
                generation_config.prompt_template
            ),
        },
    }
    if generation_config.model and isinstance(generation_config.model, GenerativeModel):
        step["model"] = {
            "model_name_or_endpoint": generation_config.model._model_name,
        }
    return step


def _parse_required_inputs(
    metric: Union[PointwiseMetric, PairwiseMetric, RubricBasedMetric],
) -> List[str]:
    """Parses required inputs from a metric object."""
    if isinstance(metric, PointwiseMetric) or isinstance(metric, PairwiseMetric):
        return list(PromptTemplate(metric.metric_prompt_template).variables)
    elif isinstance(metric, RubricBasedMetric):
        met = PromptTemplate(metric.critique_metric.metric_prompt_template).variables
        gen = PromptTemplate(metric.generation_config.prompt_template).variables
        return list(met.union(gen))
    else:
        raise ValueError(f"Unsupported metric type: {type(metric)}")


def load(
    file_path: str,
    baseline_model: Optional[Union[GenerativeModel, Callable[[str], str]]] = None,
) -> Union[PointwiseMetric, PairwiseMetric, RubricBasedMetric]:
    """Loads a metric object from a YAML file.

    Args:
      file_path: Path to the file containing the autorater metric configuration.
        Local and GCS files are supported.
      baseline_model: Optional. The baseline model to use for pairwise metrics.

    Returns:
      The metric object loaded from the file.
    """
    if file_path.startswith(utils._GCS_PREFIX):
        file_contents = utils._read_gcs_file_contents(file_path)
        return loads(file_contents, baseline_model)
    with open(file_path, "r") as f:
        return loads(f.read(), baseline_model)


def loads(
    yaml_data: str,
    baseline_model: Optional[Union[GenerativeModel, Callable[[str], str]]] = None,
) -> Union[PointwiseMetric, PairwiseMetric, RubricBasedMetric]:
    """Loads a metric object from YAML data.

    Args:
      yaml_data: YAML data containing the autorater metric configuration.
      baseline_model: Optional. The baseline model to use for pairwise metrics.

    Returns:
      The metric object loaded from the YAML data.
    """
    yaml = ruamel.yaml.YAML(typ="safe")
    yaml_obj = yaml.load(yaml_data)
    try:
        _schema_validator.validate(yaml_obj)
    except jsonschema.exceptions.ValidationError as e:
        raise ValueError(
            f"Invalid autorater metric config: {e.message} for {e.path.pop()}"
        ) from e
    metadata = yaml_obj["metadata"]
    steps = yaml_obj["steps"]
    required_inputs = set(metadata["required_inputs"])
    metric = None
    rubric = None
    for step in steps:
        _validate_template(step["prompt"]["template"], required_inputs)
        model_name = None
        flip = None
        sampling = None
        if "model" in step:
            model_name = _parse_model_name(step["model"]["model_name_or_endpoint"])
        if "options" in step:
            flip = step["options"].get("flip_enabled", False)
            sampling = step["options"].get("sample_count", 1)
        autorater = None
        if model_name:
            autorater = AutoraterConfig(
                autorater_model=model_name,
                flip_enabled=flip,
                sampling_count=sampling,
            )
        system_instruction = step["prompt"].get("system_instruction")
        custom_output = None
        if "output" in step and step["output"]["type"] == "raw":
            custom_output = CustomOutputConfig(return_raw_output=True)
        if step["type"] == "pointwise_metric":
            if metric is not None:
                raise ValueError("Only one metric step is supported.")
            if baseline_model:
                raise ValueError("Baseline model provided for pointwise metric.")
            metric = PointwiseMetric(
                metric=metadata["name"],
                metric_prompt_template=step["prompt"]["template"],
                system_instruction=system_instruction,
                autorater_config=autorater,
                custom_output_config=custom_output,
            )
        elif step["type"] == "pairwise_metric":
            if metric is not None:
                raise ValueError("Only one metric step is supported.")
            metric = PairwiseMetric(
                metric=metadata["name"],
                metric_prompt_template=step["prompt"]["template"],
                system_instruction=system_instruction,
                baseline_model=baseline_model,
                autorater_config=autorater,
                custom_output_config=custom_output,
            )
        elif step["type"] == "rubric":
            if rubric is not None:
                raise ValueError("Only one rubric step is supported.")
            model = None
            if model_name:
                model = generative_models.GenerativeModel(model_name=model_name)
            rubric = RubricGenerationConfig(
                prompt_template=step["prompt"]["template"],
                model=model,
            )
    if metric is None:
        raise ValueError("A metric step must be provided.")
    if rubric is not None:
        return RubricBasedMetric(
            generation_config=rubric,
            critique_metric=metric,
        )
    return metric


def _parse_model_name(model_name_or_endpoint: str) -> str:
    """Parses model name or endpoint.

    Args:
      model_name_or_endpoint: Model Garden model  name or tuned model endpoint
        resource name can be provided.

    Returns:
      The model resource name.
    """
    project = initializer.global_config.project
    location = initializer.global_config.location
    model_name = _generative_models._reconcile_model_name(
        model_name_or_endpoint, project, location
    )
    return _generative_models._get_resource_name_from_model_name(
        model_name, project, location
    )


def _validate_template(template: str, required_inputs: List[str]) -> None:
    """Validates the template contains only required inputs."""
    placeholders = PromptTemplate(template).variables
    if not placeholders.issubset(required_inputs):
        raise ValueError(
            "Template contains placeholders that are not in required inputs:"
            f" {placeholders - required_inputs}"
        )
