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
"""Schema for autorater metric configuration."""

AUTORATER_METRIC_SCHEMA = """
  $schema: https://json-schema.org/draft/2020-12/schema
  title: AutoRater Metric Configuration
  description: A metric definition for model-based evaluation.
  type: object
  properties:
    metadata:
      description: Useful information about the metric.
      type: object
      properties:
        name:
          description: Name of the metric.
          type: string
        description:
          description: Description of the metric.
          type: string
        author:
          description: Author of the metric.
          type: string
        contact:
          description: PoC for the metric.
          type: string
        version:
          description: Version of the metric.
          type: string
        classification:
          description: Classification of the metric.
          type: string
          enum:
            - experimental
            - benchmarked
            - deprecated
        required_inputs:
          description: Input fields used in the metric prompt template.
          type: array
          items:
            type: string
          minItems: 1
          uniqueItems: true
        benchmarks:
          description: List of benchmarks used for the metric.
          type: array
          items:
            type: object
            properties:
              dataset:
                description: Dataset used for benchmarking.
                type: string
              results:
                description: Results from benchmarking.
                type: string
            required:
              - results
          minItems: 1
          uniqueItems: true
        usage:
          description: Links to documentation or notebooks with example usage.
          type: array
          items:
            type: string
          minItems: 1
          uniqueItems: true
      required:
        - name
        - version
        - required_inputs
    steps:
      description: List of steps used for the autorater workflow.
      type: array
      items:
        type: object
        properties:
          type:
            description: Type of the step.
            type: string
            enum:
              - pointwise_metric
              - pairwise_metric
              - rubric
          prompt:
            description: Prompt template for the step.
            type: object
            properties:
              system_instruction:
                description: System instruction for the model.
                type: string
              template:
                description: Template to populate with inputs from the dataset.
                type: string
            required:
              - template
          model:
            description: Configuration of the model for the step.
            type: object
            properties:
              model_name_or_endpoint:
                description: Name or endpoint of the model.
                type: string
            required:
              - model_name_or_endpoint
          options:
            description: Options for the step.
            type: object
            properties:
              sample_count:
                description: Number of samples for each instance in the dataset.
                type: integer
              flip_enabled:
                description: Whether to flip candidate and baseline responses.
                type: boolean
          output:
            description: Output of the step.
            type: object
            properties:
              type:
                description: Type of the output.
                type: string
                enum:
                  - raw
            required:
              - type
        required:
          - type
          - prompt
      minItems: 1
      uniqueItems: true
  required:
    - metadata
    - steps
"""
