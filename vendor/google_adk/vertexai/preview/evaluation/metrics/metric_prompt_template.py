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
"""Metric prompt template classes for model-based metrics evaluation."""

from typing import Dict, List, Optional

from google.cloud.aiplatform import base
from vertexai.preview.evaluation import (
    prompt_template,
)


_LOGGER = base.Logger(__name__)
_NEWLINE = "\n"


def serialize_dict_in_order(elements: Optional[Dict[str, str]]):
    """Serializes dictionary to ordered string value without brackets."""
    if elements is None:
        return ""
    return _NEWLINE.join(f"{key}: {value}" for key, value in sorted(elements.items()))


class _MetricPromptTemplate(prompt_template.PromptTemplate):
    """Metric prompt template for generic model-based metrics evaluation."""

    def __init__(
        self,
        *,
        criteria: Dict[str, str],
        rating_rubric: Dict[str, str],
        input_variables: List[str],
        instruction: Optional[str] = None,
        evaluation_steps: Optional[Dict[str, str]] = None,
        metric_definition: Optional[str] = None,
        few_shot_examples: Optional[List[str]] = None,
    ):
        """Initializes a metric prompt template."""
        self._input_variables = input_variables

        self._instruction = instruction
        self._metric_definition = metric_definition
        self._criteria = criteria
        self._rating_rubric = rating_rubric
        self._evaluation_steps = evaluation_steps
        self._few_shot_examples = few_shot_examples

        self.template = self.__str__()

    @property
    def prompt_data(self) -> str:
        return self.template


class PointwiseMetricPromptTemplate(_MetricPromptTemplate):
    """Pointwise metric prompt template for pointwise model-based metrics."""

    def __init__(
        self,
        *,
        criteria: Dict[str, str],
        rating_rubric: Dict[str, str],
        input_variables: Optional[List[str]] = None,
        instruction: Optional[str] = None,
        metric_definition: Optional[str] = None,
        evaluation_steps: Optional[Dict[str, str]] = None,
        few_shot_examples: Optional[List[str]] = None,
    ):
        """Initializes a pointwise metric prompt template.

        Args:
            criteria: The standards and measures used to evaluate the model
              responses. It is a dictionary of criterion names and criterion
              definitions.
            rating_rubric: A dictionary mapping of rating name and rating
              definition, used to assign ratings or scores based on specific
              criteria.
            input_variables: An optional list of input fields to use in the metric
              prompt template for generating model-based evaluation results. Model
              "response" column is included by default. If metric_column_mapping is
              provided, the mapping values of the input fields will be used to
              retrieve data from the evaluation dataset.
            instruction: The general instruction to the model that performs the
              evaluation. If not provided, a default pointwise metric instruction
              will be used.
            metric_definition: The optional metric definition. It is a string
              describing the metric to be evaluated at a high level. If not
              provided, this field will not be included in the prompt template.
            evaluation_steps: The optional gudelines of evaluation steps. A
              dictionary of evaluation step name and evaluation step definition. If
              not provided, a default pointwise metric evaluation steps will be
              used.
            few_shot_examples: The optional list of few-shot examples to be used in
              the prompt, to provide the model with demonstrations of how to perform
              the evaluation, and improve the evaluation accuracy. If not provided,
              this field will not be included in the prompt template.
        """
        if not input_variables:
            input_variables = []
            _LOGGER.info(
                "The `input_variables` parameter is empty. Only the `response`"
                " column is used for computing this model-based metric."
            )
        input_variables = list(set(input_variables + ["response"]))

        instruction = instruction or self.get_default_pointwise_instruction()

        evaluation_steps = (
            evaluation_steps or self.get_default_pointwise_evaluation_steps()
        )

        super().__init__(
            input_variables=input_variables,
            criteria=criteria,
            rating_rubric=rating_rubric,
            instruction=instruction,
            metric_definition=metric_definition,
            evaluation_steps=evaluation_steps,
            few_shot_examples=few_shot_examples,
        )

    def get_default_pointwise_instruction(self) -> str:
        """Returns the default instruction for the metric prompt template."""

        return (
            "You are an expert evaluator. Your task is to evaluate the quality of"
            " the responses generated by AI models. We will provide you with the"
            " user prompt and an AI-generated responses.\nYou should first read"
            " the user input carefully for analyzing the task, and then evaluate"
            " the quality of the responses based on the Criteria provided in the"
            " Evaluation section below.\nYou will assign the response a rating"
            " following the Rating Rubric and Evaluation Steps. Give step by step"
            " explanations for your rating, and only choose ratings from the Rating"
            " Rubric."
        )

    def get_default_pointwise_evaluation_steps(self) -> Dict[str, str]:
        """Returns the default evaluation steps for the metric prompt template."""
        return {
            "Step 1": (
                "Assess the response in aspects of all criteria provided. Provide"
                " assessment according to each criterion."
            ),
            "Step 2": (
                "Score based on the rating rubric. Give a brief rationale to"
                " explain your evaluation considering each individual criterion."
            ),
        }

    def __str__(self):
        """Serializes the pointwise metric prompt template to a string."""
        metric_prompt_template_str = [
            "# Instruction",
            f"{self._instruction}",
            _NEWLINE,
            "# Evaluation",
        ]
        if self._metric_definition:
            metric_prompt_template_str.extend(
                [
                    "## Metric Definition",
                    f"{self._metric_definition}\n",
                ]
            )
        metric_prompt_template_str.extend(
            [
                "## Criteria",
                f"{serialize_dict_in_order(self._criteria)}\n",
                "## Rating Rubric",
                f"{serialize_dict_in_order(self._rating_rubric)}\n",
            ]
        )
        if self._evaluation_steps:
            metric_prompt_template_str.extend(
                [
                    "## Evaluation Steps",
                    f"{serialize_dict_in_order(self._evaluation_steps)}\n",
                ]
            )
        if self._few_shot_examples:
            metric_prompt_template_str.extend(
                [
                    "## Evaluation Examples",
                    f"{_NEWLINE.join(self._few_shot_examples)}\n",
                ]
            )
        metric_prompt_template_str.extend(
            ["\n# User Inputs and AI-generated Response", "## User Inputs"]
        )
        for input_variable in self._input_variables:
            if input_variable == "response":
                continue
            metric_prompt_template_str.extend(
                [
                    f"### {input_variable}",
                    f"{{{input_variable}}}\n",
                ]
            )
        metric_prompt_template_str.extend(
            [
                _NEWLINE,
                "\n## AI-generated Response",
                "{response}",
            ]
        )
        return _NEWLINE.join(metric_prompt_template_str)

    def __repr__(self):
        return (
            f"PointwiseMetricPromptTemplate(prompt_data={self.prompt_data},"
            f" variables={self.variables})"
        )


class PairwiseMetricPromptTemplate(_MetricPromptTemplate):
    """Pairwise metric prompt template for pairwise model-based metrics."""

    def __init__(
        self,
        *,
        criteria: Dict[str, str],
        rating_rubric: Dict[str, str],
        input_variables: Optional[List[str]] = None,
        instruction: Optional[str] = None,
        metric_definition: Optional[str] = None,
        evaluation_steps: Optional[Dict[str, str]] = None,
        few_shot_examples: Optional[List[str]] = None,
    ):
        """Initializes a pairwise metric prompt template.

        Args:
            criteria: The standards and measures used to evaluate the model
              responses. It is a dictionary of criterion names and criterion
              definitions.
            rating_rubric: A dictionary mapping of rating name and rating
              definition, used to assign ratings or scores based on specific
              criteria.
            input_variables: An optional list of input fields to use in the metric
              prompt template for generating model-based evaluation results.
              Candidate model "response" column and "baseline_model_response" column
              are included by default. If metric_column_mapping is provided, the
              mapping values of the input fields will be used to retrieve data from
              the evaluation dataset.
            instruction: The general instruction to the model that performs the
              evaluation. If not provided, a default pairwise metric instruction
              will be used.
            metric_definition: The optional metric definition. It is a string
              describing the metric to be evaluated at a high level. If not
              provided, this field will not be included in the prompt template.
            evaluation_steps: The optional gudelines of evaluation steps. A
              dictionary of evaluation step name and evaluation step definition. If
              not provided, a default pairwise metric evaluation steps will be used.
            few_shot_examples: The optional list of few-shot examples to be used in
              the prompt, to provide the model with demonstrations of how to perform
              the evaluation, and improve the evaluation accuracy. If not provided,
              this field will not be included in the prompt template.
        """
        if not input_variables:
            input_variables = []
            _LOGGER.info(
                "The `input_variables` parameter is empty. Only the `response`"
                " column and `baseline_model_response` columns are used for"
                " computing this model-based metric."
            )
        input_variables = list(
            set(input_variables + ["response", "baseline_model_response"])
        )

        instruction = instruction or self.get_default_pairwise_instruction()

        evaluation_steps = (
            evaluation_steps or self.get_default_pairwise_evaluation_steps()
        )

        super().__init__(
            input_variables=input_variables,
            criteria=criteria,
            rating_rubric=rating_rubric,
            instruction=instruction,
            metric_definition=metric_definition,
            evaluation_steps=evaluation_steps,
            few_shot_examples=few_shot_examples,
        )

    def get_default_pairwise_instruction(self) -> str:
        """Returns the default instruction for the metric prompt template."""

        return (
            "You are an expert evaluator. Your task is to evaluate the quality of"
            " the responses generated by two AI models. We will provide you with"
            " the user input and a pair of AI-generated responses (Response A and"
            " Response B).\nYou should first read the user input carefully for"
            " analyzing the task, and then evaluate the quality of the responses"
            " based on based on the Criteria provided in the Evaluation section"
            " below.\nYou will first judge responses individually, following the"
            " Rating Rubric and Evaluation Steps. Then you will give step by step"
            " explanations for your judgement, compare results to declare the"
            " winner based on the Rating Rubric and Evaluation Steps."
        )

    def get_default_pairwise_evaluation_steps(self) -> Dict[str, str]:
        """Returns the default evaluation steps for the metric prompt template."""
        return {
            "Step 1": "Analyze Response A based on all the Criteria.",
            "Step 2": "Analyze Response B based on all the Criteria.",
            "Step 3": (
                "Compare the overall performance of Response A and Response B based"
                " on your analyses and assessment."
            ),
            "Step 4": (
                'Output your preference of "A", "SAME" or "B" to the'
                " pairwise_choice field according to the Rating Rubrics."
            ),
            "Step 5": "Output your assessment reasoning in the explanation field",
        }

    def __str__(self):
        """Serializes the pairwise metric prompt template to a string."""
        metric_prompt_template_str = [
            "# Instruction",
            f"{self._instruction}",
            _NEWLINE,
            "# Evaluation",
        ]
        if self._metric_definition:
            metric_prompt_template_str.extend(
                [
                    "## Metric Definition",
                    f"{self._metric_definition}\n",
                ]
            )
        metric_prompt_template_str.extend(
            [
                "## Criteria",
                f"{serialize_dict_in_order(self._criteria)}\n",
                "## Rating Rubric",
                f"{serialize_dict_in_order(self._rating_rubric)}\n",
            ]
        )
        if self._evaluation_steps:
            metric_prompt_template_str.extend(
                [
                    "## Evaluation Steps",
                    f"{serialize_dict_in_order(self._evaluation_steps)}\n",
                ]
            )
        if self._few_shot_examples:
            metric_prompt_template_str.extend(
                [
                    "## Evaluation Examples",
                    f"{_NEWLINE.join(self._few_shot_examples)}\n",
                ]
            )
        metric_prompt_template_str.extend(
            ["\n# User Inputs and AI-generated Responses", "## User Inputs"]
        )
        for input_variable in self._input_variables:
            if input_variable in ["response", "baseline_model_response"]:
                continue
            metric_prompt_template_str.extend(
                [
                    f"### {input_variable}",
                    f"{{{input_variable}}}\n",
                ]
            )
        metric_prompt_template_str.extend(
            [
                "\n## AI-generated Responses",
                "### Response A",
                "{baseline_model_response}\n",
                "### Response B",
                "{response}",
            ]
        )
        return _NEWLINE.join(metric_prompt_template_str)

    def __repr__(self):
        return (
            f"PairwiseMetricPromptTemplate(prompt_data={self.prompt_data},"
            f" variables={self.variables})"
        )
