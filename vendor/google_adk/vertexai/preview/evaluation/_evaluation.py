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
"""Evaluation Orchestration Library."""

import collections
from concurrent import futures
import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from google.cloud.aiplatform import base
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)
from vertexai import generative_models
from vertexai.preview import reasoning_engines
from vertexai.preview.evaluation import _base as evaluation_base
from vertexai.preview.evaluation import _pre_eval_utils
from vertexai.preview.evaluation import constants
from vertexai.preview.evaluation import (
    prompt_template as prompt_template_base,
)
from vertexai.preview.evaluation import utils
from vertexai.preview.evaluation.metrics import (
    _base as metrics_base,
)
from vertexai.preview.evaluation.metrics import (
    _instance_evaluation,
)
from vertexai.preview.evaluation.metrics import (
    custom_output_config,
)
from vertexai.preview.evaluation.metrics import (
    metric_prompt_template_examples,
)
from vertexai.preview.evaluation.metrics import pairwise_metric
from vertexai.preview.evaluation.metrics import pointwise_metric
from vertexai.preview.evaluation.metrics import (
    rubric_based_metric,
)


if TYPE_CHECKING:
    import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        'tqdm is not installed. Please install the SDK using "pip install'
        ' google-cloud-aiplatform[evaluation]"'
    )

_LOGGER = base.Logger(__name__)
_SUCCESSFUL_FINISH_REASONS = [
    gapic_content_types.Candidate.FinishReason.STOP,
    gapic_content_types.Candidate.FinishReason.MAX_TOKENS,
    # Many responses have this finish reason
    gapic_content_types.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED,
]

_RunnableType = Union[reasoning_engines.Queryable, Callable[[str], Dict[str, str]]]
_ModelType = Union[generative_models.GenerativeModel, Callable[[str], str]]


def _validate_metrics(metrics: List[Union[str, metrics_base._Metric]]) -> None:
    """Validates the metrics list.

    Args:
      metrics: The list of metric names, or Metric instances to
        evaluate.

    Raises:
      ValueError: If metric is empty or if multiple metrics of the
        same metric name are found.
    """
    if not metrics:
        raise ValueError("Metrics cannot be empty.")

    seen_strings = set()
    seen_metric_names = set()
    rubric_based_metric_present = False

    for metric in metrics:
        if isinstance(metric, str):
            if metric in seen_strings:
                raise ValueError(f"Duplicate string metric name found: '{metric}'")
            seen_strings.add(metric)
        elif isinstance(
            metric, metrics_base._Metric
        ):  # pylint: disable=protected-access
            if metric.metric_name in seen_metric_names:
                raise ValueError(
                    "Duplicate Metric instances of the same metric name found: "
                    f"'{metric.metric_name}'"
                )
            if isinstance(metric, rubric_based_metric.RubricBasedMetric):
                if rubric_based_metric_present:
                    raise ValueError("Multiple rubric based metrics are not supported.")
                rubric_based_metric_present = True
            seen_metric_names.add(metric.metric_name)


def _validate_metric_column_map(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
):
    """Validates the column map for metric prompt template usage."""
    for metric in evaluation_run_config.metrics:
        if isinstance(
            metric, metrics_base._ModelBasedMetric  # pylint: disable=protected-access
        ):
            for variable in prompt_template_base.PromptTemplate(
                metric.metric_prompt_template
            ).variables:
                if (
                    evaluation_run_config.metric_column_mapping.get(variable, "")
                    not in evaluation_run_config.dataset.columns
                ):
                    raise ValueError(
                        f"Cannot find the `{variable}` column in the evaluation"
                        " dataset to fill the metric prompt template for"
                        f" `{str(metric)}` metric. Please check if the column is"
                        " present in the evaluation dataset, or provide a"
                        " key-value pair in `metric_column_mapping` parameter"
                        " of `EvalTask` to map it to a different column name."
                        " The evaluation dataset columns are"
                        f" {list(evaluation_run_config.dataset.columns)}."
                    )


def _validate_autorater_config(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
    pairwise_metric_exists: bool = False,
):
    """Validates the fields in the autorater config.

    Args:
      evaluation_run_config: The evaluation run config.
      pairwise_metric_exists: Whether pairwise metrics are used.

    Raises:
      ValueError: If the sampling count is not in the range [1, 32], or flipping
      is enabled for pairwise metrics but the required columns are not found in
      the dataset.
    """
    if not evaluation_run_config.autorater_config:
        return
    if evaluation_run_config.autorater_config.sampling_count and (
        evaluation_run_config.autorater_config.sampling_count < 1
        or evaluation_run_config.autorater_config.sampling_count > 32
    ):
        raise ValueError(
            "autorater_config.sampling_count must be in the range [1, 32]."
        )

    if pairwise_metric_exists and evaluation_run_config.autorater_config.flip_enabled:
        for column in [
            constants.Dataset.MODEL_RESPONSE_COLUMN,
            constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN,
        ]:
            field_name = evaluation_run_config.metric_column_mapping.get(column, "")
            if (
                not field_name
                or field_name not in evaluation_run_config.dataset.columns
            ):
                raise ValueError(
                    f"Cannot find the `{column}` column in"
                    " the evaluation dataset to fill the metric prompt template."
                    " This is required for Pairwise metrics when performing"
                    " autorater config flipping. Please check if the column is"
                    " present in the evaluation dataset, or provide a key-value"
                    " pair in `metric_column_mapping` parameter of `EvalTask` to"
                    " map it to a different column name. The evaluation dataset"
                    f" columns are {list(evaluation_run_config.dataset.columns)}."
                )


def _validate_column_provided(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
    column_name: str,
) -> None:
    """Validates the required column exist in the dataset."""
    if column_name not in evaluation_run_config.metric_column_mapping:
        evaluation_run_config.metric_column_mapping[column_name] = column_name
    evaluation_run_config.validate_dataset_column(column_name)


def _validate_response_column_required(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
) -> None:
    """Validates the response column exists in the dataset."""
    for metric in evaluation_run_config.metrics:
        if metric in constants.Metric.AUTOMATIC_METRIC_LIST:
            _validate_column_provided(
                evaluation_run_config,
                constants.Dataset.MODEL_RESPONSE_COLUMN,
            )


def _validate_reference_column_required(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
) -> None:
    """Validates the reference column exists in the dataset."""
    if set(evaluation_run_config.metrics).intersection(
        set(constants.Metric.AUTOMATIC_METRIC_LIST)
    ):
        _validate_column_provided(
            evaluation_run_config,
            constants.Dataset.REFERENCE_COLUMN,
        )


def _validate_dataset(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
) -> None:
    """Validates the required columns exists in the dataset."""
    _validate_response_column_required(evaluation_run_config)
    _validate_reference_column_required(evaluation_run_config)


def _compute_custom_metrics(
    row_dict: Dict[str, Any],
    custom_metrics: List[metrics_base.CustomMetric],
    pbar: tqdm,
    executor: futures.ThreadPoolExecutor,
) -> Dict[str, Any]:
    """Computes custom metrics for a row.

    Args:
        row_dict: A dictionary of an instance in the eval dataset.
        custom_metrics: A list of CustomMetrics.
        pbar: A tqdm progress bar.
        executor: A thread pool executor.

    Returns:
        A dictionary of an instance containing custom metric results.

    Raises:
        KeyError: If the custom metric function does not return a valid output.
    """
    futures_by_metric = collections.defaultdict(list)
    for custom_metric in custom_metrics:
        future = executor.submit(custom_metric.metric_function, row_dict)
        future.add_done_callback(lambda _: pbar.update(1))
        futures_by_metric[custom_metric].append(future)

    for custom_metric, futures_list in futures_by_metric.items():
        for future in futures_list:
            metric_output = future.result()
            try:
                row_dict[f"{custom_metric.name}/{constants.MetricResult.SCORE_KEY}"] = (
                    metric_output[custom_metric.name]
                )
            except KeyError:
                raise KeyError(
                    f"Custom metric score `{custom_metric.name}` not found in"
                    f" the metric output {metric_output}. Please make sure the"
                    " custom metric function is valid, and the output"
                    f" dictionary uses `{custom_metric.name}` as the key for"
                    " metric score."
                )
            # Include additional metric results like explanation.
            for key, value in metric_output.items():
                if key != custom_metric.name:
                    row_dict[f"{custom_metric.name}/{key}"] = value
    return row_dict


def _separate_custom_metrics(
    metrics: List[Union[str, metrics_base._Metric]],
) -> Tuple[
    List[Union[str, metrics_base._Metric]],
    List[metrics_base.CustomMetric],
]:
    """Separates the metrics list into API and custom metrics."""
    custom_metrics = []
    api_metrics = []
    for metric in metrics:
        if isinstance(metric, metrics_base.CustomMetric):
            custom_metrics.append(metric)
        else:
            api_metrics.append(metric)
    return api_metrics, custom_metrics


def _aggregate_summary_metrics(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
    metrics_table: "pd.DataFrame",
) -> Dict[str, Any]:
    """Computes summary metrics.

    Args:
        evaluation_run_config: Evaluation Run Configurations.
        metrics_table: A dataframe containing per-instance metrics results.

    Returns:
        A dictionary containing summary metrics results and statistics.
    """
    summary_metrics = {}
    summary_metrics[constants.MetricResult.ROW_COUNT_KEY] = metrics_table.shape[0]

    for metric in evaluation_run_config.metrics:
        try:
            if isinstance(metric, pairwise_metric.PairwiseMetric):
                pairwise_choice_col_name = (
                    f"{metric.metric_name}/{constants.MetricResult.PAIRWISE_CHOICE_KEY}"
                )
                if pairwise_choice_col_name in metrics_table:
                    candidate_model_win_rate_choices = [
                        "CANDIDATE",
                        utils.RATING_TO_VERDICT["B>>A"],
                        utils.RATING_TO_VERDICT["A<<B"],
                        utils.RATING_TO_VERDICT["B>A"],
                        utils.RATING_TO_VERDICT["A<B"],
                    ]
                    baseline_model_win_rate_choices = [
                        "BASELINE",
                        utils.RATING_TO_VERDICT["A>>B"],
                        utils.RATING_TO_VERDICT["B<<A"],
                        utils.RATING_TO_VERDICT["B<A"],
                        utils.RATING_TO_VERDICT["A>B"],
                    ]
                    summary_metrics[
                        f"{metric.metric_name}/candidate_model_win_rate"
                    ] = (
                        metrics_table[pairwise_choice_col_name].isin(
                            candidate_model_win_rate_choices
                        )
                    ).mean()
                    summary_metrics[f"{metric.metric_name}/baseline_model_win_rate"] = (
                        metrics_table[pairwise_choice_col_name].isin(
                            baseline_model_win_rate_choices
                        )
                    ).mean()
            else:
                score_col_name = f"{str(metric)}/{constants.MetricResult.SCORE_KEY}"
                if score_col_name in metrics_table:
                    summary_metrics[f"{str(metric)}/mean"] = metrics_table.loc[
                        :, score_col_name
                    ].mean()
                    summary_metrics[f"{str(metric)}/std"] = metrics_table.loc[
                        :, score_col_name
                    ].std()
        except (ValueError, KeyError) as e:
            _LOGGER.warning(
                f"Failed to compute metric statistics for `{metric}` metric."
                f"{type(e).__name__}: {e}"
            )
            continue

    metrics_table_columns = metrics_table.columns.tolist()
    for default_metric in constants.Metric.DEFAULT_METRIC_LIST:
        if default_metric in metrics_table_columns:
            summary_metrics[f"{str(default_metric)}/mean"] = metrics_table.loc[
                :, f"{str(default_metric)}"
            ].mean()
            summary_metrics[f"{str(default_metric)}/std"] = metrics_table.loc[
                :, f"{str(default_metric)}"
            ].std()
    return summary_metrics


def _run_model_inference(
    model: _ModelType,
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
    response_column_name: str = constants.Dataset.MODEL_RESPONSE_COLUMN,
) -> None:
    """Runs model inference on dataset for evaluation.

    Args:
      model: The model or baseline model or a custom model function to
        generate responses to evaluate.
      evaluation_run_config: Evaluation Run Configurations.
      response_column_name: Column name key in metric_column_mapping. Value is
        constants.Dataset.MODEL_RESPONSE_COLUMN or
        constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN.

    Raises:
        ValueError: If the model or baseline model is not supported.
    """
    is_baseline_model = (
        response_column_name == constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN
    )
    if response_column_name not in evaluation_run_config.metric_column_mapping:
        if constants.Dataset.PROMPT_COLUMN in evaluation_run_config.dataset.columns:
            t1 = time.perf_counter()
            if isinstance(model, generative_models.GenerativeModel):
                responses = _pre_eval_utils._generate_responses_from_gemini_model(
                    model, evaluation_run_config.dataset
                )
                _pre_eval_utils.populate_eval_dataset_with_model_responses(
                    responses,
                    evaluation_run_config,
                    is_baseline_model,
                )
            elif callable(model):
                responses = _pre_eval_utils._generate_response_from_custom_model_fn(
                    model, evaluation_run_config.dataset
                )
                _pre_eval_utils.populate_eval_dataset_with_model_responses(
                    responses,
                    evaluation_run_config,
                    is_baseline_model,
                )
            else:
                raise ValueError(
                    f"Unsupported model or baseline model type: {type(model)}"
                )
            t2 = time.perf_counter()
            _LOGGER.info(f"Multithreaded Batch Inference took: {t2 - t1} seconds.")
            evaluation_run_config.metric_column_mapping[response_column_name] = (
                response_column_name
            )
        else:
            raise ValueError(
                "Missing required input `prompt` column to start model inference."
                " Please provide a `prompt_template` parameter in"
                " `EvalTask.evaluate()` function if you want to assemble a"
                " `prompt` column with variables from the dataset, or provide a"
                " `prompt` column in dataset to directly use as input to"
                " the model. Mappings in `metric_column_mapping` do not"
                " apply for model inference and are used for evaluation only."
            )
    else:
        raise ValueError(
            "The `model` parameter or `baseline_model` in pairwise metric is"
            " specified, but the evaluation `dataset` contains model response"
            " column or baseline model response column"
            f" `{evaluation_run_config.metric_column_mapping[response_column_name]}`"
            " to perform bring-your-own-response(BYOR) evaluation. If you would"
            " like to perform evaluation using the dataset with the"
            " existing model response column or baseline model response column"
            f" `{evaluation_run_config.metric_column_mapping[response_column_name]}`,"
            " please remove `model` parameter in `EvalTask.evaluate()`"
            " function or `baseline_model` in `PairwiseMetric`."
        )


def _generate_response_from_custom_agent_fn(
    runnable: Callable[[str], Dict[str, str]], prompt: str
):
    """Processes a single row from the DataFrame for a custom agent."""
    t1 = time.perf_counter()
    try:
        response_dict = runnable(prompt)
        latency = time.perf_counter() - t1
        failure = 0
    except Exception as e:  # pylint: disable=broad-exception-caught
        latency = None
        response_dict = {
            "input": prompt,
            "output": f"Error occurred during inference: {e}",
            "predicted_trajectory": None,
        }
        failure = 1
    return response_dict, latency, failure


def _generate_response_from_queryable(
    runnable: reasoning_engines.Queryable, prompt: str
):
    """Processes a single row from the DataFrame for a Queryable."""
    t1 = time.perf_counter()
    try:
        response_dict = runnable.query(input=prompt)
        output_dict = {
            "input": prompt,
            "output": response_dict["output"],
        }
        if "intermediate_steps" in response_dict:
            predicted_trajectory = utils.parse_intermediate_steps(
                response_dict["intermediate_steps"]
            )
            output_dict["predicted_trajectory"] = predicted_trajectory
        latency = time.perf_counter() - t1
        failure = 0

    except Exception as e:  # pylint: disable=broad-exception-caught
        latency = None
        output_dict = {
            "input": prompt,
            "output": f"Error occurred during inference: {e}",
            "predicted_trajectory": None,
        }
        failure = 1
    return output_dict, latency, failure


def _run_runnable_inference(
    runnable: _RunnableType,
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
    response_column_name: str = constants.Dataset.MODEL_RESPONSE_COLUMN,
    predicted_trajectory_column_name: str = constants.Dataset.PREDICTED_TRAJECTORY_COLUMN,
):
    """Runs inference with a runnable agent and adds the inference final response and predicted trajectory to the response column."""
    if constants.Dataset.PROMPT_COLUMN in evaluation_run_config.dataset.columns:
        responses = []
        trajectory_list = []
        latency_list = []
        failure_list = []
        tasks = []
        if isinstance(runnable, reasoning_engines.Queryable):
            with tqdm(total=len(evaluation_run_config.dataset)) as pbar:
                with futures.ThreadPoolExecutor(
                    max_workers=constants.MAX_WORKERS
                ) as executor:
                    for _, row in evaluation_run_config.dataset.iterrows():
                        task = executor.submit(
                            _generate_response_from_queryable,
                            runnable,
                            row[constants.Dataset.PROMPT_COLUMN],
                        )
                        task.add_done_callback(lambda _: pbar.update(1))
                        tasks.append(task)
                    for task in tasks:
                        response_dict, latency, failure = task.result()
                        responses.append(response_dict["output"])
                        latency_list.append(latency)
                        failure_list.append(failure)
                        if (
                            response_column_name
                            in evaluation_run_config.metric_column_mapping
                        ):
                            _LOGGER.warning(
                                "When a `runnable` is provided, response is"
                                " generated dynamically by the runnable, so "
                                "the pre-existing `response` column provided "
                                "in the evaluation dataset is not used."
                            )
                        if "predicted_trajectory" in response_dict:
                            trajectory_list.append(
                                response_dict["predicted_trajectory"]
                            )
                            if (
                                predicted_trajectory_column_name
                                in evaluation_run_config.metric_column_mapping
                            ):
                                _LOGGER.warning(
                                    "When a `runnable` is provided, trajectory is"
                                    "generated dynamically by the runnable, so "
                                    "the pre-existing `response` column provided "
                                    "in the evaluation dataset is not used."
                                )
        elif callable(runnable):
            with tqdm(total=len(evaluation_run_config.dataset)) as pbar:
                with futures.ThreadPoolExecutor(
                    max_workers=constants.MAX_WORKERS
                ) as executor:
                    for _, row in evaluation_run_config.dataset.iterrows():
                        task = executor.submit(
                            _generate_response_from_custom_agent_fn,
                            runnable,
                            row[constants.Dataset.PROMPT_COLUMN],
                        )
                        task.add_done_callback(lambda _: pbar.update(1))
                        tasks.append(task)
                    for task in tasks:
                        response_dict, latency, failure = task.result()
                        responses.append(response_dict.get("response", ""))
                        latency_list.append(latency)
                        failure_list.append(failure)
                        if "predicted_trajectory" in response_dict:
                            predicted_trajectory = response_dict["predicted_trajectory"]
                            trajectory_list.append(predicted_trajectory)
                            if (
                                response_column_name
                                in evaluation_run_config.metric_column_mapping
                            ):
                                _LOGGER.warning(
                                    "When a `runnable` is provided, response is"
                                    " generated dynamically by the runnable, so "
                                    "the pre-existing `response` column provided "
                                    "in the evaluation dataset is not used."
                                )
                            if (
                                predicted_trajectory_column_name
                                in evaluation_run_config.metric_column_mapping
                            ):
                                _LOGGER.warning(
                                    "When a `runnable` is provided, trajectory"
                                    " is generated dynamically by the runnable,"
                                    " so the pre-existing `predicted_trajectory`"
                                    " column provided in the evaluation dataset"
                                    " is not used."
                                )
        else:
            raise ValueError(f"Unsupported runnable type: {type(runnable)}")

        evaluation_run_config.dataset = evaluation_run_config.dataset.assign(
            response=responses
        )
        evaluation_run_config.dataset = evaluation_run_config.dataset.assign(
            latency_in_seconds=latency_list
        )
        evaluation_run_config.dataset = evaluation_run_config.dataset.assign(
            failure=failure_list
        )
        evaluation_run_config.metric_column_mapping[response_column_name] = (
            response_column_name
        )
        if trajectory_list:
            evaluation_run_config.dataset = evaluation_run_config.dataset.assign(
                predicted_trajectory=trajectory_list
            )
            evaluation_run_config.metric_column_mapping[
                predicted_trajectory_column_name
            ] = predicted_trajectory_column_name

        _LOGGER.info(
            f"All {evaluation_run_config.dataset.shape[0]} responses are successfully"
            " generated from the runnable."
        )
    else:
        raise ValueError(
            "Missing required input `prompt` column to start inference."
            " Please provide a `prompt_template` parameter in"
            " `EvalTask.evaluate()` function if you want to assemble a"
            " `prompt` column with variables from the dataset, or provide a"
            " `prompt` column in dataset to directly use as input to"
            " the runnable. Mappings in `metric_column_mapping` do not"
            " apply for runnable inference and are used for evaluation only."
        )


def _assemble_prompt_for_dataset(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
    prompt_template: Union[prompt_template_base.PromptTemplate, str],
) -> None:
    """Assembles a prompt column in metrics_table from variable columns.

    Args:
        evaluation_run_config: Evaluation Run Configurations.
        prompt_template:  A `PromptTemplate` object or a prompt template string
          with variables that can be assembled from the evaluation dataset. The
          variables can be represented in curly braces `{variable}`, and
          must be included in the dataset columns if specified. The variable
          names cannot contain spaces.

    Returns:
        The assembled prompt template string to send to the model.

    Raises:
        ValueError: If any variable names do not exist in the dataset columns
        or the prompt template is invalid.
    """
    if not prompt_template:
        raise ValueError("Prompt template cannot be an empty string.")

    _LOGGER.info(
        "Assembling prompts from the `prompt_template`. The `prompt` column in"
        " the `EvalResult.metrics_table` has the assembled prompts used for model"
        " response generation."
    )

    try:
        evaluation_run_config.dataset[constants.Dataset.PROMPT_COLUMN] = (
            evaluation_run_config.dataset.apply(
                lambda row: _pre_eval_utils._assemble_prompt(row, prompt_template),
                axis=1,
            )
        )
        if (
            constants.Dataset.PROMPT_COLUMN
            in evaluation_run_config.metric_column_mapping
            and evaluation_run_config.metric_column_mapping[
                constants.Dataset.PROMPT_COLUMN
            ]
            != constants.Dataset.PROMPT_COLUMN
        ):
            _LOGGER.warning(
                "The `prompt` column mapping provided in"
                " `metric_column_mapping` parameter is overwritten by the"
                " assembled `prompt` column because the `prompt_template`"
                " parameter is provided. Please verify that you want to use"
                " the assembled `prompt` column for evaluation."
            )
        evaluation_run_config.metric_column_mapping[constants.Dataset.PROMPT_COLUMN] = (
            constants.Dataset.PROMPT_COLUMN
        )
    except Exception as e:
        raise ValueError(
            f"Failed to assemble prompt template: {e}. Please make sure all"
            " variables in `prompt_template` are present in the evaluation"
            f" dataset columns: `{list(evaluation_run_config.dataset.columns)}`."
        ) from e


def _set_metric_table(
    metric_name: str,
    metric_results: Any,
    metrics_table: "pd.DataFrame",
    metric_result_key: str,
):
    """Parses value from metric results to metrics_table."""
    if (
        metric_result_key == constants.MetricResult.SCORE_KEY
        or metric_result_key == constants.MetricResult.IS_UNSAFE_PROBABILITY_KEY
    ):
        metric_result_items = [
            result.get(metric_result_key) if isinstance(result, dict) else None
            for result in metric_results
        ]
    else:
        metric_result_items = [
            result.get(metric_result_key) if isinstance(result, dict) else "Error"
            for result in metric_results
        ]
    metrics_table[f"{metric_name}/{metric_result_key}"] = metric_result_items


def _parse_metric_results_to_dataframe(
    instance_df: "pd.DataFrame", results: Dict[Union[str, metrics_base._Metric], Any]
) -> Dict[str, Any]:
    """Parses metric results to a pandas dataframe.

    Args:
        instance_df: A dataframe containing per-instance metrics results.
        results: A dictionary containing metric results.

    Returns:
        A dataframe containing per-instance metrics results. Each metric result
        can contain metric score, explanation, and confidence. If
        custom_output_config is specified, raw output or parsed output will
        be included instead.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[evaluation]"'
        )

    metrics_table = pd.DataFrame(dict(zip(instance_df.columns, instance_df.values.T)))
    for metric, metric_results in results.items():
        if isinstance(metric, pointwise_metric.PointwiseMetric):
            if getattr(metric, "custom_output_config", None) and getattr(
                metric.custom_output_config, "return_raw_output", False
            ):
                if getattr(metric, "custom_output_config", None) and getattr(
                    metric.custom_output_config, "parsing_fn", None
                ):
                    for key in metric_results[0].keys():
                        _set_metric_table(
                            metric.metric_name,
                            metric_results,
                            metrics_table,
                            key,
                        )
                else:
                    _set_metric_table(
                        metric.metric_name,
                        metric_results,
                        metrics_table,
                        constants.MetricResult.RAW_OUTPUT_KEY,
                    )
            else:
                _set_metric_table(
                    metric.metric_name,
                    metric_results,
                    metrics_table,
                    constants.MetricResult.EXPLANATION_KEY,
                )
                _set_metric_table(
                    metric.metric_name,
                    metric_results,
                    metrics_table,
                    constants.MetricResult.SCORE_KEY,
                )
        elif isinstance(metric, pairwise_metric.PairwiseMetric):
            if getattr(metric, "custom_output_config", None) and getattr(
                metric.custom_output_config, "return_raw_output", False
            ):
                if getattr(metric, "custom_output_config", None) and getattr(
                    metric.custom_output_config, "parsing_fn", None
                ):
                    for key in metric_results[0].keys():
                        _set_metric_table(
                            metric.metric_name,
                            metric_results,
                            metrics_table,
                            key,
                        )
                else:
                    _set_metric_table(
                        metric.metric_name,
                        metric_results,
                        metrics_table,
                        constants.MetricResult.RAW_OUTPUT_KEY,
                    )
            else:
                _set_metric_table(
                    metric.metric_name,
                    metric_results,
                    metrics_table,
                    constants.MetricResult.EXPLANATION_KEY,
                )
                _set_metric_table(
                    metric.metric_name,
                    metric_results,
                    metrics_table,
                    constants.MetricResult.PAIRWISE_CHOICE_KEY,
                )
        elif (
            str(metric)
            in constants.Metric.AUTOMATIC_METRIC_LIST
            + constants.Metric.TRAJECTORY_METRIC_LIST
        ):
            _set_metric_table(
                str(metric),
                metric_results,
                metrics_table,
                constants.MetricResult.SCORE_KEY,
            )
        elif metric == constants.Metric.RUBRIC_BASED_INSTRUCTION_FOLLOWING:
            _set_metric_table(
                constants.Metric.RUBRIC_BASED_INSTRUCTION_FOLLOWING,
                metric_results,
                metrics_table,
                constants.MetricResult.RUBRIC_LEVEL_INSTRUCTION_FOLLOWING_KEY,
            )
            _set_metric_table(
                constants.Metric.RUBRIC_BASED_INSTRUCTION_FOLLOWING,
                metric_results,
                metrics_table,
                constants.MetricResult.SCORE_KEY,
            )
        else:
            _LOGGER.warning(
                f"Metric name: {str(metric)} is not supported when parsing"
                " metric results."
            )

    return metrics_table


def _compute_metrics(
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
) -> Tuple[Dict[str, Any], "pd.DataFrame"]:
    """Computes the metrics for the dataset.

    Args:
      evaluation_run_config: Evaluation Run Configurations.

    Returns:
      The evaluation results for the input metrics.

    Raises:
      RuntimeError: The number of responses does not match the number of metrics.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[evaluation]"'
        )

    api_metrics, custom_metrics = _separate_custom_metrics(
        evaluation_run_config.metrics
    )
    row_count = len(evaluation_run_config.dataset)
    api_request_count = len(api_metrics) * row_count
    custom_metric_request_count = len(custom_metrics) * row_count
    total_request_count = api_request_count + custom_metric_request_count

    _LOGGER.info(
        f"Computing metrics with a total of {total_request_count} Vertex Gen AI"
        " Evaluation Service API requests."
    )

    instance_list = []
    futures_by_metric = collections.defaultdict(list)
    rate_limiter = utils.RateLimiter(evaluation_run_config.evaluation_service_qps)
    results_dict = collections.defaultdict(
        lambda: [None] * len(evaluation_run_config.dataset)
    )
    error_list = []
    with tqdm(total=total_request_count) as pbar:
        with futures.ThreadPoolExecutor(max_workers=constants.MAX_WORKERS) as executor:
            for idx, row in evaluation_run_config.dataset.iterrows():
                row_dict = _compute_custom_metrics(
                    row.to_dict(), custom_metrics, pbar, executor
                )
                instance_list.append(row_dict)
                for metric in api_metrics:
                    try:
                        future = executor.submit(
                            _instance_evaluation.evaluate_instances,
                            client=evaluation_run_config.client,
                            request=_instance_evaluation.build_request(
                                metric=metric,
                                row_dict=row_dict,
                                evaluation_run_config=evaluation_run_config,
                            ),
                            rate_limiter=rate_limiter,
                            retry_timeout=evaluation_run_config.retry_timeout,
                        )
                        future.add_done_callback(lambda _: pbar.update(1))
                        futures_by_metric[metric].append((future, idx))
                    except ValueError as e:
                        results_dict[metric][idx] = "Error"
                        error_list.append((metric, idx, f"Error: {e}"))

        # Retrieve results from all futures and handle errors.
        for metric, futures_list in futures_by_metric.items():
            for future, index in futures_list:
                try:
                    response = future.result()
                    results_dict[metric][index] = response
                except Exception as e:
                    results_dict[metric][index] = "Error"
                    error_list.append((metric, index, f"Error: {e}"))

    for metric, responses in results_dict.items():
        results_dict[metric] = [
            _instance_evaluation.handle_response(response, metric)
            for response in responses
        ]
    if error_list:
        _LOGGER.warning(
            f"{len(error_list)} errors encountered during evaluation. Continue to"
            " compute summary metrics for the rest of the dataset."
        )
        for metric_name, index, error in error_list:
            _LOGGER.warning(
                f"Error encountered for metric {metric_name} at dataset index"
                f" {index}: {error}"
            )
    else:
        _LOGGER.info(
            f"All {total_request_count} metric requests are successfully computed."
        )

    instance_df = pd.DataFrame.from_dict(instance_list)
    metrics_table = _parse_metric_results_to_dataframe(instance_df, results_dict)

    # Aggregate the summary metrics.
    summary_metrics = _aggregate_summary_metrics(evaluation_run_config, metrics_table)

    return evaluation_base.EvalResult(
        summary_metrics=summary_metrics, metrics_table=metrics_table
    )


def _get_baseline_model(evaluation_run_config: evaluation_base.EvaluationRunConfig):
    """Gets the baseline model from the pairwise metrics."""
    pairwise_metric_instances = [
        metric
        for metric in evaluation_run_config.metrics
        if isinstance(metric, pairwise_metric.PairwiseMetric)
    ]
    baseline_models = {
        instance.metric_name: instance.baseline_model
        for instance in pairwise_metric_instances
    }
    if len(set(baseline_models.values())) > 1:
        raise ValueError(
            "Not all `PairwiseMetric` instances have the same `baseline_model`. "
            f"Here are the detected baseline models: `{baseline_models}`. "
            "Please separate pairwise metrics with different baseline models "
            "in different `EvalTask` or use the same baseline model for "
            "all pairwise metrics."
        )
    return pairwise_metric_instances[0].baseline_model


def _convert_metric_prompt_template_example(metrics):
    """Converts string metric names to generic model-based metric instances."""
    updated_metrics = []
    for metric in metrics:
        if metric in constants.Metric.POINTWISE_METRIC_PROMPT_TEMPLATE_EXAMPLE_LIST:
            template = metric_prompt_template_examples.MetricPromptTemplateExamples.get_prompt_template(
                metric
            )
            metric = pointwise_metric.PointwiseMetric(
                metric=metric, metric_prompt_template=template
            )
        elif metric in constants.Metric.PAIRWISE_METRIC_PROMPT_TEMPLATE_EXAMPLE_LIST:
            template = metric_prompt_template_examples.MetricPromptTemplateExamples.get_prompt_template(
                metric
            )
            metric = pairwise_metric.PairwiseMetric(
                metric=metric, metric_prompt_template=template
            )
            _LOGGER.info(
                f"Pairwise metric `{metric.metric_name}` loaded from"
                " `MetricPromptTemplateExamples` does not have `baseline_model`"
                " specified and only supports Bring-Your-Own-Response(BYOR)"
                " evaluation. If you would like to run inference on the baseline model,"
                " please instantiate a `PairwiseMetric` and provide the"
                " `baseline_model` parameter."
            )
        updated_metrics.append(metric)
    return updated_metrics


def _get_rubric_metric_with_idx(
    metrics: List[Union[str, metrics_base._Metric]],
) -> Optional[Tuple[rubric_based_metric.RubricBasedMetric, int]]:
    """Gets the rubric metric with its index in the metrics list."""
    for i, metric in enumerate(metrics):
        if isinstance(metric, rubric_based_metric.RubricBasedMetric):
            return metric, i
    return None


def _safe_deepcopy_rubric_generation_config(
    generation_config: metrics_base.RubricGenerationConfig,
) -> metrics_base.RubricGenerationConfig:
    """Safely deepcopies the rubric generation config of RubricBasedMetric."""
    copied_generation_config = None

    if generation_config:
        generation_config_model = generation_config.model
        # Cannot deepcopy instance of GenerativeModel due to its cygrpc.Channel
        generation_config.model = None
        copied_generation_config = copy.deepcopy(generation_config)

        if generation_config_model:
            copied_generation_config.model = generation_config_model

    return copied_generation_config


def evaluate(
    dataset: "pd.DataFrame",
    metrics: List[Union[str, metrics_base._Metric]],
    *,
    model: Optional[_ModelType] = None,
    runnable: Optional[_RunnableType] = None,
    prompt_template: Optional[Union[str, prompt_template_base.PromptTemplate]] = None,
    metric_column_mapping: Dict[str, str],
    evaluation_service_qps: Optional[float] = None,
    retry_timeout: float = 600.0,
    autorater_config: Optional[evaluation_base.AutoraterConfig] = None,
) -> evaluation_base.EvalResult:
    """Runs the evaluation for metrics.

    Args:
      dataset: The dataset to evaluate.
      metrics: The list of metric names, or Metric instances to
        evaluate. Prompt template is required for PairwiseMetric.
      model: The GenerativeModel instance or a custom model function to generate
        responses to evaluate. If not provided, the evaluation is computed with
        the `response` column in the `dataset`.
      runnable: The runnable to generate responses to evaluate. If not provided, the
        evaluation is computed with the `response` and/or `predicted_trajectory`
        column in the `dataset`.
      prompt_template: A `PromptTemplate` or a prompt template string compatible
        with `PromptTemplate` class with variables that can be formatted with
        dataset columns to create assembled prompts. The variables can be
        represented in curly braces `{variable_name}`, and must be included in the
        dataset columns if specified. The variable names cannot contain spaces.
      metric_column_mapping: An optional dictionary column mapping that
        overrides the metric prompt template input variable names with
        mapped the evaluation dataset column names, used during evaluation.
        For example, if the input_variables of the metric prompt template
        are ["context", "reference"], the metric_column_mapping can be
              {
                  "context": "news_context",
                  "reference": "ground_truth",
                  "response": "model_1_response"
              }
            if the dataset has columns "news_context", "ground_truth" and
            "model_1_response".
      evaluation_service_qps: The custom QPS limit for the evaluation service.
      retry_timeout: How long to keep retrying the evaluation requests for the
        whole evaluation dataset, in seconds.
      autorater_config: The autorater config for model based evaluation.

    Returns:
      EvalResult with summary metrics and a metrics table for per-instance
      metrics.

    Raises:
      ValueError: If the metrics list is empty, or the prompt template is not
        provided for PairwiseMetric, or multiple baseline models are specified for
        PairwiseMetric instances, or both model and dataset model response column
        are present, or the sampling count is not in the range [1, 32], or flipping
        is enabled for pairwise metrics but the required columns are not found in
        the dataset.
    """
    _validate_metrics(metrics)
    metrics = _convert_metric_prompt_template_example(metrics)
    copied_metrics = []
    for metric in metrics:
        if isinstance(metric, pairwise_metric.PairwiseMetric):
            copied_metrics.append(
                pairwise_metric.PairwiseMetric(
                    metric=metric.metric_name,
                    metric_prompt_template=metric.metric_prompt_template,
                    baseline_model=metric.baseline_model,
                    system_instruction=metric.system_instruction,
                    custom_output_config=metric.custom_output_config,
                    autorater_config=metric.autorater_config,
                )
            )
        elif isinstance(metric, rubric_based_metric.RubricBasedMetric):
            copied_metrics.append(
                rubric_based_metric.RubricBasedMetric(
                    generation_config=_safe_deepcopy_rubric_generation_config(
                        metric.generation_config,
                    ),
                    critique_metric=copy.deepcopy(metric.critique_metric),
                )
            )
        else:
            copied_metrics.append(copy.deepcopy(metric))

    evaluation_run_config = evaluation_base.EvaluationRunConfig(
        dataset=dataset.copy(deep=True),
        metrics=copied_metrics,
        metric_column_mapping=copy.deepcopy(metric_column_mapping),
        client=utils.create_evaluation_service_client(),
        evaluation_service_qps=(
            evaluation_service_qps
            if evaluation_service_qps
            else constants.QuotaLimit.EVAL_SERVICE_QPS
        ),
        retry_timeout=retry_timeout,
        autorater_config=autorater_config,
    )

    if prompt_template:
        _assemble_prompt_for_dataset(evaluation_run_config, prompt_template)

    if model and runnable:
        raise ValueError(
            "Only one of model and runnable can be provided in a single evaluation."
        )

    if model:
        _run_model_inference(
            model=model,
            evaluation_run_config=evaluation_run_config,
            response_column_name=constants.Dataset.MODEL_RESPONSE_COLUMN,
        )

    if runnable:
        _run_runnable_inference(
            runnable=runnable,
            evaluation_run_config=evaluation_run_config,
            response_column_name=constants.Dataset.MODEL_RESPONSE_COLUMN,
            predicted_trajectory_column_name=constants.Dataset.PREDICTED_TRAJECTORY_COLUMN,
        )
    _validate_dataset(evaluation_run_config)

    rubric_metric_with_idx = _get_rubric_metric_with_idx(evaluation_run_config.metrics)

    if rubric_metric_with_idx:
        rubric_metric, idx = rubric_metric_with_idx
        eval_dataset_with_rubrics = rubric_metric.generate_rubrics(
            evaluation_run_config.dataset
        )
        for column in eval_dataset_with_rubrics.columns:
            if column not in evaluation_run_config.metric_column_mapping:
                evaluation_run_config.metric_column_mapping[column] = column
        evaluation_run_config.dataset = eval_dataset_with_rubrics
        if not rubric_metric.critique_metric.custom_output_config:
            rubric_metric.critique_metric.custom_output_config = (
                custom_output_config.CustomOutputConfig()
            )
        rubric_metric.critique_metric.custom_output_config.return_raw_output = True
        evaluation_run_config.metrics[idx] = rubric_metric.critique_metric

    pairwise_metric_exists = any(
        isinstance(metric, pairwise_metric.PairwiseMetric)
        for metric in evaluation_run_config.metrics
    )
    if pairwise_metric_exists:
        baseline_model = _get_baseline_model(evaluation_run_config)
        if baseline_model:
            _run_model_inference(
                model=baseline_model,
                evaluation_run_config=evaluation_run_config,
                response_column_name=constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN,
            )

    _validate_autorater_config(evaluation_run_config, pairwise_metric_exists)
    _validate_metric_column_map(evaluation_run_config)
    t1 = time.perf_counter()
    evaluation_result = _compute_metrics(evaluation_run_config)
    t2 = time.perf_counter()
    _LOGGER.info(f"Evaluation Took:{t2 - t1} seconds")

    return evaluation_result
