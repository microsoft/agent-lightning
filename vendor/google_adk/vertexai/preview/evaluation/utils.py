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
"""Utility functions for evaluation."""

import functools
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from google.cloud import bigquery
from google.cloud import storage
from google.cloud.aiplatform import base
from google.cloud.aiplatform import compat
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.utils import _ipython_utils
from google.cloud.aiplatform_v1beta1.services import (
    evaluation_service as gapic_evaluation_services,
)
from vertexai.evaluation import _base as eval_base
from vertexai.evaluation.metrics import _base as metrics_base
from vertexai.evaluation.metrics import (
    metric_prompt_template as metric_prompt_template_base,
)
from google.cloud.aiplatform.utils.gcs_utils import blob_from_uri


if TYPE_CHECKING:
    import pandas as pd


_BQ_PREFIX = "bq://"
_GCS_PREFIX = "gs://"
_LOGGER = base.Logger(__name__)

_QUESTION_REGEX = re.compile(r"Question:(.*?)Verdict:", re.DOTALL)
_VERDICT_REGEX = re.compile("Verdict:(.*)")
_QUESTION_BLOCK_REGEX = re.compile("<question>(.*?)</question>", re.DOTALL)
_RESPONSE_A_REGEX = re.compile(
    r"\[\[Response A Answers:\]\](.*?)\[\[Rubric Score:", re.DOTALL
)
_RESPONSE_B_REGEX = re.compile(
    r"\[\[Response B Answers:\]\](.*?)\[\[Rubric Score:", re.DOTALL
)
_SXS_RATING_REGEX = re.compile(r"\[\[(SxSRating:[AB<>=]+)\]\]", re.DOTALL)

RATING_TO_VERDICT = {
    "B>>A": "Candidate response is better than the baseline response.",
    "A<<B": "Candiate response is better than the baseline response.",
    "B>A": "Candidate response is slightly better than the baseline response.",
    "A<B": "Candidate response is slightly better than the baseline response.",
    "A=B": "Both responses are equally good.",
    "B=A": "Both responses are equally good.",
    "A>B": "Baseline response is slightly better than the candidate response.",
    "B<A": "Baseline response is slightly better than the candidate response.",
    "B<<A": "Baseline response is better than the candidate response.",
    "A>>B": "Baseline response is better than the candidate response.",
}
_RATING_TO_SCORE = {
    "B>>A": 1,
    "A<<B": 1,
    "B>A": 0.5,
    "A<B": 0.5,
    "A=B": 0,
    "B=A": 0,
    "A>B": -0.5,
    "B<A": -0.5,
    "B<<A": -1,
    "A>>B": -1,
}


class _EvaluationServiceClientWithOverride(utils.ClientWithOverride):
    _is_temporary = False
    _default_version = compat.V1
    _version_map = (
        (
            compat.V1,
            gapic_evaluation_services.EvaluationServiceClient,
        ),
    )


class RateLimiter:
    """Helper class for rate-limiting requests to Vertex AI to improve QoS.

    Attributes:
        seconds_per_event: The time interval (in seconds) between events to
          maintain the desired rate.
        last: The timestamp of the last event.
        _lock: A lock to ensure thread safety.
    """

    def __init__(self, rate: Optional[float] = None):
        """Initializes the rate limiter.

        A simple rate limiter for controlling the frequency of API calls. This class
        implements a token bucket algorithm to limit the rate at which events
        can occur. It's designed for cases where the batch size (number of events
        per call) is always 1 for traffic shaping and rate limiting.

        Args:
            rate: The number of queries allowed per second.

        Raises:
            ValueError: If the rate is not positive.
        """
        if not rate or rate <= 0:
            raise ValueError("Rate must be a positive number")
        self.seconds_per_event = 1.0 / rate
        self.last = time.time() - self.seconds_per_event
        self._lock = threading.Lock()

    def _admit(self) -> float:
        """Checks if an event can be admitted or calculates the remaining delay."""
        now = time.time()
        time_since_last = now - self.last
        if time_since_last >= self.seconds_per_event:
            self.last = now
            return 0
        else:
            return self.seconds_per_event - time_since_last

    def sleep_and_advance(self):
        """Blocks the current thread until the next event can be admitted."""
        with self._lock:
            delay = self._admit()
            if delay > 0:
                time.sleep(delay)
                self.last = time.time()


def rate_limit(rate: Optional[float] = None) -> Callable[[Any], Any]:
    """Decorator version of rate limiter."""

    def _rate_limit(method):
        limiter = RateLimiter(rate)

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            limiter.sleep_and_advance()
            return method(*args, **kwargs)

        return wrapper

    return _rate_limit


def create_evaluation_service_client(
    api_base_path_override: Optional[str] = None,
) -> _EvaluationServiceClientWithOverride:
    """Creates a client for the evaluation service.

    Args:
      api_base_path_override: Optional. Override default api base path.

    Returns:
      Instantiated Vertex AI EvaluationServiceClient with optional
      overrides.
    """
    return initializer.global_config.create_client(
        client_class=_EvaluationServiceClientWithOverride,
        location_override=initializer.global_config.location,
        api_base_path_override=api_base_path_override,
    )


def load_dataset(
    source: Union[str, "pd.DataFrame", Dict[str, Any]],
) -> "pd.DataFrame":
    """Loads dataset from various sources into a DataFrame.

    Args:
        source: The dataset source. Supports the following dataset formats:
        * pandas.DataFrame: Used directly for evaluation.
        * Dict: Converted to a pandas DataFrame before evaluation.
        * str: Interpreted as a file path or URI. Supported formats include:
            * Local JSONL or CSV files:  Loaded from the local filesystem.
            * GCS JSONL or CSV files: Loaded from Google Cloud Storage (e.g.,
            'gs://bucket/data.csv').
            * BigQuery table URI: Loaded from Google Cloud
            BigQuery (e.g., 'bq://project-id.dataset.table_name').

    Returns:
        The dataset in pandas DataFrame format.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[evaluation]"'
        )
    if "google.colab" in sys.modules:
        from google.colab import sheets

        if isinstance(source, sheets.InteractiveSheet):
            return source.as_df().copy()

    if isinstance(source, pd.DataFrame):
        return source.copy()
    elif isinstance(source, dict):
        return pd.DataFrame(source)
    elif isinstance(source, str):
        if source.startswith(_BQ_PREFIX):
            return _load_bigquery(source[len(_BQ_PREFIX) :])

        _, extension = os.path.splitext(source)
        file_type = extension.lower()[1:]

        if file_type == "jsonl":
            return _load_jsonl(source)
        elif file_type == "csv":
            return _load_csv(source)
        else:
            raise ValueError(
                f"Unsupported file type: {file_type} from {source}. Please"
                " provide a valid GCS path with `jsonl` or `csv` suffix or a valid"
                " BigQuery table URI."
            )
    else:
        raise TypeError(
            "Unsupported dataset type. Must be a `pd.DataFrame`, Python dictionary,"
            " valid GCS path with  `jsonl` or `csv` suffix or a valid BigQuery"
            " table URI."
        )


def _load_jsonl(filepath: str) -> "pd.DataFrame":
    """Loads data from a JSONL file into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[evaluation]"'
        )
    if filepath.startswith(_GCS_PREFIX):
        file_contents = _read_gcs_file_contents(filepath)
        return pd.read_json(file_contents, lines=True)
    else:
        with open(filepath, "r") as f:
            return pd.read_json(f, lines=True)


def _load_csv(filepath: str) -> "pd.DataFrame":
    """Loads data from a CSV file into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[evaluation]"'
        )
    if filepath.startswith(_GCS_PREFIX):
        file_contents = _read_gcs_file_contents(filepath)
        return pd.read_csv(io.StringIO(file_contents), encoding="utf-8")
    else:
        return pd.read_csv(filepath, encoding="utf-8")


def _load_bigquery(table_id: str) -> "pd.DataFrame":
    """Loads data from a BigQuery table into a DataFrame."""

    bigquery_client = bigquery.Client(project=initializer.global_config.project)
    table = bigquery_client.get_table(table_id)
    return bigquery_client.list_rows(table).to_dataframe()


def _read_gcs_file_contents(filepath: str) -> str:
    """Reads the contents of a file from Google Cloud Storage.

    Args:
        filepath: The GCS file path (e.g., 'gs://bucket_name/file.csv')

    Returns:
        str: The contents of the file.
    """

    storage_client = storage.Client(
        project=initializer.global_config.project,
        credentials=initializer.global_config.credentials,
    )
    bucket_name, blob_path = filepath[len(_GCS_PREFIX) :].split("/", 1)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_string().decode("utf-8")


def _upload_file_to_gcs(upload_gcs_path: str, filename: str) -> None:
    storage_client = storage.Client(
        project=initializer.global_config.project,
        credentials=initializer.global_config.credentials,
    )
    blob_from_uri(uri=upload_gcs_path, client=storage_client).upload_from_filename(
        filename
    )


def _upload_string_to_gcs(upload_gcs_path: str, contents: str) -> None:
    """Uploads the provided string to a GCS bucket."""
    storage_client = storage.Client(
        project=initializer.global_config.project,
        credentials=initializer.global_config.credentials,
    )
    blob_from_uri(uri=upload_gcs_path, client=storage_client).upload_from_string(
        contents
    )


def _upload_pandas_df_to_gcs(
    df: "pd.DataFrame", upload_gcs_path: str, file_type: str
) -> None:
    """Uploads the provided Pandas DataFrame to a GCS bucket.

    Args:
        df: The Pandas DataFrame to upload.
        upload_gcs_path: The GCS path to upload the data file.
        file_type: The file type of the data file.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        if file_type == "csv":
            local_dataset_path = os.path.join(temp_dir, "metrics_table.csv")
            df.to_csv(path_or_buf=local_dataset_path)
        elif file_type == "jsonl":
            local_dataset_path = os.path.join(temp_dir, "metrics_table.jsonl")
            df.to_json(path_or_buf=local_dataset_path, orient="records", lines=True)
        else:
            raise ValueError(
                f"Unsupported file type: {file_type} from {upload_gcs_path}."
                " Please provide a valid GCS path with `jsonl` or `csv` suffix."
            )

        storage_client = storage.Client(
            project=initializer.global_config.project,
            credentials=initializer.global_config.credentials,
        )
        blob_from_uri(uri=upload_gcs_path, client=storage_client).upload_from_filename(
            filename=local_dataset_path
        )


def _upload_evaluation_summary_to_gcs(
    summary_metrics: Dict[str, float],
    upload_gcs_path: str,
    candidate_model_name: Optional[str] = None,
    baseline_model_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    metrics: Optional[List[Union[str, metrics_base._Metric]]] = None,
) -> None:
    """Uploads the evaluation summary to a GCS bucket."""
    summary = {
        "summary_metrics": summary_metrics,
    }
    if candidate_model_name:
        summary["candidate_model_name"] = candidate_model_name
    if baseline_model_name:
        summary["baseline_model_name"] = baseline_model_name
    if dataset_uri:
        summary["dataset_uri"] = dataset_uri

    if metrics:
        metric_descriptions = {}
        for metric in metrics:
            if isinstance(metric, metrics_base._ModelBasedMetric) and isinstance(
                metric._raw_metric_prompt_template,
                metric_prompt_template_base._MetricPromptTemplate,
            ):
                metric_descriptions[metric.metric_name] = {
                    "criteria": metric._raw_metric_prompt_template._criteria,
                    "rating_rubric": metric._raw_metric_prompt_template._rating_rubric,
                }
        summary["metric_descriptions"] = metric_descriptions

    with tempfile.TemporaryDirectory() as temp_dir:
        local_summary_path = os.path.join(temp_dir, "summary_metrics.json")
        json.dump(summary, open(local_summary_path, "w"))
        _upload_file_to_gcs(upload_gcs_path, local_summary_path)


def upload_evaluation_results(
    eval_result: eval_base.EvalResult,
    destination_uri_prefix: str,
    file_name: Optional[str] = None,
    candidate_model_name: Optional[str] = None,
    baseline_model_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    metrics: Optional[List[Union[str, metrics_base._Metric]]] = None,
) -> None:
    """Uploads eval results to GCS destination.

    Args:
        eval_result: Eval results to upload.
        destination_uri_prefix: GCS folder to store the data.
        file_name: Optional. File name to store the metrics table.
        candidate_model_name: Optional. Candidate model name.
        baseline_model_name: Optional. Baseline model name.
        dataset_uri: Optional. URI pointing to the dataset.
        metrics: Optional. List of metrics used for evaluation.
    """
    if not destination_uri_prefix:
        _ipython_utils.display_gen_ai_evaluation_results_button()
        return
    if eval_result.metrics_table is None:
        return
    if destination_uri_prefix.startswith(_GCS_PREFIX):
        if file_name:
            base_name, extension = os.path.splitext(file_name)
            file_type = extension.lower()[1:]
            output_folder = destination_uri_prefix + "/" + base_name
            metrics_table_path = output_folder + "/" + file_name
            _upload_pandas_df_to_gcs(
                eval_result.metrics_table, metrics_table_path, file_type
            )
            _upload_evaluation_summary_to_gcs(
                eval_result.summary_metrics,
                output_folder + "/summary_metrics.json",
                candidate_model_name,
                baseline_model_name,
                dataset_uri,
                metrics,
            )
            _ipython_utils.display_gen_ai_evaluation_results_button(
                metrics_table_path.split(_GCS_PREFIX)[1]
            )
    else:
        raise ValueError(
            f"Unsupported destination URI: {destination_uri_prefix}."
            f" Please provide a valid GCS bucket URI prefix starting with"
            f" {_GCS_PREFIX}."
        )


def initialize_metric_column_mapping(
    metric_column_mapping: Optional[Dict[str, str]], dataset: "pd.DataFrame"
):
    """Initializes metric column mapping with dataset columns."""
    initialized_metric_column_mapping = {}
    for column in dataset.columns:
        initialized_metric_column_mapping[column] = column
    if metric_column_mapping:
        for key, value in metric_column_mapping.items():
            if key in initialized_metric_column_mapping:
                _LOGGER.warning(
                    f"Cannot override `{key}` column with `{key}:{value}` mapping"
                    f" because `{key}` column is present in the evaluation"
                    " dataset. `metric_column_mapping` cannot override keys"
                    " that are already in evaluation dataset columns."
                )
            else:
                initialized_metric_column_mapping[key] = value
    return initialized_metric_column_mapping


def parse_intermediate_steps(intermediate_steps: List[Dict[str, Any]]):
    """Parses intermediate steps from the response to create trajectory."""
    trajectory = []
    try:
        for step in intermediate_steps:
            step_input, _ = step[0], step[1]
            tool_name = step_input["kwargs"]["tool"]
            tool_input = step_input["kwargs"]["tool_input"]
            trajectory.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                }
            )
    except Exception as e:  # pylint: disable=broad-exception-caught
        _LOGGER.error(
            f"Failed to parse intermediate steps: {e}. The runnable you are using"
            " is likely not compatible with the evaluation service. Please ensure"
            " that the runnable you are using is compatible with the evaluation"
            " service, if not, consider building a custom runnable function."
        )
    return trajectory


def parse_rubrics(rubric_generation_response: str) -> Dict[str, Any]:
    """Parses the rubric generation responses."""
    try:
        _, response = rubric_generation_response.split("```json")
    except ValueError:
        _LOGGER.warning(
            "Failed to parse rubric generation response. Does not contain ```json"
        )
        return {"questions": ""}
    try:
        result = json.loads(response.strip("\n` "))
    except json.JSONDecodeError:
        _LOGGER.warning(
            "Failed to parse rubric generation response. Does not contain valid"
            " JSON."
        )
        return {"questions": ""}
    return result


def parse_pairwise_rubric_verdict_pairs(prediction: str, regex: Pattern[str]) -> str:
    """Parses the pairwise rubric critique responses."""
    response = "Unable to parse rubric verdict pairs from response."
    response_matches = regex.findall(prediction)
    if response_matches:
        response_pairs = parse_question_blocks(response_matches[0])
        response = "\n".join(f"{q}: {v}" for q, v in response_pairs)
    return response


def parse_pairwise_rubric_result(
    predictions: List[str],
) -> Dict[str, Any]:
    """Parses the pairwise rubric critique responses."""
    prediction = predictions[0]  # currently only supports one sample
    rating_str = "Unable to parse verdict."

    response_a = parse_pairwise_rubric_verdict_pairs(prediction, _RESPONSE_A_REGEX)
    response_b = parse_pairwise_rubric_verdict_pairs(prediction, _RESPONSE_B_REGEX)

    sxs_rating_matches = _SXS_RATING_REGEX.findall(prediction.replace(" ", ""))

    if sxs_rating_matches:
        rating_str = sxs_rating_matches[0].strip("[]")
        rating_str = rating_str[rating_str.find(":") + 1 :]

    return {
        "pairwise_choice": (
            RATING_TO_VERDICT[rating_str]
            if rating_str in RATING_TO_VERDICT
            else rating_str
        ),
        "score": (
            _RATING_TO_SCORE[rating_str] if rating_str in _RATING_TO_SCORE else None
        ),
        "baseline_rubric_verdict_pairs": response_a,
        "candidate_rubric_verdict_pairs": response_b,
        "raw_outputs": predictions,
    }


def parse_verdict(txt: str):
    """Parses the verdict from the rubric critique response."""
    if not isinstance(txt, str) or not txt:
        return None
    try:
        if verdict := _VERDICT_REGEX.findall(txt):
            verdict = verdict[0]
            if "yes" in verdict.lower():
                return True
            elif "no" in verdict.lower():
                return False
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def parse_question(txt: str):
    """Parses the question from the rubric critique response."""
    if not isinstance(txt, str) or not txt:
        return None
    try:
        txt = txt.split("Verdict:")[0]
        if "Question:" in txt:
            return txt.split("Question:")[-1].strip()

        if not (question := _QUESTION_REGEX.findall(txt)):
            return txt.strip().split("\n")[0].removeprefix("STEP 1:").strip()
        return question[0].strip()
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def parse_question_blocks(txt: str) -> List[Tuple[str, bool]]:
    """Parses the question blocks from the rubric critique response."""
    if not txt.startswith("<question>\n"):
        txt = "<question>\n" + txt
    responses = []
    question_blocks = _QUESTION_BLOCK_REGEX.findall(txt)
    if not question_blocks:
        question_blocks = [txt]
    for block in question_blocks:
        q = parse_question(block)
        v = parse_verdict(block)
        if q is not None and v is not None:
            responses.append((q, v))
    return responses


def parse_pointwise_rubric_result(results: List[str]) -> Dict[str, Any]:
    """Parses the pointwise rubric critique responses."""
    self_consistency_results = {}
    for sample_result in results:
        rubric_verdict_pairs = parse_question_blocks(sample_result)
        for rubric, verdict in rubric_verdict_pairs:
            if rubric not in self_consistency_results:
                self_consistency_results[rubric] = 0

            self_consistency_results[rubric] += 1 if verdict else -1

    rubric_results = {}
    for rubric, verdict_counts in self_consistency_results.items():
        rubric_results[rubric] = verdict_counts > 0

    rubric_results_str = "\n".join(f"{q}: {v}" for q, v in rubric_results.items())
    row_results = {
        "score": (
            sum(rubric_results.values()) / len(rubric_results) if rubric_results else 0
        )
    }
    row_results["rubric_verdict_pairs"] = rubric_results_str
    row_results["raw_outputs"] = results
    return row_results
