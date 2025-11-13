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
"""Utility functions for evals."""

import abc
import json
import logging
import os
import re
from typing import Any, Optional, Union

from google.cloud import bigquery
from google.genai._api_client import BaseApiClient
from google.genai._common import get_value_by_path as getv
from google.genai._common import set_value_by_path as setv
import pandas as pd
import yaml

from . import _evals_constant
from . import _transformers
from . import types
from . import _gcs_utils


logger = logging.getLogger(__name__)


GCS_PREFIX = "gs://"
BQ_PREFIX = "bq://"


class BigQueryUtils:
    """Handles BigQuery operations."""

    def __init__(self, api_client: BaseApiClient):
        self.api_client = api_client
        self.bigquery_client = bigquery.Client(
            project=self.api_client.project,
            credentials=self.api_client._credentials,
        )

    def load_bigquery_to_dataframe(self, table_uri: str) -> "pd.DataFrame":
        """Loads data from a BigQuery table into a DataFrame."""
        table = self.bigquery_client.get_table(table_uri)
        return self.bigquery_client.list_rows(table).to_dataframe()

    def upload_dataframe_to_bigquery(
        self, df: "pd.DataFrame", bq_table_uri: str
    ) -> None:
        """Uploads a Pandas DataFrame to a BigQuery table."""
        job = self.bigquery_client.load_table_from_dataframe(df, bq_table_uri)
        job.result()
        logger.info(
            f"DataFrame successfully uploaded to BigQuery table: {bq_table_uri}"
        )


class EvalDatasetLoader:
    """A loader for datasets from various sources, using a shared client."""

    def __init__(self, api_client: BaseApiClient):
        self.api_client = api_client
        self.gcs_utils = _gcs_utils.GcsUtils(self.api_client)
        self.bigquery_utils = BigQueryUtils(self.api_client)

    def _load_file(
        self, filepath: str, file_type: str
    ) -> Union[list[dict[str, Any]], Any]:
        """Loads data from a file into a list of dictionaries."""
        if filepath.startswith(GCS_PREFIX):
            df = self.gcs_utils.read_gcs_file_to_dataframe(filepath, file_type)
            return df.to_dict(orient="records")
        else:
            if file_type == "jsonl":
                df = pd.read_json(filepath, lines=True)
                return df.to_dict(orient="records")
            elif file_type == "csv":
                df = pd.read_csv(filepath, encoding="utf-8")
                return df.to_dict(orient="records")
            else:
                raise ValueError(
                    f"Unsupported file type: '{file_type}'. Please provide 'jsonl' or"
                    " 'csv'."
                )

    def load(
        self, source: Union[str, "pd.DataFrame"]
    ) -> Union[list[dict[str, Any]], Any]:
        """Loads dataset from various sources into a list of dictionaries."""
        if isinstance(source, pd.DataFrame):
            return source.to_dict(orient="records")
        elif isinstance(source, str):
            if source.startswith(BQ_PREFIX):
                df = self.bigquery_utils.load_bigquery_to_dataframe(
                    source[len(BQ_PREFIX) :]
                )
                return df.to_dict(orient="records")

            _, extension = os.path.splitext(source)
            file_type = extension.lower()[1:]

            if file_type == "jsonl":
                return self._load_file(source, "jsonl")
            elif file_type == "csv":
                return self._load_file(source, "csv")
            else:
                raise TypeError(
                    f"Unsupported file type: {file_type} from {source}. Please"
                    " provide a valid GCS path with `jsonl` or `csv` suffix, "
                    "a local file path, or a valid BigQuery table URI."
                )
        else:
            raise TypeError(
                "Unsupported dataset type. Must be a `pd.DataFrame`, Python"
                " a valid GCS path with `jsonl` or `csv` suffix, a local"
                " file path, or a valid BigQuery table URI."
            )


class LazyLoadedPrebuiltMetric:
    """A proxy object representing a prebuilt metric to be loaded on demand.

    This can resolve to either an API Predefined Metric or an LLM Metric
    loaded from GCS.
    """

    _cache: dict[str, types.Metric] = {}
    _base_gcs_path = (
        "gs://vertex-ai-generative-ai-eval-sdk-resources/metrics/{metric_name}/"
    )

    def __init__(self, name: str, version: Optional[str] = None, **kwargs):
        self.name = name.upper()
        self.version = version
        self.metric_kwargs = kwargs
        self._resolved_metric: Optional[types.Metric] = None

    def _get_api_metric_spec_name(self) -> Optional[str]:
        """Constructs the metric_spec_name for API Predefined Metrics."""
        base_name = self.name.lower()
        if self.version:
            # Explicit version provided
            version = self.version.lower()
            potential_name = f"{base_name}_{version}"
            return (
                potential_name
                if potential_name in _evals_constant.SUPPORTED_PREDEFINED_METRICS
                else None
            )
        else:
            # Default versioning: Try _v1, then base name
            v1_name = f"{base_name}_v1"
            if v1_name in _evals_constant.SUPPORTED_PREDEFINED_METRICS:
                return v1_name
            if base_name in _evals_constant.SUPPORTED_PREDEFINED_METRICS:
                return base_name
        return None

    def _resolve_api_predefined(self) -> Optional[types.Metric]:
        """Attempts to resolve as an API Predefined Metric."""
        metric_spec_name = self._get_api_metric_spec_name()
        if metric_spec_name:
            logger.info(
                "Resolving '%s' as API Predefined Metric with spec name: %s",
                self.name,
                metric_spec_name,
            )
            return types.Metric(name=metric_spec_name, **self.metric_kwargs)
        return None

    def _get_latest_version_uri(self, api_client: Any, metric_gcs_dir: str) -> str:
        """Lists files in GCS directory and determines the latest version URI."""
        gcs_utils = _gcs_utils.GcsUtils(api_client)
        bucket_name, prefix = gcs_utils.parse_gcs_path(metric_gcs_dir)

        blobs = gcs_utils.storage_client.list_blobs(bucket_name, prefix=prefix)

        version_files: list[dict[str, Union[list[int], str]]] = (
            []
        )  # {'version_parts': [1,0,0], 'filename': 'v1.0.0.yaml'}

        version_pattern = re.compile(
            r"v(\d+)(?:\.(\d+))?(?:\.(\d+))?\.(yaml|yml|json)$", re.IGNORECASE
        )

        for blob in blobs:
            match = version_pattern.match(os.path.basename(blob.name))
            if match:
                major = int(match.group(1))
                minor = int(match.group(2)) if match.group(2) else 0
                patch = int(match.group(3)) if match.group(3) else 0
                version_files.append(
                    {
                        "version_parts": [major, minor, patch],
                        "filename": os.path.basename(blob.name),
                    }
                )

        if not version_files:
            raise IOError(f"No versioned metric files found in {metric_gcs_dir}")

        version_files.sort(key=lambda x: x["version_parts"], reverse=True)

        latest_filename = version_files[0]["filename"]
        return os.path.join(metric_gcs_dir, latest_filename)

    def _fetch_and_parse(self, api_client: Any) -> types.LLMMetric:
        """Fetches and parses the metric definition from GCS."""
        metric_gcs_dir = self._base_gcs_path.format(metric_name=self.name.lower())
        uri: str
        if self.version == "latest" or self.version is None:
            uri = self._get_latest_version_uri(api_client, metric_gcs_dir)
            resolved_version_match = re.match(
                r"(v\d+(?:\.\d+)*)\.(?:yaml|yml|json)",
                os.path.basename(uri),
                re.IGNORECASE,
            )
            if resolved_version_match:
                self.version = resolved_version_match.group(1)
            else:
                # Fallback if regex fails
                self.version = os.path.splitext(os.path.basename(uri))[0]
        else:
            yaml_uri = os.path.join(metric_gcs_dir, f"{self.version}.yaml")
            json_uri = os.path.join(metric_gcs_dir, f"{self.version}.json")

            gcs_utils = _gcs_utils.GcsUtils(api_client)
            try:
                bucket_name, blob_path = gcs_utils.parse_gcs_path(yaml_uri)
                if (
                    gcs_utils.storage_client.bucket(bucket_name)
                    .blob(blob_path)
                    .exists()
                ):
                    uri = yaml_uri
                else:
                    bucket_name_json, blob_path_json = gcs_utils.parse_gcs_path(
                        json_uri
                    )
                    if (
                        gcs_utils.storage_client.bucket(bucket_name_json)
                        .blob(blob_path_json)
                        .exists()
                    ):
                        uri = json_uri
                    else:
                        raise IOError(
                            f"Metric file for version '{self.version}' "
                            f"not found as .yaml or .json in {metric_gcs_dir}"
                        )
            except Exception as e:
                raise IOError(
                    f"Error checking for metric file version '{self.version}' in"
                    f" {metric_gcs_dir}: {e}"
                ) from e

        logger.info(
            "Fetching predefined metric '%s@%s' from %s...",
            self.name,
            self.version,
            uri,
        )

        gcs_utils = _gcs_utils.GcsUtils(api_client)
        content_str = gcs_utils.read_file_contents(uri)

        file_extension = os.path.splitext(uri)[1].lower()
        data: dict[str, Any]
        if file_extension == ".yaml" or file_extension == ".yml":
            if yaml is None:
                raise ImportError(
                    "YAML parsing requires the pyyaml library. Please install it"
                    " with `pip install google-cloud-aiplatform[evaluation]`."
                )
            data = yaml.safe_load(content_str)
        elif file_extension == ".json":
            data = json.loads(content_str)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        if not isinstance(data, dict):
            raise ValueError("Metric config content did not parse into a dictionary.")

        metric_obj = types.LLMMetric.model_validate({**data, **self.metric_kwargs})
        metric_obj._is_predefined = True
        metric_obj._config_source = uri
        metric_obj._version = self.version
        return metric_obj

    def resolve(self, api_client: Any) -> types.Metric:
        """Resolves the metric by checking API Predefined, then GCS, caching results."""
        if self._resolved_metric:
            return self._resolved_metric

        cache_key = f"{self.name}@{self.version or 'default'}"
        if cache_key in LazyLoadedPrebuiltMetric._cache:
            self._resolved_metric = LazyLoadedPrebuiltMetric._cache[cache_key]
            logger.debug("Metric '%s' found in cache.", cache_key)
            return self._resolved_metric

        # Try resolving as API Predefined Metric first
        api_metric = self._resolve_api_predefined()
        if api_metric:
            self._resolved_metric = api_metric
            LazyLoadedPrebuiltMetric._cache[cache_key] = self._resolved_metric
            return self._resolved_metric

        # Fallback to GCS loading for custom LLM-based Prebuilt Metrics
        logger.debug(
            "Metric '%s' not an API Predefined Metric, trying GCS...", self.name
        )
        try:
            gcs_metric = self._fetch_and_parse(api_client)
            final_cache_key = f"{self.name}@{self.version}"
            LazyLoadedPrebuiltMetric._cache[final_cache_key] = gcs_metric
            self._resolved_metric = gcs_metric
            return self._resolved_metric
        except Exception as e:
            logger.error(
                "Error loading metric %s (requested version: %s) from GCS: %s",
                self.name,
                self.version,
                e,
            )
            raise ValueError(
                f"Metric '{self.name}' could not be resolved as an API "
                "Predefined Metric or loaded from GCS."
            ) from e

    def __call__(
        self, version: Optional[str] = None, **kwargs
    ) -> "LazyLoadedPrebuiltMetric":
        """Allows setting a specific version and other metric attributes."""
        updated_kwargs = self.metric_kwargs.copy()
        updated_kwargs.update(kwargs)
        return LazyLoadedPrebuiltMetric(
            name=self.name, version=version or self.version, **updated_kwargs
        )


class PrebuiltMetricLoader:
    """Provides access to predefined evaluation metrics via attributes.

    This class provides a set of predefined LLM-based metrics (Autorater recipes)
    for evaluation. These metrics are lazily loaded from a GCS repository
    when they are first accessed.

    Example:
      from vertexai import types
      text_quality_metric = types.RubricMetric.TEXT_QUALITY
    """

    def __getattr__(
        self, name: str, version: Optional[str] = None, **kwargs
    ) -> LazyLoadedPrebuiltMetric:
        return LazyLoadedPrebuiltMetric(name=name, version=version, **kwargs)

    @property
    def GENERAL_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("GENERAL_QUALITY")

    @property
    def TEXT_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("TEXT_QUALITY")

    @property
    def INSTRUCTION_FOLLOWING(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("INSTRUCTION_FOLLOWING")

    @property
    def SAFETY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("SAFETY")

    @property
    def MULTI_TURN_GENERAL_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("MULTI_TURN_GENERAL_QUALITY")

    @property
    def MULTI_TURN_TEXT_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("MULTI_TURN_TEXT_QUALITY")

    @property
    def FINAL_RESPONSE_MATCH(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("FINAL_RESPONSE_MATCH", version="v2")

    @property
    def FINAL_RESPONSE_REFERENCE_FREE(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("FINAL_RESPONSE_REFERENCE_FREE")

    @property
    def COHERENCE(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("COHERENCE")

    @property
    def FLUENCY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("FLUENCY")

    @property
    def VERBOSITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("VERBOSITY")

    @property
    def SUMMARIZATION_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("SUMMARIZATION_QUALITY")

    @property
    def QUESTION_ANSWERING_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("QUESTION_ANSWERING_QUALITY")

    @property
    def MULTI_TURN_CHAT_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("MULTI_TURN_CHAT_QUALITY")

    @property
    def MULTI_TURN_SAFETY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("MULTI_TURN_SAFETY")

    @property
    def FINAL_RESPONSE_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("FINAL_RESPONSE_QUALITY")

    @property
    def HALLUCINATION(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("HALLUCINATION")

    @property
    def TOOL_USE_QUALITY(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("TOOL_USE_QUALITY")

    @property
    def GECKO_TEXT2IMAGE(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("GECKO_TEXT2IMAGE")

    @property
    def GECKO_TEXT2VIDEO(self) -> LazyLoadedPrebuiltMetric:
        return self.__getattr__("GECKO_TEXT2VIDEO")


PrebuiltMetric = PrebuiltMetricLoader()
RubricMetric = PrebuiltMetric


class BatchEvaluateRequestPreparer:
    """Prepares data for requests."""

    @staticmethod
    def _EvaluationDataset_to_vertex(
        from_object: Union[dict[str, Any], object],
        parent_object: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        to_object: dict[str, Any] = {}

        if getv(from_object, ["gcs_source"]) is not None:
            setv(
                to_object,
                ["gcs_source"],
                getv(from_object, ["gcs_source"]),
            )

        if getv(from_object, ["bigquery_source"]) is not None:
            setv(
                to_object,
                ["bigquery_source"],
                getv(from_object, ["bigquery_source"]),
            )

        return to_object

    @staticmethod
    def _Metric_to_vertex(
        from_object: Union[dict[str, Any], object],
        parent_object: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        to_object: dict[str, Any] = {}

        if getv(from_object, ["prompt_template"]) is not None:
            setv(
                to_object,
                ["pointwise_metric_spec", "prompt_template"],
                getv(from_object, ["prompt_template"]),
            )

        if getv(from_object, ["judge_model"]) is not None:
            setv(
                parent_object,
                ["autorater_config", "autorater_model"],
                getv(from_object, ["judge_model"]),
            )

        if getv(from_object, ["judge_model_sampling_count"]) is not None:
            setv(
                parent_object,
                ["autorater_config", "sampling_count"],
                getv(from_object, ["judge_model_sampling_count"]),
            )

        if getv(from_object, ["judge_model_system_instruction"]) is not None:
            setv(
                to_object,
                ["pointwise_metric_spec", "system_instruction"],
                getv(from_object, ["judge_model_system_instruction"]),
            )

        if getv(from_object, ["return_raw_output"]) is not None:
            setv(
                to_object,
                [
                    "pointwise_metric_spec",
                    "custom_output_format_config",
                    "return_raw_output",
                ],
                getv(from_object, ["return_raw_output"]),
            )

        return to_object

    @staticmethod
    def _OutputConfig_to_vertex(
        from_object: Union[dict[str, Any], object],
        parent_object: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        to_object: dict[str, Any] = {}
        if getv(from_object, ["gcs_destination"]) is not None:
            setv(
                to_object,
                ["gcsDestination"],
                getv(from_object, ["gcs_destination"]),
            )

        return to_object

    @staticmethod
    def _EvaluationDataset_from_vertex(
        from_object: Union[dict[str, Any], object],
        parent_object: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        to_object: dict[str, Any] = {}

        if getv(from_object, ["dataset", "gcs_source"]) is not None:
            setv(
                to_object,
                ["gcs_source"],
                getv(from_object, ["dataset", "gcs_source"]),
            )

        if getv(from_object, ["dataset", "bigquery_source"]) is not None:
            setv(
                to_object,
                ["bigquery_source"],
                getv(from_object, ["dataset", "bigquery_source"]),
            )

        return to_object

    @staticmethod
    def _AutoraterConfig_to_vertex(
        from_object: Union[dict[str, Any], object],
        parent_object: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        to_object: dict[str, Any] = {}
        if getv(from_object, ["sampling_count"]) is not None:
            setv(to_object, ["samplingCount"], getv(from_object, ["sampling_count"]))

        if getv(from_object, ["flip_enabled"]) is not None:
            setv(to_object, ["flipEnabled"], getv(from_object, ["flip_enabled"]))

        if getv(from_object, ["autorater_model"]) is not None:
            setv(
                to_object,
                ["autoraterModel"],
                getv(from_object, ["autorater_model"]),
            )

        return to_object

    @staticmethod
    def EvaluateDatasetOperation_from_vertex(
        from_object: Union[dict[str, Any], object],
        parent_object: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        to_object: dict[str, Any] = {}
        if getv(from_object, ["name"]) is not None:
            setv(to_object, ["name"], getv(from_object, ["name"]))

        if getv(from_object, ["metadata"]) is not None:
            setv(to_object, ["metadata"], getv(from_object, ["metadata"]))

        if getv(from_object, ["done"]) is not None:
            setv(to_object, ["done"], getv(from_object, ["done"]))

        if getv(from_object, ["error"]) is not None:
            setv(to_object, ["error"], getv(from_object, ["error"]))

        if getv(from_object, ["response"]) is not None:
            setv(
                to_object,
                ["response"],
                BatchEvaluateRequestPreparer._EvaluationDataset_from_vertex(
                    getv(from_object, ["response"]), to_object
                ),
            )

        return to_object

    @staticmethod
    def EvaluateDatasetRequestParameters_to_vertex(
        from_object: Union[dict[str, Any], object],
        parent_object: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        to_object: dict[str, Any] = {}
        if getv(from_object, ["dataset"]) is not None:
            setv(
                to_object,
                ["dataset"],
                BatchEvaluateRequestPreparer._EvaluationDataset_to_vertex(
                    getv(from_object, ["dataset"]), to_object
                ),
            )

        if getv(from_object, ["metrics"]) is not None:
            setv(
                to_object,
                ["metrics"],
                [
                    BatchEvaluateRequestPreparer._Metric_to_vertex(item, to_object)
                    for item in getv(from_object, ["metrics"])
                ],
            )

        if getv(from_object, ["output_config"]) is not None:
            setv(
                to_object,
                ["outputConfig"],
                BatchEvaluateRequestPreparer._OutputConfig_to_vertex(
                    getv(from_object, ["output_config"]), to_object
                ),
            )

        if getv(from_object, ["autorater_config"]) is not None:
            setv(
                to_object,
                ["autoraterConfig"],
                BatchEvaluateRequestPreparer._AutoraterConfig_to_vertex(
                    getv(from_object, ["autorater_config"]), to_object
                ),
            )

        if getv(from_object, ["config"]) is not None:
            setv(to_object, ["config"], getv(from_object, ["config"]))

        return to_object

    @staticmethod
    def prepare_metric_payload(
        request_dict: dict[str, Any], resolved_metrics: list["types.MetricSubclass"]
    ) -> dict[str, Any]:
        """Prepares the metric payload for the evaluation request.

        Args:
            request_dict: The dictionary containing the request details.
            resolved_metrics: A list of resolved metric objects.

        Returns:
            The updated request dictionary with the prepared metric payload.
        """
        request_dict["metrics"] = _transformers.t_metrics(
            resolved_metrics, set_default_aggregation_metrics=True
        )
        return request_dict


class EvalDataConverter(abc.ABC):
    """Abstract base class for dataset converters."""

    @abc.abstractmethod
    def convert(self, raw_data: Any) -> "types.EvaluationDataset":
        """Converts a loaded raw dataset into an EvaluationDataset."""
        raise NotImplementedError()
