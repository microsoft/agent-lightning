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

import json
import logging
import os
import re
from typing import Any, Optional, Union, TYPE_CHECKING

from google.genai._api_client import BaseApiClient
import pandas as pd
import yaml

from . import _evals_constant
from . import _gcs_utils
from . import _bigquery_utils

if TYPE_CHECKING:
    from . import types


logger = logging.getLogger(__name__)


GCS_PREFIX = "gs://"
BQ_PREFIX = "bq://"


class EvalDatasetLoader:
    """A loader for datasets from various sources, using a shared client."""

    def __init__(self, api_client: BaseApiClient):
        self.api_client = api_client
        self.gcs_utils = _gcs_utils.GcsUtils(self.api_client)
        self.bigquery_utils = _bigquery_utils.BigQueryUtils(self.api_client)

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

    _cache: dict[str, "types.Metric"] = {}
    _base_gcs_path = (
        "gs://vertex-ai-generative-ai-eval-sdk-resources/metrics/{metric_name}/"
    )

    def __init__(self, name: str, version: Optional[str] = None, **kwargs):
        self.name = name.upper()
        self.version = version
        self.metric_kwargs = kwargs
        self._resolved_metric: Optional["types.Metric"] = None

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

    def _resolve_api_predefined(self) -> Optional["types.Metric"]:
        """Attempts to resolve as an API Predefined Metric."""
        from . import types

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

    def _fetch_and_parse(self, api_client: Any) -> "types.LLMMetric":
        """Fetches and parses the metric definition from GCS."""

        from . import types

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

    def resolve(self, api_client: Any) -> "types.Metric":
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


PrebuiltMetric = PrebuiltMetricLoader()
RubricMetric = PrebuiltMetric
