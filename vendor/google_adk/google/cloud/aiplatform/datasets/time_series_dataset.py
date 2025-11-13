# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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

from typing import Dict, Optional, Sequence, Tuple, Union

from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import datasets
from google.cloud.aiplatform.datasets import _datasources
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import schema
from google.cloud.aiplatform import utils


class TimeSeriesDataset(datasets._ColumnNamesDataset):
    """A managed time series dataset resource for Vertex AI.

    Use this class to work with time series datasets. A time series is a dataset
    that contains data recorded at different time intervals. The dataset
    includes time and at least one variable that's dependent on time. You use a
    time series dataset for forecasting predictions. For more information, see
    [Forecasting overview](https://cloud.google.com/vertex-ai/docs/tabular-data/forecasting/overview).

    You can create a managed time series dataset from CSV files in a Cloud
    Storage bucket or from a BigQuery table.

    The following code shows you how to create a `TimeSeriesDataset` with a CSV
    file that has the time series dataset:

    ```py
    my_dataset = aiplatform.TimeSeriesDataset.create(
        display_name="my-dataset",
        gcs_source=['gs://path/to/my/dataset.csv'],
    )
    ```

    The following code shows you how to create with a `TimeSeriesDataset` with a
    BigQuery table file that has the time series dataset:

    ```py
    my_dataset = aiplatform.TimeSeriesDataset.create(
        display_name="my-dataset",
        bq_source=['bq://path/to/my/bigquerydataset.train'],
    )
    ```

    """

    _supported_metadata_schema_uris: Optional[Tuple[str]] = (
        schema.dataset.metadata.time_series,
    )

    @classmethod
    def create(
        cls,
        display_name: Optional[str] = None,
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        bq_source: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "TimeSeriesDataset":
        """Creates a new time series dataset.

        Args:
            display_name (str):
                Optional. The user-defined name of the dataset. The name must
                contain 128 or fewer UTF-8 characters.
            gcs_source (Union[str, Sequence[str]]):
                The URI to one or more Google Cloud Storage buckets that contain
                your datasets. For example, `str: "gs://bucket/file.csv"` or
                `Sequence[str]: ["gs://bucket/file1.csv",
                "gs://bucket/file2.csv"]`.
            bq_source (str):
                A BigQuery URI for the input table. For example,
                `bq://project.dataset.table_name`.
            project (str):
                The name of the Google Cloud project to which this
                `TimeSeriesDataset` is uploaded. This overrides the project that
                was set by `aiplatform.init`.
            location (str):
                The Google Cloud region where this dataset is uploaded. This
                region overrides the region that was set by `aiplatform.init`.
            credentials (auth_credentials.Credentials):
                The credentials that are used to upload the `TimeSeriesDataset`.
                These credentials override the credentials set by
                `aiplatform.init`.
            request_metadata (Sequence[Tuple[str, str]]):
                Strings that contain metadata that's sent with the request.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your
                Vertex AI Tensorboards. The maximum length of a key and of a
                value is 64 unicode characters. Labels and keys can contain only
                lowercase letters, numeric characters, underscores, and dashes.
                International characters are allowed. No more than 64 user
                labels can be associated with one Tensorboard (system labels are
                excluded). For more information and examples of using labels, see
                [Using labels to organize Google Cloud Platform resources](https://goo.gl/xmQnxf).
                System reserved label keys are prefixed with
                `aiplatform.googleapis.com/` and are immutable.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key that's used to protect the dataset. The
                format of the key is
                `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key`.
                The key needs to be in the same region as where the compute
                resource is created.

                If `encryption_spec_key_name` is set, this time series dataset
                and all of its sub-resources are secured by this key.

                This `encryption_spec_key_name` overrides the
                `encryption_spec_key_name` set by `aiplatform.init`.
            create_request_timeout (float):
                 Optional. The number of seconds for the timeout of the create
                request.
            sync (bool):
                If `true`, the `create` method creates a time series dataset
                synchronously. If `false`, the `create` method creates a time
                series dataset asynchronously.

        Returns:
            time_series_dataset (TimeSeriesDataset):
                An instantiated representation of the managed
                `TimeSeriesDataset` resource.

        """
        if not display_name:
            display_name = cls._generate_display_name()
        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        metadata_schema_uri = schema.dataset.metadata.time_series

        datasource = _datasources.create_datasource(
            metadata_schema_uri=metadata_schema_uri,
            gcs_source=gcs_source,
            bq_source=bq_source,
        )

        return cls._create_and_import(
            api_client=api_client,
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            display_name=display_name,
            metadata_schema_uri=metadata_schema_uri,
            datasource=datasource,
            project=project or initializer.global_config.project,
            location=location or initializer.global_config.location,
            credentials=credentials or initializer.global_config.credentials,
            request_metadata=request_metadata,
            labels=labels,
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    def import_data(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not support 'import_data'"
        )
