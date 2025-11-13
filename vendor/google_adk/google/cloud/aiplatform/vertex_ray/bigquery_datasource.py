# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from typing import List, Optional

from google.api_core import client_info
from google.api_core import exceptions
from google.api_core.gapic_v1 import client_info as v1_client_info
from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud.aiplatform import initializer
from google.cloud.bigquery_storage import types
from ray.data.block import Block
from ray.data.block import BlockMetadata
from ray.data.datasource.datasource import Datasource
from ray.data.datasource.datasource import ReadTask


_BQ_GAPIC_VERSION = bigquery.__version__ + "+vertex_ray"
_BQS_GAPIC_VERSION = bigquery_storage.__version__ + "+vertex_ray"
bq_info = client_info.ClientInfo(
    gapic_version=_BQ_GAPIC_VERSION, user_agent=f"ray-on-vertex/{_BQ_GAPIC_VERSION}"
)
bqstorage_info = v1_client_info.ClientInfo(
    gapic_version=_BQS_GAPIC_VERSION, user_agent=f"ray-on-vertex/{_BQS_GAPIC_VERSION}"
)

DEFAULT_MAX_RETRY_CNT = 10
RATE_LIMIT_EXCEEDED_SLEEP_TIME = 11


class _BigQueryDatasource(Datasource):
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
    ):
        self._project_id = project_id or initializer.global_config.project
        self._dataset = dataset
        self._query = query

        if query is not None and dataset is not None:
            raise ValueError(
                "[Ray on Vertex AI]: Query and dataset kwargs cannot both be provided (must be mutually exclusive)."
            )

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        # Executed by a worker node
        def _read_single_partition(stream) -> Block:
            client = bigquery_storage.BigQueryReadClient(client_info=bqstorage_info)
            reader = client.read_rows(stream.name)
            return reader.to_arrow()

        if self._query:
            query_client = bigquery.Client(
                project=self._project_id, client_info=bq_info
            )
            query_job = query_client.query(self._query)
            query_job.result()
            destination = str(query_job.destination)
            dataset_id = destination.split(".")[-2]
            table_id = destination.split(".")[-1]
        else:
            self._validate_dataset_table_exist(self._project_id, self._dataset)
            dataset_id = self._dataset.split(".")[0]
            table_id = self._dataset.split(".")[1]

        bqs_client = bigquery_storage.BigQueryReadClient(client_info=bqstorage_info)
        table = f"projects/{self._project_id}/datasets/{dataset_id}/tables/{table_id}"

        if parallelism == -1:
            parallelism = None
        requested_session = types.ReadSession(
            table=table,
            data_format=types.DataFormat.ARROW,
        )
        read_session = bqs_client.create_read_session(
            parent=f"projects/{self._project_id}",
            read_session=requested_session,
            max_stream_count=parallelism,
        )

        read_tasks = []
        print("[Ray on Vertex AI]: Created streams:", len(read_session.streams))
        if len(read_session.streams) < parallelism:
            print(
                "[Ray on Vertex AI]: The number of streams created by the "
                + "BigQuery Storage Read API is less than the requested "
                + "parallelism due to the size of the dataset."
            )

        for stream in read_session.streams:
            # Create a metadata block object to store schema, etc.
            metadata = BlockMetadata(
                num_rows=None,
                size_bytes=None,
                schema=None,
                input_files=None,
                exec_stats=None,
            )

            # Create a no-arg wrapper read function which returns a block
            read_single_partition = lambda stream=stream: [  # noqa: E731
                _read_single_partition(stream)
            ]

            # Create the read task and pass the wrapper and metadata in
            read_task = ReadTask(read_single_partition, metadata)
            read_tasks.append(read_task)

        return read_tasks

    def estimate_inmemory_data_size(self) -> Optional[int]:
        # TODO(b/281891467): Implement this method
        return None

    def _validate_dataset_table_exist(self, project_id: str, dataset: str) -> None:
        client = bigquery.Client(project=project_id, client_info=bq_info)
        dataset_id = dataset.split(".")[0]
        try:
            client.get_dataset(dataset_id)
        except exceptions.NotFound:
            raise ValueError(
                "[Ray on Vertex AI]: Dataset {} is not found. Please ensure that it exists.".format(
                    dataset_id
                )
            )

        try:
            client.get_table(dataset)
        except exceptions.NotFound:
            raise ValueError(
                "[Ray on Vertex AI]: Table {} is not found. Please ensure that it exists.".format(
                    dataset
                )
            )
