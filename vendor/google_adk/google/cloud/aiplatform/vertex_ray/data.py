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
import warnings
import ray.data
from ray.data.dataset import Dataset
from typing import Any, Dict, Optional

from google.cloud.aiplatform.vertex_ray.bigquery_datasource import (
    _BigQueryDatasource,
)

try:
    from google.cloud.aiplatform.vertex_ray.bigquery_datasink import (
        _BigQueryDatasink,
    )
except ImportError:
    _BigQueryDatasink = None

from google.cloud.aiplatform.vertex_ray.util._validation_utils import (
    _V2_4_WARNING_MESSAGE,
    _V2_9_WARNING_MESSAGE,
)


def read_bigquery(
    project_id: Optional[str] = None,
    dataset: Optional[str] = None,
    query: Optional[str] = None,
    *,
    parallelism: int = -1,
    ray_remote_args: Dict[str, Any] = None,
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
) -> Dataset:
    """Create a dataset from BigQuery.

    The data to read from is specified via the ``project_id``, ``dataset``
    and/or ``query`` parameters.

    Args:
        project_id: The name of the associated Google Cloud Project that hosts
            the dataset to read.
        dataset: The name of the dataset hosted in BigQuery in the format of
            ``dataset_id.table_id``. Both the dataset_id and table_id must exist
            otherwise an exception will be raised.
        query: The query to execute. The dataset is created from the results of
            executing the query if provided. Otherwise, the entire dataset is read.
            For query syntax guidelines, see
            https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax
        parallelism: 2.33.0, 2.42.0, 2.47.1: This argument is deprecated. Use
            ``override_num_blocks`` argument. 2.9.3: The requested parallelism of
            the read. If -1, it will be automatically chosen based on the available
            cluster resources and estimated in-memory data size.
        ray_remote_args: kwargs passed to ray.remote in the read tasks.
        concurrency: Supported for 2.33.0, 2.42.0 and 2.47.1 only: The maximum number of
            Ray tasks to run concurrently. Set this to control number of tasks to
            run concurrently. This doesn't change the total number of tasks run or
            the total number of output blocks. By default, concurrency is
            dynamically decided based on the available resources.
        override_num_blocks: Supported for 2.33.0, 2.42.0 and 2.47.1 only: Override the
            number of output blocks from all read tasks. By default, the number of
            output blocks is dynamically decided based on input data size and
            available resources. You shouldn't manually set this value in most
            cases.

    Returns:
        Dataset producing rows from the results of executing the query
        or reading the entire dataset on the specified BigQuery dataset.
    """
    datasource = _BigQueryDatasource(
        project_id=project_id,
        dataset=dataset,
        query=query,
    )

    if ray.__version__ == "2.9.3":
        warnings.warn(_V2_9_WARNING_MESSAGE, DeprecationWarning, stacklevel=1)
        # Concurrency and override_num_blocks are not supported in 2.9.3
        return ray.data.read_datasource(
            datasource=datasource,
            parallelism=parallelism,
            ray_remote_args=ray_remote_args,
        )
    elif ray.__version__ in ("2.33.0", "2.42.0", "2.47.1"):
        return ray.data.read_datasource(
            datasource=datasource,
            parallelism=parallelism,
            ray_remote_args=ray_remote_args,
            concurrency=concurrency,
            override_num_blocks=override_num_blocks,
        )
    else:
        raise ImportError(
            f"[Ray on Vertex AI]: Unsupported version {ray.__version__}."
            + "Only 2.47.1, 2.42.0, 2.33.0, and 2.9.3 are supported."
        )


def write_bigquery(
    ds: Dataset,
    project_id: Optional[str] = None,
    dataset: Optional[str] = None,
    max_retry_cnt: int = 10,
    ray_remote_args: Dict[str, Any] = None,
    overwrite_table: Optional[bool] = True,
    concurrency: Optional[int] = None,
) -> Any:
    """Write the dataset to a BigQuery dataset table.

    Args:
        ds: The dataset to write.
        project_id: The name of the associated Google Cloud Project that hosts
            the dataset table to write to.
        dataset: The name of the dataset table hosted in BigQuery in the format of
            ``dataset_id.table_id``.
            The dataset table is created if it doesn't already exist.
            In 2.9.3, the table_id is overwritten if it exists.
        max_retry_cnt: The maximum number of retries that an individual block write
            is retried due to BigQuery rate limiting errors.
            The default number of retries is 10.
        ray_remote_args: kwargs passed to ray.remote in the write tasks.
        overwrite_table: Not supported in 2.9.3.
            2.33.0, 2.42.0, 2.47.1: Whether the write will overwrite the table
            if it already exists. The default behavior is to overwrite the table.
            If false, will append to the table if it exists.
        concurrency: Not supported in 2.9.3.
            2.33.0, 2.42.0, 2.47.1: The maximum number of Ray tasks to run concurrently.
            Set this to control number of tasks to run concurrently. This doesn't
            change the total number of tasks run or the total number of output blocks.
            By default, concurrency is dynamically decided based on the available
            resources.
    """
    if ray.__version__ == "2.4.0":
        raise RuntimeError(_V2_4_WARNING_MESSAGE)

    elif ray.__version__ in ("2.9.3", "2.33.0", "2.42.0", "2.47.1"):
        if ray.__version__ == "2.9.3":
            warnings.warn(_V2_9_WARNING_MESSAGE, DeprecationWarning, stacklevel=1)
        if ray_remote_args is None:
            ray_remote_args = {}

        # Each write task will launch individual remote tasks to write each block
        # To avoid duplicate block writes, the write task should not be retried
        if ray_remote_args.get("max_retries", 0) != 0:
            print(
                "[Ray on Vertex AI]: The max_retries of a BigQuery Write "
                "Task should be set to 0 to avoid duplicate writes."
            )
        else:
            ray_remote_args["max_retries"] = 0

        if ray.__version__ == "2.9.3":
            # Concurrency and overwrite_table are not supported in 2.9.3
            datasink = _BigQueryDatasink(
                project_id=project_id,
                dataset=dataset,
                max_retry_cnt=max_retry_cnt,
            )
            return ds.write_datasink(
                datasink=datasink,
                ray_remote_args=ray_remote_args,
            )
        elif ray.__version__ in ("2.33.0", "2.42.0", "2.47.1"):
            datasink = _BigQueryDatasink(
                project_id=project_id,
                dataset=dataset,
                max_retry_cnt=max_retry_cnt,
                overwrite_table=overwrite_table,
            )
            return ds.write_datasink(
                datasink=datasink,
                ray_remote_args=ray_remote_args,
                concurrency=concurrency,
            )
    else:
        raise ImportError(
            f"[Ray on Vertex AI]: Unsupported version {ray.__version__}."
            + "Only 2.47.1, 2.42.0, 2.33.0 and 2.9.3 are supported."
        )
