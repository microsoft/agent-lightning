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

import logging

from google.cloud import bigquery
from google.genai._api_client import BaseApiClient
import pandas as pd


logger = logging.getLogger(__name__)


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
