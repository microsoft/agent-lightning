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

from google.cloud.aiplatform.compat.types import (
    io_v1beta1 as io,
    model_monitoring_spec_v1beta1 as model_monitoring_spec,
)


class OutputSpec:
    """Initializer for OutputSpec.

    Args:
        data_source (str):
            Optional. Google Cloud Storage base folder path for metrics, error
            logs, etc.
    """

    def __init__(
        self,
        gcs_base_dir: str,
    ):
        self.gcs_base_dir = gcs_base_dir

    def _as_proto(self) -> model_monitoring_spec.ModelMonitoringOutputSpec:
        """Converts ModelMonitoringOutputSpec to a proto message.

        Returns:
           The GAPIC representation of the notification alert config.
        """
        user_gcs_base_dir = io.GcsDestination(output_uri_prefix=self.gcs_base_dir)
        return model_monitoring_spec.ModelMonitoringOutputSpec(
            gcs_base_directory=user_gcs_base_dir,
        )
