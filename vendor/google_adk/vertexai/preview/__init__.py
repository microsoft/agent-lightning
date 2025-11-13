# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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

from google.cloud.aiplatform.metadata import metadata


# For Vertex AI Experiment.

# ExperimentRun manipulation.
start_run = metadata._experiment_tracker.start_run
end_run = metadata._experiment_tracker.end_run
get_experiment_df = metadata._experiment_tracker.get_experiment_df

# Experiment logging.
log_params = metadata._experiment_tracker.log_params
log_metrics = metadata._experiment_tracker.log_metrics
log_time_series_metrics = metadata._experiment_tracker.log_time_series_metrics
log_classification_metrics = metadata._experiment_tracker.log_classification_metrics


__all__ = (
    "start_run",
    "end_run",
    "get_experiment_df",
    "log_params",
    "log_metrics",
    "log_time_series_metrics",
    "log_classification_metrics",
)
