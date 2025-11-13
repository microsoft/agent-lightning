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

BUILT_IN_METRICS_METER_NAME = "gax-python"
NATIVE_METRICS_PREFIX = "spanner.googleapis.com/internal/client"
SPANNER_RESOURCE_TYPE = "spanner_instance_client"
SPANNER_SERVICE_NAME = "spanner-python"
GOOGLE_CLOUD_RESOURCE_KEY = "google-cloud-resource-prefix"
GOOGLE_CLOUD_REGION_KEY = "cloud.region"
GOOGLE_CLOUD_REGION_GLOBAL = "global"
SPANNER_METHOD_PREFIX = "/google.spanner.v1."
ENABLE_SPANNER_METRICS_ENV_VAR = "SPANNER_ENABLE_BUILTIN_METRICS"

# Monitored resource labels
MONITORED_RES_LABEL_KEY_PROJECT = "project_id"
MONITORED_RES_LABEL_KEY_INSTANCE = "instance_id"
MONITORED_RES_LABEL_KEY_INSTANCE_CONFIG = "instance_config"
MONITORED_RES_LABEL_KEY_LOCATION = "location"
MONITORED_RES_LABEL_KEY_CLIENT_HASH = "client_hash"
MONITORED_RESOURCE_LABELS = [
    MONITORED_RES_LABEL_KEY_PROJECT,
    MONITORED_RES_LABEL_KEY_INSTANCE,
    MONITORED_RES_LABEL_KEY_INSTANCE_CONFIG,
    MONITORED_RES_LABEL_KEY_LOCATION,
    MONITORED_RES_LABEL_KEY_CLIENT_HASH,
]

# Metric labels
METRIC_LABEL_KEY_CLIENT_UID = "client_uid"
METRIC_LABEL_KEY_CLIENT_NAME = "client_name"
METRIC_LABEL_KEY_DATABASE = "database"
METRIC_LABEL_KEY_METHOD = "method"
METRIC_LABEL_KEY_STATUS = "status"
METRIC_LABEL_KEY_DIRECT_PATH_ENABLED = "directpath_enabled"
METRIC_LABEL_KEY_DIRECT_PATH_USED = "directpath_used"
METRIC_LABELS = [
    METRIC_LABEL_KEY_CLIENT_UID,
    METRIC_LABEL_KEY_CLIENT_NAME,
    METRIC_LABEL_KEY_DATABASE,
    METRIC_LABEL_KEY_METHOD,
    METRIC_LABEL_KEY_STATUS,
    METRIC_LABEL_KEY_DIRECT_PATH_ENABLED,
    METRIC_LABEL_KEY_DIRECT_PATH_USED,
]

# Metric names
METRIC_NAME_OPERATION_LATENCIES = "operation_latencies"
METRIC_NAME_ATTEMPT_LATENCIES = "attempt_latencies"
METRIC_NAME_OPERATION_COUNT = "operation_count"
METRIC_NAME_ATTEMPT_COUNT = "attempt_count"
METRIC_NAME_GFE_LATENCY = "gfe_latency"
METRIC_NAME_GFE_MISSING_HEADER_COUNT = "gfe_missing_header_count"
METRIC_NAMES = [
    METRIC_NAME_OPERATION_LATENCIES,
    METRIC_NAME_ATTEMPT_LATENCIES,
    METRIC_NAME_OPERATION_COUNT,
    METRIC_NAME_ATTEMPT_COUNT,
]

METRIC_EXPORT_INTERVAL_MS = 60000  # 1 Minute
