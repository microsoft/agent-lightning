# Copyright 2016 Google LLC
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

"""Python :mod:`logging` handlers for Google Cloud Logging."""

from google.cloud.logging_v2.handlers.app_engine import AppEngineHandler
from google.cloud.logging_v2.handlers.container_engine import ContainerEngineHandler
from google.cloud.logging_v2.handlers.structured_log import StructuredLogHandler
from google.cloud.logging_v2.handlers.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers.handlers import CloudLoggingFilter
from google.cloud.logging_v2.handlers.handlers import setup_logging

__all__ = [
    "AppEngineHandler",
    "CloudLoggingFilter",
    "CloudLoggingHandler",
    "ContainerEngineHandler",
    "StructuredLogHandler",
    "setup_logging",
]
