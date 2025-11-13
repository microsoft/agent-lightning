# -*- coding: utf-8 -*-

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

"""Custom output config for model-based metrics."""

from typing import Any, Callable, Dict, Optional


class CustomOutputConfig:
    """Custom output config for model-based metrics.

    Attributes:
        return_raw_output: Whether to return the raw output of the metric
            function.
        parsing_fn: Function to parse the raw output of the metric.
    """

    def __init__(
        self,
        return_raw_output: bool = False,
        parsing_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        """Initializes CustomOutputConfig."""
        self.return_raw_output = return_raw_output
        self.parsing_fn = parsing_fn
