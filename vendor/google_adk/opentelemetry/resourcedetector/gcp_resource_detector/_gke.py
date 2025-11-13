# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from typing import Literal

from opentelemetry.resourcedetector.gcp_resource_detector import (
    _gce,
    _metadata,
)

KUBERNETES_SERVICE_HOST_ENV = "KUBERNETES_SERVICE_HOST"


def on_gke() -> bool:
    return os.environ.get(KUBERNETES_SERVICE_HOST_ENV) is not None


def host_id() -> str:
    return _gce.host_id()


def cluster_name() -> str:
    return _metadata.get_metadata()["instance"]["attributes"]["cluster-name"]


@dataclass
class ZoneOrRegion:
    type: Literal["zone", "region"]
    value: str


def availability_zone_or_region() -> ZoneOrRegion:
    cluster_location = _metadata.get_metadata()["instance"]["attributes"][
        "cluster-location"
    ]
    hyphen_count = cluster_location.count("-")
    if hyphen_count == 1:
        return ZoneOrRegion(type="region", value=cluster_location)
    if hyphen_count == 2:
        return ZoneOrRegion(type="zone", value=cluster_location)
    raise Exception(
        f"unrecognized format for cluster location: {cluster_location}"
    )
