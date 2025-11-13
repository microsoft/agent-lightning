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

import logging
import re
from dataclasses import dataclass

from opentelemetry.resourcedetector.gcp_resource_detector import _metadata

# Format described in
# https://cloud.google.com/compute/docs/metadata/default-metadata-values#vm_instance_metadata
_ZONE_REGION_RE = re.compile(
    r"projects\/\d+\/zones\/(?P<zone>(?P<region>\w+-\w+)-\w+)"
)

_logger = logging.getLogger(__name__)


def on_gce() -> bool:
    try:
        _metadata.get_metadata()["instance"]["machineType"]
    except (_metadata.MetadataAccessException, KeyError):
        _logger.debug(
            "Could not fetch metadata attribute instance/machineType, "
            "assuming not on GCE.",
            exc_info=True,
        )
        return False
    return True


def host_type() -> str:
    return _metadata.get_metadata()["instance"]["machineType"]


def host_id() -> str:
    return str(_metadata.get_metadata()["instance"]["id"])


def host_name() -> str:
    return _metadata.get_metadata()["instance"]["name"]


@dataclass
class ZoneAndRegion:
    zone: str
    region: str


def availability_zone_and_region() -> ZoneAndRegion:
    full_zone = _metadata.get_metadata()["instance"]["zone"]
    match = _ZONE_REGION_RE.search(full_zone)
    if not match:
        raise Exception(
            "zone was not in the expected format: "
            f"projects/PROJECT_NUM/zones/COUNTRY-REGION-ZONE. Got {full_zone}"
        )

    return ZoneAndRegion(
        zone=match.group("zone"), region=match.group("region")
    )
