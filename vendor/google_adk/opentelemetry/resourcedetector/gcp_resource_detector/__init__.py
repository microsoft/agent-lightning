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

from typing import Mapping

from opentelemetry.resourcedetector.gcp_resource_detector import (
    _faas,
    _gae,
    _gce,
    _gke,
    _metadata,
)
from opentelemetry.resourcedetector.gcp_resource_detector._constants import (
    ResourceAttributes,
)
from opentelemetry.sdk.resources import Resource, ResourceDetector
from opentelemetry.util.types import AttributeValue


class GoogleCloudResourceDetector(ResourceDetector):
    def detect(self) -> Resource:
        # pylint: disable=too-many-return-statements
        if not _metadata.is_available():
            return Resource.get_empty()

        if _gke.on_gke():
            return _gke_resource()
        if _faas.on_cloud_functions():
            return _cloud_functions_resource()
        if _faas.on_cloud_run():
            return _cloud_run_resource()
        if _gae.on_app_engine():
            return _gae_resource()
        if _gce.on_gce():
            return _gce_resource()

        return Resource.get_empty()


def _gke_resource() -> Resource:
    zone_or_region = _gke.availability_zone_or_region()
    zone_or_region_key = (
        ResourceAttributes.CLOUD_AVAILABILITY_ZONE
        if zone_or_region.type == "zone"
        else ResourceAttributes.CLOUD_REGION
    )
    return _make_resource(
        {
            ResourceAttributes.CLOUD_PLATFORM_KEY: ResourceAttributes.GCP_KUBERNETES_ENGINE,
            zone_or_region_key: zone_or_region.value,
            ResourceAttributes.K8S_CLUSTER_NAME: _gke.cluster_name(),
            ResourceAttributes.HOST_ID: _gke.host_id(),
        }
    )


def _gce_resource() -> Resource:
    zone_and_region = _gce.availability_zone_and_region()
    return _make_resource(
        {
            ResourceAttributes.CLOUD_PLATFORM_KEY: ResourceAttributes.GCP_COMPUTE_ENGINE,
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE: zone_and_region.zone,
            ResourceAttributes.CLOUD_REGION: zone_and_region.region,
            ResourceAttributes.HOST_TYPE: _gce.host_type(),
            ResourceAttributes.HOST_ID: _gce.host_id(),
            ResourceAttributes.HOST_NAME: _gce.host_name(),
        }
    )


def _cloud_run_resource() -> Resource:
    return _make_resource(
        {
            ResourceAttributes.CLOUD_PLATFORM_KEY: ResourceAttributes.GCP_CLOUD_RUN,
            ResourceAttributes.FAAS_NAME: _faas.faas_name(),
            ResourceAttributes.FAAS_VERSION: _faas.faas_version(),
            ResourceAttributes.FAAS_INSTANCE: _faas.faas_instance(),
            ResourceAttributes.CLOUD_REGION: _faas.faas_cloud_region(),
        }
    )


def _cloud_functions_resource() -> Resource:
    return _make_resource(
        {
            ResourceAttributes.CLOUD_PLATFORM_KEY: ResourceAttributes.GCP_CLOUD_FUNCTIONS,
            ResourceAttributes.FAAS_NAME: _faas.faas_name(),
            ResourceAttributes.FAAS_VERSION: _faas.faas_version(),
            ResourceAttributes.FAAS_INSTANCE: _faas.faas_instance(),
            ResourceAttributes.CLOUD_REGION: _faas.faas_cloud_region(),
        }
    )


def _gae_resource() -> Resource:
    if _gae.on_app_engine_standard():
        zone = _gae.standard_availability_zone()
        region = _gae.standard_cloud_region()
    else:
        zone_and_region = _gae.flex_availability_zone_and_region()
        zone = zone_and_region.zone
        region = zone_and_region.region

    faas_name = _gae.service_name()
    faas_version = _gae.service_version()
    faas_instance = _gae.service_instance()

    return _make_resource(
        {
            ResourceAttributes.CLOUD_PLATFORM_KEY: ResourceAttributes.GCP_APP_ENGINE,
            ResourceAttributes.FAAS_NAME: faas_name,
            ResourceAttributes.FAAS_VERSION: faas_version,
            ResourceAttributes.FAAS_INSTANCE: faas_instance,
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE: zone,
            ResourceAttributes.CLOUD_REGION: region,
        }
    )


def _make_resource(attrs: Mapping[str, AttributeValue]) -> Resource:
    return Resource(
        {
            ResourceAttributes.CLOUD_PROVIDER: "gcp",
            ResourceAttributes.CLOUD_ACCOUNT_ID: _metadata.get_metadata()[
                "project"
            ]["projectId"],
            **attrs,
        }
    )


__all__ = ["GoogleCloudResourceDetector"]
