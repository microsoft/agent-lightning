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

import json
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from opentelemetry.resourcedetector.gcp_resource_detector import _constants
from opentelemetry.resourcedetector.gcp_resource_detector._constants import (
    ResourceAttributes,
)
from opentelemetry.sdk.resources import Attributes, Resource


class MapConfig:
    otel_keys: Tuple[str, ...]
    """
    OTel resource keys to try and populate the resource label from. For entries with multiple
    OTel resource keys, the keys' values will be coalesced in order until there is a non-empty
    value.
    """

    fallback: str
    """If none of the otelKeys are present in the Resource, fallback to this literal value"""

    def __init__(self, *otel_keys: str, fallback: str = ""):
        self.otel_keys = otel_keys
        self.fallback = fallback


# Mappings of GCM resource label keys onto mapping config from OTel resource for a given
# monitored resource type. Copied from Go impl:
# https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/blob/v1.8.0/internal/resourcemapping/resourcemapping.go#L51
MAPPINGS = {
    _constants.GCE_INSTANCE: {
        _constants.ZONE: MapConfig(ResourceAttributes.CLOUD_AVAILABILITY_ZONE),
        _constants.INSTANCE_ID: MapConfig(ResourceAttributes.HOST_ID),
    },
    _constants.K8S_CONTAINER: {
        _constants.LOCATION: MapConfig(
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE,
            ResourceAttributes.CLOUD_REGION,
        ),
        _constants.CLUSTER_NAME: MapConfig(
            ResourceAttributes.K8S_CLUSTER_NAME
        ),
        _constants.NAMESPACE_NAME: MapConfig(
            ResourceAttributes.K8S_NAMESPACE_NAME
        ),
        _constants.POD_NAME: MapConfig(ResourceAttributes.K8S_POD_NAME),
        _constants.CONTAINER_NAME: MapConfig(
            ResourceAttributes.K8S_CONTAINER_NAME
        ),
    },
    _constants.K8S_POD: {
        _constants.LOCATION: MapConfig(
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE,
            ResourceAttributes.CLOUD_REGION,
        ),
        _constants.CLUSTER_NAME: MapConfig(
            ResourceAttributes.K8S_CLUSTER_NAME
        ),
        _constants.NAMESPACE_NAME: MapConfig(
            ResourceAttributes.K8S_NAMESPACE_NAME
        ),
        _constants.POD_NAME: MapConfig(ResourceAttributes.K8S_POD_NAME),
    },
    _constants.K8S_NODE: {
        _constants.LOCATION: MapConfig(
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE,
            ResourceAttributes.CLOUD_REGION,
        ),
        _constants.CLUSTER_NAME: MapConfig(
            ResourceAttributes.K8S_CLUSTER_NAME
        ),
        _constants.NODE_NAME: MapConfig(ResourceAttributes.K8S_NODE_NAME),
    },
    _constants.K8S_CLUSTER: {
        _constants.LOCATION: MapConfig(
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE,
            ResourceAttributes.CLOUD_REGION,
        ),
        _constants.CLUSTER_NAME: MapConfig(
            ResourceAttributes.K8S_CLUSTER_NAME
        ),
    },
    _constants.AWS_EC2_INSTANCE: {
        _constants.INSTANCE_ID: MapConfig(ResourceAttributes.HOST_ID),
        _constants.REGION: MapConfig(
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE,
            ResourceAttributes.CLOUD_REGION,
        ),
        _constants.AWS_ACCOUNT: MapConfig(ResourceAttributes.CLOUD_ACCOUNT_ID),
    },
    _constants.GENERIC_TASK: {
        _constants.LOCATION: MapConfig(
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE,
            ResourceAttributes.CLOUD_REGION,
            fallback="global",
        ),
        _constants.NAMESPACE: MapConfig(ResourceAttributes.SERVICE_NAMESPACE),
        _constants.JOB: MapConfig(
            ResourceAttributes.SERVICE_NAME,
            ResourceAttributes.FAAS_NAME,
        ),
        _constants.TASK_ID: MapConfig(
            ResourceAttributes.SERVICE_INSTANCE_ID,
            ResourceAttributes.FAAS_INSTANCE,
        ),
    },
    _constants.GENERIC_NODE: {
        _constants.LOCATION: MapConfig(
            ResourceAttributes.CLOUD_AVAILABILITY_ZONE,
            ResourceAttributes.CLOUD_REGION,
            fallback="global",
        ),
        _constants.NAMESPACE: MapConfig(ResourceAttributes.SERVICE_NAMESPACE),
        _constants.NODE_ID: MapConfig(
            ResourceAttributes.HOST_ID, ResourceAttributes.HOST_NAME
        ),
    },
}


@dataclass
class MonitoredResourceData:
    """Dataclass representing a protobuf monitored resource. Make sure to convert to a protobuf
    if needed."""

    type: str
    labels: Mapping[str, str]


def get_monitored_resource(
    resource: Resource,
) -> Optional[MonitoredResourceData]:
    """Add Google resource specific information (e.g. instance id, region).

    See
    https://cloud.google.com/monitoring/custom-metrics/creating-metrics#custom-metric-resources
    for supported types
    Args:
            resource: OTel resource
    """

    attrs = resource.attributes

    platform = attrs.get(ResourceAttributes.CLOUD_PLATFORM_KEY)
    if platform == ResourceAttributes.GCP_COMPUTE_ENGINE:
        mr = _create_monitored_resource(_constants.GCE_INSTANCE, attrs)
    elif platform == ResourceAttributes.GCP_KUBERNETES_ENGINE:
        if ResourceAttributes.K8S_CONTAINER_NAME in attrs:
            mr = _create_monitored_resource(_constants.K8S_CONTAINER, attrs)
        elif ResourceAttributes.K8S_POD_NAME in attrs:
            mr = _create_monitored_resource(_constants.K8S_POD, attrs)
        elif ResourceAttributes.K8S_NODE_NAME in attrs:
            mr = _create_monitored_resource(_constants.K8S_NODE, attrs)
        else:
            mr = _create_monitored_resource(_constants.K8S_CLUSTER, attrs)
    elif platform == ResourceAttributes.AWS_EC2:
        mr = _create_monitored_resource(_constants.AWS_EC2_INSTANCE, attrs)
    else:
        # fallback to generic_task
        if (
            ResourceAttributes.SERVICE_NAME in attrs
            or ResourceAttributes.FAAS_NAME in attrs
        ) and (
            ResourceAttributes.SERVICE_INSTANCE_ID in attrs
            or ResourceAttributes.FAAS_INSTANCE in attrs
        ):
            mr = _create_monitored_resource(_constants.GENERIC_TASK, attrs)
        else:
            mr = _create_monitored_resource(_constants.GENERIC_NODE, attrs)

    return mr


def _create_monitored_resource(
    monitored_resource_type: str, resource_attrs: Attributes
) -> MonitoredResourceData:
    mapping = MAPPINGS[monitored_resource_type]
    labels: Dict[str, str] = {}

    for mr_key, map_config in mapping.items():
        mr_value = None
        for otel_key in map_config.otel_keys:
            if otel_key in resource_attrs and not str(
                resource_attrs[otel_key]
            ).startswith(_constants.UNKNOWN_SERVICE_PREFIX):
                mr_value = resource_attrs[otel_key]
                break

        if (
            mr_value is None
            and ResourceAttributes.SERVICE_NAME in map_config.otel_keys
        ):
            # The service name started with unknown_service, and was ignored above.
            mr_value = resource_attrs.get(ResourceAttributes.SERVICE_NAME)

        if mr_value is None:
            mr_value = map_config.fallback

        # OTel attribute values can be any of str, bool, int, float, or Sequence of any of
        # them. Encode any non-strings as json string
        if not isinstance(mr_value, str):
            mr_value = json.dumps(
                mr_value, sort_keys=True, indent=None, separators=(",", ":")
            )
        labels[mr_key] = mr_value

    return MonitoredResourceData(type=monitored_resource_type, labels=labels)
