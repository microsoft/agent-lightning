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
import dataclasses
from typing import Dict, List, Optional
from google.cloud.aiplatform_v1beta1.types import PersistentResource


@dataclasses.dataclass
class AutoscalingSpec:
    """Autoscaling spec for a ray cluster node.

    Attributes:
        min_replica_count: The minimum number of replicas in the cluster.
        max_replica_count: The maximum number of replicas in the cluster.
    """

    min_replica_count: int = 1
    max_replica_count: int = 2


@dataclasses.dataclass
class Resources:
    """Resources for a ray cluster node.

    Attributes:
        machine_type: See the list of machine types:
            https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types
        node_count: This argument represents how many nodes to start for the
            ray cluster.
        accelerator_type: e.g. "NVIDIA_TESLA_P4".
            Vertex AI supports the following types of GPU:
            https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus
        accelerator_count: The number of accelerators to attach to the machine.
        boot_disk_type: Type of the boot disk (default is "pd-ssd").
            Valid values: "pd-ssd" (Persistent Disk Solid State Drive) or
            "pd-standard" (Persistent Disk Hard Disk Drive).
        boot_disk_size_gb: Size in GB of the boot disk (default is 100GB). Must
            be either unspecified or within the range of [100, 64000].
        custom_image: Custom image for this resource (e.g.
            us-docker.pkg.dev/my-project/ray-gpu.2-9.py310-tf:latest).
        autoscaling_spec: Autoscaling spec for this resource.
    """

    machine_type: Optional[str] = "n1-standard-16"
    node_count: Optional[int] = 1
    accelerator_type: Optional[str] = None
    accelerator_count: Optional[int] = 0
    boot_disk_type: Optional[str] = "pd-ssd"
    boot_disk_size_gb: Optional[int] = 100
    custom_image: Optional[str] = None
    autoscaling_spec: Optional[AutoscalingSpec] = None


@dataclasses.dataclass
class NodeImages:
    """Custom images for a ray cluster.

    We currently support Ray v2.9, v2.33, v2.42 and python v3.10.
    We also support python v3.11 for Ray v2.42 and v2.47.
    The custom images must be extended from the following base images:
    "{region}-docker.pkg.dev/vertex-ai/training/ray-cpu.2-9.py310:latest",
    "{region}-docker.pkg.dev/vertex-ai/training/ray-gpu.2-9.py310:latest",
    "{region}-docker.pkg.dev/vertex-ai/training/ray-cpu.2-33.py310:latest",
    "{region}-docker.pkg.dev/vertex-ai/training/ray-gpu.2-33.py310:latest",
    "{region}-docker.pkg.dev/vertex-ai/training/ray-cpu.2-42.py310:latest",
    "{region}-docker.pkg.dev/vertex-ai/training/ray-gpu.2-42.py310:latest",
    "{region}-docker.pkg.dev/vertex-ai/training/ray-cpu.2-42.py311:latest",
    "{region}-docker.pkg.dev/vertex-ai/training/ray-gpu.2-42.py311:latest".
    "{region}-docker.pkg.dev/vertex-ai/training/ray-cpu.2-47.py311:latest", or
    "{region}-docker.pkg.dev/vertex-ai/training/ray-gpu.2-47.py311:latest". In
    order to use custom images, need to specify both head and worker images.

    Attributes:
        head: image for head node (eg. us-docker.pkg.dev/my-project/ray-cpu.2-47.py311:latest).
        worker: image for all worker nodes (eg. us-docker.pkg.dev/my-project/ray-gpu.2-47.py311:latest).
    """

    head: str = None
    worker: str = None


@dataclasses.dataclass
class PscIConfig:
    """PSC-I config.

    Attributes:
        network_attachment: Optional. The name or full name of the Compute Engine
            `network attachment <https://cloud.google.com/vpc/docs/about-network-attachments>`
            to attach to the resource. It has a format:
            ``projects/{project}/regions/{region}/networkAttachments/{networkAttachment}``.
            Where {project} is a project number, as in ``12345``, and
            {networkAttachment} is a network attachment name. To specify
            this field, you must have already [created a network
            attachment]
            (https://cloud.google.com/vpc/docs/create-manage-network-attachments#create-network-attachments).
            This field is only used for resources using PSC-I. Make sure you do not
            specify the network here for VPC peering.
    """

    network_attachment: str = None


@dataclasses.dataclass
class NfsMount:
    """NFS mount.

    Attributes:
        server: Required. IP address of the NFS server.
        path: Required. Source path exported from NFS server. Has to start
            with '/', and combined with the ip address, it indicates the
            source mount path in the form of ``server:path``.
        mount_point: Required. Destination mount path. The NFS will be mounted
            for the user under /mnt/nfs/<mount_point>.
    """

    server: str = None
    path: str = None
    mount_point: str = None


@dataclasses.dataclass
class Cluster:
    """Ray cluster (output only).

    Attributes:
        cluster_resource_name: It has a format:
            "projects/<project_num>/locations/<region>/persistentResources/<pr_id>".
        network: Virtual private cloud (VPC) network. It has a format:
            "projects/<project_num>/global/networks/<network_name>".
            For Ray Client, VPC peering is required to connect to the cluster
            managed in the Vertex API service. For Ray Job API, VPC network is
            not required because cluster connection can be accessed through
            dashboard address.
        reserved_ip_ranges: A list of names for the reserved IP ranges under
            the VPC network that can be used for this cluster. If set, we will
            deploy the cluster within the provided IP ranges. Otherwise, the
            cluster is deployed to any IP ranges under the provided VPC network.
            Example: ["vertex-ai-ip-range"].
        service_account: Service account to be used for running Ray programs on
            the cluster.
        state: Describes the cluster state (defined in PersistentResource.State).
        python_version: Python version for the ray cluster (e.g. "3.11").
        ray_version: Ray version for the ray cluster (e.g. "2.47").
        head_node_type: The head node resource. Resources.node_count must be 1.
            If not set, by default it is a CPU node with machine_type of n1-standard-8.
        worker_node_types: The list of Resources of the worker nodes. Should not
            duplicate the elements in the list.
        dashboard_address: For Ray Job API (JobSubmissionClient), with this
           cluster connection doesn't require VPC peering.
        labels:
            The labels with user-defined metadata to organize Ray cluster.

            Label keys and values can be no longer than 64 characters (Unicode
            codepoints), can only contain lowercase letters, numeric characters,
            underscores and dashes. International characters are allowed.

            See https://goo.gl/xmQnxf for more information and examples of labels.
    """

    cluster_resource_name: str = None
    network: str = None
    reserved_ip_ranges: List[str] = None
    service_account: str = None
    state: PersistentResource.State = None
    python_version: str = None
    ray_version: str = None
    head_node_type: Resources = None
    worker_node_types: List[Resources] = None
    dashboard_address: str = None
    ray_metric_enabled: bool = True
    ray_logs_enabled: bool = True
    psc_interface_config: PscIConfig = None
    labels: Dict[str, str] = None


def _check_machine_spec_identical(
    node_type_1: Resources,
    node_type_2: Resources,
) -> int:
    """Check if node_type_1 and node_type_2 have the same machine_spec.
    If they are identical, return additional_replica_count."""
    additional_replica_count = 0

    # Check if machine_spec are the same
    if (
        node_type_1.machine_type == node_type_2.machine_type
        and node_type_1.accelerator_type == node_type_2.accelerator_type
        and node_type_1.accelerator_count == node_type_2.accelerator_count
    ):
        if node_type_1.boot_disk_type != node_type_2.boot_disk_type:
            raise ValueError(
                "Worker disk type must match the head node's disk type if"
                " sharing the same machine_type, accelerator_type, and"
                " accelerator_count"
            )
        if node_type_1.boot_disk_size_gb != node_type_2.boot_disk_size_gb:
            raise ValueError(
                "Worker disk size must match the head node's disk size if"
                " sharing the same machine_type, accelerator_type, and"
                " accelerator_count"
            )
        additional_replica_count = node_type_2.node_count
        return additional_replica_count

    return additional_replica_count
