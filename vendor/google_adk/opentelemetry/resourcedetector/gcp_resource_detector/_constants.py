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


# TODO: use opentelemetry-semantic-conventions package for these constants once it has
# stabilized. Right now, pinning an unstable version would cause dependency conflicts for
# users so these are copied in.
class ResourceAttributes:
    AWS_EC2 = "aws_ec2"
    CLOUD_ACCOUNT_ID = "cloud.account.id"
    CLOUD_AVAILABILITY_ZONE = "cloud.availability_zone"
    CLOUD_PLATFORM_KEY = "cloud.platform"
    CLOUD_PROVIDER = "cloud.provider"
    CLOUD_REGION = "cloud.region"
    FAAS_INSTANCE = "faas.instance"
    FAAS_NAME = "faas.name"
    FAAS_VERSION = "faas.version"
    GCP_APP_ENGINE = "gcp_app_engine"
    GCP_CLOUD_FUNCTIONS = "gcp_cloud_functions"
    GCP_CLOUD_RUN = "gcp_cloud_run"
    GCP_COMPUTE_ENGINE = "gcp_compute_engine"
    GCP_KUBERNETES_ENGINE = "gcp_kubernetes_engine"
    HOST_ID = "host.id"
    HOST_NAME = "host.name"
    HOST_TYPE = "host.type"
    K8S_CLUSTER_NAME = "k8s.cluster.name"
    K8S_CONTAINER_NAME = "k8s.container.name"
    K8S_NAMESPACE_NAME = "k8s.namespace.name"
    K8S_NODE_NAME = "k8s.node.name"
    K8S_POD_NAME = "k8s.pod.name"
    SERVICE_INSTANCE_ID = "service.instance.id"
    SERVICE_NAME = "service.name"
    SERVICE_NAMESPACE = "service.namespace"


AWS_ACCOUNT = "aws_account"
AWS_EC2_INSTANCE = "aws_ec2_instance"
CLUSTER_NAME = "cluster_name"
CONTAINER_NAME = "container_name"
GCE_INSTANCE = "gce_instance"
GENERIC_NODE = "generic_node"
GENERIC_TASK = "generic_task"
INSTANCE_ID = "instance_id"
JOB = "job"
K8S_CLUSTER = "k8s_cluster"
K8S_CONTAINER = "k8s_container"
K8S_NODE = "k8s_node"
K8S_POD = "k8s_pod"
LOCATION = "location"
NAMESPACE = "namespace"
NAMESPACE_NAME = "namespace_name"
NODE_ID = "node_id"
NODE_NAME = "node_name"
POD_NAME = "pod_name"
REGION = "region"
TASK_ID = "task_id"
ZONE = "zone"
UNKNOWN_SERVICE_PREFIX = "unknown_service"
