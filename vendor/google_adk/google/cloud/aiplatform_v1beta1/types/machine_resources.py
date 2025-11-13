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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import (
    accelerator_type as gca_accelerator_type,
)
from google.cloud.aiplatform_v1beta1.types import (
    reservation_affinity as gca_reservation_affinity,
)
from google.protobuf import duration_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "MachineSpec",
        "DedicatedResources",
        "AutomaticResources",
        "BatchDedicatedResources",
        "ResourcesConsumed",
        "DiskSpec",
        "PersistentDiskSpec",
        "NfsMount",
        "AutoscalingMetricSpec",
        "ShieldedVmConfig",
        "FlexStart",
    },
)


class MachineSpec(proto.Message):
    r"""Specification of a single machine.

    Attributes:
        machine_type (str):
            Immutable. The type of the machine.

            See the `list of machine types supported for
            prediction <https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types>`__

            See the `list of machine types supported for custom
            training <https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types>`__.

            For
            [DeployedModel][google.cloud.aiplatform.v1beta1.DeployedModel]
            this field is optional, and the default value is
            ``n1-standard-2``. For
            [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob]
            or as part of
            [WorkerPoolSpec][google.cloud.aiplatform.v1beta1.WorkerPoolSpec]
            this field is required.
        accelerator_type (google.cloud.aiplatform_v1beta1.types.AcceleratorType):
            Immutable. The type of accelerator(s) that may be attached
            to the machine as per
            [accelerator_count][google.cloud.aiplatform.v1beta1.MachineSpec.accelerator_count].
        accelerator_count (int):
            The number of accelerators to attach to the
            machine.
        gpu_partition_size (str):
            Optional. Immutable. The Nvidia GPU partition size.

            When specified, the requested accelerators will be
            partitioned into smaller GPU partitions. For example, if the
            request is for 8 units of NVIDIA A100 GPUs, and
            gpu_partition_size="1g.10gb", the service will create 8 \* 7
            = 56 partitioned MIG instances.

            The partition size must be a value supported by the
            requested accelerator. Refer to `Nvidia GPU
            Partitioning <https://cloud.google.com/kubernetes-engine/docs/how-to/gpus-multi#multi-instance_gpu_partitions>`__
            for the available partition sizes.

            If set, the accelerator_count should be set to 1.
        tpu_topology (str):
            Immutable. The topology of the TPUs. Corresponds to the TPU
            topologies available from GKE. (Example: tpu_topology:
            "2x2x1").
        multihost_gpu_node_count (int):
            Optional. Immutable. The number of nodes per
            replica for multihost GPU deployments.
        reservation_affinity (google.cloud.aiplatform_v1beta1.types.ReservationAffinity):
            Optional. Immutable. Configuration
            controlling how this resource pool consumes
            reservation.
    """

    machine_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    accelerator_type: gca_accelerator_type.AcceleratorType = proto.Field(
        proto.ENUM,
        number=2,
        enum=gca_accelerator_type.AcceleratorType,
    )
    accelerator_count: int = proto.Field(
        proto.INT32,
        number=3,
    )
    gpu_partition_size: str = proto.Field(
        proto.STRING,
        number=7,
    )
    tpu_topology: str = proto.Field(
        proto.STRING,
        number=4,
    )
    multihost_gpu_node_count: int = proto.Field(
        proto.INT32,
        number=6,
    )
    reservation_affinity: gca_reservation_affinity.ReservationAffinity = proto.Field(
        proto.MESSAGE,
        number=5,
        message=gca_reservation_affinity.ReservationAffinity,
    )


class DedicatedResources(proto.Message):
    r"""A description of resources that are dedicated to a
    DeployedModel or DeployedIndex, and that need a higher degree of
    manual configuration.

    Attributes:
        machine_spec (google.cloud.aiplatform_v1beta1.types.MachineSpec):
            Required. Immutable. The specification of a
            single machine being used.
        min_replica_count (int):
            Required. Immutable. The minimum number of
            machine replicas that will be always deployed
            on. This value must be greater than or equal to
            1.

            If traffic increases, it may dynamically be
            deployed onto more replicas, and as traffic
            decreases, some of these extra replicas may be
            freed.
        max_replica_count (int):
            Immutable. The maximum number of replicas that may be
            deployed on when the traffic against it increases. If the
            requested value is too large, the deployment will error, but
            if deployment succeeds then the ability to scale to that
            many replicas is guaranteed (barring service outages). If
            traffic increases beyond what its replicas at maximum may
            handle, a portion of the traffic will be dropped. If this
            value is not provided, will use
            [min_replica_count][google.cloud.aiplatform.v1beta1.DedicatedResources.min_replica_count]
            as the default value.

            The value of this field impacts the charge against Vertex
            CPU and GPU quotas. Specifically, you will be charged for
            (max_replica_count \* number of cores in the selected
            machine type) and (max_replica_count \* number of GPUs per
            replica in the selected machine type).
        required_replica_count (int):
            Optional. Number of required available replicas for the
            deployment to succeed. This field is only needed when
            partial deployment/mutation is desired. If set, the
            deploy/mutate operation will succeed once
            available_replica_count reaches required_replica_count, and
            the rest of the replicas will be retried. If not set, the
            default required_replica_count will be min_replica_count.
        initial_replica_count (int):
            Immutable. Number of initial replicas being deployed on when
            scaling the workload up from zero or when creating the
            workload in case
            [min_replica_count][google.cloud.aiplatform.v1beta1.DedicatedResources.min_replica_count]
            = 0. When
            [min_replica_count][google.cloud.aiplatform.v1beta1.DedicatedResources.min_replica_count]

               0 (meaning that the scale-to-zero feature is not
               enabled),
               [initial_replica_count][google.cloud.aiplatform.v1beta1.DedicatedResources.initial_replica_count]
               should not be set. When
               [min_replica_count][google.cloud.aiplatform.v1beta1.DedicatedResources.min_replica_count]
               = 0 (meaning that the scale-to-zero feature is enabled),
               [initial_replica_count][google.cloud.aiplatform.v1beta1.DedicatedResources.initial_replica_count]
               should be larger than zero, but no greater than
               [max_replica_count][google.cloud.aiplatform.v1beta1.DedicatedResources.max_replica_count].
        autoscaling_metric_specs (MutableSequence[google.cloud.aiplatform_v1beta1.types.AutoscalingMetricSpec]):
            Immutable. The metric specifications that overrides a
            resource utilization metric (CPU utilization, accelerator's
            duty cycle, and so on) target value (default to 60 if not
            set). At most one entry is allowed per metric.

            If
            [machine_spec.accelerator_count][google.cloud.aiplatform.v1beta1.MachineSpec.accelerator_count]
            is above 0, the autoscaling will be based on both CPU
            utilization and accelerator's duty cycle metrics and scale
            up when either metrics exceeds its target value while scale
            down if both metrics are under their target value. The
            default target value is 60 for both metrics.

            If
            [machine_spec.accelerator_count][google.cloud.aiplatform.v1beta1.MachineSpec.accelerator_count]
            is 0, the autoscaling will be based on CPU utilization
            metric only with default target value 60 if not explicitly
            set.

            For example, in the case of Online Prediction, if you want
            to override target CPU utilization to 80, you should set
            [autoscaling_metric_specs.metric_name][google.cloud.aiplatform.v1beta1.AutoscalingMetricSpec.metric_name]
            to
            ``aiplatform.googleapis.com/prediction/online/cpu/utilization``
            and
            [autoscaling_metric_specs.target][google.cloud.aiplatform.v1beta1.AutoscalingMetricSpec.target]
            to ``80``.
        spot (bool):
            Optional. If true, schedule the deployment workload on `spot
            VMs <https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms>`__.
        flex_start (google.cloud.aiplatform_v1beta1.types.FlexStart):
            Optional. Immutable. If set, use DWS resource
            to schedule the deployment workload. reference:

            (https://cloud.google.com/blog/products/compute/introducing-dynamic-workload-scheduler)
        scale_to_zero_spec (google.cloud.aiplatform_v1beta1.types.DedicatedResources.ScaleToZeroSpec):
            Optional. Specification for scale-to-zero
            feature.
    """

    class ScaleToZeroSpec(proto.Message):
        r"""Specification for scale-to-zero feature.

        Attributes:
            min_scaleup_period (google.protobuf.duration_pb2.Duration):
                Optional. Minimum duration that a deployment will be scaled
                up before traffic is evaluated for potential scale-down.
                [MinValue=300] (5 minutes) [MaxValue=28800] (8 hours)
            idle_scaledown_period (google.protobuf.duration_pb2.Duration):
                Optional. Duration of no traffic before scaling to zero.
                [MinValue=3600] (5 minutes) [MaxValue=28800] (8 hours)
        """

        min_scaleup_period: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=1,
            message=duration_pb2.Duration,
        )
        idle_scaledown_period: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=2,
            message=duration_pb2.Duration,
        )

    machine_spec: "MachineSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="MachineSpec",
    )
    min_replica_count: int = proto.Field(
        proto.INT32,
        number=2,
    )
    max_replica_count: int = proto.Field(
        proto.INT32,
        number=3,
    )
    required_replica_count: int = proto.Field(
        proto.INT32,
        number=9,
    )
    initial_replica_count: int = proto.Field(
        proto.INT32,
        number=6,
    )
    autoscaling_metric_specs: MutableSequence["AutoscalingMetricSpec"] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=4,
            message="AutoscalingMetricSpec",
        )
    )
    spot: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    flex_start: "FlexStart" = proto.Field(
        proto.MESSAGE,
        number=10,
        message="FlexStart",
    )
    scale_to_zero_spec: ScaleToZeroSpec = proto.Field(
        proto.MESSAGE,
        number=11,
        message=ScaleToZeroSpec,
    )


class AutomaticResources(proto.Message):
    r"""A description of resources that to large degree are decided
    by Vertex AI, and require only a modest additional
    configuration. Each Model supporting these resources documents
    its specific guidelines.

    Attributes:
        min_replica_count (int):
            Immutable. The minimum number of replicas that will be
            always deployed on. If traffic against it increases, it may
            dynamically be deployed onto more replicas up to
            [max_replica_count][google.cloud.aiplatform.v1beta1.AutomaticResources.max_replica_count],
            and as traffic decreases, some of these extra replicas may
            be freed. If the requested value is too large, the
            deployment will error.
        max_replica_count (int):
            Immutable. The maximum number of replicas
            that may be deployed on when the traffic against
            it increases. If the requested value is too
            large, the deployment will error, but if
            deployment succeeds then the ability to scale to
            that many replicas is guaranteed (barring
            service outages). If traffic increases beyond
            what its replicas at maximum may handle, a
            portion of the traffic will be dropped. If this
            value is not provided, a no upper bound for
            scaling under heavy traffic will be assume,
            though Vertex AI may be unable to scale beyond
            certain replica number.
    """

    min_replica_count: int = proto.Field(
        proto.INT32,
        number=1,
    )
    max_replica_count: int = proto.Field(
        proto.INT32,
        number=2,
    )


class BatchDedicatedResources(proto.Message):
    r"""A description of resources that are used for performing batch
    operations, are dedicated to a Model, and need manual
    configuration.

    Attributes:
        machine_spec (google.cloud.aiplatform_v1beta1.types.MachineSpec):
            Required. Immutable. The specification of a
            single machine.
        starting_replica_count (int):
            Immutable. The number of machine replicas used at the start
            of the batch operation. If not set, Vertex AI decides
            starting number, not greater than
            [max_replica_count][google.cloud.aiplatform.v1beta1.BatchDedicatedResources.max_replica_count]
        max_replica_count (int):
            Immutable. The maximum number of machine
            replicas the batch operation may be scaled to.
            The default value is 10.
        flex_start (google.cloud.aiplatform_v1beta1.types.FlexStart):
            Optional. Immutable. If set, use DWS resource
            to schedule the deployment workload. reference:

            (https://cloud.google.com/blog/products/compute/introducing-dynamic-workload-scheduler)
        spot (bool):
            Optional. If true, schedule the deployment workload on `spot
            VMs <https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms>`__.
    """

    machine_spec: "MachineSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="MachineSpec",
    )
    starting_replica_count: int = proto.Field(
        proto.INT32,
        number=2,
    )
    max_replica_count: int = proto.Field(
        proto.INT32,
        number=3,
    )
    flex_start: "FlexStart" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="FlexStart",
    )
    spot: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


class ResourcesConsumed(proto.Message):
    r"""Statistics information about resource consumption.

    Attributes:
        replica_hours (float):
            Output only. The number of replica hours
            used. Note that many replicas may run in
            parallel, and additionally any given work may be
            queued for some time. Therefore this value is
            not strictly related to wall time.
    """

    replica_hours: float = proto.Field(
        proto.DOUBLE,
        number=1,
    )


class DiskSpec(proto.Message):
    r"""Represents the spec of disk options.

    Attributes:
        boot_disk_type (str):
            Type of the boot disk. For non-A3U machines,
            the default value is "pd-ssd", for A3U machines,
            the default value is "hyperdisk-balanced". Valid
            values: "pd-ssd" (Persistent Disk Solid State
            Drive), "pd-standard" (Persistent Disk Hard Disk
            Drive) or "hyperdisk-balanced".
        boot_disk_size_gb (int):
            Size in GB of the boot disk (default is
            100GB).
    """

    boot_disk_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    boot_disk_size_gb: int = proto.Field(
        proto.INT32,
        number=2,
    )


class PersistentDiskSpec(proto.Message):
    r"""Represents the spec of [persistent
    disk][https://cloud.google.com/compute/docs/disks/persistent-disks]
    options.

    Attributes:
        disk_type (str):
            Type of the disk (default is "pd-standard").
            Valid values: "pd-ssd" (Persistent Disk Solid
            State Drive) "pd-standard" (Persistent Disk Hard
            Disk Drive) "pd-balanced" (Balanced Persistent
            Disk)
            "pd-extreme" (Extreme Persistent Disk)
        disk_size_gb (int):
            Size in GB of the disk (default is 100GB).
    """

    disk_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    disk_size_gb: int = proto.Field(
        proto.INT64,
        number=2,
    )


class NfsMount(proto.Message):
    r"""Represents a mount configuration for Network File System
    (NFS) to mount.

    Attributes:
        server (str):
            Required. IP address of the NFS server.
        path (str):
            Required. Source path exported from NFS server. Has to start
            with '/', and combined with the ip address, it indicates the
            source mount path in the form of ``server:path``
        mount_point (str):
            Required. Destination mount path. The NFS will be mounted
            for the user under /mnt/nfs/<mount_point>
    """

    server: str = proto.Field(
        proto.STRING,
        number=1,
    )
    path: str = proto.Field(
        proto.STRING,
        number=2,
    )
    mount_point: str = proto.Field(
        proto.STRING,
        number=3,
    )


class AutoscalingMetricSpec(proto.Message):
    r"""The metric specification that defines the target resource
    utilization (CPU utilization, accelerator's duty cycle, and so
    on) for calculating the desired replica count.

    Attributes:
        metric_name (str):
            Required. The resource metric name. Supported metrics:

            - For Online Prediction:
            - ``aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle``
            - ``aiplatform.googleapis.com/prediction/online/cpu/utilization``
            - ``aiplatform.googleapis.com/prediction/online/request_count``
        target (int):
            The target resource utilization in percentage
            (1% - 100%) for the given metric; once the real
            usage deviates from the target by a certain
            percentage, the machine replicas change. The
            default value is 60 (representing 60%) if not
            provided.
        monitored_resource_labels (MutableMapping[str, str]):
            Optional. The Cloud Monitoring monitored
            resource labels as key value pairs used for
            metrics filtering. See Cloud Monitoring Labels
            https://cloud.google.com/monitoring/api/v3/metric-model#generic-label-info
    """

    metric_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    target: int = proto.Field(
        proto.INT32,
        number=2,
    )
    monitored_resource_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=3,
    )


class ShieldedVmConfig(proto.Message):
    r"""A set of Shielded Instance options. See `Images using supported
    Shielded VM
    features <https://cloud.google.com/compute/docs/instances/modifying-shielded-vm>`__.

    Attributes:
        enable_secure_boot (bool):
            Defines whether the instance has `Secure
            Boot <https://cloud.google.com/compute/shielded-vm/docs/shielded-vm#secure-boot>`__
            enabled.

            Secure Boot helps ensure that the system only runs authentic
            software by verifying the digital signature of all boot
            components, and halting the boot process if signature
            verification fails.
    """

    enable_secure_boot: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


class FlexStart(proto.Message):
    r"""FlexStart is used to schedule the deployment workload on DWS
    resource. It contains the max duration of the deployment.

    Attributes:
        max_runtime_duration (google.protobuf.duration_pb2.Duration):
            The max duration of the deployment is max_runtime_duration.
            The deployment will be terminated after the duration. The
            max_runtime_duration can be set up to 7 days.
    """

    max_runtime_duration: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
