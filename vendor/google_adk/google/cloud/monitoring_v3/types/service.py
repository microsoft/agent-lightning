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

from google.protobuf import duration_pb2  # type: ignore
from google.type import calendar_period_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "Service",
        "ServiceLevelObjective",
        "ServiceLevelIndicator",
        "BasicSli",
        "Range",
        "RequestBasedSli",
        "TimeSeriesRatio",
        "DistributionCut",
        "WindowsBasedSli",
    },
)


class Service(proto.Message):
    r"""A ``Service`` is a discrete, autonomous, and network-accessible
    unit, designed to solve an individual concern
    (`Wikipedia <https://en.wikipedia.org/wiki/Service-orientation>`__).
    In Cloud Monitoring, a ``Service`` acts as the root resource under
    which operational aspects of the service are accessible.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Identifier. Resource name for this Service. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
        display_name (str):
            Name used for UI elements listing this
            Service.
        custom (google.cloud.monitoring_v3.types.Service.Custom):
            Custom service type.

            This field is a member of `oneof`_ ``identifier``.
        app_engine (google.cloud.monitoring_v3.types.Service.AppEngine):
            Type used for App Engine services.

            This field is a member of `oneof`_ ``identifier``.
        cloud_endpoints (google.cloud.monitoring_v3.types.Service.CloudEndpoints):
            Type used for Cloud Endpoints services.

            This field is a member of `oneof`_ ``identifier``.
        cluster_istio (google.cloud.monitoring_v3.types.Service.ClusterIstio):
            Type used for Istio services that live in a
            Kubernetes cluster.

            This field is a member of `oneof`_ ``identifier``.
        mesh_istio (google.cloud.monitoring_v3.types.Service.MeshIstio):
            Type used for Istio services scoped to an
            Istio mesh.

            This field is a member of `oneof`_ ``identifier``.
        istio_canonical_service (google.cloud.monitoring_v3.types.Service.IstioCanonicalService):
            Type used for canonical services scoped to an Istio mesh.
            Metrics for Istio are `documented
            here <https://istio.io/latest/docs/reference/config/metrics/>`__

            This field is a member of `oneof`_ ``identifier``.
        cloud_run (google.cloud.monitoring_v3.types.Service.CloudRun):
            Type used for Cloud Run services.

            This field is a member of `oneof`_ ``identifier``.
        gke_namespace (google.cloud.monitoring_v3.types.Service.GkeNamespace):
            Type used for GKE Namespaces.

            This field is a member of `oneof`_ ``identifier``.
        gke_workload (google.cloud.monitoring_v3.types.Service.GkeWorkload):
            Type used for GKE Workloads.

            This field is a member of `oneof`_ ``identifier``.
        gke_service (google.cloud.monitoring_v3.types.Service.GkeService):
            Type used for GKE Services (the Kubernetes
            concept of a service).

            This field is a member of `oneof`_ ``identifier``.
        basic_service (google.cloud.monitoring_v3.types.Service.BasicService):
            Message that contains the service type and service labels of
            this service if it is a basic service. Documentation and
            examples
            `here <https://cloud.google.com/stackdriver/docs/solutions/slo-monitoring/api/api-structures#basic-svc-w-basic-sli>`__.
        telemetry (google.cloud.monitoring_v3.types.Service.Telemetry):
            Configuration for how to query telemetry on a
            Service.
        user_labels (MutableMapping[str, str]):
            Labels which have been used to annotate the
            service. Label keys must start with a letter.
            Label keys and values may contain lowercase
            letters, numbers, underscores, and dashes. Label
            keys and values have a maximum length of 63
            characters, and must be less than 128 bytes in
            size. Up to 64 label entries may be stored. For
            labels which do not have a semantic value, the
            empty string may be supplied for the label
            value.
    """

    class Custom(proto.Message):
        r"""Use a custom service to designate a service that you want to
        monitor when none of the other service types (like App Engine,
        Cloud Run, or a GKE type) matches your intended service.

        """

    class AppEngine(proto.Message):
        r"""App Engine service. Learn more at
        https://cloud.google.com/appengine.

        Attributes:
            module_id (str):
                The ID of the App Engine module underlying this service.
                Corresponds to the ``module_id`` resource label in the
                ```gae_app`` monitored
                resource <https://cloud.google.com/monitoring/api/resources#tag_gae_app>`__.
        """

        module_id: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class CloudEndpoints(proto.Message):
        r"""Cloud Endpoints service. Learn more at
        https://cloud.google.com/endpoints.

        Attributes:
            service (str):
                The name of the Cloud Endpoints service underlying this
                service. Corresponds to the ``service`` resource label in
                the ```api`` monitored
                resource <https://cloud.google.com/monitoring/api/resources#tag_api>`__.
        """

        service: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class ClusterIstio(proto.Message):
        r"""Istio service scoped to a single Kubernetes cluster. Learn
        more at https://istio.io. Clusters running OSS Istio will have
        their services ingested as this type.

        Attributes:
            location (str):
                The location of the Kubernetes cluster in which this Istio
                service is defined. Corresponds to the ``location`` resource
                label in ``k8s_cluster`` resources.
            cluster_name (str):
                The name of the Kubernetes cluster in which this Istio
                service is defined. Corresponds to the ``cluster_name``
                resource label in ``k8s_cluster`` resources.
            service_namespace (str):
                The namespace of the Istio service underlying this service.
                Corresponds to the ``destination_service_namespace`` metric
                label in Istio metrics.
            service_name (str):
                The name of the Istio service underlying this service.
                Corresponds to the ``destination_service_name`` metric label
                in Istio metrics.
        """

        location: str = proto.Field(
            proto.STRING,
            number=1,
        )
        cluster_name: str = proto.Field(
            proto.STRING,
            number=2,
        )
        service_namespace: str = proto.Field(
            proto.STRING,
            number=3,
        )
        service_name: str = proto.Field(
            proto.STRING,
            number=4,
        )

    class MeshIstio(proto.Message):
        r"""Istio service scoped to an Istio mesh. Anthos clusters
        running ASM < 1.6.8 will have their services ingested as this
        type.

        Attributes:
            mesh_uid (str):
                Identifier for the mesh in which this Istio service is
                defined. Corresponds to the ``mesh_uid`` metric label in
                Istio metrics.
            service_namespace (str):
                The namespace of the Istio service underlying this service.
                Corresponds to the ``destination_service_namespace`` metric
                label in Istio metrics.
            service_name (str):
                The name of the Istio service underlying this service.
                Corresponds to the ``destination_service_name`` metric label
                in Istio metrics.
        """

        mesh_uid: str = proto.Field(
            proto.STRING,
            number=1,
        )
        service_namespace: str = proto.Field(
            proto.STRING,
            number=3,
        )
        service_name: str = proto.Field(
            proto.STRING,
            number=4,
        )

    class IstioCanonicalService(proto.Message):
        r"""Canonical service scoped to an Istio mesh. Anthos clusters
        running ASM >= 1.6.8 will have their services ingested as this
        type.

        Attributes:
            mesh_uid (str):
                Identifier for the Istio mesh in which this canonical
                service is defined. Corresponds to the ``mesh_uid`` metric
                label in `Istio
                metrics <https://cloud.google.com/monitoring/api/metrics_istio>`__.
            canonical_service_namespace (str):
                The namespace of the canonical service underlying this
                service. Corresponds to the
                ``destination_canonical_service_namespace`` metric label in
                `Istio
                metrics <https://cloud.google.com/monitoring/api/metrics_istio>`__.
            canonical_service (str):
                The name of the canonical service underlying this service.
                Corresponds to the ``destination_canonical_service_name``
                metric label in label in `Istio
                metrics <https://cloud.google.com/monitoring/api/metrics_istio>`__.
        """

        mesh_uid: str = proto.Field(
            proto.STRING,
            number=1,
        )
        canonical_service_namespace: str = proto.Field(
            proto.STRING,
            number=3,
        )
        canonical_service: str = proto.Field(
            proto.STRING,
            number=4,
        )

    class CloudRun(proto.Message):
        r"""Cloud Run service. Learn more at
        https://cloud.google.com/run.

        Attributes:
            service_name (str):
                The name of the Cloud Run service. Corresponds to the
                ``service_name`` resource label in the
                ```cloud_run_revision`` monitored
                resource <https://cloud.google.com/monitoring/api/resources#tag_cloud_run_revision>`__.
            location (str):
                The location the service is run. Corresponds to the
                ``location`` resource label in the ```cloud_run_revision``
                monitored
                resource <https://cloud.google.com/monitoring/api/resources#tag_cloud_run_revision>`__.
        """

        service_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        location: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class GkeNamespace(proto.Message):
        r"""GKE Namespace. The field names correspond to the resource metadata
        labels on monitored resources that fall under a namespace (for
        example, ``k8s_container`` or ``k8s_pod``).

        Attributes:
            project_id (str):
                Output only. The project this resource lives in. For legacy
                services migrated from the ``Custom`` type, this may be a
                distinct project from the one parenting the service itself.
            location (str):
                The location of the parent cluster. This may
                be a zone or region.
            cluster_name (str):
                The name of the parent cluster.
            namespace_name (str):
                The name of this namespace.
        """

        project_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        location: str = proto.Field(
            proto.STRING,
            number=2,
        )
        cluster_name: str = proto.Field(
            proto.STRING,
            number=3,
        )
        namespace_name: str = proto.Field(
            proto.STRING,
            number=4,
        )

    class GkeWorkload(proto.Message):
        r"""A GKE Workload (Deployment, StatefulSet, etc). The field names
        correspond to the metadata labels on monitored resources that fall
        under a workload (for example, ``k8s_container`` or ``k8s_pod``).

        Attributes:
            project_id (str):
                Output only. The project this resource lives in. For legacy
                services migrated from the ``Custom`` type, this may be a
                distinct project from the one parenting the service itself.
            location (str):
                The location of the parent cluster. This may
                be a zone or region.
            cluster_name (str):
                The name of the parent cluster.
            namespace_name (str):
                The name of the parent namespace.
            top_level_controller_type (str):
                The type of this workload (for example,
                "Deployment" or "DaemonSet")
            top_level_controller_name (str):
                The name of this workload.
        """

        project_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        location: str = proto.Field(
            proto.STRING,
            number=2,
        )
        cluster_name: str = proto.Field(
            proto.STRING,
            number=3,
        )
        namespace_name: str = proto.Field(
            proto.STRING,
            number=4,
        )
        top_level_controller_type: str = proto.Field(
            proto.STRING,
            number=5,
        )
        top_level_controller_name: str = proto.Field(
            proto.STRING,
            number=6,
        )

    class GkeService(proto.Message):
        r"""GKE Service. The "service" here represents a `Kubernetes service
        object <https://kubernetes.io/docs/concepts/services-networking/service>`__.
        The field names correspond to the resource labels on
        ```k8s_service`` monitored
        resources <https://cloud.google.com/monitoring/api/resources#tag_k8s_service>`__.

        Attributes:
            project_id (str):
                Output only. The project this resource lives in. For legacy
                services migrated from the ``Custom`` type, this may be a
                distinct project from the one parenting the service itself.
            location (str):
                The location of the parent cluster. This may
                be a zone or region.
            cluster_name (str):
                The name of the parent cluster.
            namespace_name (str):
                The name of the parent namespace.
            service_name (str):
                The name of this service.
        """

        project_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        location: str = proto.Field(
            proto.STRING,
            number=2,
        )
        cluster_name: str = proto.Field(
            proto.STRING,
            number=3,
        )
        namespace_name: str = proto.Field(
            proto.STRING,
            number=4,
        )
        service_name: str = proto.Field(
            proto.STRING,
            number=5,
        )

    class BasicService(proto.Message):
        r"""A well-known service type, defined by its service type and service
        labels. Documentation and examples
        `here <https://cloud.google.com/stackdriver/docs/solutions/slo-monitoring/api/api-structures#basic-svc-w-basic-sli>`__.

        Attributes:
            service_type (str):
                The type of service that this basic service defines, e.g.
                APP_ENGINE service type. Documentation and valid values
                `here <https://cloud.google.com/stackdriver/docs/solutions/slo-monitoring/api/api-structures#basic-svc-w-basic-sli>`__.
            service_labels (MutableMapping[str, str]):
                Labels that specify the resource that emits the monitoring
                data which is used for SLO reporting of this ``Service``.
                Documentation and valid values for given service types
                `here <https://cloud.google.com/stackdriver/docs/solutions/slo-monitoring/api/api-structures#basic-svc-w-basic-sli>`__.
        """

        service_type: str = proto.Field(
            proto.STRING,
            number=1,
        )
        service_labels: MutableMapping[str, str] = proto.MapField(
            proto.STRING,
            proto.STRING,
            number=2,
        )

    class Telemetry(proto.Message):
        r"""Configuration for how to query telemetry on a Service.

        Attributes:
            resource_name (str):
                The full name of the resource that defines this service.
                Formatted as described in
                https://cloud.google.com/apis/design/resource_names.
        """

        resource_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    custom: Custom = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="identifier",
        message=Custom,
    )
    app_engine: AppEngine = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="identifier",
        message=AppEngine,
    )
    cloud_endpoints: CloudEndpoints = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="identifier",
        message=CloudEndpoints,
    )
    cluster_istio: ClusterIstio = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="identifier",
        message=ClusterIstio,
    )
    mesh_istio: MeshIstio = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="identifier",
        message=MeshIstio,
    )
    istio_canonical_service: IstioCanonicalService = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="identifier",
        message=IstioCanonicalService,
    )
    cloud_run: CloudRun = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="identifier",
        message=CloudRun,
    )
    gke_namespace: GkeNamespace = proto.Field(
        proto.MESSAGE,
        number=15,
        oneof="identifier",
        message=GkeNamespace,
    )
    gke_workload: GkeWorkload = proto.Field(
        proto.MESSAGE,
        number=16,
        oneof="identifier",
        message=GkeWorkload,
    )
    gke_service: GkeService = proto.Field(
        proto.MESSAGE,
        number=17,
        oneof="identifier",
        message=GkeService,
    )
    basic_service: BasicService = proto.Field(
        proto.MESSAGE,
        number=19,
        message=BasicService,
    )
    telemetry: Telemetry = proto.Field(
        proto.MESSAGE,
        number=13,
        message=Telemetry,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=14,
    )


class ServiceLevelObjective(proto.Message):
    r"""A Service-Level Objective (SLO) describes a level of desired
    good service. It consists of a service-level indicator (SLI), a
    performance goal, and a period over which the objective is to be
    evaluated against that goal. The SLO can use SLIs defined in a
    number of different manners. Typical SLOs might include "99% of
    requests in each rolling week have latency below 200
    milliseconds" or "99.5% of requests in each calendar month
    return successfully."

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Identifier. Resource name for this
            ``ServiceLevelObjective``. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]/serviceLevelObjectives/[SLO_NAME]
        display_name (str):
            Name used for UI elements listing this SLO.
        service_level_indicator (google.cloud.monitoring_v3.types.ServiceLevelIndicator):
            The definition of good service, used to measure and
            calculate the quality of the ``Service``'s performance with
            respect to a single aspect of service quality.
        goal (float):
            The fraction of service that must be good in order for this
            objective to be met. ``0 < goal <= 0.9999``.
        rolling_period (google.protobuf.duration_pb2.Duration):
            A rolling time period, semantically "in the past
            ``<rolling_period>``". Must be an integer multiple of 1 day
            no larger than 30 days.

            This field is a member of `oneof`_ ``period``.
        calendar_period (google.type.calendar_period_pb2.CalendarPeriod):
            A calendar period, semantically "since the start of the
            current ``<calendar_period>``". At this time, only ``DAY``,
            ``WEEK``, ``FORTNIGHT``, and ``MONTH`` are supported.

            This field is a member of `oneof`_ ``period``.
        user_labels (MutableMapping[str, str]):
            Labels which have been used to annotate the
            service-level objective. Label keys must start
            with a letter. Label keys and values may contain
            lowercase letters, numbers, underscores, and
            dashes. Label keys and values have a maximum
            length of 63 characters, and must be less than
            128 bytes in size. Up to 64 label entries may be
            stored. For labels which do not have a semantic
            value, the empty string may be supplied for the
            label value.
    """

    class View(proto.Enum):
        r"""``ServiceLevelObjective.View`` determines what form of
        ``ServiceLevelObjective`` is returned from
        ``GetServiceLevelObjective``, ``ListServiceLevelObjectives``, and
        ``ListServiceLevelObjectiveVersions`` RPCs.

        Values:
            VIEW_UNSPECIFIED (0):
                Same as FULL.
            FULL (2):
                Return the embedded ``ServiceLevelIndicator`` in the form in
                which it was defined. If it was defined using a
                ``BasicSli``, return that ``BasicSli``.
            EXPLICIT (1):
                For ``ServiceLevelIndicator``\ s using ``BasicSli``
                articulation, instead return the ``ServiceLevelIndicator``
                with its mode of computation fully spelled out as a
                ``RequestBasedSli``. For ``ServiceLevelIndicator``\ s using
                ``RequestBasedSli`` or ``WindowsBasedSli``, return the
                ``ServiceLevelIndicator`` as it was provided.
        """
        VIEW_UNSPECIFIED = 0
        FULL = 2
        EXPLICIT = 1

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=11,
    )
    service_level_indicator: "ServiceLevelIndicator" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="ServiceLevelIndicator",
    )
    goal: float = proto.Field(
        proto.DOUBLE,
        number=4,
    )
    rolling_period: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="period",
        message=duration_pb2.Duration,
    )
    calendar_period: calendar_period_pb2.CalendarPeriod = proto.Field(
        proto.ENUM,
        number=6,
        oneof="period",
        enum=calendar_period_pb2.CalendarPeriod,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=12,
    )


class ServiceLevelIndicator(proto.Message):
    r"""A Service-Level Indicator (SLI) describes the "performance" of a
    service. For some services, the SLI is well-defined. In such cases,
    the SLI can be described easily by referencing the well-known SLI
    and providing the needed parameters. Alternatively, a "custom" SLI
    can be defined with a query to the underlying metric store. An SLI
    is defined to be ``good_service / total_service`` over any queried
    time interval. The value of performance always falls into the range
    ``0 <= performance <= 1``. A custom SLI describes how to compute
    this ratio, whether this is by dividing values from a pair of time
    series, cutting a ``Distribution`` into good and bad counts, or
    counting time windows in which the service complies with a
    criterion. For separation of concerns, a single Service-Level
    Indicator measures performance for only one aspect of service
    quality, such as fraction of successful queries or fast-enough
    queries.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        basic_sli (google.cloud.monitoring_v3.types.BasicSli):
            Basic SLI on a well-known service type.

            This field is a member of `oneof`_ ``type``.
        request_based (google.cloud.monitoring_v3.types.RequestBasedSli):
            Request-based SLIs

            This field is a member of `oneof`_ ``type``.
        windows_based (google.cloud.monitoring_v3.types.WindowsBasedSli):
            Windows-based SLIs

            This field is a member of `oneof`_ ``type``.
    """

    basic_sli: "BasicSli" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="type",
        message="BasicSli",
    )
    request_based: "RequestBasedSli" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="type",
        message="RequestBasedSli",
    )
    windows_based: "WindowsBasedSli" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="type",
        message="WindowsBasedSli",
    )


class BasicSli(proto.Message):
    r"""An SLI measuring performance on a well-known service type.
    Performance will be computed on the basis of pre-defined metrics.
    The type of the ``service_resource`` determines the metrics to use
    and the ``service_resource.labels`` and ``metric_labels`` are used
    to construct a monitoring filter to filter that metric down to just
    the data relevant to this service.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        method (MutableSequence[str]):
            OPTIONAL: The set of RPCs to which this SLI
            is relevant. Telemetry from other methods will
            not be used to calculate performance for this
            SLI. If omitted, this SLI applies to all the
            Service's methods. For service types that don't
            support breaking down by method, setting this
            field will result in an error.
        location (MutableSequence[str]):
            OPTIONAL: The set of locations to which this
            SLI is relevant. Telemetry from other locations
            will not be used to calculate performance for
            this SLI. If omitted, this SLI applies to all
            locations in which the Service has activity. For
            service types that don't support breaking down
            by location, setting this field will result in
            an error.
        version (MutableSequence[str]):
            OPTIONAL: The set of API versions to which
            this SLI is relevant. Telemetry from other API
            versions will not be used to calculate
            performance for this SLI. If omitted, this SLI
            applies to all API versions. For service types
            that don't support breaking down by version,
            setting this field will result in an error.
        availability (google.cloud.monitoring_v3.types.BasicSli.AvailabilityCriteria):
            Good service is defined to be the count of
            requests made to this service that return
            successfully.

            This field is a member of `oneof`_ ``sli_criteria``.
        latency (google.cloud.monitoring_v3.types.BasicSli.LatencyCriteria):
            Good service is defined to be the count of requests made to
            this service that are fast enough with respect to
            ``latency.threshold``.

            This field is a member of `oneof`_ ``sli_criteria``.
    """

    class AvailabilityCriteria(proto.Message):
        r"""Future parameters for the availability SLI."""

    class LatencyCriteria(proto.Message):
        r"""Parameters for a latency threshold SLI.

        Attributes:
            threshold (google.protobuf.duration_pb2.Duration):
                Good service is defined to be the count of requests made to
                this service that return in no more than ``threshold``.
        """

        threshold: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=3,
            message=duration_pb2.Duration,
        )

    method: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=7,
    )
    location: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=8,
    )
    version: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=9,
    )
    availability: AvailabilityCriteria = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="sli_criteria",
        message=AvailabilityCriteria,
    )
    latency: LatencyCriteria = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="sli_criteria",
        message=LatencyCriteria,
    )


class Range(proto.Message):
    r"""Range of numerical values within ``min`` and ``max``.

    Attributes:
        min_ (float):
            Range minimum.
        max_ (float):
            Range maximum.
    """

    min_: float = proto.Field(
        proto.DOUBLE,
        number=1,
    )
    max_: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )


class RequestBasedSli(proto.Message):
    r"""Service Level Indicators for which atomic units of service
    are counted directly.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        good_total_ratio (google.cloud.monitoring_v3.types.TimeSeriesRatio):
            ``good_total_ratio`` is used when the ratio of
            ``good_service`` to ``total_service`` is computed from two
            ``TimeSeries``.

            This field is a member of `oneof`_ ``method``.
        distribution_cut (google.cloud.monitoring_v3.types.DistributionCut):
            ``distribution_cut`` is used when ``good_service`` is a
            count of values aggregated in a ``Distribution`` that fall
            into a good range. The ``total_service`` is the total count
            of all values aggregated in the ``Distribution``.

            This field is a member of `oneof`_ ``method``.
    """

    good_total_ratio: "TimeSeriesRatio" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="method",
        message="TimeSeriesRatio",
    )
    distribution_cut: "DistributionCut" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="method",
        message="DistributionCut",
    )


class TimeSeriesRatio(proto.Message):
    r"""A ``TimeSeriesRatio`` specifies two ``TimeSeries`` to use for
    computing the ``good_service / total_service`` ratio. The specified
    ``TimeSeries`` must have ``ValueType = DOUBLE`` or
    ``ValueType = INT64`` and must have ``MetricKind = DELTA`` or
    ``MetricKind = CUMULATIVE``. The ``TimeSeriesRatio`` must specify
    exactly two of good, bad, and total, and the relationship
    ``good_service + bad_service = total_service`` will be assumed.

    Attributes:
        good_service_filter (str):
            A `monitoring
            filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            specifying a ``TimeSeries`` quantifying good service
            provided. Must have ``ValueType = DOUBLE`` or
            ``ValueType = INT64`` and must have ``MetricKind = DELTA``
            or ``MetricKind = CUMULATIVE``.
        bad_service_filter (str):
            A `monitoring
            filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            specifying a ``TimeSeries`` quantifying bad service, either
            demanded service that was not provided or demanded service
            that was of inadequate quality. Must have
            ``ValueType = DOUBLE`` or ``ValueType = INT64`` and must
            have ``MetricKind = DELTA`` or ``MetricKind = CUMULATIVE``.
        total_service_filter (str):
            A `monitoring
            filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            specifying a ``TimeSeries`` quantifying total demanded
            service. Must have ``ValueType = DOUBLE`` or
            ``ValueType = INT64`` and must have ``MetricKind = DELTA``
            or ``MetricKind = CUMULATIVE``.
    """

    good_service_filter: str = proto.Field(
        proto.STRING,
        number=4,
    )
    bad_service_filter: str = proto.Field(
        proto.STRING,
        number=5,
    )
    total_service_filter: str = proto.Field(
        proto.STRING,
        number=6,
    )


class DistributionCut(proto.Message):
    r"""A ``DistributionCut`` defines a ``TimeSeries`` and thresholds used
    for measuring good service and total service. The ``TimeSeries``
    must have ``ValueType = DISTRIBUTION`` and ``MetricKind = DELTA`` or
    ``MetricKind = CUMULATIVE``. The computed ``good_service`` will be
    the estimated count of values in the ``Distribution`` that fall
    within the specified ``min`` and ``max``.

    Attributes:
        distribution_filter (str):
            A `monitoring
            filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            specifying a ``TimeSeries`` aggregating values. Must have
            ``ValueType = DISTRIBUTION`` and ``MetricKind = DELTA`` or
            ``MetricKind = CUMULATIVE``.
        range_ (google.cloud.monitoring_v3.types.Range):
            Range of values considered "good." For a
            one-sided range, set one bound to an infinite
            value.
    """

    distribution_filter: str = proto.Field(
        proto.STRING,
        number=4,
    )
    range_: "Range" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="Range",
    )


class WindowsBasedSli(proto.Message):
    r"""A ``WindowsBasedSli`` defines ``good_service`` as the count of time
    windows for which the provided service was of good quality. Criteria
    for determining if service was good are embedded in the
    ``window_criterion``.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        good_bad_metric_filter (str):
            A `monitoring
            filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            specifying a ``TimeSeries`` with ``ValueType = BOOL``. The
            window is good if any ``true`` values appear in the window.

            This field is a member of `oneof`_ ``window_criterion``.
        good_total_ratio_threshold (google.cloud.monitoring_v3.types.WindowsBasedSli.PerformanceThreshold):
            A window is good if its ``performance`` is high enough.

            This field is a member of `oneof`_ ``window_criterion``.
        metric_mean_in_range (google.cloud.monitoring_v3.types.WindowsBasedSli.MetricRange):
            A window is good if the metric's value is in
            a good range, averaged across returned streams.

            This field is a member of `oneof`_ ``window_criterion``.
        metric_sum_in_range (google.cloud.monitoring_v3.types.WindowsBasedSli.MetricRange):
            A window is good if the metric's value is in
            a good range, summed across returned streams.

            This field is a member of `oneof`_ ``window_criterion``.
        window_period (google.protobuf.duration_pb2.Duration):
            Duration over which window quality is evaluated. Must be an
            integer fraction of a day and at least ``60s``.
    """

    class PerformanceThreshold(proto.Message):
        r"""A ``PerformanceThreshold`` is used when each window is good when
        that window has a sufficiently high ``performance``.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            performance (google.cloud.monitoring_v3.types.RequestBasedSli):
                ``RequestBasedSli`` to evaluate to judge window quality.

                This field is a member of `oneof`_ ``type``.
            basic_sli_performance (google.cloud.monitoring_v3.types.BasicSli):
                ``BasicSli`` to evaluate to judge window quality.

                This field is a member of `oneof`_ ``type``.
            threshold (float):
                If window ``performance >= threshold``, the window is
                counted as good.
        """

        performance: "RequestBasedSli" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="type",
            message="RequestBasedSli",
        )
        basic_sli_performance: "BasicSli" = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="type",
            message="BasicSli",
        )
        threshold: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )

    class MetricRange(proto.Message):
        r"""A ``MetricRange`` is used when each window is good when the value x
        of a single ``TimeSeries`` satisfies
        ``range.min <= x <= range.max``. The provided ``TimeSeries`` must
        have ``ValueType = INT64`` or ``ValueType = DOUBLE`` and
        ``MetricKind = GAUGE``.

        Attributes:
            time_series (str):
                A `monitoring
                filter <https://cloud.google.com/monitoring/api/v3/filters>`__
                specifying the ``TimeSeries`` to use for evaluating window
                quality.
            range_ (google.cloud.monitoring_v3.types.Range):
                Range of values considered "good." For a
                one-sided range, set one bound to an infinite
                value.
        """

        time_series: str = proto.Field(
            proto.STRING,
            number=1,
        )
        range_: "Range" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="Range",
        )

    good_bad_metric_filter: str = proto.Field(
        proto.STRING,
        number=5,
        oneof="window_criterion",
    )
    good_total_ratio_threshold: PerformanceThreshold = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="window_criterion",
        message=PerformanceThreshold,
    )
    metric_mean_in_range: MetricRange = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="window_criterion",
        message=MetricRange,
    )
    metric_sum_in_range: MetricRange = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="window_criterion",
        message=MetricRange,
    )
    window_period: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=4,
        message=duration_pb2.Duration,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
