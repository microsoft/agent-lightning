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

from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.monitoring_v3.types import service as gm_service

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "CreateServiceRequest",
        "GetServiceRequest",
        "ListServicesRequest",
        "ListServicesResponse",
        "UpdateServiceRequest",
        "DeleteServiceRequest",
        "CreateServiceLevelObjectiveRequest",
        "GetServiceLevelObjectiveRequest",
        "ListServiceLevelObjectivesRequest",
        "ListServiceLevelObjectivesResponse",
        "UpdateServiceLevelObjectiveRequest",
        "DeleteServiceLevelObjectiveRequest",
    },
)


class CreateServiceRequest(proto.Message):
    r"""The ``CreateService`` request.

    Attributes:
        parent (str):
            Required. Resource
            `name <https://cloud.google.com/monitoring/api/v3#project_name>`__
            of the parent Metrics Scope. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        service_id (str):
            Optional. The Service id to use for this Service. If
            omitted, an id will be generated instead. Must match the
            pattern ``[a-z0-9\-]+``
        service (google.cloud.monitoring_v3.types.Service):
            Required. The ``Service`` to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    service_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    service: gm_service.Service = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gm_service.Service,
    )


class GetServiceRequest(proto.Message):
    r"""The ``GetService`` request.

    Attributes:
        name (str):
            Required. Resource name of the ``Service``. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListServicesRequest(proto.Message):
    r"""The ``ListServices`` request.

    Attributes:
        parent (str):
            Required. Resource name of the parent containing the listed
            services, either a
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            or a Monitoring Metrics Scope. The formats are:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
                workspaces/[HOST_PROJECT_ID_OR_NUMBER]
        filter (str):
            A filter specifying what ``Service``\ s to return. The
            filter supports filtering on a particular service-identifier
            type or one of its attributes.

            To filter on a particular service-identifier type, the
            ``identifier_case`` refers to which option in the
            ``identifier`` field is populated. For example, the filter
            ``identifier_case = "CUSTOM"`` would match all services with
            a value for the ``custom`` field. Valid options include
            "CUSTOM", "APP_ENGINE", "MESH_ISTIO", and the other options
            listed at
            https://cloud.google.com/monitoring/api/ref_v3/rest/v3/services#Service

            To filter on an attribute of a service-identifier type,
            apply the filter name by using the snake case of the
            service-identifier type and the attribute of that
            service-identifier type, and join the two with a period. For
            example, to filter by the ``meshUid`` field of the
            ``MeshIstio`` service-identifier type, you must filter on
            ``mesh_istio.mesh_uid = "123"`` to match all services with
            mesh UID "123". Service-identifier types and their
            attributes are described at
            https://cloud.google.com/monitoring/api/ref_v3/rest/v3/services#Service
        page_size (int):
            A non-negative number that is the maximum
            number of results to return. When 0, use default
            page size.
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return
            additional results from the previous method call.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListServicesResponse(proto.Message):
    r"""The ``ListServices`` response.

    Attributes:
        services (MutableSequence[google.cloud.monitoring_v3.types.Service]):
            The ``Service``\ s matching the specified filter.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
    """

    @property
    def raw_page(self):
        return self

    services: MutableSequence[gm_service.Service] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gm_service.Service,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateServiceRequest(proto.Message):
    r"""The ``UpdateService`` request.

    Attributes:
        service (google.cloud.monitoring_v3.types.Service):
            Required. The ``Service`` to draw updates from. The given
            ``name`` specifies the resource to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            A set of field paths defining which fields to
            use for the update.
    """

    service: gm_service.Service = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gm_service.Service,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteServiceRequest(proto.Message):
    r"""The ``DeleteService`` request.

    Attributes:
        name (str):
            Required. Resource name of the ``Service`` to delete. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateServiceLevelObjectiveRequest(proto.Message):
    r"""The ``CreateServiceLevelObjective`` request.

    Attributes:
        parent (str):
            Required. Resource name of the parent ``Service``. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
        service_level_objective_id (str):
            Optional. The ServiceLevelObjective id to use for this
            ServiceLevelObjective. If omitted, an id will be generated
            instead. Must match the pattern ``^[a-zA-Z0-9-_:.]+$``
        service_level_objective (google.cloud.monitoring_v3.types.ServiceLevelObjective):
            Required. The ``ServiceLevelObjective`` to create. The
            provided ``name`` will be respected if no
            ``ServiceLevelObjective`` exists with this name.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    service_level_objective_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    service_level_objective: gm_service.ServiceLevelObjective = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gm_service.ServiceLevelObjective,
    )


class GetServiceLevelObjectiveRequest(proto.Message):
    r"""The ``GetServiceLevelObjective`` request.

    Attributes:
        name (str):
            Required. Resource name of the ``ServiceLevelObjective`` to
            get. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]/serviceLevelObjectives/[SLO_NAME]
        view (google.cloud.monitoring_v3.types.ServiceLevelObjective.View):
            View of the ``ServiceLevelObjective`` to return. If
            ``DEFAULT``, return the ``ServiceLevelObjective`` as
            originally defined. If ``EXPLICIT`` and the
            ``ServiceLevelObjective`` is defined in terms of a
            ``BasicSli``, replace the ``BasicSli`` with a
            ``RequestBasedSli`` spelling out how the SLI is computed.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    view: gm_service.ServiceLevelObjective.View = proto.Field(
        proto.ENUM,
        number=2,
        enum=gm_service.ServiceLevelObjective.View,
    )


class ListServiceLevelObjectivesRequest(proto.Message):
    r"""The ``ListServiceLevelObjectives`` request.

    Attributes:
        parent (str):
            Required. Resource name of the parent containing the listed
            SLOs, either a project or a Monitoring Metrics Scope. The
            formats are:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
                workspaces/[HOST_PROJECT_ID_OR_NUMBER]/services/-
        filter (str):
            A filter specifying what ``ServiceLevelObjective``\ s to
            return.
        page_size (int):
            A non-negative number that is the maximum
            number of results to return. When 0, use default
            page size.
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return
            additional results from the previous method call.
        view (google.cloud.monitoring_v3.types.ServiceLevelObjective.View):
            View of the ``ServiceLevelObjective``\ s to return. If
            ``DEFAULT``, return each ``ServiceLevelObjective`` as
            originally defined. If ``EXPLICIT`` and the
            ``ServiceLevelObjective`` is defined in terms of a
            ``BasicSli``, replace the ``BasicSli`` with a
            ``RequestBasedSli`` spelling out how the SLI is computed.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    view: gm_service.ServiceLevelObjective.View = proto.Field(
        proto.ENUM,
        number=5,
        enum=gm_service.ServiceLevelObjective.View,
    )


class ListServiceLevelObjectivesResponse(proto.Message):
    r"""The ``ListServiceLevelObjectives`` response.

    Attributes:
        service_level_objectives (MutableSequence[google.cloud.monitoring_v3.types.ServiceLevelObjective]):
            The ``ServiceLevelObjective``\ s matching the specified
            filter.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
    """

    @property
    def raw_page(self):
        return self

    service_level_objectives: MutableSequence[
        gm_service.ServiceLevelObjective
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gm_service.ServiceLevelObjective,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateServiceLevelObjectiveRequest(proto.Message):
    r"""The ``UpdateServiceLevelObjective`` request.

    Attributes:
        service_level_objective (google.cloud.monitoring_v3.types.ServiceLevelObjective):
            Required. The ``ServiceLevelObjective`` to draw updates
            from. The given ``name`` specifies the resource to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            A set of field paths defining which fields to
            use for the update.
    """

    service_level_objective: gm_service.ServiceLevelObjective = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gm_service.ServiceLevelObjective,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteServiceLevelObjectiveRequest(proto.Message):
    r"""The ``DeleteServiceLevelObjective`` request.

    Attributes:
        name (str):
            Required. Resource name of the ``ServiceLevelObjective`` to
            delete. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]/serviceLevelObjectives/[SLO_NAME]
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
