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

from google.api import monitored_resource_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.monitoring_v3.types import common
from google.cloud.monitoring_v3.types import group as gm_group

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "ListGroupsRequest",
        "ListGroupsResponse",
        "GetGroupRequest",
        "CreateGroupRequest",
        "UpdateGroupRequest",
        "DeleteGroupRequest",
        "ListGroupMembersRequest",
        "ListGroupMembersResponse",
    },
)


class ListGroupsRequest(proto.Message):
    r"""The ``ListGroup`` request.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            whose groups are to be listed. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        children_of_group (str):
            A group name. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

            Returns groups whose ``parent_name`` field contains the
            group name. If no groups have this parent, the results are
            empty.

            This field is a member of `oneof`_ ``filter``.
        ancestors_of_group (str):
            A group name. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

            Returns groups that are ancestors of the specified group.
            The groups are returned in order, starting with the
            immediate parent and ending with the most distant ancestor.
            If the specified group has no immediate parent, the results
            are empty.

            This field is a member of `oneof`_ ``filter``.
        descendants_of_group (str):
            A group name. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

            Returns the descendants of the specified group. This is a
            superset of the results returned by the
            ``children_of_group`` filter, and includes
            children-of-children, and so forth.

            This field is a member of `oneof`_ ``filter``.
        page_size (int):
            A positive number that is the maximum number
            of results to return.
        page_token (str):
            If this field is not empty then it must contain the
            ``next_page_token`` value returned by a previous call to
            this method. Using this field causes the method to return
            additional results from the previous method call.
    """

    name: str = proto.Field(
        proto.STRING,
        number=7,
    )
    children_of_group: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="filter",
    )
    ancestors_of_group: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="filter",
    )
    descendants_of_group: str = proto.Field(
        proto.STRING,
        number=4,
        oneof="filter",
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=5,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListGroupsResponse(proto.Message):
    r"""The ``ListGroups`` response.

    Attributes:
        group (MutableSequence[google.cloud.monitoring_v3.types.Group]):
            The groups that match the specified filters.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
    """

    @property
    def raw_page(self):
        return self

    group: MutableSequence[gm_group.Group] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gm_group.Group,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetGroupRequest(proto.Message):
    r"""The ``GetGroup`` request.

    Attributes:
        name (str):
            Required. The group to retrieve. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreateGroupRequest(proto.Message):
    r"""The ``CreateGroup`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            in which to create the group. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        group (google.cloud.monitoring_v3.types.Group):
            Required. A group definition. It is an error to define the
            ``name`` field because the system assigns the name.
        validate_only (bool):
            If true, validate this request but do not
            create the group.
    """

    name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    group: gm_group.Group = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gm_group.Group,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class UpdateGroupRequest(proto.Message):
    r"""The ``UpdateGroup`` request.

    Attributes:
        group (google.cloud.monitoring_v3.types.Group):
            Required. The new definition of the group. All fields of the
            existing group, excepting ``name``, are replaced with the
            corresponding fields of this group.
        validate_only (bool):
            If true, validate this request but do not
            update the existing group.
    """

    group: gm_group.Group = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gm_group.Group,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class DeleteGroupRequest(proto.Message):
    r"""The ``DeleteGroup`` request. The default behavior is to be able to
    delete a single group without any descendants.

    Attributes:
        name (str):
            Required. The group to delete. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]
        recursive (bool):
            If this field is true, then the request means
            to delete a group with all its descendants.
            Otherwise, the request means to delete a group
            only when it has no descendants. The default
            value is false.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    recursive: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class ListGroupMembersRequest(proto.Message):
    r"""The ``ListGroupMembers`` request.

    Attributes:
        name (str):
            Required. The group whose members are listed. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]
        page_size (int):
            A positive number that is the maximum number
            of results to return.
        page_token (str):
            If this field is not empty then it must contain the
            ``next_page_token`` value returned by a previous call to
            this method. Using this field causes the method to return
            additional results from the previous method call.
        filter (str):
            An optional `list
            filter <https://cloud.google.com/monitoring/api/learn_more#filtering>`__
            describing the members to be returned. The filter may
            reference the type, labels, and metadata of monitored
            resources that comprise the group. For example, to return
            only resources representing Compute Engine VM instances, use
            this filter:

            ::

                `resource.type = "gce_instance"`
        interval (google.cloud.monitoring_v3.types.TimeInterval):
            An optional time interval for which results
            should be returned. Only members that were part
            of the group during the specified interval are
            included in the response.  If no interval is
            provided then the group membership over the last
            minute is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=7,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=5,
    )
    interval: common.TimeInterval = proto.Field(
        proto.MESSAGE,
        number=6,
        message=common.TimeInterval,
    )


class ListGroupMembersResponse(proto.Message):
    r"""The ``ListGroupMembers`` response.

    Attributes:
        members (MutableSequence[google.api.monitored_resource_pb2.MonitoredResource]):
            A set of monitored resources in the group.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
        total_size (int):
            The total number of elements matching this
            request.
    """

    @property
    def raw_page(self):
        return self

    members: MutableSequence[
        monitored_resource_pb2.MonitoredResource
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=monitored_resource_pb2.MonitoredResource,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    total_size: int = proto.Field(
        proto.INT32,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
