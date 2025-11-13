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

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "Group",
    },
)


class Group(proto.Message):
    r"""The description of a dynamic collection of monitored resources. Each
    group has a filter that is matched against monitored resources and
    their associated metadata. If a group's filter matches an available
    monitored resource, then that resource is a member of that group.
    Groups can contain any number of monitored resources, and each
    monitored resource can be a member of any number of groups.

    Groups can be nested in parent-child hierarchies. The ``parentName``
    field identifies an optional parent for each group. If a group has a
    parent, then the only monitored resources available to be matched by
    the group's filter are the resources contained in the parent group.
    In other words, a group contains the monitored resources that match
    its filter and the filters of all the group's ancestors. A group
    without a parent can contain any monitored resource.

    For example, consider an infrastructure running a set of instances
    with two user-defined tags: ``"environment"`` and ``"role"``. A
    parent group has a filter, ``environment="production"``. A child of
    that parent group has a filter, ``role="transcoder"``. The parent
    group contains all instances in the production environment,
    regardless of their roles. The child group contains instances that
    have the transcoder role *and* are in the production environment.

    The monitored resources contained in a group can change at any
    moment, depending on what resources exist and what filters are
    associated with the group and its ancestors.

    Attributes:
        name (str):
            Output only. The name of this group. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

            When creating a group, this field is ignored and a new name
            is created consisting of the project specified in the call
            to ``CreateGroup`` and a unique ``[GROUP_ID]`` that is
            generated automatically.
        display_name (str):
            A user-assigned name for this group, used
            only for display purposes.
        parent_name (str):
            The name of the group's parent, if it has one. The format
            is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

            For groups with no parent, ``parent_name`` is the empty
            string, ``""``.
        filter (str):
            The filter used to determine which monitored
            resources belong to this group.
        is_cluster (bool):
            If true, the members of this group are
            considered to be a cluster. The system can
            perform additional analysis on groups that are
            clusters.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    parent_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=5,
    )
    is_cluster: bool = proto.Field(
        proto.BOOL,
        number=6,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
