# Copyright 2018 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""User-friendly container for Google Cloud Bigtable AppProfile."""


import re

from google.cloud.bigtable.enums import RoutingPolicyType
from google.cloud.bigtable_admin_v2.types import instance
from google.protobuf import field_mask_pb2
from google.api_core.exceptions import NotFound

_APP_PROFILE_NAME_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/"
    r"instances/(?P<instance>[^/]+)/"
    r"appProfiles/(?P<app_profile_id>[_a-zA-Z0-9][-_.a-zA-Z0-9]*)$"
)


class AppProfile(object):
    """Representation of a Google Cloud Bigtable AppProfile.

    We can use a :class:`AppProfile` to:

    * :meth:`reload` itself
    * :meth:`create` itself
    * :meth:`update` itself
    * :meth:`delete` itself

    :type app_profile_id: str
    :param app_profile_id: The ID of the AppProfile. Must be of the form
                           ``[_a-zA-Z0-9][-_.a-zA-Z0-9]*``.

    :type: routing_policy_type: int
    :param: routing_policy_type: (Optional) The type of the routing policy.
                                 Possible values are represented
                                 by the following constants:
                                 :data:`google.cloud.bigtable.enums.RoutingPolicyType.ANY`
                                 :data:`google.cloud.bigtable.enums.RoutingPolicyType.SINGLE`

    :type: description: str
    :param: description: (Optional) Long form description of the use
                         case for this AppProfile.

    :type: cluster_id: str
    :param: cluster_id: (Optional) Unique cluster_id which is only required
                        when routing_policy_type is
                        ROUTING_POLICY_TYPE_SINGLE.

    :type: multi_cluster_ids: list
    :param: multi_cluster_ids: (Optional) The set of clusters to route to.
                            The order is ignored; clusters will be tried in order of distance.
                            If left empty, all clusters are eligible.

    :type: allow_transactional_writes: bool
    :param: allow_transactional_writes: (Optional) If true, allow
                                        transactional writes for
                                        ROUTING_POLICY_TYPE_SINGLE.
    """

    def __init__(
        self,
        app_profile_id,
        instance,
        routing_policy_type=None,
        description=None,
        cluster_id=None,
        multi_cluster_ids=None,
        allow_transactional_writes=None,
    ):
        self.app_profile_id = app_profile_id
        self._instance = instance
        self.routing_policy_type = routing_policy_type
        self.description = description
        self.cluster_id = cluster_id
        self.multi_cluster_ids = multi_cluster_ids
        self.allow_transactional_writes = allow_transactional_writes

    @property
    def name(self):
        """AppProfile name used in requests.

        .. note::

          This property will not change if ``app_profile_id`` does not, but
          the return value is not cached.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_app_profile_name]
            :end-before: [END bigtable_api_app_profile_name]
            :dedent: 4

        The AppProfile name is of the form
            ``"projects/../instances/../app_profile/{app_profile_id}"``

        :rtype: str
        :returns: The AppProfile name.
        """
        return self.instance_admin_client.app_profile_path(
            self._instance._client.project,
            self._instance.instance_id,
            self.app_profile_id,
        )

    @property
    def instance_admin_client(self):
        """Shortcut to instance_admin_client

        :rtype: :class:`.bigtable_admin_pb2.BigtableInstanceAdmin`
        :returns: A BigtableInstanceAdmin instance.
        """
        return self._instance._client.instance_admin_client

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        # NOTE: This does not compare the configuration values, such as
        #       the routing_policy_type. Instead, it only compares
        #       identifying values instance, AppProfile ID and client. This is
        #       intentional, since the same AppProfile can be in different
        #       states if not synchronized.
        return (
            other.app_profile_id == self.app_profile_id
            and other._instance == self._instance
        )

    def __ne__(self, other):
        return not self == other

    @classmethod
    def from_pb(cls, app_profile_pb, instance):
        """Creates an instance app_profile from a protobuf.

        :type app_profile_pb: :class:`instance.app_profile_pb`
        :param app_profile_pb: An instance protobuf object.

        :type instance: :class:`google.cloud.bigtable.instance.Instance`
        :param instance: The instance that owns the cluster.

        :rtype: :class:`AppProfile`
        :returns: The AppProfile parsed from the protobuf response.

        :raises: :class:`ValueError <exceptions.ValueError>` if the AppProfile
                 name does not match
                 ``projects/{project}/instances/{instance_id}/appProfiles/{app_profile_id}``
                 or if the parsed instance ID does not match the istance ID
                 on the client.
                 or if the parsed project ID does not match the project ID
                 on the client.
        """
        match_app_profile_name = _APP_PROFILE_NAME_RE.match(app_profile_pb.name)
        if match_app_profile_name is None:
            raise ValueError(
                "AppProfile protobuf name was not in the " "expected format.",
                app_profile_pb.name,
            )
        if match_app_profile_name.group("instance") != instance.instance_id:
            raise ValueError(
                "Instance ID on app_profile does not match the "
                "instance ID on the client"
            )
        if match_app_profile_name.group("project") != instance._client.project:
            raise ValueError(
                "Project ID on app_profile does not match the "
                "project ID on the client"
            )
        app_profile_id = match_app_profile_name.group("app_profile_id")

        result = cls(app_profile_id, instance)
        result._update_from_pb(app_profile_pb)
        return result

    def _update_from_pb(self, app_profile_pb):
        """Refresh self from the server-provided protobuf.
        Helper for :meth:`from_pb` and :meth:`reload`.
        """
        self.routing_policy_type = None
        self.allow_transactional_writes = None
        self.cluster_id = None
        self.multi_cluster_ids = None
        self.description = app_profile_pb.description

        routing_policy_type = None
        if app_profile_pb._pb.HasField("multi_cluster_routing_use_any"):
            routing_policy_type = RoutingPolicyType.ANY
            self.allow_transactional_writes = False
            if app_profile_pb.multi_cluster_routing_use_any.cluster_ids:
                self.multi_cluster_ids = (
                    app_profile_pb.multi_cluster_routing_use_any.cluster_ids
                )
        else:
            routing_policy_type = RoutingPolicyType.SINGLE
            self.cluster_id = app_profile_pb.single_cluster_routing.cluster_id
            self.allow_transactional_writes = (
                app_profile_pb.single_cluster_routing.allow_transactional_writes
            )
        self.routing_policy_type = routing_policy_type

    def _to_pb(self):
        """Create an AppProfile proto buff message for API calls
        :rtype: :class:`.instance.AppProfile`
        :returns: The converted current object.

        :raises: :class:`ValueError <exceptions.ValueError>` if the AppProfile
                 routing_policy_type is not set
        """
        if not self.routing_policy_type:
            raise ValueError("AppProfile required routing policy.")

        single_cluster_routing = None
        multi_cluster_routing_use_any = None

        if self.routing_policy_type == RoutingPolicyType.ANY:
            multi_cluster_routing_use_any = (
                instance.AppProfile.MultiClusterRoutingUseAny(
                    cluster_ids=self.multi_cluster_ids
                )
            )
        else:
            single_cluster_routing = instance.AppProfile.SingleClusterRouting(
                cluster_id=self.cluster_id,
                allow_transactional_writes=self.allow_transactional_writes,
            )

        app_profile_pb = instance.AppProfile(
            name=self.name,
            description=self.description,
            multi_cluster_routing_use_any=multi_cluster_routing_use_any,
            single_cluster_routing=single_cluster_routing,
        )
        return app_profile_pb

    def reload(self):
        """Reload the metadata for this cluster

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_reload_app_profile]
            :end-before: [END bigtable_api_reload_app_profile]
            :dedent: 4
        """

        app_profile_pb = self.instance_admin_client.get_app_profile(
            request={"name": self.name}
        )

        # NOTE: _update_from_pb does not check that the project and
        #       app_profile ID on the response match the request.
        self._update_from_pb(app_profile_pb)

    def exists(self):
        """Check whether the AppProfile already exists.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_app_profile_exists]
            :end-before: [END bigtable_api_app_profile_exists]
            :dedent: 4

        :rtype: bool
        :returns: True if the AppProfile exists, else False.
        """
        try:
            self.instance_admin_client.get_app_profile(request={"name": self.name})
            return True
        # NOTE: There could be other exceptions that are returned to the user.
        except NotFound:
            return False

    def create(self, ignore_warnings=None):
        """Create this AppProfile.

        .. note::

            Uses the ``instance`` and ``app_profile_id`` on the current
            :class:`AppProfile` in addition to the ``routing_policy_type``,
            ``description``, ``cluster_id`` and ``allow_transactional_writes``.
            To change them before creating, reset the values via

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_create_app_profile]
            :end-before: [END bigtable_api_create_app_profile]
            :dedent: 4

        :type: ignore_warnings: bool
        :param: ignore_warnings: (Optional) If true, ignore safety checks when
                                 creating the AppProfile.
        """
        return self.from_pb(
            self.instance_admin_client.create_app_profile(
                request={
                    "parent": self._instance.name,
                    "app_profile_id": self.app_profile_id,
                    "app_profile": self._to_pb(),
                    "ignore_warnings": ignore_warnings,
                }
            ),
            self._instance,
        )

    def update(self, ignore_warnings=None):
        """Update this app_profile.

        .. note::

            Update any or all of the following values:
            ``routing_policy_type``
            ``description``
            ``cluster_id``
            ``multi_cluster_ids``
            ``allow_transactional_writes``

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_update_app_profile]
            :end-before: [END bigtable_api_update_app_profile]
            :dedent: 4
        """
        update_mask_pb = field_mask_pb2.FieldMask()

        if self.description is not None:
            update_mask_pb.paths.append("description")

        if self.routing_policy_type == RoutingPolicyType.ANY:
            update_mask_pb.paths.append("multi_cluster_routing_use_any")
        else:
            update_mask_pb.paths.append("single_cluster_routing")

        return self.instance_admin_client.update_app_profile(
            request={
                "app_profile": self._to_pb(),
                "update_mask": update_mask_pb,
                "ignore_warnings": ignore_warnings,
            }
        )

    def delete(self, ignore_warnings=None):
        """Delete this AppProfile.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_delete_app_profile]
            :end-before: [END bigtable_api_delete_app_profile]
            :dedent: 4

        :type: ignore_warnings: bool
        :param: ignore_warnings: If true, ignore safety checks when deleting
                the AppProfile.

        :raises: google.api_core.exceptions.GoogleAPICallError: If the request
                 failed for any reason. google.api_core.exceptions.RetryError:
                 If the request failed due to a retryable error and retry
                 attempts failed. ValueError: If the parameters are invalid.
        """
        self.instance_admin_client.delete_app_profile(
            request={"name": self.name, "ignore_warnings": ignore_warnings}
        )
