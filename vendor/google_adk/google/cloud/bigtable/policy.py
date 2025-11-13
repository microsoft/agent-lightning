# Copyright 2018 Google LLC
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

import base64

from google.api_core.iam import Policy as BasePolicy
from google.cloud._helpers import _to_bytes  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore

"""IAM roles supported by Bigtable Instance resource"""
BIGTABLE_ADMIN_ROLE = "roles/bigtable.admin"
"""Administers all instances within a project, including the data stored
within tables. Can create new instances. Intended for project administrators.
"""
BIGTABLE_USER_ROLE = "roles/bigtable.user"
"""Provides read-write access to the data stored within tables. Intended for
application developers or service accounts.
"""
BIGTABLE_READER_ROLE = "roles/bigtable.reader"
"""Provides read-only access to the data stored within tables. Intended for
data scientists, dashboard generators, and other data-analysis scenarios.
"""
BIGTABLE_VIEWER_ROLE = "roles/bigtable.viewer"
"""Provides no data access. Intended as a minimal set of permissions to access
the GCP Console for Cloud Bigtable.
"""
"""For detailed information
See
https://cloud.google.com/bigtable/docs/access-control#roles
"""


class Policy(BasePolicy):
    """IAM Policy

    See
    https://cloud.google.com/bigtable/docs/reference/admin/rpc/google.iam.v1#policy

    A Policy consists of a list of bindings. A binding binds a list of
    members to a role, where the members can be user accounts, Google
    groups, Google domains, and service accounts. A role is a named list
    of permissions defined by IAM.
    For more information about predefined roles currently supoprted
    by Bigtable Instance please see
    `Predefined roles
    <https://cloud.google.com/bigtable/docs/access-control#roles>`_.
    For more information about custom roles please see
    `Custom roles
    <https://cloud.google.com/bigtable/docs/access-control#custom-roles>`_.

    :type etag: str
    :param etag: etag is used for optimistic concurrency control as a way to
                 help prevent simultaneous updates of a policy from overwriting
                 each other. It is strongly suggested that systems make use
                 of the etag in the read-modify-write cycle to perform policy
                 updates in order to avoid race conditions:
                 An etag is returned in the response to getIamPolicy, and
                 systems are expected to put that etag in the request to
                 setIamPolicy to ensure that their change will be applied to
                 the same version of the policy.

                 If no etag is provided in the call to setIamPolicy, then the
                 existing policy is overwritten blindly.
    :type version: int
    :param version: The syntax schema version of the policy.

    Note:
        Using conditions in bindings requires the policy's version to be set
        to `3` or greater, depending on the versions that are currently supported.

        Accessing the policy using dict operations will raise InvalidOperationException
        when the policy's version is set to 3.

        Use the policy.bindings getter/setter to retrieve and modify the policy's bindings.

    See:
        IAM Policy https://cloud.google.com/iam/reference/rest/v1/Policy
        Policy versions https://cloud.google.com/iam/docs/policies#versions
        Conditions overview https://cloud.google.com/iam/docs/conditions-overview.
    """

    def __init__(self, etag=None, version=None):
        BasePolicy.__init__(
            self, etag=etag if etag is None else _to_bytes(etag), version=version
        )

    @property
    def bigtable_admins(self):
        """Access to bigtable.admin role memebers

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_admins_policy]
            :end-before: [END bigtable_api_admins_policy]
            :dedent: 4
        """
        result = set()
        for member in self.get(BIGTABLE_ADMIN_ROLE, ()):
            result.add(member)
        return frozenset(result)

    @property
    def bigtable_readers(self):
        """Access to bigtable.reader role memebers

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_readers_policy]
            :end-before: [END bigtable_api_readers_policy]
            :dedent: 4
        """
        result = set()
        for member in self.get(BIGTABLE_READER_ROLE, ()):
            result.add(member)
        return frozenset(result)

    @property
    def bigtable_users(self):
        """Access to bigtable.user role memebers

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_users_policy]
            :end-before: [END bigtable_api_users_policy]
            :dedent: 4
        """
        result = set()
        for member in self.get(BIGTABLE_USER_ROLE, ()):
            result.add(member)
        return frozenset(result)

    @property
    def bigtable_viewers(self):
        """Access to bigtable.viewer role memebers

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        For example:

        .. literalinclude:: snippets.py
            :start-after: [START bigtable_api_viewers_policy]
            :end-before: [END bigtable_api_viewers_policy]
            :dedent: 4
        """
        result = set()
        for member in self.get(BIGTABLE_VIEWER_ROLE, ()):
            result.add(member)
        return frozenset(result)

    @classmethod
    def from_pb(cls, policy_pb):
        """Factory: create a policy from a protobuf message.

        Args:
            policy_pb (google.iam.policy_pb2.Policy): message returned by
            ``get_iam_policy`` gRPC API.

        Returns:
            :class:`Policy`: the parsed policy
        """
        policy = cls(policy_pb.etag, policy_pb.version)

        policy.bindings = bindings = []
        for binding_pb in policy_pb.bindings:
            binding = {"role": binding_pb.role, "members": set(binding_pb.members)}
            condition = binding_pb.condition
            if condition and condition.expression:
                binding["condition"] = {
                    "title": condition.title,
                    "description": condition.description,
                    "expression": condition.expression,
                }
            bindings.append(binding)

        return policy

    def to_pb(self):
        """Render a protobuf message.

        Returns:
            google.iam.policy_pb2.Policy: a message to be passed to the
            ``set_iam_policy`` gRPC API.
        """

        return policy_pb2.Policy(
            etag=self.etag,
            version=self.version or 0,
            bindings=[
                policy_pb2.Binding(
                    role=binding["role"],
                    members=sorted(binding["members"]),
                    condition=binding.get("condition"),
                )
                for binding in self.bindings
                if binding["members"]
            ],
        )

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: create a policy from a JSON resource.

        Overrides the base class version to store :attr:`etag` as bytes.

        Args:
            resource (dict): JSON policy resource returned by the
            ``getIamPolicy`` REST API.

        Returns:
            :class:`Policy`: the parsed policy
        """
        etag = resource.get("etag")

        if etag is not None:
            resource = resource.copy()
            resource["etag"] = base64.b64decode(etag.encode("ascii"))

        return super(Policy, cls).from_api_repr(resource)

    def to_api_repr(self):
        """Render a JSON policy resource.

        Overrides the base class version to convert :attr:`etag` from bytes
        to JSON-compatible base64-encoded text.

        Returns:
            dict: a JSON resource to be passed to the
            ``setIamPolicy`` REST API.
        """
        resource = super(Policy, self).to_api_repr()

        if self.etag is not None:
            resource["etag"] = base64.b64encode(self.etag).decode("ascii")

        return resource
