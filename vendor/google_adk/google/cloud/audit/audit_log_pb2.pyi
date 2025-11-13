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

from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.rpc import status_pb2 as _status_pb2
from google.rpc.context import attribute_context_pb2 as _attribute_context_pb2

DESCRIPTOR: _descriptor.FileDescriptor

class AuditLog(_message.Message):
    __slots__ = (
        "service_name",
        "method_name",
        "resource_name",
        "resource_location",
        "resource_original_state",
        "num_response_items",
        "status",
        "authentication_info",
        "authorization_info",
        "policy_violation_info",
        "request_metadata",
        "request",
        "response",
        "metadata",
        "service_data",
    )
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_NAME_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ORIGINAL_STATE_FIELD_NUMBER: _ClassVar[int]
    NUM_RESPONSE_ITEMS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    AUTHENTICATION_INFO_FIELD_NUMBER: _ClassVar[int]
    AUTHORIZATION_INFO_FIELD_NUMBER: _ClassVar[int]
    POLICY_VIOLATION_INFO_FIELD_NUMBER: _ClassVar[int]
    REQUEST_METADATA_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SERVICE_DATA_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    method_name: str
    resource_name: str
    resource_location: ResourceLocation
    resource_original_state: _struct_pb2.Struct
    num_response_items: int
    status: _status_pb2.Status
    authentication_info: AuthenticationInfo
    authorization_info: _containers.RepeatedCompositeFieldContainer[AuthorizationInfo]
    policy_violation_info: PolicyViolationInfo
    request_metadata: RequestMetadata
    request: _struct_pb2.Struct
    response: _struct_pb2.Struct
    metadata: _struct_pb2.Struct
    service_data: _any_pb2.Any
    def __init__(
        self,
        service_name: _Optional[str] = ...,
        method_name: _Optional[str] = ...,
        resource_name: _Optional[str] = ...,
        resource_location: _Optional[_Union[ResourceLocation, _Mapping]] = ...,
        resource_original_state: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        num_response_items: _Optional[int] = ...,
        status: _Optional[_Union[_status_pb2.Status, _Mapping]] = ...,
        authentication_info: _Optional[_Union[AuthenticationInfo, _Mapping]] = ...,
        authorization_info: _Optional[
            _Iterable[_Union[AuthorizationInfo, _Mapping]]
        ] = ...,
        policy_violation_info: _Optional[_Union[PolicyViolationInfo, _Mapping]] = ...,
        request_metadata: _Optional[_Union[RequestMetadata, _Mapping]] = ...,
        request: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        response: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        service_data: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...,
    ) -> None: ...

class AuthenticationInfo(_message.Message):
    __slots__ = (
        "principal_email",
        "authority_selector",
        "third_party_principal",
        "service_account_key_name",
        "service_account_delegation_info",
        "principal_subject",
    )
    PRINCIPAL_EMAIL_FIELD_NUMBER: _ClassVar[int]
    AUTHORITY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    THIRD_PARTY_PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ACCOUNT_KEY_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ACCOUNT_DELEGATION_INFO_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_SUBJECT_FIELD_NUMBER: _ClassVar[int]
    principal_email: str
    authority_selector: str
    third_party_principal: _struct_pb2.Struct
    service_account_key_name: str
    service_account_delegation_info: _containers.RepeatedCompositeFieldContainer[
        ServiceAccountDelegationInfo
    ]
    principal_subject: str
    def __init__(
        self,
        principal_email: _Optional[str] = ...,
        authority_selector: _Optional[str] = ...,
        third_party_principal: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        service_account_key_name: _Optional[str] = ...,
        service_account_delegation_info: _Optional[
            _Iterable[_Union[ServiceAccountDelegationInfo, _Mapping]]
        ] = ...,
        principal_subject: _Optional[str] = ...,
    ) -> None: ...

class AuthorizationInfo(_message.Message):
    __slots__ = (
        "resource",
        "permission",
        "granted",
        "resource_attributes",
        "permission_type",
    )

    class PermissionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        PERMISSION_TYPE_UNSPECIFIED: _ClassVar[AuthorizationInfo.PermissionType]
        ADMIN_READ: _ClassVar[AuthorizationInfo.PermissionType]
        ADMIN_WRITE: _ClassVar[AuthorizationInfo.PermissionType]
        DATA_READ: _ClassVar[AuthorizationInfo.PermissionType]
        DATA_WRITE: _ClassVar[AuthorizationInfo.PermissionType]
    PERMISSION_TYPE_UNSPECIFIED: AuthorizationInfo.PermissionType
    ADMIN_READ: AuthorizationInfo.PermissionType
    ADMIN_WRITE: AuthorizationInfo.PermissionType
    DATA_READ: AuthorizationInfo.PermissionType
    DATA_WRITE: AuthorizationInfo.PermissionType
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    PERMISSION_FIELD_NUMBER: _ClassVar[int]
    GRANTED_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    PERMISSION_TYPE_FIELD_NUMBER: _ClassVar[int]
    resource: str
    permission: str
    granted: bool
    resource_attributes: _attribute_context_pb2.AttributeContext.Resource
    permission_type: AuthorizationInfo.PermissionType
    def __init__(
        self,
        resource: _Optional[str] = ...,
        permission: _Optional[str] = ...,
        granted: bool = ...,
        resource_attributes: _Optional[
            _Union[_attribute_context_pb2.AttributeContext.Resource, _Mapping]
        ] = ...,
        permission_type: _Optional[_Union[AuthorizationInfo.PermissionType, str]] = ...,
    ) -> None: ...

class RequestMetadata(_message.Message):
    __slots__ = (
        "caller_ip",
        "caller_supplied_user_agent",
        "caller_network",
        "request_attributes",
        "destination_attributes",
    )
    CALLER_IP_FIELD_NUMBER: _ClassVar[int]
    CALLER_SUPPLIED_USER_AGENT_FIELD_NUMBER: _ClassVar[int]
    CALLER_NETWORK_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    caller_ip: str
    caller_supplied_user_agent: str
    caller_network: str
    request_attributes: _attribute_context_pb2.AttributeContext.Request
    destination_attributes: _attribute_context_pb2.AttributeContext.Peer
    def __init__(
        self,
        caller_ip: _Optional[str] = ...,
        caller_supplied_user_agent: _Optional[str] = ...,
        caller_network: _Optional[str] = ...,
        request_attributes: _Optional[
            _Union[_attribute_context_pb2.AttributeContext.Request, _Mapping]
        ] = ...,
        destination_attributes: _Optional[
            _Union[_attribute_context_pb2.AttributeContext.Peer, _Mapping]
        ] = ...,
    ) -> None: ...

class ResourceLocation(_message.Message):
    __slots__ = ("current_locations", "original_locations")
    CURRENT_LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    current_locations: _containers.RepeatedScalarFieldContainer[str]
    original_locations: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        current_locations: _Optional[_Iterable[str]] = ...,
        original_locations: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class ServiceAccountDelegationInfo(_message.Message):
    __slots__ = ("principal_subject", "first_party_principal", "third_party_principal")

    class FirstPartyPrincipal(_message.Message):
        __slots__ = ("principal_email", "service_metadata")
        PRINCIPAL_EMAIL_FIELD_NUMBER: _ClassVar[int]
        SERVICE_METADATA_FIELD_NUMBER: _ClassVar[int]
        principal_email: str
        service_metadata: _struct_pb2.Struct
        def __init__(
            self,
            principal_email: _Optional[str] = ...,
            service_metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        ) -> None: ...

    class ThirdPartyPrincipal(_message.Message):
        __slots__ = ("third_party_claims",)
        THIRD_PARTY_CLAIMS_FIELD_NUMBER: _ClassVar[int]
        third_party_claims: _struct_pb2.Struct
        def __init__(
            self,
            third_party_claims: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        ) -> None: ...
    PRINCIPAL_SUBJECT_FIELD_NUMBER: _ClassVar[int]
    FIRST_PARTY_PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    THIRD_PARTY_PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    principal_subject: str
    first_party_principal: ServiceAccountDelegationInfo.FirstPartyPrincipal
    third_party_principal: ServiceAccountDelegationInfo.ThirdPartyPrincipal
    def __init__(
        self,
        principal_subject: _Optional[str] = ...,
        first_party_principal: _Optional[
            _Union[ServiceAccountDelegationInfo.FirstPartyPrincipal, _Mapping]
        ] = ...,
        third_party_principal: _Optional[
            _Union[ServiceAccountDelegationInfo.ThirdPartyPrincipal, _Mapping]
        ] = ...,
    ) -> None: ...

class PolicyViolationInfo(_message.Message):
    __slots__ = ("org_policy_violation_info",)
    ORG_POLICY_VIOLATION_INFO_FIELD_NUMBER: _ClassVar[int]
    org_policy_violation_info: OrgPolicyViolationInfo
    def __init__(
        self,
        org_policy_violation_info: _Optional[
            _Union[OrgPolicyViolationInfo, _Mapping]
        ] = ...,
    ) -> None: ...

class OrgPolicyViolationInfo(_message.Message):
    __slots__ = ("payload", "resource_type", "resource_tags", "violation_info")

    class ResourceTagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_TAGS_FIELD_NUMBER: _ClassVar[int]
    VIOLATION_INFO_FIELD_NUMBER: _ClassVar[int]
    payload: _struct_pb2.Struct
    resource_type: str
    resource_tags: _containers.ScalarMap[str, str]
    violation_info: _containers.RepeatedCompositeFieldContainer[ViolationInfo]
    def __init__(
        self,
        payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        resource_type: _Optional[str] = ...,
        resource_tags: _Optional[_Mapping[str, str]] = ...,
        violation_info: _Optional[_Iterable[_Union[ViolationInfo, _Mapping]]] = ...,
    ) -> None: ...

class ViolationInfo(_message.Message):
    __slots__ = ("constraint", "error_message", "checked_value", "policy_type")

    class PolicyType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        POLICY_TYPE_UNSPECIFIED: _ClassVar[ViolationInfo.PolicyType]
        BOOLEAN_CONSTRAINT: _ClassVar[ViolationInfo.PolicyType]
        LIST_CONSTRAINT: _ClassVar[ViolationInfo.PolicyType]
        CUSTOM_CONSTRAINT: _ClassVar[ViolationInfo.PolicyType]
    POLICY_TYPE_UNSPECIFIED: ViolationInfo.PolicyType
    BOOLEAN_CONSTRAINT: ViolationInfo.PolicyType
    LIST_CONSTRAINT: ViolationInfo.PolicyType
    CUSTOM_CONSTRAINT: ViolationInfo.PolicyType
    CONSTRAINT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CHECKED_VALUE_FIELD_NUMBER: _ClassVar[int]
    POLICY_TYPE_FIELD_NUMBER: _ClassVar[int]
    constraint: str
    error_message: str
    checked_value: str
    policy_type: ViolationInfo.PolicyType
    def __init__(
        self,
        constraint: _Optional[str] = ...,
        error_message: _Optional[str] = ...,
        checked_value: _Optional[str] = ...,
        policy_type: _Optional[_Union[ViolationInfo.PolicyType, str]] = ...,
    ) -> None: ...
