# Copyright 2024 Google LLC
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

import datetime
import json
from typing import List, Optional
from typing_extensions import override

from google.cloud.aiplatform import base as aiplatform_base
from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform import utils as aiplatform_utils
from google.cloud.aiplatform.compat.types import (
    cached_content_v1beta1 as gca_cached_content,
)
from google.cloud.aiplatform_v1.services import (
    gen_ai_cache_service as gen_ai_cache_service_v1,
)
from google.cloud.aiplatform_v1beta1.types.cached_content import (
    CachedContent as GapicCachedContent,
)
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)
from google.cloud.aiplatform_v1beta1.types.gen_ai_cache_service import (
    CreateCachedContentRequest,
    GetCachedContentRequest,
    UpdateCachedContentRequest,
)
from google.cloud.aiplatform_v1 import types as types_v1
from google.cloud.aiplatform_v1beta1.types import EncryptionSpec
from vertexai.generative_models import _generative_models
from vertexai.generative_models._generative_models import (
    Content,
    PartsType,
    Tool,
    ToolConfig,
    ContentsType,
)
from google.protobuf import field_mask_pb2
from vertexai._utils import warning_logs


def _prepare_create_request(
    model_name: str,
    *,
    system_instruction: Optional[PartsType] = None,
    tools: Optional[List[Tool]] = None,
    tool_config: Optional[ToolConfig] = None,
    contents: Optional[ContentsType] = None,
    expire_time: Optional[datetime.datetime] = None,
    ttl: Optional[datetime.timedelta] = None,
    display_name: Optional[str] = None,
    kms_key_name: Optional[str] = None,
) -> CreateCachedContentRequest:
    """Prepares the request create_cached_content RPC."""
    (
        project,
        location,
    ) = aiplatform_initializer.global_config._get_default_project_and_location()
    if contents:
        _generative_models._validate_contents_type_as_valid_sequence(contents)
    if tools:
        _generative_models._validate_tools_type_as_valid_sequence(tools)
    if tool_config:
        _generative_models._validate_tool_config_type(tool_config)

    # contents can either be a list of Content objects (most generic case)
    if contents:
        contents = _generative_models._content_types_to_gapic_contents(contents)

    gapic_system_instruction: Optional[gapic_content_types.Content] = None
    if system_instruction:
        gapic_system_instruction = _generative_models._to_content(system_instruction)

    gapic_tools = None
    if tools:
        gapic_tools = _generative_models._tool_types_to_gapic_tools(tools)

    gapic_tool_config = None
    if tool_config:
        gapic_tool_config = tool_config._gapic_tool_config

    if ttl and expire_time:
        raise ValueError("Only one of ttl and expire_time can be set.")

    request_v1beta1 = CreateCachedContentRequest(
        parent=f"projects/{project}/locations/{location}",
        cached_content=GapicCachedContent(
            model=model_name,
            system_instruction=gapic_system_instruction,
            tools=gapic_tools,
            tool_config=gapic_tool_config,
            contents=contents,
            expire_time=expire_time,
            ttl=ttl,
            display_name=display_name,
            encryption_spec=(
                EncryptionSpec(kms_key_name=kms_key_name) if kms_key_name else None
            ),
        ),
    )
    serialized_message_v1beta1 = type(request_v1beta1).serialize(request_v1beta1)
    try:
        request_v1 = types_v1.CreateCachedContentRequest.deserialize(
            serialized_message_v1beta1
        )
    except Exception as ex:
        raise ValueError(
            "Failed to convert CreateCachedContentRequest from v1beta1 to v1:\n"
            f"{serialized_message_v1beta1}"
        ) from ex
    return request_v1


def _prepare_get_cached_content_request(name: str) -> GetCachedContentRequest:
    return types_v1.GetCachedContentRequest(name=name)


class CachedContent(aiplatform_base._VertexAiResourceNounPlus):
    """A cached content resource."""

    _resource_noun = "cachedContent"
    _getter_method = "get_cached_content"
    _list_method = "list_cached_contents"
    _delete_method = "delete_cached_content"
    _parse_resource_name_method = "parse_cached_content_path"
    _format_resource_name_method = "cached_content_path"

    client_class = aiplatform_utils.GenAiCacheServiceClientWithOverride

    _gen_ai_cache_service_client_value: Optional[
        gen_ai_cache_service_v1.GenAiCacheServiceClient
    ] = None

    def __init__(self, cached_content_name: str):
        """Represents a cached content resource.

        This resource can be used with vertexai.generative_models.GenerativeModel
        to cache the prefix so it can be used across multiple generate_content
        requests.

        Args:
            cached_content_name (str):
                Required. The name of the cached content resource. It could be a
                fully-qualified CachedContent resource name or a CachedContent
                ID. Example: "projects/.../locations/../cachedContents/456" or
                "456".
        """
        warning_logs.show_deprecation_warning()
        super().__init__(resource_name=cached_content_name)
        self._gca_resource = self._get_gca_resource(cached_content_name)

    @property
    def _raw_cached_content(self) -> gca_cached_content.CachedContent:
        return self._gca_resource

    @property
    def model_name(self) -> str:
        return self._gca_resource.model

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        system_instruction: Optional[Content] = None,
        tools: Optional[List[Tool]] = None,
        tool_config: Optional[ToolConfig] = None,
        contents: Optional[List[Content]] = None,
        expire_time: Optional[datetime.datetime] = None,
        ttl: Optional[datetime.timedelta] = None,
        display_name: Optional[str] = None,
        kms_key_name: Optional[str] = None,
    ) -> "CachedContent":
        """Creates a new cached content through the gen ai cache service.

        Usage:

        Args:
            model:
                Immutable. The name of the publisher model to use for cached
                content.
                Allowed formats:

                projects/{project}/locations/{location}/publishers/{publisher}/models/{model}, or
                publishers/{publisher}/models/{model}, or
                a single model name.
            system_instruction:
                Optional. Immutable. Developer-set system instruction.
                Currently, text only.
            contents:
                Optional. Immutable. The content to cache as a list of Content.
            tools:
                Optional. Immutable. A list of ``Tools`` the model may use to
                generate the next response.
            tool_config:
                Optional. Immutable. Tool config. This config is shared for all
                tools.
            expire_time:
                Timestamp of when this resource is considered expired.

                At most one of expire_time and ttl can be set. If neither is set,
                default TTL on the API side will be used (currently 1 hour).
            ttl:
                The TTL for this resource. If provided, the expiration time is
                computed: created_time + TTL.

                At most one of expire_time and ttl can be set. If neither is set,
                default TTL on the API side will be used (currently 1 hour).
            display_name:
                The user-generated meaningful display name of the cached content.
            kms_key_name:
                Optional. Customer-managed encryption key. See
                https://cloud.google.com/vertex-ai/docs/general/cmek for more
                details. If this is set, then all created CachedContent objects
                will be encrypted with the provided encryption key.
                Allowed formats:

                projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}
        Returns:
            A CachedContent object with only name and model_name specified.
        Raises:
            ValueError: If both expire_time and ttl are set.
        """
        project = aiplatform_initializer.global_config.project
        location = aiplatform_initializer.global_config.location
        if model_name.startswith("publishers/"):
            model_name = f"projects/{project}/locations/{location}/{model_name}"
        elif not model_name.startswith("projects/"):
            model_name = f"projects/{project}/locations/{location}/publishers/google/models/{model_name}"

        if ttl and expire_time:
            raise ValueError("Only one of ttl and expire_time can be set.")

        request = _prepare_create_request(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
            contents=contents,
            expire_time=expire_time,
            ttl=ttl,
            display_name=display_name,
            kms_key_name=kms_key_name,
        )
        client = cls._instantiate_client(location=location)
        cached_content_resource = client.create_cached_content(request)
        obj = cls(cached_content_resource.name)
        obj._gca_resource = cached_content_resource
        return obj

    def refresh(self):
        """Syncs the local cached content with the remote resource."""
        self._sync_gca_resource()

    def update(
        self,
        *,
        expire_time: Optional[datetime.datetime] = None,
        ttl: Optional[datetime.timedelta] = None,
    ):
        """Updates the expiration time of the cached content."""
        if expire_time and ttl:
            raise ValueError("Only one of ttl and expire_time can be set.")
        update_mask: List[str] = []

        if ttl:
            update_mask.append("ttl")

        if expire_time:
            update_mask.append("expire_time")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)
        request_v1beta1 = UpdateCachedContentRequest(
            cached_content=GapicCachedContent(
                name=self.resource_name,
                expire_time=expire_time,
                ttl=ttl,
            ),
            update_mask=update_mask,
        )
        serialized_message_v1beta1 = type(request_v1beta1).serialize(request_v1beta1)
        try:
            request_v1 = types_v1.UpdateCachedContentRequest.deserialize(
                serialized_message_v1beta1
            )
        except Exception as ex:
            raise ValueError(
                "Failed to convert UpdateCachedContentRequest from v1beta1 to v1:\n"
                f"{serialized_message_v1beta1}"
            ) from ex
        self.api_client.update_cached_content(request_v1)

    @property
    def expire_time(self) -> datetime.datetime:
        """Time this resource is considered expired.

        The returned value may be stale. Use refresh() to get the latest value.

        Returns:
            The expiration time of the cached content resource.
        """
        return self._gca_resource.expire_time

    def delete(self):
        """Deletes the current cached content resource."""
        self._delete()

    @override
    def __repr__(self) -> str:
        return f"{object.__repr__(self)}: {json.dumps(self.to_dict(), indent=2)}"

    @classmethod
    def list(cls) -> List["CachedContent"]:
        """Lists the active cached content resources."""
        # TODO(b/345326114): Make list() interface richer after aligning with
        # Google AI SDK
        return cls._list()

    @classmethod
    def get(cls, cached_content_name: str) -> "CachedContent":
        """Retrieves an existing cached content resource."""
        cache = cls(cached_content_name)
        return cache

    @override
    @property
    def display_name(self) -> str:
        """Display name of this resource."""
        return self._gca_resource.display_name
