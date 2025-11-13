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

from google.cloud.aiplatform_v1beta1.types import content
from google.cloud.aiplatform_v1beta1.types import example
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ExampleStore",
        "ExampleStoreConfig",
        "StoredContentsExampleFilter",
        "StoredContentsExampleParameters",
        "ExamplesArrayFilter",
    },
)


class ExampleStore(proto.Message):
    r"""Represents an executable service to manage and retrieve
    examples.

    Attributes:
        name (str):
            Identifier. The resource name of the ExampleStore. This is a
            unique identifier. Format:
            projects/{project}/locations/{location}/exampleStores/{example_store}
        display_name (str):
            Required. Display name of the ExampleStore.
        description (str):
            Optional. Description of the ExampleStore.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this ExampleStore
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this ExampleStore
            was most recently updated.
        example_store_config (google.cloud.aiplatform_v1beta1.types.ExampleStoreConfig):
            Required. Example Store config.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    example_store_config: "ExampleStoreConfig" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="ExampleStoreConfig",
    )


class ExampleStoreConfig(proto.Message):
    r"""Configuration for the Example Store.

    Attributes:
        vertex_embedding_model (str):
            Required. The embedding model to be used for vector
            embedding. Immutable. Supported models:

            - "textembedding-gecko@003"
            - "text-embedding-004"
            - "text-embedding-005"
            - "text-multilingual-embedding-002".
    """

    vertex_embedding_model: str = proto.Field(
        proto.STRING,
        number=1,
    )


class StoredContentsExampleFilter(proto.Message):
    r"""The metadata filters that will be used to remove or fetch
    StoredContentsExamples. If a field is unspecified, then no
    filtering for that field will be applied.

    Attributes:
        search_keys (MutableSequence[str]):
            Optional. The search keys for filtering. Only examples with
            one of the specified search keys
            ([StoredContentsExample.search_key][google.cloud.aiplatform.v1beta1.StoredContentsExample.search_key])
            are eligible to be returned.
        function_names (google.cloud.aiplatform_v1beta1.types.ExamplesArrayFilter):
            Optional. The function names for filtering.
    """

    search_keys: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    function_names: "ExamplesArrayFilter" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ExamplesArrayFilter",
    )


class StoredContentsExampleParameters(proto.Message):
    r"""The metadata filters that will be used to search
    StoredContentsExamples. If a field is unspecified, then no
    filtering for that field will be applied

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        search_key (str):
            The exact search key to use for retrieval.

            This field is a member of `oneof`_ ``query``.
        content_search_key (google.cloud.aiplatform_v1beta1.types.StoredContentsExampleParameters.ContentSearchKey):
            The chat history to use to generate the
            search key for retrieval.

            This field is a member of `oneof`_ ``query``.
        function_names (google.cloud.aiplatform_v1beta1.types.ExamplesArrayFilter):
            Optional. The function names for filtering.
    """

    class ContentSearchKey(proto.Message):
        r"""The chat history to use to generate the search key for
        retrieval.

        Attributes:
            contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
                Required. The conversation for generating a
                search key.
            search_key_generation_method (google.cloud.aiplatform_v1beta1.types.StoredContentsExample.SearchKeyGenerationMethod):
                Required. The method of generating a search
                key.
        """

        contents: MutableSequence[content.Content] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=content.Content,
        )
        search_key_generation_method: (
            example.StoredContentsExample.SearchKeyGenerationMethod
        ) = proto.Field(
            proto.MESSAGE,
            number=2,
            message=example.StoredContentsExample.SearchKeyGenerationMethod,
        )

    search_key: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="query",
    )
    content_search_key: ContentSearchKey = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="query",
        message=ContentSearchKey,
    )
    function_names: "ExamplesArrayFilter" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="ExamplesArrayFilter",
    )


class ExamplesArrayFilter(proto.Message):
    r"""Filters for examples' array metadata fields. An array field
    is example metadata where multiple values are attributed to a
    single example.

    Attributes:
        values (MutableSequence[str]):
            Required. The values by which to filter
            examples.
        array_operator (google.cloud.aiplatform_v1beta1.types.ExamplesArrayFilter.ArrayOperator):
            Required. The operator logic to use for
            filtering.
    """

    class ArrayOperator(proto.Enum):
        r"""The logic to use for filtering.

        Values:
            ARRAY_OPERATOR_UNSPECIFIED (0):
                Not specified. This value should not be used.
            CONTAINS_ANY (1):
                The metadata array field in the example must
                contain at least one of the values.
            CONTAINS_ALL (2):
                The metadata array field in the example must
                contain all of the values.
        """

        ARRAY_OPERATOR_UNSPECIFIED = 0
        CONTAINS_ANY = 1
        CONTAINS_ALL = 2

    values: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    array_operator: ArrayOperator = proto.Field(
        proto.ENUM,
        number=2,
        enum=ArrayOperator,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
