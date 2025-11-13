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

from google.cloud.aiplatform_v1beta1.types import example as gca_example
from google.cloud.aiplatform_v1beta1.types import example_store as gca_example_store
from google.cloud.aiplatform_v1beta1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateExampleStoreRequest",
        "CreateExampleStoreOperationMetadata",
        "GetExampleStoreRequest",
        "UpdateExampleStoreRequest",
        "UpdateExampleStoreOperationMetadata",
        "DeleteExampleStoreRequest",
        "DeleteExampleStoreOperationMetadata",
        "ListExampleStoresRequest",
        "ListExampleStoresResponse",
        "Example",
        "UpsertExamplesRequest",
        "UpsertExamplesResponse",
        "RemoveExamplesRequest",
        "RemoveExamplesResponse",
        "SearchExamplesRequest",
        "SearchExamplesResponse",
        "FetchExamplesRequest",
        "FetchExamplesResponse",
    },
)


class CreateExampleStoreRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.CreateExampleStore][google.cloud.aiplatform.v1beta1.ExampleStoreService.CreateExampleStore].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            ExampleStore in. Format:
            ``projects/{project}/locations/{location}``
        example_store (google.cloud.aiplatform_v1beta1.types.ExampleStore):
            Required. The Example Store to be created.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    example_store: gca_example_store.ExampleStore = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_example_store.ExampleStore,
    )


class CreateExampleStoreOperationMetadata(proto.Message):
    r"""Details of
    [ExampleStoreService.CreateExampleStore][google.cloud.aiplatform.v1beta1.ExampleStoreService.CreateExampleStore]
    operation.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetExampleStoreRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.GetExampleStore][google.cloud.aiplatform.v1beta1.ExampleStoreService.GetExampleStore].

    Attributes:
        name (str):
            Required. The resource name of the ExampleStore. Format:
            ``projects/{project}/locations/{location}/exampleStores/{example_store}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateExampleStoreRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.UpdateExampleStore][google.cloud.aiplatform.v1beta1.ExampleStoreService.UpdateExampleStore].

    Attributes:
        example_store (google.cloud.aiplatform_v1beta1.types.ExampleStore):
            Required. The Example Store which replaces
            the resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Mask specifying which fields to update. Supported
            fields:

            ::

               * `display_name`
               * `description`
    """

    example_store: gca_example_store.ExampleStore = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_example_store.ExampleStore,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateExampleStoreOperationMetadata(proto.Message):
    r"""Details of
    [ExampleStoreService.UpdateExampleStore][google.cloud.aiplatform.v1beta1.ExampleStoreService.UpdateExampleStore]
    operation.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class DeleteExampleStoreRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.DeleteExampleStore][google.cloud.aiplatform.v1beta1.ExampleStoreService.DeleteExampleStore].

    Attributes:
        name (str):
            Required. The resource name of the ExampleStore to be
            deleted. Format:
            ``projects/{project}/locations/{location}/exampleStores/{example_store}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteExampleStoreOperationMetadata(proto.Message):
    r"""Details of
    [ExampleStoreService.DeleteExampleStore][google.cloud.aiplatform.v1beta1.ExampleStoreService.DeleteExampleStore]
    operation.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class ListExampleStoresRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.ListExampleStores][google.cloud.aiplatform.v1beta1.ExampleStoreService.ListExampleStores].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            ExampleStores from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Optional. The standard list filter. More detail in
            `AIP-160 <https://google.aip.dev/160>`__.
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token.
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


class ListExampleStoresResponse(proto.Message):
    r"""Response message for
    [ExampleStoreService.ListExampleStores][google.cloud.aiplatform.v1beta1.ExampleStoreService.ListExampleStores].

    Attributes:
        example_stores (MutableSequence[google.cloud.aiplatform_v1beta1.types.ExampleStore]):
            List of ExampleStore in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListExampleStoresRequest.page_token][google.cloud.aiplatform.v1beta1.ListExampleStoresRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    example_stores: MutableSequence[gca_example_store.ExampleStore] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=gca_example_store.ExampleStore,
        )
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class Example(proto.Message):
    r"""A single example to upload or read from the Example Store.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        stored_contents_example (google.cloud.aiplatform_v1beta1.types.StoredContentsExample):
            An example of chat history and its expected
            outcome to be used with GenerateContent.

            This field is a member of `oneof`_ ``example_type``.
        display_name (str):
            Optional. The display name for Example.
        example_id (str):
            Optional. Immutable. Unique identifier of an example. If not
            specified when upserting new examples, the example_id will
            be generated.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Example was
            created.
    """

    stored_contents_example: gca_example.StoredContentsExample = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="example_type",
        message=gca_example.StoredContentsExample,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    example_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )


class UpsertExamplesRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.UpsertExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.UpsertExamples].

    Attributes:
        example_store (str):
            Required. The name of the ExampleStore resource that
            examples are added to or updated in. Format:
            ``projects/{project}/locations/{location}/exampleStores/{example_store}``
        examples (MutableSequence[google.cloud.aiplatform_v1beta1.types.Example]):
            Required. A list of examples to be
            created/updated.
        overwrite (bool):
            Optional. A flag indicating whether an
            example can be overwritten if it already exists.
            If False (default) and the example already
            exists, the example will not be updated. This
            does not affect behavior if the example does not
            exist already.
    """

    example_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    examples: MutableSequence["Example"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Example",
    )
    overwrite: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class UpsertExamplesResponse(proto.Message):
    r"""Response message for
    [ExampleStoreService.UpsertExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.UpsertExamples].

    Attributes:
        results (MutableSequence[google.cloud.aiplatform_v1beta1.types.UpsertExamplesResponse.UpsertResult]):
            A list of results for creating/updating. It's
            either a successfully created/updated example or
            a status with an error message.
    """

    class UpsertResult(proto.Message):
        r"""The result for creating/updating a single example.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            example (google.cloud.aiplatform_v1beta1.types.Example):
                The example created/updated successfully.

                This field is a member of `oneof`_ ``result``.
            status (google.rpc.status_pb2.Status):
                The error message of the example that was not
                created/updated successfully.

                This field is a member of `oneof`_ ``result``.
        """

        example: "Example" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="result",
            message="Example",
        )
        status: status_pb2.Status = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="result",
            message=status_pb2.Status,
        )

    results: MutableSequence[UpsertResult] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=UpsertResult,
    )


class RemoveExamplesRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.RemoveExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.RemoveExamples].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        stored_contents_example_filter (google.cloud.aiplatform_v1beta1.types.StoredContentsExampleFilter):
            The metadata filters for
            StoredContentsExamples.

            This field is a member of `oneof`_ ``metadata_filter``.
        example_store (str):
            Required. The name of the ExampleStore resource that the
            examples should be removed from. Format:
            ``projects/{project}/locations/{location}/exampleStores/{example_store}``
        example_ids (MutableSequence[str]):
            Optional. Example IDs to remove. If both
            metadata filters and Example IDs are specified,
            the metadata filters will be applied to the
            specified examples in order to identify which
            should be removed.
    """

    stored_contents_example_filter: gca_example_store.StoredContentsExampleFilter = (
        proto.Field(
            proto.MESSAGE,
            number=8,
            oneof="metadata_filter",
            message=gca_example_store.StoredContentsExampleFilter,
        )
    )
    example_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    example_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=6,
    )


class RemoveExamplesResponse(proto.Message):
    r"""Response message for
    [ExampleStoreService.RemoveExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.RemoveExamples].

    Attributes:
        example_ids (MutableSequence[str]):
            The IDs for the removed examples.
    """

    example_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )


class SearchExamplesRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.SearchExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.SearchExamples].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        stored_contents_example_parameters (google.cloud.aiplatform_v1beta1.types.StoredContentsExampleParameters):
            The parameters of StoredContentsExamples to
            be searched.

            This field is a member of `oneof`_ ``parameters``.
        example_store (str):
            Required. The name of the ExampleStore resource that
            examples are retrieved from. Format:
            ``projects/{project}/locations/{location}/exampleStores/{example_store}``
        top_k (int):
            Optional. The number of similar examples to
            return.
    """

    stored_contents_example_parameters: (
        gca_example_store.StoredContentsExampleParameters
    ) = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="parameters",
        message=gca_example_store.StoredContentsExampleParameters,
    )
    example_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    top_k: int = proto.Field(
        proto.INT64,
        number=2,
    )


class SearchExamplesResponse(proto.Message):
    r"""Response message for
    [ExampleStoreService.SearchExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.SearchExamples].

    Attributes:
        results (MutableSequence[google.cloud.aiplatform_v1beta1.types.SearchExamplesResponse.SimilarExample]):
            The results of searching for similar
            examples.
    """

    class SimilarExample(proto.Message):
        r"""The result of the similar example.

        Attributes:
            example (google.cloud.aiplatform_v1beta1.types.Example):
                The example that is similar to the searched
                query.
            similarity_score (float):
                The similarity score of this example.
        """

        example: "Example" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Example",
        )
        similarity_score: float = proto.Field(
            proto.FLOAT,
            number=2,
        )

    results: MutableSequence[SimilarExample] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=SimilarExample,
    )


class FetchExamplesRequest(proto.Message):
    r"""Request message for
    [ExampleStoreService.FetchExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.FetchExamples].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        stored_contents_example_filter (google.cloud.aiplatform_v1beta1.types.StoredContentsExampleFilter):
            The metadata filters for
            StoredContentsExamples.

            This field is a member of `oneof`_ ``metadata_filter``.
        example_store (str):
            Required. The name of the ExampleStore resource that the
            examples should be fetched from. Format:
            ``projects/{project}/locations/{location}/exampleStores/{example_store}``
        page_size (int):
            Optional. The maximum number of examples to
            return. The service may return fewer than this
            value. If unspecified, at most 100 examples will
            be returned.
        page_token (str):
            Optional. The
            [next_page_token][google.cloud.aiplatform.v1beta1.FetchExamplesResponse.next_page_token]
            value returned from a previous list
            [ExampleStoreService.FetchExamplesResponse][] call.
        example_ids (MutableSequence[str]):
            Optional. Example IDs to fetch. If both
            metadata filters and Example IDs are specified,
            then both ID and metadata filtering will be
            applied.
    """

    stored_contents_example_filter: gca_example_store.StoredContentsExampleFilter = (
        proto.Field(
            proto.MESSAGE,
            number=8,
            oneof="metadata_filter",
            message=gca_example_store.StoredContentsExampleFilter,
        )
    )
    example_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )
    example_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=6,
    )


class FetchExamplesResponse(proto.Message):
    r"""Response message for
    [ExampleStoreService.FetchExamples][google.cloud.aiplatform.v1beta1.ExampleStoreService.FetchExamples].

    Attributes:
        examples (MutableSequence[google.cloud.aiplatform_v1beta1.types.Example]):
            The examples in the Example Store that
            satisfy the metadata filters.
        next_page_token (str):
            A token, which can be sent as
            [FetchExamplesRequest.page_token][] to retrieve the next
            page. Absence of this field indicates there are no
            subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    examples: MutableSequence["Example"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Example",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
