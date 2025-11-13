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
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1alpha.types import document as gcd_document

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "GetDocumentRequest",
        "ListDocumentsRequest",
        "ListDocumentsResponse",
        "CreateDocumentRequest",
        "UpdateDocumentRequest",
        "DeleteDocumentRequest",
        "GetProcessedDocumentRequest",
        "BatchGetDocumentsMetadataRequest",
        "BatchGetDocumentsMetadataResponse",
    },
)


class GetDocumentRequest(proto.Message):
    r"""Request message for
    [DocumentService.GetDocument][google.cloud.discoveryengine.v1alpha.DocumentService.GetDocument]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}/documents/{document}``.

            If the caller does not have permission to access the
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            If the requested
            [Document][google.cloud.discoveryengine.v1alpha.Document]
            does not exist, a ``NOT_FOUND`` error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListDocumentsRequest(proto.Message):
    r"""Request message for
    [DocumentService.ListDocuments][google.cloud.discoveryengine.v1alpha.DocumentService.ListDocuments]
    method.

    Attributes:
        parent (str):
            Required. The parent branch resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}``.
            Use ``default_branch`` as the branch ID, to list documents
            under the default branch.

            If the caller does not have permission to list
            [Document][google.cloud.discoveryengine.v1alpha.Document]s
            under this branch, regardless of whether or not this branch
            exists, a ``PERMISSION_DENIED`` error is returned.
        page_size (int):
            Maximum number of
            [Document][google.cloud.discoveryengine.v1alpha.Document]s
            to return. If unspecified, defaults to 100. The maximum
            allowed value is 1000. Values above 1000 are set to 1000.

            If this field is negative, an ``INVALID_ARGUMENT`` error is
            returned.
        page_token (str):
            A page token
            [ListDocumentsResponse.next_page_token][google.cloud.discoveryengine.v1alpha.ListDocumentsResponse.next_page_token],
            received from a previous
            [DocumentService.ListDocuments][google.cloud.discoveryengine.v1alpha.DocumentService.ListDocuments]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [DocumentService.ListDocuments][google.cloud.discoveryengine.v1alpha.DocumentService.ListDocuments]
            must match the call that provided the page token. Otherwise,
            an ``INVALID_ARGUMENT`` error is returned.
    """

    parent: str = proto.Field(
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


class ListDocumentsResponse(proto.Message):
    r"""Response message for
    [DocumentService.ListDocuments][google.cloud.discoveryengine.v1alpha.DocumentService.ListDocuments]
    method.

    Attributes:
        documents (MutableSequence[google.cloud.discoveryengine_v1alpha.types.Document]):
            The
            [Document][google.cloud.discoveryengine.v1alpha.Document]s.
        next_page_token (str):
            A token that can be sent as
            [ListDocumentsRequest.page_token][google.cloud.discoveryengine.v1alpha.ListDocumentsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    documents: MutableSequence[gcd_document.Document] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_document.Document,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateDocumentRequest(proto.Message):
    r"""Request message for
    [DocumentService.CreateDocument][google.cloud.discoveryengine.v1alpha.DocumentService.CreateDocument]
    method.

    Attributes:
        parent (str):
            Required. The parent resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}``.
        document (google.cloud.discoveryengine_v1alpha.types.Document):
            Required. The
            [Document][google.cloud.discoveryengine.v1alpha.Document] to
            create.
        document_id (str):
            Required. The ID to use for the
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            which becomes the final component of the
            [Document.name][google.cloud.discoveryengine.v1alpha.Document.name].

            If the caller does not have permission to create the
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            This field must be unique among all
            [Document][google.cloud.discoveryengine.v1alpha.Document]s
            with the same
            [parent][google.cloud.discoveryengine.v1alpha.CreateDocumentRequest.parent].
            Otherwise, an ``ALREADY_EXISTS`` error is returned.

            This field must conform to
            `RFC-1034 <https://tools.ietf.org/html/rfc1034>`__ standard
            with a length limit of 63 characters. Otherwise, an
            ``INVALID_ARGUMENT`` error is returned.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    document: gcd_document.Document = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_document.Document,
    )
    document_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class UpdateDocumentRequest(proto.Message):
    r"""Request message for
    [DocumentService.UpdateDocument][google.cloud.discoveryengine.v1alpha.DocumentService.UpdateDocument]
    method.

    Attributes:
        document (google.cloud.discoveryengine_v1alpha.types.Document):
            Required. The document to update/create.

            If the caller does not have permission to update the
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            If the
            [Document][google.cloud.discoveryengine.v1alpha.Document] to
            update does not exist and
            [allow_missing][google.cloud.discoveryengine.v1alpha.UpdateDocumentRequest.allow_missing]
            is not set, a ``NOT_FOUND`` error is returned.
        allow_missing (bool):
            If set to ``true`` and the
            [Document][google.cloud.discoveryengine.v1alpha.Document] is
            not found, a new
            [Document][google.cloud.discoveryengine.v1alpha.Document] is
            be created.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            imported 'document' to update. If not set, by
            default updates all fields.
    """

    document: gcd_document.Document = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_document.Document,
    )
    allow_missing: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=3,
        message=field_mask_pb2.FieldMask,
    )


class DeleteDocumentRequest(proto.Message):
    r"""Request message for
    [DocumentService.DeleteDocument][google.cloud.discoveryengine.v1alpha.DocumentService.DeleteDocument]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}/documents/{document}``.

            If the caller does not have permission to delete the
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            If the
            [Document][google.cloud.discoveryengine.v1alpha.Document] to
            delete does not exist, a ``NOT_FOUND`` error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetProcessedDocumentRequest(proto.Message):
    r"""Request message for
    [DocumentService.GetDocument][google.cloud.discoveryengine.v1alpha.DocumentService.GetDocument]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}/documents/{document}``.

            If the caller does not have permission to access the
            [Document][google.cloud.discoveryengine.v1alpha.Document],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            If the requested
            [Document][google.cloud.discoveryengine.v1alpha.Document]
            does not exist, a ``NOT_FOUND`` error is returned.
        processed_document_type (google.cloud.discoveryengine_v1alpha.types.GetProcessedDocumentRequest.ProcessedDocumentType):
            Required. What type of processing to return.
        processed_document_format (google.cloud.discoveryengine_v1alpha.types.GetProcessedDocumentRequest.ProcessedDocumentFormat):
            What format output should be.  If
            unspecified, defaults to JSON.
    """

    class ProcessedDocumentType(proto.Enum):
        r"""The type of processing to return in the response.

        Values:
            PROCESSED_DOCUMENT_TYPE_UNSPECIFIED (0):
                Default value.
            PARSED_DOCUMENT (1):
                Available for all data store parsing configs.
            CHUNKED_DOCUMENT (2):
                Only available if ChunkingConfig is enabled
                on the data store.
            PNG_CONVERTED_DOCUMENT (3):
                Returns the converted Image bytes (as JPEG or
                PNG) if available.
        """
        PROCESSED_DOCUMENT_TYPE_UNSPECIFIED = 0
        PARSED_DOCUMENT = 1
        CHUNKED_DOCUMENT = 2
        PNG_CONVERTED_DOCUMENT = 3

    class ProcessedDocumentFormat(proto.Enum):
        r"""The format of the returned processed document. If
        unspecified, defaults to JSON.

        Values:
            PROCESSED_DOCUMENT_FORMAT_UNSPECIFIED (0):
                Default value.
            JSON (1):
                Output format is a JSON string representation
                of processed document.
        """
        PROCESSED_DOCUMENT_FORMAT_UNSPECIFIED = 0
        JSON = 1

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    processed_document_type: ProcessedDocumentType = proto.Field(
        proto.ENUM,
        number=2,
        enum=ProcessedDocumentType,
    )
    processed_document_format: ProcessedDocumentFormat = proto.Field(
        proto.ENUM,
        number=3,
        enum=ProcessedDocumentFormat,
    )


class BatchGetDocumentsMetadataRequest(proto.Message):
    r"""Request message for
    [DocumentService.BatchGetDocumentsMetadata][google.cloud.discoveryengine.v1alpha.DocumentService.BatchGetDocumentsMetadata]
    method.

    Attributes:
        parent (str):
            Required. The parent branch resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}``.
        matcher (google.cloud.discoveryengine_v1alpha.types.BatchGetDocumentsMetadataRequest.Matcher):
            Required. Matcher for the
            [Document][google.cloud.discoveryengine.v1alpha.Document]s.
    """

    class UrisMatcher(proto.Message):
        r"""Matcher for the
        [Document][google.cloud.discoveryengine.v1alpha.Document]s by exact
        uris.

        Attributes:
            uris (MutableSequence[str]):
                The exact URIs to match by.
        """

        uris: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    class Matcher(proto.Message):
        r"""Matcher for the
        [Document][google.cloud.discoveryengine.v1alpha.Document]s.
        Currently supports matching by exact URIs.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            uris_matcher (google.cloud.discoveryengine_v1alpha.types.BatchGetDocumentsMetadataRequest.UrisMatcher):
                Matcher by exact URIs.

                This field is a member of `oneof`_ ``matcher``.
        """

        uris_matcher: "BatchGetDocumentsMetadataRequest.UrisMatcher" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="matcher",
            message="BatchGetDocumentsMetadataRequest.UrisMatcher",
        )

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    matcher: Matcher = proto.Field(
        proto.MESSAGE,
        number=2,
        message=Matcher,
    )


class BatchGetDocumentsMetadataResponse(proto.Message):
    r"""Response message for
    [DocumentService.BatchGetDocumentsMetadata][google.cloud.discoveryengine.v1alpha.DocumentService.BatchGetDocumentsMetadata]
    method.

    Attributes:
        documents_metadata (MutableSequence[google.cloud.discoveryengine_v1alpha.types.BatchGetDocumentsMetadataResponse.DocumentMetadata]):
            The metadata of the
            [Document][google.cloud.discoveryengine.v1alpha.Document]s.
    """

    class State(proto.Enum):
        r"""The state of the
        [Document][google.cloud.discoveryengine.v1alpha.Document].

        Values:
            STATE_UNSPECIFIED (0):
                Should never be set.
            INDEXED (1):
                The
                [Document][google.cloud.discoveryengine.v1alpha.Document] is
                indexed.
            NOT_IN_TARGET_SITE (2):
                The
                [Document][google.cloud.discoveryengine.v1alpha.Document] is
                not indexed because its URI is not in the
                [TargetSite][google.cloud.discoveryengine.v1alpha.TargetSite].
            NOT_IN_INDEX (3):
                The
                [Document][google.cloud.discoveryengine.v1alpha.Document] is
                not indexed.
        """
        STATE_UNSPECIFIED = 0
        INDEXED = 1
        NOT_IN_TARGET_SITE = 2
        NOT_IN_INDEX = 3

    class DocumentMetadata(proto.Message):
        r"""The metadata of a
        [Document][google.cloud.discoveryengine.v1alpha.Document].

        Attributes:
            matcher_value (google.cloud.discoveryengine_v1alpha.types.BatchGetDocumentsMetadataResponse.DocumentMetadata.MatcherValue):
                The value of the matcher that was used to match the
                [Document][google.cloud.discoveryengine.v1alpha.Document].
            state (google.cloud.discoveryengine_v1alpha.types.BatchGetDocumentsMetadataResponse.State):
                The state of the document.
            last_refreshed_time (google.protobuf.timestamp_pb2.Timestamp):
                The timestamp of the last time the
                [Document][google.cloud.discoveryengine.v1alpha.Document]
                was last indexed.
        """

        class MatcherValue(proto.Message):
            r"""The value of the matcher that was used to match the
            [Document][google.cloud.discoveryengine.v1alpha.Document].


            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                uri (str):
                    If match by URI, the URI of the
                    [Document][google.cloud.discoveryengine.v1alpha.Document].

                    This field is a member of `oneof`_ ``matcher_value``.
            """

            uri: str = proto.Field(
                proto.STRING,
                number=1,
                oneof="matcher_value",
            )

        matcher_value: "BatchGetDocumentsMetadataResponse.DocumentMetadata.MatcherValue" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="BatchGetDocumentsMetadataResponse.DocumentMetadata.MatcherValue",
        )
        state: "BatchGetDocumentsMetadataResponse.State" = proto.Field(
            proto.ENUM,
            number=3,
            enum="BatchGetDocumentsMetadataResponse.State",
        )
        last_refreshed_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=4,
            message=timestamp_pb2.Timestamp,
        )

    documents_metadata: MutableSequence[DocumentMetadata] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=DocumentMetadata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
