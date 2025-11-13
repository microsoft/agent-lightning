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

from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1beta.types import import_config

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "PurgeUserEventsRequest",
        "PurgeUserEventsResponse",
        "PurgeUserEventsMetadata",
        "PurgeErrorConfig",
        "PurgeDocumentsRequest",
        "PurgeDocumentsResponse",
        "PurgeDocumentsMetadata",
        "PurgeSuggestionDenyListEntriesRequest",
        "PurgeSuggestionDenyListEntriesResponse",
        "PurgeSuggestionDenyListEntriesMetadata",
        "PurgeCompletionSuggestionsRequest",
        "PurgeCompletionSuggestionsResponse",
        "PurgeCompletionSuggestionsMetadata",
    },
)


class PurgeUserEventsRequest(proto.Message):
    r"""Request message for PurgeUserEvents method.

    Attributes:
        parent (str):
            Required. The resource name of the catalog under which the
            events are created. The format is
            ``projects/{project}/locations/global/collections/{collection}/dataStores/{dataStore}``.
        filter (str):
            Required. The filter string to specify the events to be
            deleted with a length limit of 5,000 characters. The
            eligible fields for filtering are:

            - ``eventType``: Double quoted
              [UserEvent.event_type][google.cloud.discoveryengine.v1beta.UserEvent.event_type]
              string.
            - ``eventTime``: in ISO 8601 "zulu" format.
            - ``userPseudoId``: Double quoted string. Specifying this
              will delete all events associated with a visitor.
            - ``userId``: Double quoted string. Specifying this will
              delete all events associated with a user.

            Examples:

            - Deleting all events in a time range:
              ``eventTime > "2012-04-23T18:25:43.511Z" eventTime < "2012-04-23T18:30:43.511Z"``
            - Deleting specific eventType: ``eventType = "search"``
            - Deleting all events for a specific visitor:
              ``userPseudoId = "visitor1024"``
            - Deleting all events inside a DataStore: ``*``

            The filtering fields are assumed to have an implicit AND.
        force (bool):
            The ``force`` field is currently not supported. Purge user
            event requests will permanently delete all purgeable events.
            Once the development is complete: If ``force`` is set to
            false, the method will return the expected purge count
            without deleting any user events. This field will default to
            false if not included in the request.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class PurgeUserEventsResponse(proto.Message):
    r"""Response of the PurgeUserEventsRequest. If the long running
    operation is successfully done, then this message is returned by
    the google.longrunning.Operations.response field.

    Attributes:
        purge_count (int):
            The total count of events purged as a result
            of the operation.
    """

    purge_count: int = proto.Field(
        proto.INT64,
        number=1,
    )


class PurgeUserEventsMetadata(proto.Message):
    r"""Metadata related to the progress of the PurgeUserEvents
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
        success_count (int):
            Count of entries that were deleted
            successfully.
        failure_count (int):
            Count of entries that encountered errors
            while processing.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    success_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    failure_count: int = proto.Field(
        proto.INT64,
        number=4,
    )


class PurgeErrorConfig(proto.Message):
    r"""Configuration of destination for Purge related errors.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_prefix (str):
            Cloud Storage prefix for purge errors. This must be an
            empty, existing Cloud Storage directory. Purge errors are
            written to sharded files in this directory, one per line, as
            a JSON-encoded ``google.rpc.Status`` message.

            This field is a member of `oneof`_ ``destination``.
    """

    gcs_prefix: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="destination",
    )


class PurgeDocumentsRequest(proto.Message):
    r"""Request message for
    [DocumentService.PurgeDocuments][google.cloud.discoveryengine.v1beta.DocumentService.PurgeDocuments]
    method.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_source (google.cloud.discoveryengine_v1beta.types.GcsSource):
            Cloud Storage location for the input content. Supported
            ``data_schema``:

            - ``document_id``: One valid
              [Document.id][google.cloud.discoveryengine.v1beta.Document.id]
              per line.

            This field is a member of `oneof`_ ``source``.
        inline_source (google.cloud.discoveryengine_v1beta.types.PurgeDocumentsRequest.InlineSource):
            Inline source for the input content for
            purge.

            This field is a member of `oneof`_ ``source``.
        parent (str):
            Required. The parent resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}``.
        filter (str):
            Required. Filter matching documents to purge. Only currently
            supported value is ``*`` (all items).
        error_config (google.cloud.discoveryengine_v1beta.types.PurgeErrorConfig):
            The desired location of errors incurred
            during the purge.
        force (bool):
            Actually performs the purge. If ``force`` is set to false,
            return the expected purge count without deleting any
            documents.
    """

    class InlineSource(proto.Message):
        r"""The inline source for the input config for
        [DocumentService.PurgeDocuments][google.cloud.discoveryengine.v1beta.DocumentService.PurgeDocuments]
        method.

        Attributes:
            documents (MutableSequence[str]):
                Required. A list of full resource name of documents to
                purge. In the format
                ``projects/*/locations/*/collections/*/dataStores/*/branches/*/documents/*``.
                Recommended max of 100 items.
        """

        documents: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    gcs_source: import_config.GcsSource = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="source",
        message=import_config.GcsSource,
    )
    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="source",
        message=InlineSource,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    error_config: "PurgeErrorConfig" = proto.Field(
        proto.MESSAGE,
        number=7,
        message="PurgeErrorConfig",
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class PurgeDocumentsResponse(proto.Message):
    r"""Response message for
    [DocumentService.PurgeDocuments][google.cloud.discoveryengine.v1beta.DocumentService.PurgeDocuments]
    method. If the long running operation is successfully done, then
    this message is returned by the
    google.longrunning.Operations.response field.

    Attributes:
        purge_count (int):
            The total count of documents purged as a
            result of the operation.
        purge_sample (MutableSequence[str]):
            A sample of document names that will be deleted. Only
            populated if ``force`` is set to false. A max of 100 names
            will be returned and the names are chosen at random.
    """

    purge_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    purge_sample: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class PurgeDocumentsMetadata(proto.Message):
    r"""Metadata related to the progress of the PurgeDocuments
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
        success_count (int):
            Count of entries that were deleted
            successfully.
        failure_count (int):
            Count of entries that encountered errors
            while processing.
        ignored_count (int):
            Count of entries that were ignored as entries
            were not found.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    success_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    failure_count: int = proto.Field(
        proto.INT64,
        number=4,
    )
    ignored_count: int = proto.Field(
        proto.INT64,
        number=5,
    )


class PurgeSuggestionDenyListEntriesRequest(proto.Message):
    r"""Request message for
    [CompletionService.PurgeSuggestionDenyListEntries][google.cloud.discoveryengine.v1beta.CompletionService.PurgeSuggestionDenyListEntries]
    method.

    Attributes:
        parent (str):
            Required. The parent data store resource name for which to
            import denylist entries. Follows pattern
            projects/*/locations/*/collections/*/dataStores/*.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )


class PurgeSuggestionDenyListEntriesResponse(proto.Message):
    r"""Response message for
    [CompletionService.PurgeSuggestionDenyListEntries][google.cloud.discoveryengine.v1beta.CompletionService.PurgeSuggestionDenyListEntries]
    method.

    Attributes:
        purge_count (int):
            Number of suggestion deny list entries
            purged.
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
    """

    purge_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=status_pb2.Status,
    )


class PurgeSuggestionDenyListEntriesMetadata(proto.Message):
    r"""Metadata related to the progress of the
    PurgeSuggestionDenyListEntries operation. This is returned by
    the google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class PurgeCompletionSuggestionsRequest(proto.Message):
    r"""Request message for
    [CompletionService.PurgeCompletionSuggestions][google.cloud.discoveryengine.v1beta.CompletionService.PurgeCompletionSuggestions]
    method.

    Attributes:
        parent (str):
            Required. The parent data store resource name for which to
            purge completion suggestions. Follows pattern
            projects/*/locations/*/collections/*/dataStores/*.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )


class PurgeCompletionSuggestionsResponse(proto.Message):
    r"""Response message for
    [CompletionService.PurgeCompletionSuggestions][google.cloud.discoveryengine.v1beta.CompletionService.PurgeCompletionSuggestions]
    method.

    Attributes:
        purge_succeeded (bool):
            Whether the completion suggestions were
            successfully purged.
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
    """

    purge_succeeded: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=status_pb2.Status,
    )


class PurgeCompletionSuggestionsMetadata(proto.Message):
    r"""Metadata related to the progress of the
    PurgeCompletionSuggestions operation. This is returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
