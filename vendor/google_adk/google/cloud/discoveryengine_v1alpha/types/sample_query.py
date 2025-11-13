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
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "SampleQuery",
    },
)


class SampleQuery(proto.Message):
    r"""Sample Query captures metadata to be used for evaluation.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        query_entry (google.cloud.discoveryengine_v1alpha.types.SampleQuery.QueryEntry):
            The query entry.

            This field is a member of `oneof`_ ``content``.
        name (str):
            Identifier. The full resource name of the sample query, in
            the format of
            ``projects/{project}/locations/{location}/sampleQuerySets/{sample_query_set}/sampleQueries/{sample_query}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]
            was created at.
    """

    class QueryEntry(proto.Message):
        r"""Query Entry captures metadata to be used for search
        evaluation.

        Attributes:
            query (str):
                Required. The query.
            targets (MutableSequence[google.cloud.discoveryengine_v1alpha.types.SampleQuery.QueryEntry.Target]):
                List of targets for the query.
        """

        class Target(proto.Message):
            r"""Defines the parameters of the query's expected outcome.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                uri (str):
                    Expected uri of the target.

                    This field must be a UTF-8 encoded string with a length
                    limit of 2048 characters.

                    Example of valid uris: ``https://example.com/abc``,
                    ``gcs://example/example.pdf``.
                page_numbers (MutableSequence[int]):
                    Expected page numbers of the target.

                    Each page number must be non negative.
                score (float):
                    Relevance score of the target.

                    This field is a member of `oneof`_ ``_score``.
            """

            uri: str = proto.Field(
                proto.STRING,
                number=1,
            )
            page_numbers: MutableSequence[int] = proto.RepeatedField(
                proto.INT32,
                number=2,
            )
            score: float = proto.Field(
                proto.DOUBLE,
                number=3,
                optional=True,
            )

        query: str = proto.Field(
            proto.STRING,
            number=1,
        )
        targets: MutableSequence["SampleQuery.QueryEntry.Target"] = proto.RepeatedField(
            proto.MESSAGE,
            number=3,
            message="SampleQuery.QueryEntry.Target",
        )

    query_entry: QueryEntry = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="content",
        message=QueryEntry,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
