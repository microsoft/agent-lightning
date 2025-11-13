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
    package="google.cloud.discoveryengine.v1",
    manifest={
        "SiteSearchEngine",
        "TargetSite",
        "SiteVerificationInfo",
        "Sitemap",
    },
)


class SiteSearchEngine(proto.Message):
    r"""SiteSearchEngine captures DataStore level site search
    persisting configurations. It is a singleton value per data
    store.

    Attributes:
        name (str):
            The fully qualified resource name of the site search engine.
            Format:
            ``projects/*/locations/*/dataStores/*/siteSearchEngine``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class TargetSite(proto.Message):
    r"""A target site for the SiteSearchEngine.

    Attributes:
        name (str):
            Output only. The fully qualified resource name of the target
            site.
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine/targetSites/{target_site}``
            The ``target_site_id`` is system-generated.
        provided_uri_pattern (str):
            Required. Input only. The user provided URI pattern from
            which the ``generated_uri_pattern`` is generated.
        type_ (google.cloud.discoveryengine_v1.types.TargetSite.Type):
            The type of the target site, e.g., whether
            the site is to be included or excluded.
        exact_match (bool):
            Immutable. If set to false, a uri_pattern is generated to
            include all pages whose address contains the
            provided_uri_pattern. If set to true, an uri_pattern is
            generated to try to be an exact match of the
            provided_uri_pattern or just the specific page if the
            provided_uri_pattern is a specific one. provided_uri_pattern
            is always normalized to generate the URI pattern to be used
            by the search engine.
        generated_uri_pattern (str):
            Output only. This is system-generated based on the
            provided_uri_pattern.
        root_domain_uri (str):
            Output only. Root domain of the provided_uri_pattern.
        site_verification_info (google.cloud.discoveryengine_v1.types.SiteVerificationInfo):
            Output only. Site ownership and validity
            verification status.
        indexing_status (google.cloud.discoveryengine_v1.types.TargetSite.IndexingStatus):
            Output only. Indexing status.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The target site's last updated
            time.
        failure_reason (google.cloud.discoveryengine_v1.types.TargetSite.FailureReason):
            Output only. Failure reason.
    """

    class Type(proto.Enum):
        r"""Possible target site types.

        Values:
            TYPE_UNSPECIFIED (0):
                This value is unused. In this case, server behavior defaults
                to
                [Type.INCLUDE][google.cloud.discoveryengine.v1.TargetSite.Type.INCLUDE].
            INCLUDE (1):
                Include the target site.
            EXCLUDE (2):
                Exclude the target site.
        """
        TYPE_UNSPECIFIED = 0
        INCLUDE = 1
        EXCLUDE = 2

    class IndexingStatus(proto.Enum):
        r"""Target site indexing status enumeration.

        Values:
            INDEXING_STATUS_UNSPECIFIED (0):
                Defaults to SUCCEEDED.
            PENDING (1):
                The target site is in the update queue and
                will be picked up by indexing pipeline.
            FAILED (2):
                The target site fails to be indexed.
            SUCCEEDED (3):
                The target site has been indexed.
            DELETING (4):
                The previously indexed target site has been
                marked to be deleted. This is a transitioning
                state which will resulted in either:

                1. target site deleted if unindexing is
                    successful;
                2. state reverts to SUCCEEDED if the unindexing
                    fails.
            CANCELLABLE (5):
                The target site change is pending but
                cancellable.
            CANCELLED (6):
                The target site change is cancelled.
        """
        INDEXING_STATUS_UNSPECIFIED = 0
        PENDING = 1
        FAILED = 2
        SUCCEEDED = 3
        DELETING = 4
        CANCELLABLE = 5
        CANCELLED = 6

    class FailureReason(proto.Message):
        r"""Site search indexing failure reasons.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            quota_failure (google.cloud.discoveryengine_v1.types.TargetSite.FailureReason.QuotaFailure):
                Failed due to insufficient quota.

                This field is a member of `oneof`_ ``failure``.
        """

        class QuotaFailure(proto.Message):
            r"""Failed due to insufficient quota.

            Attributes:
                total_required_quota (int):
                    This number is an estimation on how much
                    total quota this project needs to successfully
                    complete indexing.
            """

            total_required_quota: int = proto.Field(
                proto.INT64,
                number=1,
            )

        quota_failure: "TargetSite.FailureReason.QuotaFailure" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="failure",
            message="TargetSite.FailureReason.QuotaFailure",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    provided_uri_pattern: str = proto.Field(
        proto.STRING,
        number=2,
    )
    type_: Type = proto.Field(
        proto.ENUM,
        number=3,
        enum=Type,
    )
    exact_match: bool = proto.Field(
        proto.BOOL,
        number=6,
    )
    generated_uri_pattern: str = proto.Field(
        proto.STRING,
        number=4,
    )
    root_domain_uri: str = proto.Field(
        proto.STRING,
        number=10,
    )
    site_verification_info: "SiteVerificationInfo" = proto.Field(
        proto.MESSAGE,
        number=7,
        message="SiteVerificationInfo",
    )
    indexing_status: IndexingStatus = proto.Field(
        proto.ENUM,
        number=8,
        enum=IndexingStatus,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    failure_reason: FailureReason = proto.Field(
        proto.MESSAGE,
        number=9,
        message=FailureReason,
    )


class SiteVerificationInfo(proto.Message):
    r"""Verification information for target sites in advanced site
    search.

    Attributes:
        site_verification_state (google.cloud.discoveryengine_v1.types.SiteVerificationInfo.SiteVerificationState):
            Site verification state indicating the
            ownership and validity.
        verify_time (google.protobuf.timestamp_pb2.Timestamp):
            Latest site verification time.
    """

    class SiteVerificationState(proto.Enum):
        r"""Site verification state.

        Values:
            SITE_VERIFICATION_STATE_UNSPECIFIED (0):
                Defaults to VERIFIED.
            VERIFIED (1):
                Site ownership verified.
            UNVERIFIED (2):
                Site ownership pending verification or
                verification failed.
            EXEMPTED (3):
                Site exempt from verification, e.g., a public
                website that opens to all.
        """
        SITE_VERIFICATION_STATE_UNSPECIFIED = 0
        VERIFIED = 1
        UNVERIFIED = 2
        EXEMPTED = 3

    site_verification_state: SiteVerificationState = proto.Field(
        proto.ENUM,
        number=1,
        enum=SiteVerificationState,
    )
    verify_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class Sitemap(proto.Message):
    r"""A sitemap for the SiteSearchEngine.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        uri (str):
            Public URI for the sitemap, e.g.
            ``www.example.com/sitemap.xml``.

            This field is a member of `oneof`_ ``feed``.
        name (str):
            Output only. The fully qualified resource name of the
            sitemap.
            ``projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine/sitemaps/*``
            The ``sitemap_id`` suffix is system-generated.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The sitemap's creation time.
    """

    uri: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="feed",
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
