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

from google.cloud.discoveryengine_v1.types import (
    site_search_engine as gcd_site_search_engine,
)

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "GetSiteSearchEngineRequest",
        "CreateTargetSiteRequest",
        "CreateTargetSiteMetadata",
        "BatchCreateTargetSitesRequest",
        "GetTargetSiteRequest",
        "UpdateTargetSiteRequest",
        "UpdateTargetSiteMetadata",
        "DeleteTargetSiteRequest",
        "DeleteTargetSiteMetadata",
        "ListTargetSitesRequest",
        "ListTargetSitesResponse",
        "BatchCreateTargetSiteMetadata",
        "BatchCreateTargetSitesResponse",
        "CreateSitemapRequest",
        "DeleteSitemapRequest",
        "FetchSitemapsRequest",
        "CreateSitemapMetadata",
        "DeleteSitemapMetadata",
        "FetchSitemapsResponse",
        "EnableAdvancedSiteSearchRequest",
        "EnableAdvancedSiteSearchResponse",
        "EnableAdvancedSiteSearchMetadata",
        "DisableAdvancedSiteSearchRequest",
        "DisableAdvancedSiteSearchResponse",
        "DisableAdvancedSiteSearchMetadata",
        "RecrawlUrisRequest",
        "RecrawlUrisResponse",
        "RecrawlUrisMetadata",
        "BatchVerifyTargetSitesRequest",
        "BatchVerifyTargetSitesResponse",
        "BatchVerifyTargetSitesMetadata",
        "FetchDomainVerificationStatusRequest",
        "FetchDomainVerificationStatusResponse",
    },
)


class GetSiteSearchEngineRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.GetSiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngineService.GetSiteSearchEngine]
    method.

    Attributes:
        name (str):
            Required. Resource name of
            [SiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngine],
            such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.

            If the caller does not have permission to access the
            [SiteSearchEngine], regardless of whether or not it exists,
            a PERMISSION_DENIED error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateTargetSiteRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.CreateTargetSite][google.cloud.discoveryengine.v1.SiteSearchEngineService.CreateTargetSite]
    method.

    Attributes:
        parent (str):
            Required. Parent resource name of
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite],
            such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.
        target_site (google.cloud.discoveryengine_v1.types.TargetSite):
            Required. The
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite] to
            create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    target_site: gcd_site_search_engine.TargetSite = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_site_search_engine.TargetSite,
    )


class CreateTargetSiteMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.CreateTargetSite][google.cloud.discoveryengine.v1.SiteSearchEngineService.CreateTargetSite]
    operation. This will be returned by the
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


class BatchCreateTargetSitesRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.BatchCreateTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.BatchCreateTargetSites]
    method.

    Attributes:
        parent (str):
            Required. The parent resource shared by all TargetSites
            being created.
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.
            The parent field in the CreateBookRequest messages must
            either be empty or match this field.
        requests (MutableSequence[google.cloud.discoveryengine_v1.types.CreateTargetSiteRequest]):
            Required. The request message specifying the
            resources to create. A maximum of 20 TargetSites
            can be created in a batch.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    requests: MutableSequence["CreateTargetSiteRequest"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="CreateTargetSiteRequest",
    )


class GetTargetSiteRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.GetTargetSite][google.cloud.discoveryengine.v1.SiteSearchEngineService.GetTargetSite]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite],
            such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine/targetSites/{target_site}``.

            If the caller does not have permission to access the
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the requested
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite]
            does not exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateTargetSiteRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.UpdateTargetSite][google.cloud.discoveryengine.v1.SiteSearchEngineService.UpdateTargetSite]
    method.

    Attributes:
        target_site (google.cloud.discoveryengine_v1.types.TargetSite):
            Required. The target site to update. If the caller does not
            have permission to update the
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite] to
            update does not exist, a NOT_FOUND error is returned.
    """

    target_site: gcd_site_search_engine.TargetSite = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_site_search_engine.TargetSite,
    )


class UpdateTargetSiteMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.UpdateTargetSite][google.cloud.discoveryengine.v1.SiteSearchEngineService.UpdateTargetSite]
    operation. This will be returned by the
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


class DeleteTargetSiteRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.DeleteTargetSite][google.cloud.discoveryengine.v1.SiteSearchEngineService.DeleteTargetSite]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite],
            such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine/targetSites/{target_site}``.

            If the caller does not have permission to access the
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the requested
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite]
            does not exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteTargetSiteMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.DeleteTargetSite][google.cloud.discoveryengine.v1.SiteSearchEngineService.DeleteTargetSite]
    operation. This will be returned by the
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


class ListTargetSitesRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.ListTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.ListTargetSites]
    method.

    Attributes:
        parent (str):
            Required. The parent site search engine resource name, such
            as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.

            If the caller does not have permission to list
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite]s
            under this site search engine, regardless of whether or not
            this branch exists, a PERMISSION_DENIED error is returned.
        page_size (int):
            Requested page size. Server may return fewer items than
            requested. If unspecified, server will pick an appropriate
            default. The maximum value is 1000; values above 1000 will
            be coerced to 1000.

            If this field is negative, an INVALID_ARGUMENT error is
            returned.
        page_token (str):
            A page token, received from a previous ``ListTargetSites``
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``ListTargetSites`` must match the call that provided the
            page token.
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


class ListTargetSitesResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.ListTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.ListTargetSites]
    method.

    Attributes:
        target_sites (MutableSequence[google.cloud.discoveryengine_v1.types.TargetSite]):
            List of TargetSites.
        next_page_token (str):
            A token that can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
        total_size (int):
            The total number of items matching the
            request. This will always be populated in the
            response.
    """

    @property
    def raw_page(self):
        return self

    target_sites: MutableSequence[
        gcd_site_search_engine.TargetSite
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_site_search_engine.TargetSite,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    total_size: int = proto.Field(
        proto.INT32,
        number=3,
    )


class BatchCreateTargetSiteMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.BatchCreateTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.BatchCreateTargetSites]
    operation. This will be returned by the
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


class BatchCreateTargetSitesResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.BatchCreateTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.BatchCreateTargetSites]
    method.

    Attributes:
        target_sites (MutableSequence[google.cloud.discoveryengine_v1.types.TargetSite]):
            TargetSites created.
    """

    target_sites: MutableSequence[
        gcd_site_search_engine.TargetSite
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_site_search_engine.TargetSite,
    )


class CreateSitemapRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.CreateSitemap][google.cloud.discoveryengine.v1.SiteSearchEngineService.CreateSitemap]
    method.

    Attributes:
        parent (str):
            Required. Parent resource name of the
            [SiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngine],
            such as
            ``projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine``.
        sitemap (google.cloud.discoveryengine_v1.types.Sitemap):
            Required. The
            [Sitemap][google.cloud.discoveryengine.v1.Sitemap] to
            create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    sitemap: gcd_site_search_engine.Sitemap = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_site_search_engine.Sitemap,
    )


class DeleteSitemapRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.DeleteSitemap][google.cloud.discoveryengine.v1.SiteSearchEngineService.DeleteSitemap]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Sitemap][google.cloud.discoveryengine.v1.Sitemap], such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine/sitemaps/{sitemap}``.

            If the caller does not have permission to access the
            [Sitemap][google.cloud.discoveryengine.v1.Sitemap],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the requested
            [Sitemap][google.cloud.discoveryengine.v1.Sitemap] does not
            exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class FetchSitemapsRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.FetchSitemaps][google.cloud.discoveryengine.v1.SiteSearchEngineService.FetchSitemaps]
    method.

    Attributes:
        parent (str):
            Required. Parent resource name of the
            [SiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngine],
            such as
            ``projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine``.
        matcher (google.cloud.discoveryengine_v1.types.FetchSitemapsRequest.Matcher):
            Optional. If specified, fetches the matching
            [Sitemap][google.cloud.discoveryengine.v1.Sitemap]s. If not
            specified, fetches all
            [Sitemap][google.cloud.discoveryengine.v1.Sitemap]s in the
            [DataStore][google.cloud.discoveryengine.v1.DataStore].
    """

    class UrisMatcher(proto.Message):
        r"""Matcher for the [Sitemap][google.cloud.discoveryengine.v1.Sitemap]s
        by their uris.

        Attributes:
            uris (MutableSequence[str]):
                The [Sitemap][google.cloud.discoveryengine.v1.Sitemap] uris.
        """

        uris: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    class Matcher(proto.Message):
        r"""Matcher for the [Sitemap][google.cloud.discoveryengine.v1.Sitemap]s.
        Currently only supports uris matcher.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            uris_matcher (google.cloud.discoveryengine_v1.types.FetchSitemapsRequest.UrisMatcher):
                Matcher by sitemap URIs.

                This field is a member of `oneof`_ ``matcher``.
        """

        uris_matcher: "FetchSitemapsRequest.UrisMatcher" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="matcher",
            message="FetchSitemapsRequest.UrisMatcher",
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


class CreateSitemapMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.CreateSitemap][google.cloud.discoveryengine.v1.SiteSearchEngineService.CreateSitemap]
    operation. This will be returned by the
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


class DeleteSitemapMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.DeleteSitemap][google.cloud.discoveryengine.v1.SiteSearchEngineService.DeleteSitemap]
    operation. This will be returned by the
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


class FetchSitemapsResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.FetchSitemaps][google.cloud.discoveryengine.v1.SiteSearchEngineService.FetchSitemaps]
    method.

    Attributes:
        sitemaps_metadata (MutableSequence[google.cloud.discoveryengine_v1.types.FetchSitemapsResponse.SitemapMetadata]):
            List of [Sitemap][google.cloud.discoveryengine.v1.Sitemap]s
            fetched.
    """

    class SitemapMetadata(proto.Message):
        r"""Contains a [Sitemap][google.cloud.discoveryengine.v1.Sitemap] and
        its metadata.

        Attributes:
            sitemap (google.cloud.discoveryengine_v1.types.Sitemap):
                The [Sitemap][google.cloud.discoveryengine.v1.Sitemap].
        """

        sitemap: gcd_site_search_engine.Sitemap = proto.Field(
            proto.MESSAGE,
            number=1,
            message=gcd_site_search_engine.Sitemap,
        )

    sitemaps_metadata: MutableSequence[SitemapMetadata] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=SitemapMetadata,
    )


class EnableAdvancedSiteSearchRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.EnableAdvancedSiteSearch][google.cloud.discoveryengine.v1.SiteSearchEngineService.EnableAdvancedSiteSearch]
    method.

    Attributes:
        site_search_engine (str):
            Required. Full resource name of the
            [SiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngine],
            such as
            ``projects/{project}/locations/{location}/dataStores/{data_store_id}/siteSearchEngine``.
    """

    site_search_engine: str = proto.Field(
        proto.STRING,
        number=1,
    )


class EnableAdvancedSiteSearchResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.EnableAdvancedSiteSearch][google.cloud.discoveryengine.v1.SiteSearchEngineService.EnableAdvancedSiteSearch]
    method.

    """


class EnableAdvancedSiteSearchMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.EnableAdvancedSiteSearch][google.cloud.discoveryengine.v1.SiteSearchEngineService.EnableAdvancedSiteSearch]
    operation. This will be returned by the
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


class DisableAdvancedSiteSearchRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.DisableAdvancedSiteSearch][google.cloud.discoveryengine.v1.SiteSearchEngineService.DisableAdvancedSiteSearch]
    method.

    Attributes:
        site_search_engine (str):
            Required. Full resource name of the
            [SiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngine],
            such as
            ``projects/{project}/locations/{location}/dataStores/{data_store_id}/siteSearchEngine``.
    """

    site_search_engine: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DisableAdvancedSiteSearchResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.DisableAdvancedSiteSearch][google.cloud.discoveryengine.v1.SiteSearchEngineService.DisableAdvancedSiteSearch]
    method.

    """


class DisableAdvancedSiteSearchMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.DisableAdvancedSiteSearch][google.cloud.discoveryengine.v1.SiteSearchEngineService.DisableAdvancedSiteSearch]
    operation. This will be returned by the
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


class RecrawlUrisRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.RecrawlUris][google.cloud.discoveryengine.v1.SiteSearchEngineService.RecrawlUris]
    method.

    Attributes:
        site_search_engine (str):
            Required. Full resource name of the
            [SiteSearchEngine][google.cloud.discoveryengine.v1.SiteSearchEngine],
            such as
            ``projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine``.
        uris (MutableSequence[str]):
            Required. List of URIs to crawl. At most 10K URIs are
            supported, otherwise an INVALID_ARGUMENT error is thrown.
            Each URI should match at least one
            [TargetSite][google.cloud.discoveryengine.v1.TargetSite] in
            ``site_search_engine``.
        site_credential (str):
            Optional. Credential id to use for crawling.
    """

    site_search_engine: str = proto.Field(
        proto.STRING,
        number=1,
    )
    uris: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    site_credential: str = proto.Field(
        proto.STRING,
        number=5,
    )


class RecrawlUrisResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.RecrawlUris][google.cloud.discoveryengine.v1.SiteSearchEngineService.RecrawlUris]
    method.

    Attributes:
        failure_samples (MutableSequence[google.cloud.discoveryengine_v1.types.RecrawlUrisResponse.FailureInfo]):
            Details for a sample of up to 10 ``failed_uris``.
        failed_uris (MutableSequence[str]):
            URIs that were not crawled before the LRO
            terminated.
    """

    class FailureInfo(proto.Message):
        r"""Details about why a particular URI failed to be crawled. Each
        FailureInfo contains one FailureReason per CorpusType.

        Attributes:
            uri (str):
                URI that failed to be crawled.
            failure_reasons (MutableSequence[google.cloud.discoveryengine_v1.types.RecrawlUrisResponse.FailureInfo.FailureReason]):
                List of failure reasons by corpus type (e.g.
                desktop, mobile).
        """

        class FailureReason(proto.Message):
            r"""Details about why crawling failed for a particular
            CorpusType, e.g., DESKTOP and MOBILE crawling may fail for
            different reasons.

            Attributes:
                corpus_type (google.cloud.discoveryengine_v1.types.RecrawlUrisResponse.FailureInfo.FailureReason.CorpusType):
                    DESKTOP, MOBILE, or CORPUS_TYPE_UNSPECIFIED.
                error_message (str):
                    Reason why the URI was not crawled.
            """

            class CorpusType(proto.Enum):
                r"""CorpusType for the failed crawling operation.

                Values:
                    CORPUS_TYPE_UNSPECIFIED (0):
                        Default value.
                    DESKTOP (1):
                        Denotes a crawling attempt for the desktop
                        version of a page.
                    MOBILE (2):
                        Denotes a crawling attempt for the mobile
                        version of a page.
                """
                CORPUS_TYPE_UNSPECIFIED = 0
                DESKTOP = 1
                MOBILE = 2

            corpus_type: "RecrawlUrisResponse.FailureInfo.FailureReason.CorpusType" = (
                proto.Field(
                    proto.ENUM,
                    number=1,
                    enum="RecrawlUrisResponse.FailureInfo.FailureReason.CorpusType",
                )
            )
            error_message: str = proto.Field(
                proto.STRING,
                number=2,
            )

        uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        failure_reasons: MutableSequence[
            "RecrawlUrisResponse.FailureInfo.FailureReason"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message="RecrawlUrisResponse.FailureInfo.FailureReason",
        )

    failure_samples: MutableSequence[FailureInfo] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=FailureInfo,
    )
    failed_uris: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class RecrawlUrisMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.RecrawlUris][google.cloud.discoveryengine.v1.SiteSearchEngineService.RecrawlUris]
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
        invalid_uris (MutableSequence[str]):
            Unique URIs in the request that have invalid
            format. Sample limited to 1000.
        invalid_uris_count (int):
            Total number of unique URIs in the request
            that have invalid format.
        noindex_uris (MutableSequence[str]):
            URIs that have no index meta tag. Sample
            limited to 1000.
        noindex_uris_count (int):
            Total number of URIs that have no index meta
            tag.
        uris_not_matching_target_sites (MutableSequence[str]):
            Unique URIs in the request that don't match
            any TargetSite in the DataStore, only match
            TargetSites that haven't been fully indexed, or
            match a TargetSite with type EXCLUDE. Sample
            limited to 1000.
        uris_not_matching_target_sites_count (int):
            Total number of URIs that don't match any
            TargetSites.
        valid_uris_count (int):
            Total number of unique URIs in the request that are not in
            invalid_uris.
        success_count (int):
            Total number of URIs that have been crawled
            so far.
        pending_count (int):
            Total number of URIs that have yet to be
            crawled.
        quota_exceeded_count (int):
            Total number of URIs that were rejected due
            to insufficient indexing resources.
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
    invalid_uris: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    invalid_uris_count: int = proto.Field(
        proto.INT32,
        number=8,
    )
    noindex_uris: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )
    noindex_uris_count: int = proto.Field(
        proto.INT32,
        number=12,
    )
    uris_not_matching_target_sites: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=9,
    )
    uris_not_matching_target_sites_count: int = proto.Field(
        proto.INT32,
        number=10,
    )
    valid_uris_count: int = proto.Field(
        proto.INT32,
        number=4,
    )
    success_count: int = proto.Field(
        proto.INT32,
        number=5,
    )
    pending_count: int = proto.Field(
        proto.INT32,
        number=6,
    )
    quota_exceeded_count: int = proto.Field(
        proto.INT32,
        number=7,
    )


class BatchVerifyTargetSitesRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.BatchVerifyTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.BatchVerifyTargetSites]
    method.

    Attributes:
        parent (str):
            Required. The parent resource shared by all TargetSites
            being verified.
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )


class BatchVerifyTargetSitesResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.BatchVerifyTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.BatchVerifyTargetSites]
    method.

    """


class BatchVerifyTargetSitesMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [SiteSearchEngineService.BatchVerifyTargetSites][google.cloud.discoveryengine.v1.SiteSearchEngineService.BatchVerifyTargetSites]
    operation. This will be returned by the
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


class FetchDomainVerificationStatusRequest(proto.Message):
    r"""Request message for
    [SiteSearchEngineService.FetchDomainVerificationStatus][google.cloud.discoveryengine.v1.SiteSearchEngineService.FetchDomainVerificationStatus]
    method.

    Attributes:
        site_search_engine (str):
            Required. The site search engine resource under which we
            fetch all the domain verification status.
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/siteSearchEngine``.
        page_size (int):
            Requested page size. Server may return fewer items than
            requested. If unspecified, server will pick an appropriate
            default. The maximum value is 1000; values above 1000 will
            be coerced to 1000.

            If this field is negative, an INVALID_ARGUMENT error is
            returned.
        page_token (str):
            A page token, received from a previous
            ``FetchDomainVerificationStatus`` call. Provide this to
            retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``FetchDomainVerificationStatus`` must match the call that
            provided the page token.
    """

    site_search_engine: str = proto.Field(
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


class FetchDomainVerificationStatusResponse(proto.Message):
    r"""Response message for
    [SiteSearchEngineService.FetchDomainVerificationStatus][google.cloud.discoveryengine.v1.SiteSearchEngineService.FetchDomainVerificationStatus]
    method.

    Attributes:
        target_sites (MutableSequence[google.cloud.discoveryengine_v1.types.TargetSite]):
            List of TargetSites containing the site
            verification status.
        next_page_token (str):
            A token that can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
        total_size (int):
            The total number of items matching the
            request. This will always be populated in the
            response.
    """

    @property
    def raw_page(self):
        return self

    target_sites: MutableSequence[
        gcd_site_search_engine.TargetSite
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_site_search_engine.TargetSite,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    total_size: int = proto.Field(
        proto.INT32,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
