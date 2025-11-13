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

from google.api import monitored_resource_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "UptimeCheckRegion",
        "GroupResourceType",
        "InternalChecker",
        "SyntheticMonitorTarget",
        "UptimeCheckConfig",
        "UptimeCheckIp",
    },
)


class UptimeCheckRegion(proto.Enum):
    r"""The regions from which an Uptime check can be run.

    Values:
        REGION_UNSPECIFIED (0):
            Default value if no region is specified. Will
            result in Uptime checks running from all
            regions.
        USA (1):
            Allows checks to run from locations within
            the United States of America.
        EUROPE (2):
            Allows checks to run from locations within
            the continent of Europe.
        SOUTH_AMERICA (3):
            Allows checks to run from locations within
            the continent of South America.
        ASIA_PACIFIC (4):
            Allows checks to run from locations within
            the Asia Pacific area (ex: Singapore).
        USA_OREGON (5):
            Allows checks to run from locations within
            the western United States of America
        USA_IOWA (6):
            Allows checks to run from locations within
            the central United States of America
        USA_VIRGINIA (7):
            Allows checks to run from locations within
            the eastern United States of America
    """
    REGION_UNSPECIFIED = 0
    USA = 1
    EUROPE = 2
    SOUTH_AMERICA = 3
    ASIA_PACIFIC = 4
    USA_OREGON = 5
    USA_IOWA = 6
    USA_VIRGINIA = 7


class GroupResourceType(proto.Enum):
    r"""The supported resource types that can be used as values of
    ``group_resource.resource_type``. ``INSTANCE`` includes
    ``gce_instance`` and ``aws_ec2_instance`` resource types. The
    resource types ``gae_app`` and ``uptime_url`` are not valid here
    because group checks on App Engine modules and URLs are not allowed.

    Values:
        RESOURCE_TYPE_UNSPECIFIED (0):
            Default value (not valid).
        INSTANCE (1):
            A group of instances from Google Cloud
            Platform (GCP) or Amazon Web Services (AWS).
        AWS_ELB_LOAD_BALANCER (2):
            A group of Amazon ELB load balancers.
    """
    RESOURCE_TYPE_UNSPECIFIED = 0
    INSTANCE = 1
    AWS_ELB_LOAD_BALANCER = 2


class InternalChecker(proto.Message):
    r"""An internal checker allows Uptime checks to run on
    private/internal GCP resources.

    Attributes:
        name (str):
            A unique resource name for this InternalChecker. The format
            is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/internalCheckers/[INTERNAL_CHECKER_ID]

            ``[PROJECT_ID_OR_NUMBER]`` is the Cloud Monitoring Metrics
            Scope project for the Uptime check config associated with
            the internal checker.
        display_name (str):
            The checker's human-readable name. The
            display name should be unique within a Cloud
            Monitoring Metrics Scope in order to make it
            easier to identify; however, uniqueness is not
            enforced.
        network (str):
            The `GCP VPC
            network <https://cloud.google.com/vpc/docs/vpc>`__ where the
            internal resource lives (ex: "default").
        gcp_zone (str):
            The GCP zone the Uptime check should egress from. Only
            respected for internal Uptime checks, where internal_network
            is specified.
        peer_project_id (str):
            The GCP project ID where the internal checker
            lives. Not necessary the same as the Metrics
            Scope project.
        state (google.cloud.monitoring_v3.types.InternalChecker.State):
            The current operational state of the internal
            checker.
    """

    class State(proto.Enum):
        r"""Operational states for an internal checker.

        Values:
            UNSPECIFIED (0):
                An internal checker should never be in the
                unspecified state.
            CREATING (1):
                The checker is being created, provisioned, and configured. A
                checker in this state can be returned by
                ``ListInternalCheckers`` or ``GetInternalChecker``, as well
                as by examining the `long running
                Operation <https://cloud.google.com/apis/design/design_patterns#long_running_operations>`__
                that created it.
            RUNNING (2):
                The checker is running and available for use. A checker in
                this state can be returned by ``ListInternalCheckers`` or
                ``GetInternalChecker`` as well as by examining the `long
                running
                Operation <https://cloud.google.com/apis/design/design_patterns#long_running_operations>`__
                that created it. If a checker is being torn down, it is
                neither visible nor usable, so there is no "deleting" or
                "down" state.
        """
        UNSPECIFIED = 0
        CREATING = 1
        RUNNING = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    network: str = proto.Field(
        proto.STRING,
        number=3,
    )
    gcp_zone: str = proto.Field(
        proto.STRING,
        number=4,
    )
    peer_project_id: str = proto.Field(
        proto.STRING,
        number=6,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=7,
        enum=State,
    )


class SyntheticMonitorTarget(proto.Message):
    r"""Describes a Synthetic Monitor to be invoked by Uptime.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        cloud_function_v2 (google.cloud.monitoring_v3.types.SyntheticMonitorTarget.CloudFunctionV2Target):
            Target a Synthetic Monitor GCFv2 instance.

            This field is a member of `oneof`_ ``target``.
    """

    class CloudFunctionV2Target(proto.Message):
        r"""A Synthetic Monitor deployed to a Cloud Functions V2
        instance.

        Attributes:
            name (str):
                Required. Fully qualified GCFv2 resource name i.e.
                ``projects/{project}/locations/{location}/functions/{function}``
                Required.
            cloud_run_revision (google.api.monitored_resource_pb2.MonitoredResource):
                Output only. The ``cloud_run_revision`` Monitored Resource
                associated with the GCFv2. The Synthetic Monitor execution
                results (metrics, logs, and spans) are reported against this
                Monitored Resource. This field is output only.
        """

        name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        cloud_run_revision: monitored_resource_pb2.MonitoredResource = proto.Field(
            proto.MESSAGE,
            number=2,
            message=monitored_resource_pb2.MonitoredResource,
        )

    cloud_function_v2: CloudFunctionV2Target = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="target",
        message=CloudFunctionV2Target,
    )


class UptimeCheckConfig(proto.Message):
    r"""This message configures which resources and services to
    monitor for availability.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Identifier. A unique resource name for this Uptime check
            configuration. The format is:

            ::

                 projects/[PROJECT_ID_OR_NUMBER]/uptimeCheckConfigs/[UPTIME_CHECK_ID]

            ``[PROJECT_ID_OR_NUMBER]`` is the Workspace host project
            associated with the Uptime check.

            This field should be omitted when creating the Uptime check
            configuration; on create, the resource name is assigned by
            the server and included in the response.
        display_name (str):
            A human-friendly name for the Uptime check
            configuration. The display name should be unique
            within a Cloud Monitoring Workspace in order to
            make it easier to identify; however, uniqueness
            is not enforced. Required.
        monitored_resource (google.api.monitored_resource_pb2.MonitoredResource):
            The `monitored
            resource <https://cloud.google.com/monitoring/api/resources>`__
            associated with the configuration. The following monitored
            resource types are valid for this field: ``uptime_url``,
            ``gce_instance``, ``gae_app``, ``aws_ec2_instance``,
            ``aws_elb_load_balancer`` ``k8s_service``
            ``servicedirectory_service`` ``cloud_run_revision``

            This field is a member of `oneof`_ ``resource``.
        resource_group (google.cloud.monitoring_v3.types.UptimeCheckConfig.ResourceGroup):
            The group resource associated with the
            configuration.

            This field is a member of `oneof`_ ``resource``.
        synthetic_monitor (google.cloud.monitoring_v3.types.SyntheticMonitorTarget):
            Specifies a Synthetic Monitor to invoke.

            This field is a member of `oneof`_ ``resource``.
        http_check (google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck):
            Contains information needed to make an HTTP
            or HTTPS check.

            This field is a member of `oneof`_ ``check_request_type``.
        tcp_check (google.cloud.monitoring_v3.types.UptimeCheckConfig.TcpCheck):
            Contains information needed to make a TCP
            check.

            This field is a member of `oneof`_ ``check_request_type``.
        period (google.protobuf.duration_pb2.Duration):
            How often, in seconds, the Uptime check is performed.
            Currently, the only supported values are ``60s`` (1 minute),
            ``300s`` (5 minutes), ``600s`` (10 minutes), and ``900s``
            (15 minutes). Optional, defaults to ``60s``.
        timeout (google.protobuf.duration_pb2.Duration):
            The maximum amount of time to wait for the
            request to complete (must be between 1 and 60
            seconds). Required.
        content_matchers (MutableSequence[google.cloud.monitoring_v3.types.UptimeCheckConfig.ContentMatcher]):
            The content that is expected to appear in the data returned
            by the target server against which the check is run.
            Currently, only the first entry in the ``content_matchers``
            list is supported, and additional entries will be ignored.
            This field is optional and should only be specified if a
            content match is required as part of the/ Uptime check.
        checker_type (google.cloud.monitoring_v3.types.UptimeCheckConfig.CheckerType):
            The type of checkers to use to execute the
            Uptime check.
        selected_regions (MutableSequence[google.cloud.monitoring_v3.types.UptimeCheckRegion]):
            The list of regions from which the check will
            be run. Some regions contain one location, and
            others contain more than one. If this field is
            specified, enough regions must be provided to
            include a minimum of 3 locations.  Not
            specifying this field will result in Uptime
            checks running from all available regions.
        is_internal (bool):
            If this is ``true``, then checks are made only from the
            'internal_checkers'. If it is ``false``, then checks are
            made only from the 'selected_regions'. It is an error to
            provide 'selected_regions' when is_internal is ``true``, or
            to provide 'internal_checkers' when is_internal is
            ``false``.
        internal_checkers (MutableSequence[google.cloud.monitoring_v3.types.InternalChecker]):
            The internal checkers that this check will egress from. If
            ``is_internal`` is ``true`` and this list is empty, the
            check will egress from all the InternalCheckers configured
            for the project that owns this ``UptimeCheckConfig``.
        user_labels (MutableMapping[str, str]):
            User-supplied key/value data to be used for organizing and
            identifying the ``UptimeCheckConfig`` objects.

            The field can contain up to 64 entries. Each key and value
            is limited to 63 Unicode characters or 128 bytes, whichever
            is smaller. Labels and values can contain only lowercase
            letters, numerals, underscores, and dashes. Keys must begin
            with a letter.
    """

    class CheckerType(proto.Enum):
        r"""What kind of checkers are available to be used by the check.

        Values:
            CHECKER_TYPE_UNSPECIFIED (0):
                The default checker type. Currently converted to
                ``STATIC_IP_CHECKERS`` on creation, the default conversion
                behavior may change in the future.
            STATIC_IP_CHECKERS (1):
                ``STATIC_IP_CHECKERS`` are used for uptime checks that
                perform egress across the public internet.
                ``STATIC_IP_CHECKERS`` use the static IP addresses returned
                by ``ListUptimeCheckIps``.
            VPC_CHECKERS (3):
                ``VPC_CHECKERS`` are used for uptime checks that perform
                egress using Service Directory and private network access.
                When using ``VPC_CHECKERS``, the monitored resource type
                must be ``servicedirectory_service``.
        """
        CHECKER_TYPE_UNSPECIFIED = 0
        STATIC_IP_CHECKERS = 1
        VPC_CHECKERS = 3

    class ResourceGroup(proto.Message):
        r"""The resource submessage for group checks. It can be used
        instead of a monitored resource, when multiple resources are
        being monitored.

        Attributes:
            group_id (str):
                The group of resources being monitored. Should be only the
                ``[GROUP_ID]``, and not the full-path
                ``projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]``.
            resource_type (google.cloud.monitoring_v3.types.GroupResourceType):
                The resource type of the group members.
        """

        group_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        resource_type: "GroupResourceType" = proto.Field(
            proto.ENUM,
            number=2,
            enum="GroupResourceType",
        )

    class PingConfig(proto.Message):
        r"""Information involved in sending ICMP pings alongside public
        HTTP/TCP checks. For HTTP, the pings are performed for each part
        of the redirect chain.

        Attributes:
            pings_count (int):
                Number of ICMP pings. A maximum of 3 ICMP
                pings is currently supported.
        """

        pings_count: int = proto.Field(
            proto.INT32,
            number=1,
        )

    class HttpCheck(proto.Message):
        r"""Information involved in an HTTP/HTTPS Uptime check request.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            request_method (google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck.RequestMethod):
                The HTTP request method to use for the check. If set to
                ``METHOD_UNSPECIFIED`` then ``request_method`` defaults to
                ``GET``.
            use_ssl (bool):
                If ``true``, use HTTPS instead of HTTP to run the check.
            path (str):
                Optional (defaults to "/"). The path to the page against
                which to run the check. Will be combined with the ``host``
                (specified within the ``monitored_resource``) and ``port``
                to construct the full URL. If the provided path does not
                begin with "/", a "/" will be prepended automatically.
            port (int):
                Optional (defaults to 80 when ``use_ssl`` is ``false``, and
                443 when ``use_ssl`` is ``true``). The TCP port on the HTTP
                server against which to run the check. Will be combined with
                host (specified within the ``monitored_resource``) and
                ``path`` to construct the full URL.
            auth_info (google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck.BasicAuthentication):
                The authentication information. Optional when creating an
                HTTP check; defaults to empty. Do not set both
                ``auth_method`` and ``auth_info``.
            mask_headers (bool):
                Boolean specifying whether to encrypt the header
                information. Encryption should be specified for any headers
                related to authentication that you do not wish to be seen
                when retrieving the configuration. The server will be
                responsible for encrypting the headers. On Get/List calls,
                if ``mask_headers`` is set to ``true`` then the headers will
                be obscured with ``******.``
            headers (MutableMapping[str, str]):
                The list of headers to send as part of the
                Uptime check request. If two headers have the
                same key and different values, they should be
                entered as a single header, with the value being
                a comma-separated list of all the desired values
                as described at
                https://www.w3.org/Protocols/rfc2616/rfc2616.txt
                (page 31). Entering two separate headers with
                the same key in a Create call will cause the
                first to be overwritten by the second. The
                maximum number of headers allowed is 100.
            content_type (google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck.ContentType):
                The content type header to use for the check. The following
                configurations result in errors:

                1. Content type is specified in both the ``headers`` field
                   and the ``content_type`` field.
                2. Request method is ``GET`` and ``content_type`` is not
                   ``TYPE_UNSPECIFIED``
                3. Request method is ``POST`` and ``content_type`` is
                   ``TYPE_UNSPECIFIED``.
                4. Request method is ``POST`` and a "Content-Type" header is
                   provided via ``headers`` field. The ``content_type``
                   field should be used instead.
            custom_content_type (str):
                A user provided content type header to use for the check.
                The invalid configurations outlined in the ``content_type``
                field apply to ``custom_content_type``, as well as the
                following:

                1. ``content_type`` is ``URL_ENCODED`` and
                   ``custom_content_type`` is set.
                2. ``content_type`` is ``USER_PROVIDED`` and
                   ``custom_content_type`` is not set.
            validate_ssl (bool):
                Boolean specifying whether to include SSL certificate
                validation as a part of the Uptime check. Only applies to
                checks where ``monitored_resource`` is set to
                ``uptime_url``. If ``use_ssl`` is ``false``, setting
                ``validate_ssl`` to ``true`` has no effect.
            body (bytes):
                The request body associated with the HTTP POST request. If
                ``content_type`` is ``URL_ENCODED``, the body passed in must
                be URL-encoded. Users can provide a ``Content-Length``
                header via the ``headers`` field or the API will do so. If
                the ``request_method`` is ``GET`` and ``body`` is not empty,
                the API will return an error. The maximum byte size is 1
                megabyte.

                Note: If client libraries aren't used (which performs the
                conversion automatically) base64 encode your ``body`` data
                since the field is of ``bytes`` type.
            accepted_response_status_codes (MutableSequence[google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck.ResponseStatusCode]):
                If present, the check will only pass if the
                HTTP response status code is in this set of
                status codes. If empty, the HTTP status code
                will only pass if the HTTP status code is
                200-299.
            ping_config (google.cloud.monitoring_v3.types.UptimeCheckConfig.PingConfig):
                Contains information needed to add pings to
                an HTTP check.
            service_agent_authentication (google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck.ServiceAgentAuthentication):
                If specified, Uptime will generate and attach an OIDC JWT
                token for the Monitoring service agent service account as an
                ``Authorization`` header in the HTTP request when probing.

                This field is a member of `oneof`_ ``auth_method``.
        """

        class RequestMethod(proto.Enum):
            r"""The HTTP request method options.

            Values:
                METHOD_UNSPECIFIED (0):
                    No request method specified.
                GET (1):
                    GET request.
                POST (2):
                    POST request.
            """
            METHOD_UNSPECIFIED = 0
            GET = 1
            POST = 2

        class ContentType(proto.Enum):
            r"""Header options corresponding to the content type of a HTTP
            request body.

            Values:
                TYPE_UNSPECIFIED (0):
                    No content type specified.
                URL_ENCODED (1):
                    ``body`` is in URL-encoded form. Equivalent to setting the
                    ``Content-Type`` to ``application/x-www-form-urlencoded`` in
                    the HTTP request.
                USER_PROVIDED (2):
                    ``body`` is in ``custom_content_type`` form. Equivalent to
                    setting the ``Content-Type`` to the contents of
                    ``custom_content_type`` in the HTTP request.
            """
            TYPE_UNSPECIFIED = 0
            URL_ENCODED = 1
            USER_PROVIDED = 2

        class BasicAuthentication(proto.Message):
            r"""The authentication parameters to provide to the specified resource
            or URL that requires a username and password. Currently, only `Basic
            HTTP authentication <https://tools.ietf.org/html/rfc7617>`__ is
            supported in Uptime checks.

            Attributes:
                username (str):
                    The username to use when authenticating with
                    the HTTP server.
                password (str):
                    The password to use when authenticating with
                    the HTTP server.
            """

            username: str = proto.Field(
                proto.STRING,
                number=1,
            )
            password: str = proto.Field(
                proto.STRING,
                number=2,
            )

        class ResponseStatusCode(proto.Message):
            r"""A status to accept. Either a status code class like "2xx", or
            an integer status code like "200".

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                status_value (int):
                    A status code to accept.

                    This field is a member of `oneof`_ ``status_code``.
                status_class (google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck.ResponseStatusCode.StatusClass):
                    A class of status codes to accept.

                    This field is a member of `oneof`_ ``status_code``.
            """

            class StatusClass(proto.Enum):
                r"""An HTTP status code class.

                Values:
                    STATUS_CLASS_UNSPECIFIED (0):
                        Default value that matches no status codes.
                    STATUS_CLASS_1XX (100):
                        The class of status codes between 100 and
                        199.
                    STATUS_CLASS_2XX (200):
                        The class of status codes between 200 and
                        299.
                    STATUS_CLASS_3XX (300):
                        The class of status codes between 300 and
                        399.
                    STATUS_CLASS_4XX (400):
                        The class of status codes between 400 and
                        499.
                    STATUS_CLASS_5XX (500):
                        The class of status codes between 500 and
                        599.
                    STATUS_CLASS_ANY (1000):
                        The class of all status codes.
                """
                STATUS_CLASS_UNSPECIFIED = 0
                STATUS_CLASS_1XX = 100
                STATUS_CLASS_2XX = 200
                STATUS_CLASS_3XX = 300
                STATUS_CLASS_4XX = 400
                STATUS_CLASS_5XX = 500
                STATUS_CLASS_ANY = 1000

            status_value: int = proto.Field(
                proto.INT32,
                number=1,
                oneof="status_code",
            )
            status_class: "UptimeCheckConfig.HttpCheck.ResponseStatusCode.StatusClass" = proto.Field(
                proto.ENUM,
                number=2,
                oneof="status_code",
                enum="UptimeCheckConfig.HttpCheck.ResponseStatusCode.StatusClass",
            )

        class ServiceAgentAuthentication(proto.Message):
            r"""Contains information needed for generating either an `OpenID Connect
            token <https://developers.google.com/identity/protocols/OpenIDConnect>`__
            or `OAuth
            token <https://developers.google.com/identity/protocols/oauth2>`__.
            The token will be generated for the Monitoring service agent service
            account.

            Attributes:
                type_ (google.cloud.monitoring_v3.types.UptimeCheckConfig.HttpCheck.ServiceAgentAuthentication.ServiceAgentAuthenticationType):
                    Type of authentication.
            """

            class ServiceAgentAuthenticationType(proto.Enum):
                r"""Type of authentication.

                Values:
                    SERVICE_AGENT_AUTHENTICATION_TYPE_UNSPECIFIED (0):
                        Default value, will result in OIDC
                        Authentication.
                    OIDC_TOKEN (1):
                        OIDC Authentication
                """
                SERVICE_AGENT_AUTHENTICATION_TYPE_UNSPECIFIED = 0
                OIDC_TOKEN = 1

            type_: "UptimeCheckConfig.HttpCheck.ServiceAgentAuthentication.ServiceAgentAuthenticationType" = proto.Field(
                proto.ENUM,
                number=1,
                enum="UptimeCheckConfig.HttpCheck.ServiceAgentAuthentication.ServiceAgentAuthenticationType",
            )

        request_method: "UptimeCheckConfig.HttpCheck.RequestMethod" = proto.Field(
            proto.ENUM,
            number=8,
            enum="UptimeCheckConfig.HttpCheck.RequestMethod",
        )
        use_ssl: bool = proto.Field(
            proto.BOOL,
            number=1,
        )
        path: str = proto.Field(
            proto.STRING,
            number=2,
        )
        port: int = proto.Field(
            proto.INT32,
            number=3,
        )
        auth_info: "UptimeCheckConfig.HttpCheck.BasicAuthentication" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="UptimeCheckConfig.HttpCheck.BasicAuthentication",
        )
        mask_headers: bool = proto.Field(
            proto.BOOL,
            number=5,
        )
        headers: MutableMapping[str, str] = proto.MapField(
            proto.STRING,
            proto.STRING,
            number=6,
        )
        content_type: "UptimeCheckConfig.HttpCheck.ContentType" = proto.Field(
            proto.ENUM,
            number=9,
            enum="UptimeCheckConfig.HttpCheck.ContentType",
        )
        custom_content_type: str = proto.Field(
            proto.STRING,
            number=13,
        )
        validate_ssl: bool = proto.Field(
            proto.BOOL,
            number=7,
        )
        body: bytes = proto.Field(
            proto.BYTES,
            number=10,
        )
        accepted_response_status_codes: MutableSequence[
            "UptimeCheckConfig.HttpCheck.ResponseStatusCode"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=11,
            message="UptimeCheckConfig.HttpCheck.ResponseStatusCode",
        )
        ping_config: "UptimeCheckConfig.PingConfig" = proto.Field(
            proto.MESSAGE,
            number=12,
            message="UptimeCheckConfig.PingConfig",
        )
        service_agent_authentication: "UptimeCheckConfig.HttpCheck.ServiceAgentAuthentication" = proto.Field(
            proto.MESSAGE,
            number=14,
            oneof="auth_method",
            message="UptimeCheckConfig.HttpCheck.ServiceAgentAuthentication",
        )

    class TcpCheck(proto.Message):
        r"""Information required for a TCP Uptime check request.

        Attributes:
            port (int):
                The TCP port on the server against which to run the check.
                Will be combined with host (specified within the
                ``monitored_resource``) to construct the full URL. Required.
            ping_config (google.cloud.monitoring_v3.types.UptimeCheckConfig.PingConfig):
                Contains information needed to add pings to a
                TCP check.
        """

        port: int = proto.Field(
            proto.INT32,
            number=1,
        )
        ping_config: "UptimeCheckConfig.PingConfig" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="UptimeCheckConfig.PingConfig",
        )

    class ContentMatcher(proto.Message):
        r"""Optional. Used to perform content matching. This allows
        matching based on substrings and regular expressions, together
        with their negations. Only the first 4&nbsp;MB of an HTTP or
        HTTPS check's response (and the first 1&nbsp;MB of a TCP check's
        response) are examined for purposes of content matching.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            content (str):
                String, regex or JSON content to match. Maximum 1024 bytes.
                An empty ``content`` string indicates no content matching is
                to be performed.
            matcher (google.cloud.monitoring_v3.types.UptimeCheckConfig.ContentMatcher.ContentMatcherOption):
                The type of content matcher that will be applied to the
                server output, compared to the ``content`` string when the
                check is run.
            json_path_matcher (google.cloud.monitoring_v3.types.UptimeCheckConfig.ContentMatcher.JsonPathMatcher):
                Matcher information for ``MATCHES_JSON_PATH`` and
                ``NOT_MATCHES_JSON_PATH``

                This field is a member of `oneof`_ ``additional_matcher_info``.
        """

        class ContentMatcherOption(proto.Enum):
            r"""Options to perform content matching.

            Values:
                CONTENT_MATCHER_OPTION_UNSPECIFIED (0):
                    No content matcher type specified (maintained for backward
                    compatibility, but deprecated for future use). Treated as
                    ``CONTAINS_STRING``.
                CONTAINS_STRING (1):
                    Selects substring matching. The match succeeds if the output
                    contains the ``content`` string. This is the default value
                    for checks without a ``matcher`` option, or where the value
                    of ``matcher`` is ``CONTENT_MATCHER_OPTION_UNSPECIFIED``.
                NOT_CONTAINS_STRING (2):
                    Selects negation of substring matching. The match succeeds
                    if the output does *NOT* contain the ``content`` string.
                MATCHES_REGEX (3):
                    Selects regular-expression matching. The match succeeds if
                    the output matches the regular expression specified in the
                    ``content`` string. Regex matching is only supported for
                    HTTP/HTTPS checks.
                NOT_MATCHES_REGEX (4):
                    Selects negation of regular-expression matching. The match
                    succeeds if the output does *NOT* match the regular
                    expression specified in the ``content`` string. Regex
                    matching is only supported for HTTP/HTTPS checks.
                MATCHES_JSON_PATH (5):
                    Selects JSONPath matching. See ``JsonPathMatcher`` for
                    details on when the match succeeds. JSONPath matching is
                    only supported for HTTP/HTTPS checks.
                NOT_MATCHES_JSON_PATH (6):
                    Selects JSONPath matching. See ``JsonPathMatcher`` for
                    details on when the match succeeds. Succeeds when output
                    does *NOT* match as specified. JSONPath is only supported
                    for HTTP/HTTPS checks.
            """
            CONTENT_MATCHER_OPTION_UNSPECIFIED = 0
            CONTAINS_STRING = 1
            NOT_CONTAINS_STRING = 2
            MATCHES_REGEX = 3
            NOT_MATCHES_REGEX = 4
            MATCHES_JSON_PATH = 5
            NOT_MATCHES_JSON_PATH = 6

        class JsonPathMatcher(proto.Message):
            r"""Information needed to perform a JSONPath content match. Used for
            ``ContentMatcherOption::MATCHES_JSON_PATH`` and
            ``ContentMatcherOption::NOT_MATCHES_JSON_PATH``.

            Attributes:
                json_path (str):
                    JSONPath within the response output pointing to the expected
                    ``ContentMatcher::content`` to match against.
                json_matcher (google.cloud.monitoring_v3.types.UptimeCheckConfig.ContentMatcher.JsonPathMatcher.JsonPathMatcherOption):
                    The type of JSONPath match that will be applied to the JSON
                    output (``ContentMatcher.content``)
            """

            class JsonPathMatcherOption(proto.Enum):
                r"""Options to perform JSONPath content matching.

                Values:
                    JSON_PATH_MATCHER_OPTION_UNSPECIFIED (0):
                        No JSONPath matcher type specified (not
                        valid).
                    EXACT_MATCH (1):
                        Selects 'exact string' matching. The match succeeds if the
                        content at the ``json_path`` within the output is exactly
                        the same as the ``content`` string.
                    REGEX_MATCH (2):
                        Selects regular-expression matching. The match succeeds if
                        the content at the ``json_path`` within the output matches
                        the regular expression specified in the ``content`` string.
                """
                JSON_PATH_MATCHER_OPTION_UNSPECIFIED = 0
                EXACT_MATCH = 1
                REGEX_MATCH = 2

            json_path: str = proto.Field(
                proto.STRING,
                number=1,
            )
            json_matcher: "UptimeCheckConfig.ContentMatcher.JsonPathMatcher.JsonPathMatcherOption" = proto.Field(
                proto.ENUM,
                number=2,
                enum="UptimeCheckConfig.ContentMatcher.JsonPathMatcher.JsonPathMatcherOption",
            )

        content: str = proto.Field(
            proto.STRING,
            number=1,
        )
        matcher: "UptimeCheckConfig.ContentMatcher.ContentMatcherOption" = proto.Field(
            proto.ENUM,
            number=2,
            enum="UptimeCheckConfig.ContentMatcher.ContentMatcherOption",
        )
        json_path_matcher: "UptimeCheckConfig.ContentMatcher.JsonPathMatcher" = (
            proto.Field(
                proto.MESSAGE,
                number=3,
                oneof="additional_matcher_info",
                message="UptimeCheckConfig.ContentMatcher.JsonPathMatcher",
            )
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    monitored_resource: monitored_resource_pb2.MonitoredResource = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="resource",
        message=monitored_resource_pb2.MonitoredResource,
    )
    resource_group: ResourceGroup = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="resource",
        message=ResourceGroup,
    )
    synthetic_monitor: "SyntheticMonitorTarget" = proto.Field(
        proto.MESSAGE,
        number=21,
        oneof="resource",
        message="SyntheticMonitorTarget",
    )
    http_check: HttpCheck = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="check_request_type",
        message=HttpCheck,
    )
    tcp_check: TcpCheck = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="check_request_type",
        message=TcpCheck,
    )
    period: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=7,
        message=duration_pb2.Duration,
    )
    timeout: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=8,
        message=duration_pb2.Duration,
    )
    content_matchers: MutableSequence[ContentMatcher] = proto.RepeatedField(
        proto.MESSAGE,
        number=9,
        message=ContentMatcher,
    )
    checker_type: CheckerType = proto.Field(
        proto.ENUM,
        number=17,
        enum=CheckerType,
    )
    selected_regions: MutableSequence["UptimeCheckRegion"] = proto.RepeatedField(
        proto.ENUM,
        number=10,
        enum="UptimeCheckRegion",
    )
    is_internal: bool = proto.Field(
        proto.BOOL,
        number=15,
    )
    internal_checkers: MutableSequence["InternalChecker"] = proto.RepeatedField(
        proto.MESSAGE,
        number=14,
        message="InternalChecker",
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=20,
    )


class UptimeCheckIp(proto.Message):
    r"""Contains the region, location, and list of IP
    addresses where checkers in the location run from.

    Attributes:
        region (google.cloud.monitoring_v3.types.UptimeCheckRegion):
            A broad region category in which the IP
            address is located.
        location (str):
            A more specific location within the region
            that typically encodes a particular
            city/town/metro (and its containing
            state/province or country) within the broader
            umbrella region category.
        ip_address (str):
            The IP address from which the Uptime check
            originates. This is a fully specified IP address
            (not an IP address range). Most IP addresses, as
            of this publication, are in IPv4 format;
            however, one should not rely on the IP addresses
            being in IPv4 format indefinitely, and should
            support interpreting this field in either IPv4
            or IPv6 format.
    """

    region: "UptimeCheckRegion" = proto.Field(
        proto.ENUM,
        number=1,
        enum="UptimeCheckRegion",
    )
    location: str = proto.Field(
        proto.STRING,
        number=2,
    )
    ip_address: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
