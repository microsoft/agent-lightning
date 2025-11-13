# Copyright 2023 Google LLC All rights reserved.
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
import grpc

from google.api_core import grpc_helpers
import google.auth.credentials
from google.cloud.spanner_admin_database_v1 import DatabaseDialect
from google.cloud.spanner_v1 import SpannerClient
from google.cloud.spanner_v1.database import Database, SPANNER_DATA_SCOPE
from google.cloud.spanner_v1.services.spanner.transports import (
    SpannerGrpcTransport,
    SpannerTransport,
)
from google.cloud.spanner_v1.testing.interceptors import (
    MethodCountInterceptor,
    MethodAbortInterceptor,
    XGoogRequestIDHeaderInterceptor,
)


class TestDatabase(Database):
    """Representation of a Cloud Spanner Database. This class is only used for
    system testing as there is no support for interceptors in grpc client
    currently, and we don't want to make changes in the Database class for
    testing purpose as this is a hack to use interceptors in tests."""

    _interceptors = []

    def __init__(
        self,
        database_id,
        instance,
        ddl_statements=(),
        pool=None,
        logger=None,
        encryption_config=None,
        database_dialect=DatabaseDialect.DATABASE_DIALECT_UNSPECIFIED,
        database_role=None,
        enable_drop_protection=False,
    ):
        super().__init__(
            database_id,
            instance,
            ddl_statements,
            pool,
            logger,
            encryption_config,
            database_dialect,
            database_role,
            enable_drop_protection,
        )

        self._method_count_interceptor = MethodCountInterceptor()
        self._method_abort_interceptor = MethodAbortInterceptor()
        self._interceptors = [
            self._method_count_interceptor,
            self._method_abort_interceptor,
        ]

    @property
    def spanner_api(self):
        """Helper for session-related API calls."""
        if self._spanner_api is None:
            client = self._instance._client
            client_info = client._client_info
            client_options = client._client_options
            if self._instance.emulator_host is not None:
                channel = grpc.insecure_channel(self._instance.emulator_host)
                self._x_goog_request_id_interceptor = XGoogRequestIDHeaderInterceptor()
                self._interceptors.append(self._x_goog_request_id_interceptor)
                channel = grpc.intercept_channel(channel, *self._interceptors)
                transport = SpannerGrpcTransport(channel=channel)
                self._spanner_api = SpannerClient(
                    client_info=client_info,
                    transport=transport,
                )
                return self._spanner_api
            credentials = client.credentials
            if isinstance(credentials, google.auth.credentials.Scoped):
                credentials = credentials.with_scopes((SPANNER_DATA_SCOPE,))
            self._spanner_api = self._create_spanner_client_for_tests(
                client_options,
                credentials,
            )
        return self._spanner_api

    def _create_spanner_client_for_tests(self, client_options, credentials):
        (
            api_endpoint,
            client_cert_source_func,
        ) = SpannerClient.get_mtls_endpoint_and_cert_source(client_options)
        channel = grpc_helpers.create_channel(
            api_endpoint,
            credentials=credentials,
            credentials_file=client_options.credentials_file,
            quota_project_id=client_options.quota_project_id,
            default_scopes=SpannerTransport.AUTH_SCOPES,
            scopes=client_options.scopes,
            default_host=SpannerTransport.DEFAULT_HOST,
        )
        channel = grpc.intercept_channel(channel, *self._interceptors)
        transport = SpannerGrpcTransport(channel=channel)
        return SpannerClient(
            client_options=client_options,
            transport=transport,
        )

    def reset(self):
        if self._x_goog_request_id_interceptor:
            self._x_goog_request_id_interceptor.reset()
