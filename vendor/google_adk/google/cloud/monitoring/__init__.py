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
from google.cloud.monitoring import gapic_version as package_version

__version__ = package_version.__version__


from google.cloud.monitoring_v3.services.alert_policy_service.async_client import (
    AlertPolicyServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.alert_policy_service.client import (
    AlertPolicyServiceClient,
)
from google.cloud.monitoring_v3.services.group_service.async_client import (
    GroupServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.group_service.client import GroupServiceClient
from google.cloud.monitoring_v3.services.metric_service.async_client import (
    MetricServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.metric_service.client import (
    MetricServiceClient,
)
from google.cloud.monitoring_v3.services.notification_channel_service.async_client import (
    NotificationChannelServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.notification_channel_service.client import (
    NotificationChannelServiceClient,
)
from google.cloud.monitoring_v3.services.query_service.async_client import (
    QueryServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.query_service.client import QueryServiceClient
from google.cloud.monitoring_v3.services.service_monitoring_service.async_client import (
    ServiceMonitoringServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.service_monitoring_service.client import (
    ServiceMonitoringServiceClient,
)
from google.cloud.monitoring_v3.services.snooze_service.async_client import (
    SnoozeServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.snooze_service.client import (
    SnoozeServiceClient,
)
from google.cloud.monitoring_v3.services.uptime_check_service.async_client import (
    UptimeCheckServiceAsyncClient,
)
from google.cloud.monitoring_v3.services.uptime_check_service.client import (
    UptimeCheckServiceClient,
)
from google.cloud.monitoring_v3.types.alert import AlertPolicy
from google.cloud.monitoring_v3.types.alert_service import (
    CreateAlertPolicyRequest,
    DeleteAlertPolicyRequest,
    GetAlertPolicyRequest,
    ListAlertPoliciesRequest,
    ListAlertPoliciesResponse,
    UpdateAlertPolicyRequest,
)
from google.cloud.monitoring_v3.types.common import (
    Aggregation,
    ComparisonType,
    ServiceTier,
    TimeInterval,
    TypedValue,
)
from google.cloud.monitoring_v3.types.dropped_labels import DroppedLabels
from google.cloud.monitoring_v3.types.group import Group
from google.cloud.monitoring_v3.types.group_service import (
    CreateGroupRequest,
    DeleteGroupRequest,
    GetGroupRequest,
    ListGroupMembersRequest,
    ListGroupMembersResponse,
    ListGroupsRequest,
    ListGroupsResponse,
    UpdateGroupRequest,
)
from google.cloud.monitoring_v3.types.metric import (
    LabelValue,
    Point,
    QueryError,
    TextLocator,
    TimeSeries,
    TimeSeriesData,
    TimeSeriesDescriptor,
)
from google.cloud.monitoring_v3.types.metric_service import (
    CreateMetricDescriptorRequest,
    CreateTimeSeriesError,
    CreateTimeSeriesRequest,
    CreateTimeSeriesSummary,
    DeleteMetricDescriptorRequest,
    GetMetricDescriptorRequest,
    GetMonitoredResourceDescriptorRequest,
    ListMetricDescriptorsRequest,
    ListMetricDescriptorsResponse,
    ListMonitoredResourceDescriptorsRequest,
    ListMonitoredResourceDescriptorsResponse,
    ListTimeSeriesRequest,
    ListTimeSeriesResponse,
    QueryErrorList,
    QueryTimeSeriesRequest,
    QueryTimeSeriesResponse,
)
from google.cloud.monitoring_v3.types.mutation_record import MutationRecord
from google.cloud.monitoring_v3.types.notification import (
    NotificationChannel,
    NotificationChannelDescriptor,
)
from google.cloud.monitoring_v3.types.notification_service import (
    CreateNotificationChannelRequest,
    DeleteNotificationChannelRequest,
    GetNotificationChannelDescriptorRequest,
    GetNotificationChannelRequest,
    GetNotificationChannelVerificationCodeRequest,
    GetNotificationChannelVerificationCodeResponse,
    ListNotificationChannelDescriptorsRequest,
    ListNotificationChannelDescriptorsResponse,
    ListNotificationChannelsRequest,
    ListNotificationChannelsResponse,
    SendNotificationChannelVerificationCodeRequest,
    UpdateNotificationChannelRequest,
    VerifyNotificationChannelRequest,
)
from google.cloud.monitoring_v3.types.service import (
    BasicSli,
    DistributionCut,
    Range,
    RequestBasedSli,
    Service,
    ServiceLevelIndicator,
    ServiceLevelObjective,
    TimeSeriesRatio,
    WindowsBasedSli,
)
from google.cloud.monitoring_v3.types.service_service import (
    CreateServiceLevelObjectiveRequest,
    CreateServiceRequest,
    DeleteServiceLevelObjectiveRequest,
    DeleteServiceRequest,
    GetServiceLevelObjectiveRequest,
    GetServiceRequest,
    ListServiceLevelObjectivesRequest,
    ListServiceLevelObjectivesResponse,
    ListServicesRequest,
    ListServicesResponse,
    UpdateServiceLevelObjectiveRequest,
    UpdateServiceRequest,
)
from google.cloud.monitoring_v3.types.snooze import Snooze
from google.cloud.monitoring_v3.types.snooze_service import (
    CreateSnoozeRequest,
    GetSnoozeRequest,
    ListSnoozesRequest,
    ListSnoozesResponse,
    UpdateSnoozeRequest,
)
from google.cloud.monitoring_v3.types.span_context import SpanContext
from google.cloud.monitoring_v3.types.uptime import (
    GroupResourceType,
    InternalChecker,
    SyntheticMonitorTarget,
    UptimeCheckConfig,
    UptimeCheckIp,
    UptimeCheckRegion,
)
from google.cloud.monitoring_v3.types.uptime_service import (
    CreateUptimeCheckConfigRequest,
    DeleteUptimeCheckConfigRequest,
    GetUptimeCheckConfigRequest,
    ListUptimeCheckConfigsRequest,
    ListUptimeCheckConfigsResponse,
    ListUptimeCheckIpsRequest,
    ListUptimeCheckIpsResponse,
    UpdateUptimeCheckConfigRequest,
)

__all__ = (
    "AlertPolicyServiceClient",
    "AlertPolicyServiceAsyncClient",
    "GroupServiceClient",
    "GroupServiceAsyncClient",
    "MetricServiceClient",
    "MetricServiceAsyncClient",
    "NotificationChannelServiceClient",
    "NotificationChannelServiceAsyncClient",
    "QueryServiceClient",
    "QueryServiceAsyncClient",
    "ServiceMonitoringServiceClient",
    "ServiceMonitoringServiceAsyncClient",
    "SnoozeServiceClient",
    "SnoozeServiceAsyncClient",
    "UptimeCheckServiceClient",
    "UptimeCheckServiceAsyncClient",
    "AlertPolicy",
    "CreateAlertPolicyRequest",
    "DeleteAlertPolicyRequest",
    "GetAlertPolicyRequest",
    "ListAlertPoliciesRequest",
    "ListAlertPoliciesResponse",
    "UpdateAlertPolicyRequest",
    "Aggregation",
    "TimeInterval",
    "TypedValue",
    "ComparisonType",
    "ServiceTier",
    "DroppedLabels",
    "Group",
    "CreateGroupRequest",
    "DeleteGroupRequest",
    "GetGroupRequest",
    "ListGroupMembersRequest",
    "ListGroupMembersResponse",
    "ListGroupsRequest",
    "ListGroupsResponse",
    "UpdateGroupRequest",
    "LabelValue",
    "Point",
    "QueryError",
    "TextLocator",
    "TimeSeries",
    "TimeSeriesData",
    "TimeSeriesDescriptor",
    "CreateMetricDescriptorRequest",
    "CreateTimeSeriesError",
    "CreateTimeSeriesRequest",
    "CreateTimeSeriesSummary",
    "DeleteMetricDescriptorRequest",
    "GetMetricDescriptorRequest",
    "GetMonitoredResourceDescriptorRequest",
    "ListMetricDescriptorsRequest",
    "ListMetricDescriptorsResponse",
    "ListMonitoredResourceDescriptorsRequest",
    "ListMonitoredResourceDescriptorsResponse",
    "ListTimeSeriesRequest",
    "ListTimeSeriesResponse",
    "QueryErrorList",
    "QueryTimeSeriesRequest",
    "QueryTimeSeriesResponse",
    "MutationRecord",
    "NotificationChannel",
    "NotificationChannelDescriptor",
    "CreateNotificationChannelRequest",
    "DeleteNotificationChannelRequest",
    "GetNotificationChannelDescriptorRequest",
    "GetNotificationChannelRequest",
    "GetNotificationChannelVerificationCodeRequest",
    "GetNotificationChannelVerificationCodeResponse",
    "ListNotificationChannelDescriptorsRequest",
    "ListNotificationChannelDescriptorsResponse",
    "ListNotificationChannelsRequest",
    "ListNotificationChannelsResponse",
    "SendNotificationChannelVerificationCodeRequest",
    "UpdateNotificationChannelRequest",
    "VerifyNotificationChannelRequest",
    "BasicSli",
    "DistributionCut",
    "Range",
    "RequestBasedSli",
    "Service",
    "ServiceLevelIndicator",
    "ServiceLevelObjective",
    "TimeSeriesRatio",
    "WindowsBasedSli",
    "CreateServiceLevelObjectiveRequest",
    "CreateServiceRequest",
    "DeleteServiceLevelObjectiveRequest",
    "DeleteServiceRequest",
    "GetServiceLevelObjectiveRequest",
    "GetServiceRequest",
    "ListServiceLevelObjectivesRequest",
    "ListServiceLevelObjectivesResponse",
    "ListServicesRequest",
    "ListServicesResponse",
    "UpdateServiceLevelObjectiveRequest",
    "UpdateServiceRequest",
    "Snooze",
    "CreateSnoozeRequest",
    "GetSnoozeRequest",
    "ListSnoozesRequest",
    "ListSnoozesResponse",
    "UpdateSnoozeRequest",
    "SpanContext",
    "InternalChecker",
    "SyntheticMonitorTarget",
    "UptimeCheckConfig",
    "UptimeCheckIp",
    "GroupResourceType",
    "UptimeCheckRegion",
    "CreateUptimeCheckConfigRequest",
    "DeleteUptimeCheckConfigRequest",
    "GetUptimeCheckConfigRequest",
    "ListUptimeCheckConfigsRequest",
    "ListUptimeCheckConfigsResponse",
    "ListUptimeCheckIpsRequest",
    "ListUptimeCheckIpsResponse",
    "UpdateUptimeCheckConfigRequest",
)
