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
from google.cloud.monitoring_v3 import gapic_version as package_version

__version__ = package_version.__version__


from .services.alert_policy_service import (
    AlertPolicyServiceAsyncClient,
    AlertPolicyServiceClient,
)
from .services.group_service import GroupServiceAsyncClient, GroupServiceClient
from .services.metric_service import MetricServiceAsyncClient, MetricServiceClient
from .services.notification_channel_service import (
    NotificationChannelServiceAsyncClient,
    NotificationChannelServiceClient,
)
from .services.query_service import QueryServiceAsyncClient, QueryServiceClient
from .services.service_monitoring_service import (
    ServiceMonitoringServiceAsyncClient,
    ServiceMonitoringServiceClient,
)
from .services.snooze_service import SnoozeServiceAsyncClient, SnoozeServiceClient
from .services.uptime_check_service import (
    UptimeCheckServiceAsyncClient,
    UptimeCheckServiceClient,
)
from .types.alert import AlertPolicy
from .types.alert_service import (
    CreateAlertPolicyRequest,
    DeleteAlertPolicyRequest,
    GetAlertPolicyRequest,
    ListAlertPoliciesRequest,
    ListAlertPoliciesResponse,
    UpdateAlertPolicyRequest,
)
from .types.common import (
    Aggregation,
    ComparisonType,
    ServiceTier,
    TimeInterval,
    TypedValue,
)
from .types.dropped_labels import DroppedLabels
from .types.group import Group
from .types.group_service import (
    CreateGroupRequest,
    DeleteGroupRequest,
    GetGroupRequest,
    ListGroupMembersRequest,
    ListGroupMembersResponse,
    ListGroupsRequest,
    ListGroupsResponse,
    UpdateGroupRequest,
)
from .types.metric import (
    LabelValue,
    Point,
    QueryError,
    TextLocator,
    TimeSeries,
    TimeSeriesData,
    TimeSeriesDescriptor,
)
from .types.metric_service import (
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
from .types.mutation_record import MutationRecord
from .types.notification import NotificationChannel, NotificationChannelDescriptor
from .types.notification_service import (
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
from .types.service import (
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
from .types.service_service import (
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
from .types.snooze import Snooze
from .types.snooze_service import (
    CreateSnoozeRequest,
    GetSnoozeRequest,
    ListSnoozesRequest,
    ListSnoozesResponse,
    UpdateSnoozeRequest,
)
from .types.span_context import SpanContext
from .types.uptime import (
    GroupResourceType,
    InternalChecker,
    SyntheticMonitorTarget,
    UptimeCheckConfig,
    UptimeCheckIp,
    UptimeCheckRegion,
)
from .types.uptime_service import (
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
    "AlertPolicyServiceAsyncClient",
    "GroupServiceAsyncClient",
    "MetricServiceAsyncClient",
    "NotificationChannelServiceAsyncClient",
    "QueryServiceAsyncClient",
    "ServiceMonitoringServiceAsyncClient",
    "SnoozeServiceAsyncClient",
    "UptimeCheckServiceAsyncClient",
    "Aggregation",
    "AlertPolicy",
    "AlertPolicyServiceClient",
    "BasicSli",
    "ComparisonType",
    "CreateAlertPolicyRequest",
    "CreateGroupRequest",
    "CreateMetricDescriptorRequest",
    "CreateNotificationChannelRequest",
    "CreateServiceLevelObjectiveRequest",
    "CreateServiceRequest",
    "CreateSnoozeRequest",
    "CreateTimeSeriesError",
    "CreateTimeSeriesRequest",
    "CreateTimeSeriesSummary",
    "CreateUptimeCheckConfigRequest",
    "DeleteAlertPolicyRequest",
    "DeleteGroupRequest",
    "DeleteMetricDescriptorRequest",
    "DeleteNotificationChannelRequest",
    "DeleteServiceLevelObjectiveRequest",
    "DeleteServiceRequest",
    "DeleteUptimeCheckConfigRequest",
    "DistributionCut",
    "DroppedLabels",
    "GetAlertPolicyRequest",
    "GetGroupRequest",
    "GetMetricDescriptorRequest",
    "GetMonitoredResourceDescriptorRequest",
    "GetNotificationChannelDescriptorRequest",
    "GetNotificationChannelRequest",
    "GetNotificationChannelVerificationCodeRequest",
    "GetNotificationChannelVerificationCodeResponse",
    "GetServiceLevelObjectiveRequest",
    "GetServiceRequest",
    "GetSnoozeRequest",
    "GetUptimeCheckConfigRequest",
    "Group",
    "GroupResourceType",
    "GroupServiceClient",
    "InternalChecker",
    "LabelValue",
    "ListAlertPoliciesRequest",
    "ListAlertPoliciesResponse",
    "ListGroupMembersRequest",
    "ListGroupMembersResponse",
    "ListGroupsRequest",
    "ListGroupsResponse",
    "ListMetricDescriptorsRequest",
    "ListMetricDescriptorsResponse",
    "ListMonitoredResourceDescriptorsRequest",
    "ListMonitoredResourceDescriptorsResponse",
    "ListNotificationChannelDescriptorsRequest",
    "ListNotificationChannelDescriptorsResponse",
    "ListNotificationChannelsRequest",
    "ListNotificationChannelsResponse",
    "ListServiceLevelObjectivesRequest",
    "ListServiceLevelObjectivesResponse",
    "ListServicesRequest",
    "ListServicesResponse",
    "ListSnoozesRequest",
    "ListSnoozesResponse",
    "ListTimeSeriesRequest",
    "ListTimeSeriesResponse",
    "ListUptimeCheckConfigsRequest",
    "ListUptimeCheckConfigsResponse",
    "ListUptimeCheckIpsRequest",
    "ListUptimeCheckIpsResponse",
    "MetricServiceClient",
    "MutationRecord",
    "NotificationChannel",
    "NotificationChannelDescriptor",
    "NotificationChannelServiceClient",
    "Point",
    "QueryError",
    "QueryErrorList",
    "QueryServiceClient",
    "QueryTimeSeriesRequest",
    "QueryTimeSeriesResponse",
    "Range",
    "RequestBasedSli",
    "SendNotificationChannelVerificationCodeRequest",
    "Service",
    "ServiceLevelIndicator",
    "ServiceLevelObjective",
    "ServiceMonitoringServiceClient",
    "ServiceTier",
    "Snooze",
    "SnoozeServiceClient",
    "SpanContext",
    "SyntheticMonitorTarget",
    "TextLocator",
    "TimeInterval",
    "TimeSeries",
    "TimeSeriesData",
    "TimeSeriesDescriptor",
    "TimeSeriesRatio",
    "TypedValue",
    "UpdateAlertPolicyRequest",
    "UpdateGroupRequest",
    "UpdateNotificationChannelRequest",
    "UpdateServiceLevelObjectiveRequest",
    "UpdateServiceRequest",
    "UpdateSnoozeRequest",
    "UpdateUptimeCheckConfigRequest",
    "UptimeCheckConfig",
    "UptimeCheckIp",
    "UptimeCheckRegion",
    "UptimeCheckServiceClient",
    "VerifyNotificationChannelRequest",
    "WindowsBasedSli",
)
