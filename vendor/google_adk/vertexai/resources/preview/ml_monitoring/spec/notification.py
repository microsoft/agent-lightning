# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from typing import Optional, List
from google.cloud.aiplatform.compat.types import (
    model_monitoring_spec_v1beta1 as model_monitoring_spec,
)


class NotificationSpec:
    """Initializer for NotificationSpec.

    Args:
        user_emails (List[str]):
            Optional. The email addresses to send the alert to.
        notification_channels (List[str]):
            Optional. The notification channels to send the alert to.
            Format: ``projects/{project}/notificationChannels/{channel}``
        enable_cloud_logging (bool):
            Optional. If dump the anomalies to Cloud Logging. The anomalies will
            be put to json payload. This can be further sinked to Pub/Sub or any
            other services supported by Cloud Logging.
    """

    def __init__(
        self,
        user_emails: Optional[List[str]] = None,
        notification_channels: Optional[List[str]] = None,
        enable_cloud_logging: Optional[bool] = False,
    ):
        self.user_emails = user_emails
        self.notification_channels = notification_channels
        self.enable_cloud_logging = enable_cloud_logging

    def _as_proto(self) -> model_monitoring_spec.ModelMonitoringNotificationSpec:
        """Converts ModelMonitoringNotificationSpec to a proto message.

        Returns:
           The GAPIC representation of the notification alert config.
        """
        user_email_config = None
        if self.user_emails is not None:
            user_email_config = (
                model_monitoring_spec.ModelMonitoringNotificationSpec.EmailConfig(
                    user_emails=self.user_emails
                )
            )
        user_notification_channel_config = []
        if self.notification_channels:
            for notification_channel in self.notification_channels:
                user_notification_channel_config.append(
                    model_monitoring_spec.ModelMonitoringNotificationSpec.NotificationChannelConfig(
                        notification_channel=notification_channel
                    )
                )
        return model_monitoring_spec.ModelMonitoringNotificationSpec(
            email_config=user_email_config,
            notification_channel_configs=user_notification_channel_config,
            enable_cloud_logging=self.enable_cloud_logging,
        )
