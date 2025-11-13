# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
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

import re
from typing import List, Dict, Optional, Tuple, Sequence
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base, initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    feature_monitor_v1beta1 as gca_feature_monitor,
    feature_monitor_job_v1beta1 as gca_feature_monitor_job,
)

_LOGGER = base.Logger(__name__)


class FeatureMonitor(base.VertexAiResourceNounWithFutureManager):
    """Class for managing Feature Monitor resources."""

    client_class = utils.FeatureRegistryClientV1Beta1WithOverride

    _resource_noun = "feature_monitors"
    _getter_method = "get_feature_monitor"
    _list_method = "list_feature_monitors"
    _delete_method = "delete_feature_monitor"
    _parse_resource_name_method = "parse_feature_monitor_path"
    _format_resource_name_method = "feature_monitor_path"
    _gca_resource: gca_feature_monitor.FeatureMonitor

    def __init__(
        self,
        name: str,
        feature_group_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed feature.

        Args:
            name:
                The resource name
                (`projects/.../locations/.../featureGroups/.../featureMonitors/...`) or
                ID.
            feature_group_id:
                The feature group ID. Must be passed in if name is an ID and not
                a resource path.
            project:
                Project to retrieve feature from. If not set, the project set in
                aiplatform.init will be used.
            location:
                Location to retrieve feature from. If not set, the location set
                in aiplatform.init will be used.
            credentials:
                Custom credentials to use to retrieve this feature. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=name,
        )

        if re.fullmatch(
            r"projects/.+/locations/.+/featureGroups/.+/featureMonitors/.+",
            name,
        ):
            if feature_group_id:
                raise ValueError(
                    f"Since feature monitor '{name}' is provided as a path, feature_group_id should not be specified."
                )
            feature_monitor = name
        else:
            from .feature_group import FeatureGroup

            # Construct the feature path using feature group ID if  only the
            # feature group ID is provided.
            if not feature_group_id:
                raise ValueError(
                    f"Since feature monitor '{name}' is not provided as a path, please specify feature_group_id."
                )

            feature_group_path = utils.full_resource_name(
                resource_name=feature_group_id,
                resource_noun=FeatureGroup._resource_noun,
                parse_resource_name_method=FeatureGroup._parse_resource_name,
                format_resource_name_method=FeatureGroup._format_resource_name,
            )

            feature_monitor = f"{feature_group_path}/featureMonitors/{name}"

        self._gca_resource = self._get_gca_resource(resource_name=feature_monitor)

    @property
    def description(self) -> str:
        """The description of the feature monitor."""
        return self._gca_resource.description

    @property
    def schedule_config(self) -> str:
        """The schedule config of the feature monitor."""
        return self._gca_resource.schedule_config.cron

    @property
    def feature_selection_configs(self) -> List[Tuple[str, float]]:
        """The feature and it's drift threshold configs of the feature monitor."""
        configs: List[Tuple[str, float]] = []
        for (
            feature_config
        ) in self._gca_resource.feature_selection_config.feature_configs:
            configs.append(
                (
                    feature_config.feature_id,
                    (
                        feature_config.drift_threshold
                        if feature_config.drift_threshold
                        else 0.3
                    ),
                )
            )
        return configs

    class FeatureMonitorJob(base.VertexAiResourceNounWithFutureManager):
        """Class for managing Feature Monitor Job resources."""

        client_class = utils.FeatureRegistryClientV1Beta1WithOverride

        _resource_noun = "featureMonitorJobs"
        _getter_method = "get_feature_monitor_job"
        _list_method = "list_feature_monitor_jobs"
        _delete_method = "delete_feature_monitor_job"
        _parse_resource_name_method = "parse_feature_monitor_job_path"
        _format_resource_name_method = "feature_monitor_job_path"
        _gca_resource: gca_feature_monitor_job.FeatureMonitorJob

        def __init__(
            self,
            name: str,
            project: Optional[str] = None,
            location: Optional[str] = None,
            credentials: Optional[auth_credentials.Credentials] = None,
        ):
            """Retrieves an existing managed feature monitor job.

            Args:
                name: The resource name
                  (`projects/.../locations/.../featureGroups/.../featureMonitors/.../featureMonitorJobs/...`)
                project: Project to retrieve the feature monitor job from. If
                  unset, the project set in aiplatform.init will be used.
                location: Location to retrieve the feature monitor job from. If
                  not set, location set in aiplatform.init will be used.
                credentials: Custom credentials to use to retrieve this feature
                  monitor job. Overrides credentials set in aiplatform.init.
            """
            super().__init__(
                project=project,
                location=location,
                credentials=credentials,
                resource_name=name,
            )

            if not re.fullmatch(
                r"projects/.+/locations/.+/featureGroups/.+/featureMonitors/.+/featureMonitorJobs/.+",
                name,
            ):
                raise ValueError(
                    "name need to specify the fully qualified"
                    + " feature monitor job resource path."
                )

            self._gca_resource = self._get_gca_resource(resource_name=name)

        @property
        def description(self) -> str:
            """The description of the feature monitor."""
            return self._gca_resource.description

        @property
        def feature_stats_and_anomalies(
            self,
        ) -> List[gca_feature_monitor.FeatureStatsAndAnomaly]:
            """The feature stats and anomaly of the feature monitor job."""
            if self._gca_resource.job_summary:
                return self._gca_resource.job_summary.feature_stats_and_anomalies
            return []

    def create_feature_monitor_job(
        self,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
        create_request_timeout: Optional[float] = None,
    ) -> FeatureMonitorJob:
        """Creates a new feature monitor job.

        Args:
            description: Description of the feature monitor job.
            labels:
                The labels with user-defined metadata to organize your
                FeatureMonitorJobs.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters,
                numeric characters, underscores and dashes. International
                characters are allowed.

                See https://goo.gl/xmQnxf for more information on and examples
                of labels. No more than 64 user labels can be associated with
                one FeatureMonitor (System labels are excluded)." System reserved label
                keys are prefixed with "aiplatform.googleapis.com/" and are
                immutable.
            project:
                Project to create feature in. If unset, the project set in
                aiplatform.init will be used.
            location:
                Location to create feature in. If not set, location set in
                aiplatform.init will be used.
            credentials:
                Custom credentials to use to create this feature. Overrides
                credentials set in aiplatform.init.
            request_metadata:
                Strings which should be sent along with the request as metadata.
            create_request_timeout:
                The timeout for the create request in seconds.

        Returns:
            FeatureMonitorJob - the FeatureMonitorJob resource object.
        """

        gapic_feature_monitor_job = gca_feature_monitor_job.FeatureMonitorJob()

        if description:
            gapic_feature_monitor_job.description = description

        if labels:
            utils.validate_labels(labels)
            gapic_feature_monitor_job.labels = labels

        if request_metadata is None:
            request_metadata = ()

        api_client = self.__class__._instantiate_client(
            location=location, credentials=credentials
        )

        created_feature_monitor_job = api_client.select_version(
            "v1beta1"
        ).create_feature_monitor_job(
            parent=self.resource_name,
            feature_monitor_job=gapic_feature_monitor_job,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        feature_monitor_job_obj = self.FeatureMonitorJob(
            name=created_feature_monitor_job.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return feature_monitor_job_obj

    def get_feature_monitor_job(
        self,
        feature_monitor_job_id: str,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> FeatureMonitorJob:
        """Retrieves an existing feature monitor.

        Args:
            feature_monitor_job_id: The ID of the feature monitor job.
            credentials:
                Custom credentials to use to retrieve the feature monitor job under this
                feature monitor. The order of which credentials are used is as
                follows - (1) this parameter (2) credentials passed to FeatureMonitor
                constructor (3) credentials set in aiplatform.init.

        Returns:
            FeatureMonitorJob - the Feature Monitor Job resource object under this
            feature monitor.
        """
        credentials = (
            credentials or self.credentials or initializer.global_config.credentials
        )
        return FeatureMonitor.FeatureMonitorJob(
            f"{self.resource_name}/featureMonitorJobs/{feature_monitor_job_id}",
            credentials=credentials,
        )

    def list_feature_monitor_jobs(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List[FeatureMonitorJob]:
        """Lists features monitor jobs under this feature monitor.

        Args:
            project:
                Project to list feature monitors in. If unset, the project set in
                aiplatform.init will be used.
            location:
                Location to list feature monitors in. If not set, location set in
                aiplatform.init will be used.
            credentials:
                Custom credentials to use to list feature monitors. Overrides
                credentials set in aiplatform.init.

        Returns:
            List of feature monitor jobs under this feature monitor.
        """

        return FeatureMonitor.FeatureMonitorJob.list(
            parent=self.resource_name,
            project=project,
            location=location,
            credentials=credentials,
        )
