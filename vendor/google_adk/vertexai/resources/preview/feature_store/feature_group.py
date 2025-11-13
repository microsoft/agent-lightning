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

from typing import Dict, List, Optional, Sequence, Tuple
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base, initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    feature as gca_feature,
    feature_group as gca_feature_group,
    io as gca_io,
    feature_monitor_v1beta1 as gca_feature_monitor,
)
from vertexai.resources.preview.feature_store.utils import (
    FeatureGroupBigQuerySource,
)
from vertexai.resources.preview.feature_store import (
    Feature,
)
from vertexai.resources.preview.feature_store.feature_monitor import (
    FeatureMonitor,
)


_LOGGER = base.Logger(__name__)


class FeatureGroup(base.VertexAiResourceNounWithFutureManager):
    """Class for managing Feature Group resources."""

    client_class = utils.FeatureRegistryClientWithOverride

    _resource_noun = "feature_groups"
    _getter_method = "get_feature_group"
    _list_method = "list_feature_groups"
    _delete_method = "delete_feature_group"
    _parse_resource_name_method = "parse_feature_group_path"
    _format_resource_name_method = "feature_group_path"
    _gca_resource: gca_feature_group.FeatureGroup

    def __init__(
        self,
        name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed feature group.

        Args:
            name:
                The resource name
                (`projects/.../locations/.../featureGroups/...`) or ID.
            project:
                Project to retrieve feature group from. If unset, the
                project set in aiplatform.init will be used.
            location:
                Location to retrieve feature group from. If not set,
                location set in aiplatform.init will be used.
            credentials:
                Custom credentials to use to retrieve this feature group.
                Overrides credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=name,
        )

        self._gca_resource = self._get_gca_resource(resource_name=name)

    @classmethod
    def create(
        cls,
        name: str,
        source: FeatureGroupBigQuerySource = None,
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
        create_request_timeout: Optional[float] = None,
        sync: bool = True,
    ) -> "FeatureGroup":
        """Creates a new feature group.

        Args:
            name: The name of the feature group.
            source: The BigQuery source of the feature group.
            labels:
                The labels with user-defined metadata to organize your
                FeatureGroup.

                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.

                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one
                FeatureGroup(System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            description: Description of the FeatureGroup.
            project:
                Project to create feature group in. If unset, the project set in
                aiplatform.init will be used.
            location:
                Location to create feature group in. If not set, location set in
                aiplatform.init will be used.
            credentials:
                Custom credentials to use to create this feature group.
                Overrides credentials set in aiplatform.init.
            request_metadata:
                Strings which should be sent along with the request as metadata.
            create_request_timeout:
                The timeout for the create request in seconds.
            sync:
                Whether to execute this creation synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.

        Returns:
            FeatureGroup - the FeatureGroup resource object.
        """

        if not source:
            raise ValueError("Please specify a valid source.")

        # Only BigQuery source is supported right now.
        if not isinstance(source, FeatureGroupBigQuerySource):
            raise ValueError("Only FeatureGroupBigQuerySource is a supported source.")

        # BigQuery source validation.
        if not source.uri:
            raise ValueError("Please specify URI in BigQuery source.")

        if not source.entity_id_columns:
            _LOGGER.info(
                "No entity ID columns specified in BigQuery source. Defaulting to ['entity_id']."
            )
            entity_id_columns = ["entity_id"]
        else:
            entity_id_columns = source.entity_id_columns

        gapic_feature_group = gca_feature_group.FeatureGroup(
            big_query=gca_feature_group.FeatureGroup.BigQuery(
                big_query_source=gca_io.BigQuerySource(input_uri=source.uri),
                entity_id_columns=entity_id_columns,
            ),
            name=name,
            description=description,
        )

        if labels:
            utils.validate_labels(labels)
            gapic_feature_group.labels = labels

        if request_metadata is None:
            request_metadata = ()

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        create_feature_group_lro = api_client.create_feature_group(
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            feature_group=gapic_feature_group,
            feature_group_id=name,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, create_feature_group_lro)

        created_feature_group = create_feature_group_lro.result()

        _LOGGER.log_create_complete(cls, created_feature_group, "feature_group")

        feature_group_obj = cls(
            name=created_feature_group.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return feature_group_obj

    @base.optional_sync()
    def delete(self, force: bool = False, sync: bool = True) -> None:
        """Deletes this feature group.

        WARNING: This deletion is permanent.

        Args:
            force:
                If set to True, all features under this online store will be
                deleted prior to online store deletion. Otherwise, deletion
                will only succeed if the online store has no FeatureViews.

                If set to true, any Features under this FeatureGroup will also
                be deleted. (Otherwise, the request will only work if the
                FeatureGroup has no Features.)
            sync:
                Whether to execute this deletion synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.
        """

        lro = getattr(self.api_client, self._delete_method)(
            name=self.resource_name,
            force=force,
        )
        _LOGGER.log_delete_with_lro(self, lro)
        lro.result()
        _LOGGER.log_delete_complete(self)

    def get_feature(
        self,
        feature_id: str,
        latest_stats_count: Optional[int] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> Feature:
        """Retrieves an existing managed feature.

        Args:
            feature_id: The ID of the feature.
            latest_stats_count:
                The number of latest stats to retrieve. Only returns stats if
                Feature Monitor is created, and historical stats were generated.
            credentials:
                Custom credentials to use to retrieve the feature under this
                feature group. The order of which credentials are used is as
                follows: (1) this parameter (2) credentials passed to FeatureGroup
                constructor (3) credentials set in aiplatform.init.

        Returns:
            Feature - the Feature resource object under this feature group.
        """
        credentials = (
            credentials or self.credentials or initializer.global_config.credentials
        )
        if latest_stats_count is not None:
            return Feature(
                name=f"{self.resource_name}/features/{feature_id}",
                latest_stats_count=latest_stats_count,
                credentials=credentials,
            )
        return Feature(
            f"{self.resource_name}/features/{feature_id}", credentials=credentials
        )

    def create_feature(
        self,
        name: str,
        version_column_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        point_of_contact: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
        create_request_timeout: Optional[float] = None,
        sync: bool = True,
    ) -> Feature:
        """Creates a new feature.

        Args:
            name: The name of the feature.
            version_column_name:
                The name of the BigQuery Table/View column hosting data for this
                version. If no value is provided, will use feature_id.
            description: Description of the feature.
            labels:
                The labels with user-defined metadata to organize your Features.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters,
                numeric characters, underscores and dashes. International
                characters are allowed.

                See https://goo.gl/xmQnxf for more information on and examples
                of labels. No more than 64 user labels can be associated with
                one Feature (System labels are excluded)." System reserved label
                keys are prefixed with "aiplatform.googleapis.com/" and are
                immutable.
            point_of_contact:
                Entity responsible for maintaining this feature. Can be comma
                separated list of email addresses or URIs.
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
            sync:
                Whether to execute this creation synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.

        Returns:
            Feature - the Feature resource object.
        """

        gapic_feature = gca_feature.Feature()

        if version_column_name:
            gapic_feature.version_column_name = version_column_name

        if description:
            gapic_feature.description = description

        if labels:
            utils.validate_labels(labels)
            gapic_feature.labels = labels

        if point_of_contact:
            gapic_feature.point_of_contact = point_of_contact

        if request_metadata is None:
            request_metadata = ()

        api_client = self.__class__._instantiate_client(
            location=location, credentials=credentials
        )

        create_feature_lro = api_client.create_feature(
            parent=self.resource_name,
            feature=gapic_feature,
            feature_id=name,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(Feature, create_feature_lro)

        created_feature = create_feature_lro.result()

        _LOGGER.log_create_complete(Feature, created_feature, "feature")

        feature_obj = Feature(
            name=created_feature.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return feature_obj

    def list_features(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List[Feature]:
        """Lists features under this feature group.

        Args:
            project:
                Project to list features in. If unset, the project set in
                aiplatform.init will be used.
            location:
                Location to list features in. If not set, location set in
                aiplatform.init will be used.
            credentials:
                Custom credentials to use to list features. Overrides
                credentials set in aiplatform.init.

        Returns:
            List of features under this feature group.
        """

        return Feature.list(
            parent=self.resource_name,
            project=project,
            location=location,
            credentials=credentials,
        )

    def get_feature_monitor(
        self,
        feature_monitor_id: str,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> FeatureMonitor:
        """Retrieves an existing feature monitor.

        Args:
            feature_monitor_id: The ID of the feature monitor.
            credentials:
                Custom credentials to use to retrieve the feature monitor under this
                feature group. The order of which credentials are used is as
                follows: (1) this parameter (2) credentials passed to FeatureGroup
                constructor (3) credentials set in aiplatform.init.

        Returns:
            FeatureMonitor - the Feature Monitor resource object under this
            feature group.
        """
        credentials = (
            credentials or self.credentials or initializer.global_config.credentials
        )
        return FeatureMonitor(
            f"{self.resource_name}/featureMonitors/{feature_monitor_id}",
            credentials=credentials,
        )

    def create_feature_monitor(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        schedule_config: Optional[str] = None,
        feature_selection_configs: Optional[List[Tuple[str, float]]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
        create_request_timeout: Optional[float] = None,
    ) -> FeatureMonitor:
        """Creates a new feature monitor.

        Args:
            name: The name of the feature monitor.
            description: Description of the feature monitor.
            labels:
                The labels with user-defined metadata to organize your FeatureMonitors.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters,
                numeric characters, underscores and dashes. International
                characters are allowed.

                See https://goo.gl/xmQnxf for more information on and examples
                of labels. No more than 64 user labels can be associated with
                one FeatureMonitor (System labels are excluded)." System reserved label
                keys are prefixed with "aiplatform.googleapis.com/" and are
                immutable.
            schedule_config:
                Configures when data is to be monitored for this
                FeatureMonitor. At the end of the scheduled time,
                the stats and drift are generated for the selected features.
                Example format: "TZ=America/New_York 0 9 * * *" (monitors
                daily at 9 AM EST).
            feature_selection_configs:
                List of tuples of feature id and monitoring threshold. If unset,
                all features in the feature group will be monitored, and the
                default thresholds 0.3 will be used.
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
            FeatureMonitor - the FeatureMonitor resource object.
        """

        gapic_feature_monitor = gca_feature_monitor.FeatureMonitor()

        if description:
            gapic_feature_monitor.description = description

        if labels:
            utils.validate_labels(labels)
            gapic_feature_monitor.labels = labels

        if request_metadata is None:
            request_metadata = ()

        if schedule_config:
            gapic_feature_monitor.schedule_config = gca_feature_monitor.ScheduleConfig(
                cron=schedule_config
            )

        if feature_selection_configs is None:
            raise ValueError(
                "Please specify feature_configs: features to be monitored and"
                " their thresholds."
            )

        if feature_selection_configs is not None:
            gapic_feature_monitor.feature_selection_config.feature_configs = [
                gca_feature_monitor.FeatureSelectionConfig.FeatureConfig(
                    feature_id=feature_id,
                    drift_threshold=threshold if threshold else 0.3,
                )
                for feature_id, threshold in feature_selection_configs
            ]

        api_client = self.__class__._instantiate_client(
            location=location, credentials=credentials
        )

        create_feature_monitor_lro = api_client.select_version(
            "v1beta1"
        ).create_feature_monitor(
            parent=self.resource_name,
            feature_monitor=gapic_feature_monitor,
            feature_monitor_id=name,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(FeatureMonitor, create_feature_monitor_lro)

        created_feature_monitor = create_feature_monitor_lro.result()

        _LOGGER.log_create_complete(
            FeatureMonitor, created_feature_monitor, "feature_monitor"
        )

        feature_monitor_obj = FeatureMonitor(
            name=created_feature_monitor.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return feature_monitor_obj

    def list_feature_monitors(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List[FeatureMonitor]:
        """Lists features monitors under this feature group.

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
            List of feature monitors under this feature group.
        """

        return FeatureMonitor.list(
            parent=self.resource_name,
            project=project,
            location=location,
            credentials=credentials,
        )

    @property
    def source(self) -> FeatureGroupBigQuerySource:
        return FeatureGroupBigQuerySource(
            uri=self._gca_resource.big_query.big_query_source.input_uri,
            entity_id_columns=self._gca_resource.big_query.entity_id_columns,
        )
