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

from google.cloud.discoveryengine_v1.types import common

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "Engine",
    },
)


class Engine(proto.Message):
    r"""Metadata that describes the training and serving parameters of an
    [Engine][google.cloud.discoveryengine.v1.Engine].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        chat_engine_config (google.cloud.discoveryengine_v1.types.Engine.ChatEngineConfig):
            Configurations for the Chat Engine. Only applicable if
            [solution_type][google.cloud.discoveryengine.v1.Engine.solution_type]
            is
            [SOLUTION_TYPE_CHAT][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_CHAT].

            This field is a member of `oneof`_ ``engine_config``.
        search_engine_config (google.cloud.discoveryengine_v1.types.Engine.SearchEngineConfig):
            Configurations for the Search Engine. Only applicable if
            [solution_type][google.cloud.discoveryengine.v1.Engine.solution_type]
            is
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_SEARCH].

            This field is a member of `oneof`_ ``engine_config``.
        media_recommendation_engine_config (google.cloud.discoveryengine_v1.types.Engine.MediaRecommendationEngineConfig):
            Configurations for the Media Engine. Only applicable on the
            data stores with
            [solution_type][google.cloud.discoveryengine.v1.Engine.solution_type]
            [SOLUTION_TYPE_RECOMMENDATION][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
            and
            [IndustryVertical.MEDIA][google.cloud.discoveryengine.v1.IndustryVertical.MEDIA]
            vertical.

            This field is a member of `oneof`_ ``engine_config``.
        chat_engine_metadata (google.cloud.discoveryengine_v1.types.Engine.ChatEngineMetadata):
            Output only. Additional information of the Chat Engine. Only
            applicable if
            [solution_type][google.cloud.discoveryengine.v1.Engine.solution_type]
            is
            [SOLUTION_TYPE_CHAT][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_CHAT].

            This field is a member of `oneof`_ ``engine_metadata``.
        name (str):
            Immutable. Identifier. The fully qualified resource name of
            the engine.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.

            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}``
            engine should be 1-63 characters, and valid characters are
            /[a-z0-9][a-z0-9-\_]*/. Otherwise, an INVALID_ARGUMENT error
            is returned.
        display_name (str):
            Required. The display name of the engine.
            Should be human readable. UTF-8 encoded string
            with limit of 1024 characters.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the Recommendation
            Engine was created at.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the Recommendation
            Engine was last updated.
        data_store_ids (MutableSequence[str]):
            Optional. The data stores associated with this engine.

            For
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_SEARCH]
            and
            [SOLUTION_TYPE_RECOMMENDATION][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
            type of engines, they can only associate with at most one
            data store.

            If
            [solution_type][google.cloud.discoveryengine.v1.Engine.solution_type]
            is
            [SOLUTION_TYPE_CHAT][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_CHAT],
            multiple
            [DataStore][google.cloud.discoveryengine.v1.DataStore]s in
            the same
            [Collection][google.cloud.discoveryengine.v1.Collection] can
            be associated here.

            Note that when used in
            [CreateEngineRequest][google.cloud.discoveryengine.v1.CreateEngineRequest],
            one DataStore id must be provided as the system will use it
            for necessary initializations.
        solution_type (google.cloud.discoveryengine_v1.types.SolutionType):
            Required. The solutions of the engine.
        industry_vertical (google.cloud.discoveryengine_v1.types.IndustryVertical):
            Optional. The industry vertical that the engine registers.
            The restriction of the Engine industry vertical is based on
            [DataStore][google.cloud.discoveryengine.v1.DataStore]:
            Vertical on Engine has to match vertical of the DataStore
            linked to the engine.
        common_config (google.cloud.discoveryengine_v1.types.Engine.CommonConfig):
            Common config spec that specifies the
            metadata of the engine.
        disable_analytics (bool):
            Optional. Whether to disable analytics for
            searches performed on this engine.
    """

    class SearchEngineConfig(proto.Message):
        r"""Configurations for a Search Engine.

        Attributes:
            search_tier (google.cloud.discoveryengine_v1.types.SearchTier):
                The search feature tier of this engine.

                Different tiers might have different pricing. To learn more,
                check the pricing documentation.

                Defaults to
                [SearchTier.SEARCH_TIER_STANDARD][google.cloud.discoveryengine.v1.SearchTier.SEARCH_TIER_STANDARD]
                if not specified.
            search_add_ons (MutableSequence[google.cloud.discoveryengine_v1.types.SearchAddOn]):
                The add-on that this search engine enables.
        """

        search_tier: common.SearchTier = proto.Field(
            proto.ENUM,
            number=1,
            enum=common.SearchTier,
        )
        search_add_ons: MutableSequence[common.SearchAddOn] = proto.RepeatedField(
            proto.ENUM,
            number=2,
            enum=common.SearchAddOn,
        )

    class MediaRecommendationEngineConfig(proto.Message):
        r"""Additional config specs for a Media Recommendation engine.

        Attributes:
            type_ (str):
                Required. The type of engine. e.g., ``recommended-for-you``.

                This field together with
                [optimization_objective][google.cloud.discoveryengine.v1.Engine.MediaRecommendationEngineConfig.optimization_objective]
                describe engine metadata to use to control engine training
                and serving.

                Currently supported values: ``recommended-for-you``,
                ``others-you-may-like``, ``more-like-this``,
                ``most-popular-items``.
            optimization_objective (str):
                The optimization objective. e.g., ``cvr``.

                This field together with
                [optimization_objective][google.cloud.discoveryengine.v1.Engine.MediaRecommendationEngineConfig.type]
                describe engine metadata to use to control engine training
                and serving.

                Currently supported values: ``ctr``, ``cvr``.

                If not specified, we choose default based on engine type.
                Default depends on type of recommendation:

                ``recommended-for-you`` => ``ctr``

                ``others-you-may-like`` => ``ctr``
            optimization_objective_config (google.cloud.discoveryengine_v1.types.Engine.MediaRecommendationEngineConfig.OptimizationObjectiveConfig):
                Name and value of the custom threshold for cvr
                optimization_objective. For target_field ``watch-time``,
                target_field_value must be an integer value indicating the
                media progress time in seconds between (0, 86400] (excludes
                0, includes 86400) (e.g., 90). For target_field
                ``watch-percentage``, the target_field_value must be a valid
                float value between (0, 1.0] (excludes 0, includes 1.0)
                (e.g., 0.5).
            training_state (google.cloud.discoveryengine_v1.types.Engine.MediaRecommendationEngineConfig.TrainingState):
                The training state that the engine is in (e.g. ``TRAINING``
                or ``PAUSED``).

                Since part of the cost of running the service is frequency
                of training - this can be used to determine when to train
                engine in order to control cost. If not specified: the
                default value for ``CreateEngine`` method is ``TRAINING``.
                The default value for ``UpdateEngine`` method is to keep the
                state the same as before.
            engine_features_config (google.cloud.discoveryengine_v1.types.Engine.MediaRecommendationEngineConfig.EngineFeaturesConfig):
                Optional. Additional engine features config.
        """

        class TrainingState(proto.Enum):
            r"""The training state of the engine.

            Values:
                TRAINING_STATE_UNSPECIFIED (0):
                    Unspecified training state.
                PAUSED (1):
                    The engine training is paused.
                TRAINING (2):
                    The engine is training.
            """
            TRAINING_STATE_UNSPECIFIED = 0
            PAUSED = 1
            TRAINING = 2

        class OptimizationObjectiveConfig(proto.Message):
            r"""Custom threshold for ``cvr`` optimization_objective.

            Attributes:
                target_field (str):
                    Required. The name of the field to target. Currently
                    supported values: ``watch-percentage``, ``watch-time``.
                target_field_value_float (float):
                    Required. The threshold to be applied to the
                    target (e.g., 0.5).
            """

            target_field: str = proto.Field(
                proto.STRING,
                number=1,
            )
            target_field_value_float: float = proto.Field(
                proto.FLOAT,
                number=2,
            )

        class EngineFeaturesConfig(proto.Message):
            r"""More feature configs of the selected engine type.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                recommended_for_you_config (google.cloud.discoveryengine_v1.types.Engine.MediaRecommendationEngineConfig.RecommendedForYouFeatureConfig):
                    Recommended for you engine feature config.

                    This field is a member of `oneof`_ ``type_dedicated_config``.
                most_popular_config (google.cloud.discoveryengine_v1.types.Engine.MediaRecommendationEngineConfig.MostPopularFeatureConfig):
                    Most popular engine feature config.

                    This field is a member of `oneof`_ ``type_dedicated_config``.
            """

            recommended_for_you_config: "Engine.MediaRecommendationEngineConfig.RecommendedForYouFeatureConfig" = proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="type_dedicated_config",
                message="Engine.MediaRecommendationEngineConfig.RecommendedForYouFeatureConfig",
            )
            most_popular_config: "Engine.MediaRecommendationEngineConfig.MostPopularFeatureConfig" = proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="type_dedicated_config",
                message="Engine.MediaRecommendationEngineConfig.MostPopularFeatureConfig",
            )

        class RecommendedForYouFeatureConfig(proto.Message):
            r"""Additional feature configurations for creating a
            ``recommended-for-you`` engine.

            Attributes:
                context_event_type (str):
                    The type of event with which the engine is queried at
                    prediction time. If set to ``generic``, only ``view-item``,
                    ``media-play``,and ``media-complete`` will be used as
                    ``context-event`` in engine training. If set to
                    ``view-home-page``, ``view-home-page`` will also be used as
                    ``context-events`` in addition to ``view-item``,
                    ``media-play``, and ``media-complete``. Currently supported
                    for the ``recommended-for-you`` engine. Currently supported
                    values: ``view-home-page``, ``generic``.
            """

            context_event_type: str = proto.Field(
                proto.STRING,
                number=1,
            )

        class MostPopularFeatureConfig(proto.Message):
            r"""Feature configurations that are required for creating a Most
            Popular engine.

            Attributes:
                time_window_days (int):
                    The time window of which the engine is queried at training
                    and prediction time. Positive integers only. The value
                    translates to the last X days of events. Currently required
                    for the ``most-popular-items`` engine.
            """

            time_window_days: int = proto.Field(
                proto.INT64,
                number=1,
            )

        type_: str = proto.Field(
            proto.STRING,
            number=1,
        )
        optimization_objective: str = proto.Field(
            proto.STRING,
            number=2,
        )
        optimization_objective_config: "Engine.MediaRecommendationEngineConfig.OptimizationObjectiveConfig" = proto.Field(
            proto.MESSAGE,
            number=3,
            message="Engine.MediaRecommendationEngineConfig.OptimizationObjectiveConfig",
        )
        training_state: "Engine.MediaRecommendationEngineConfig.TrainingState" = (
            proto.Field(
                proto.ENUM,
                number=4,
                enum="Engine.MediaRecommendationEngineConfig.TrainingState",
            )
        )
        engine_features_config: "Engine.MediaRecommendationEngineConfig.EngineFeaturesConfig" = proto.Field(
            proto.MESSAGE,
            number=5,
            message="Engine.MediaRecommendationEngineConfig.EngineFeaturesConfig",
        )

    class ChatEngineConfig(proto.Message):
        r"""Configurations for a Chat Engine.

        Attributes:
            agent_creation_config (google.cloud.discoveryengine_v1.types.Engine.ChatEngineConfig.AgentCreationConfig):
                The configurationt generate the Dialogflow agent that is
                associated to this Engine.

                Note that these configurations are one-time consumed by and
                passed to Dialogflow service. It means they cannot be
                retrieved using
                [EngineService.GetEngine][google.cloud.discoveryengine.v1.EngineService.GetEngine]
                or
                [EngineService.ListEngines][google.cloud.discoveryengine.v1.EngineService.ListEngines]
                API after engine creation.
            dialogflow_agent_to_link (str):
                The resource name of an exist Dialogflow agent to link to
                this Chat Engine. Customers can either provide
                ``agent_creation_config`` to create agent or provide an
                agent name that links the agent with the Chat engine.

                Format:
                ``projects/<Project ID>/locations/<Location ID>/agents/<Agent ID>``.

                Note that the ``dialogflow_agent_to_link`` are one-time
                consumed by and passed to Dialogflow service. It means they
                cannot be retrieved using
                [EngineService.GetEngine][google.cloud.discoveryengine.v1.EngineService.GetEngine]
                or
                [EngineService.ListEngines][google.cloud.discoveryengine.v1.EngineService.ListEngines]
                API after engine creation. Use
                [ChatEngineMetadata.dialogflow_agent][google.cloud.discoveryengine.v1.Engine.ChatEngineMetadata.dialogflow_agent]
                for actual agent association after Engine is created.
            allow_cross_region (bool):
                Optional. If the flag set to true, we allow the agent and
                engine are in different locations, otherwise the agent and
                engine are required to be in the same location. The flag is
                set to false by default.

                Note that the ``allow_cross_region`` are one-time consumed
                by and passed to
                [EngineService.CreateEngine][google.cloud.discoveryengine.v1.EngineService.CreateEngine].
                It means they cannot be retrieved using
                [EngineService.GetEngine][google.cloud.discoveryengine.v1.EngineService.GetEngine]
                or
                [EngineService.ListEngines][google.cloud.discoveryengine.v1.EngineService.ListEngines]
                API after engine creation.
        """

        class AgentCreationConfig(proto.Message):
            r"""Configurations for generating a Dialogflow agent.

            Note that these configurations are one-time consumed by and passed
            to Dialogflow service. It means they cannot be retrieved using
            [EngineService.GetEngine][google.cloud.discoveryengine.v1.EngineService.GetEngine]
            or
            [EngineService.ListEngines][google.cloud.discoveryengine.v1.EngineService.ListEngines]
            API after engine creation.

            Attributes:
                business (str):
                    Name of the company, organization or other
                    entity that the agent represents. Used for
                    knowledge connector LLM prompt and for knowledge
                    search.
                default_language_code (str):
                    Required. The default language of the agent as a language
                    tag. See `Language
                    Support <https://cloud.google.com/dialogflow/docs/reference/language>`__
                    for a list of the currently supported language codes.
                time_zone (str):
                    Required. The time zone of the agent from the `time zone
                    database <https://www.iana.org/time-zones>`__, e.g.,
                    America/New_York, Europe/Paris.
                location (str):
                    Agent location for Agent creation, supported
                    values: global/us/eu. If not provided, us Engine
                    will create Agent using us-central-1 by default;
                    eu Engine will create Agent using eu-west-1 by
                    default.
            """

            business: str = proto.Field(
                proto.STRING,
                number=1,
            )
            default_language_code: str = proto.Field(
                proto.STRING,
                number=2,
            )
            time_zone: str = proto.Field(
                proto.STRING,
                number=3,
            )
            location: str = proto.Field(
                proto.STRING,
                number=4,
            )

        agent_creation_config: "Engine.ChatEngineConfig.AgentCreationConfig" = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                message="Engine.ChatEngineConfig.AgentCreationConfig",
            )
        )
        dialogflow_agent_to_link: str = proto.Field(
            proto.STRING,
            number=2,
        )
        allow_cross_region: bool = proto.Field(
            proto.BOOL,
            number=3,
        )

    class CommonConfig(proto.Message):
        r"""Common configurations for an Engine.

        Attributes:
            company_name (str):
                The name of the company, business or entity
                that is associated with the engine. Setting this
                may help improve LLM related features.
        """

        company_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class ChatEngineMetadata(proto.Message):
        r"""Additional information of a Chat Engine.
        Fields in this message are output only.

        Attributes:
            dialogflow_agent (str):
                The resource name of a Dialogflow agent, that this Chat
                Engine refers to.

                Format:
                ``projects/<Project ID>/locations/<Location ID>/agents/<Agent ID>``.
        """

        dialogflow_agent: str = proto.Field(
            proto.STRING,
            number=1,
        )

    chat_engine_config: ChatEngineConfig = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="engine_config",
        message=ChatEngineConfig,
    )
    search_engine_config: SearchEngineConfig = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="engine_config",
        message=SearchEngineConfig,
    )
    media_recommendation_engine_config: MediaRecommendationEngineConfig = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="engine_config",
        message=MediaRecommendationEngineConfig,
    )
    chat_engine_metadata: ChatEngineMetadata = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="engine_metadata",
        message=ChatEngineMetadata,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    data_store_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=5,
    )
    solution_type: common.SolutionType = proto.Field(
        proto.ENUM,
        number=6,
        enum=common.SolutionType,
    )
    industry_vertical: common.IndustryVertical = proto.Field(
        proto.ENUM,
        number=16,
        enum=common.IndustryVertical,
    )
    common_config: CommonConfig = proto.Field(
        proto.MESSAGE,
        number=15,
        message=CommonConfig,
    )
    disable_analytics: bool = proto.Field(
        proto.BOOL,
        number=26,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
