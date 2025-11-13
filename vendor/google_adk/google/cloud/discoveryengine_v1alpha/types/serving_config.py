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

from google.cloud.discoveryengine_v1alpha.types import common, search_service

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "ServingConfig",
    },
)


class ServingConfig(proto.Message):
    r"""Configures metadata that is used to generate serving time
    results (e.g. search results or recommendation predictions). The
    ServingConfig is passed in the search and predict request and
    generates results.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        media_config (google.cloud.discoveryengine_v1alpha.types.ServingConfig.MediaConfig):
            The MediaConfig of the serving configuration.

            This field is a member of `oneof`_ ``vertical_config``.
        generic_config (google.cloud.discoveryengine_v1alpha.types.ServingConfig.GenericConfig):
            The GenericConfig of the serving
            configuration.

            This field is a member of `oneof`_ ``vertical_config``.
        name (str):
            Immutable. Fully qualified name
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}/servingConfigs/{serving_config_id}``
        display_name (str):
            Required. The human readable serving config display name.
            Used in Discovery UI.

            This field must be a UTF-8 encoded string with a length
            limit of 128 characters. Otherwise, an INVALID_ARGUMENT
            error is returned.
        solution_type (google.cloud.discoveryengine_v1alpha.types.SolutionType):
            Required. Immutable. Specifies the solution
            type that a serving config can be associated
            with.
        model_id (str):
            The id of the model to use at serving time. Currently only
            RecommendationModels are supported. Can be changed but only
            to a compatible model (e.g. others-you-may-like CTR to
            others-you-may-like CVR).

            Required when
            [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
            is
            [SOLUTION_TYPE_RECOMMENDATION][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_RECOMMENDATION].
        diversity_level (str):
            How much diversity to use in recommendation model results
            e.g. ``medium-diversity`` or ``high-diversity``. Currently
            supported values:

            - ``no-diversity``
            - ``low-diversity``
            - ``medium-diversity``
            - ``high-diversity``
            - ``auto-diversity``

            If not specified, we choose default based on recommendation
            model type. Default value: ``no-diversity``.

            Can only be set if
            [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
            is
            [SOLUTION_TYPE_RECOMMENDATION][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_RECOMMENDATION].
        embedding_config (google.cloud.discoveryengine_v1alpha.types.EmbeddingConfig):
            Bring your own embedding config. The config is used for
            search semantic retrieval. The retrieval is based on the dot
            product of
            [SearchRequest.EmbeddingSpec.EmbeddingVector.vector][google.cloud.discoveryengine.v1alpha.SearchRequest.EmbeddingSpec.EmbeddingVector.vector]
            and the document embeddings that are provided by this
            EmbeddingConfig. If
            [SearchRequest.EmbeddingSpec.EmbeddingVector.vector][google.cloud.discoveryengine.v1alpha.SearchRequest.EmbeddingSpec.EmbeddingVector.vector]
            is provided, it overrides this
            [ServingConfig.embedding_config][google.cloud.discoveryengine.v1alpha.ServingConfig.embedding_config].
        ranking_expression (str):
            The ranking expression controls the customized ranking on
            retrieval documents. To leverage this, document embedding is
            required. The ranking expression setting in ServingConfig
            applies to all search requests served by the serving config.
            However, if
            [SearchRequest.ranking_expression][google.cloud.discoveryengine.v1alpha.SearchRequest.ranking_expression]
            is specified, it overrides the ServingConfig ranking
            expression.

            The ranking expression is a single function or multiple
            functions that are joined by "+".

            - ranking_expression = function, { " + ", function };

            Supported functions:

            - double \* relevance_score
            - double \* dotProduct(embedding_field_path)

            Function variables:

            - ``relevance_score``: pre-defined keywords, used for
              measure relevance between query and document.
            - ``embedding_field_path``: the document embedding field
              used with query embedding vector.
            - ``dotProduct``: embedding function between
              embedding_field_path and query embedding vector.

            Example ranking expression:

            ::

               If document has an embedding field doc_embedding, the ranking expression
               could be `0.5 * relevance_score + 0.3 * dotProduct(doc_embedding)`.
        guided_search_spec (google.cloud.discoveryengine_v1alpha.types.GuidedSearchSpec):
            Guided search configs.
        custom_fine_tuning_spec (google.cloud.discoveryengine_v1alpha.types.CustomFineTuningSpec):
            Custom fine tuning configs. If
            [SearchRequest.custom_fine_tuning_spec][google.cloud.discoveryengine.v1alpha.SearchRequest.custom_fine_tuning_spec]
            is set, it has higher priority than the configs set here.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. ServingConfig created timestamp.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. ServingConfig updated timestamp.
        filter_control_ids (MutableSequence[str]):
            Filter controls to use in serving path.
            All triggered filter controls will be applied.
            Filter controls must be in the same data store
            as the serving config. Maximum of 20 filter
            controls.
        boost_control_ids (MutableSequence[str]):
            Boost controls to use in serving path.
            All triggered boost controls will be applied.
            Boost controls must be in the same data store as
            the serving config. Maximum of 20 boost
            controls.
        redirect_control_ids (MutableSequence[str]):
            IDs of the redirect controls. Only the first triggered
            redirect action is applied, even if multiple apply. Maximum
            number of specifications is 100.

            Can only be set if
            [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
            is
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_SEARCH].
        synonyms_control_ids (MutableSequence[str]):
            Condition synonyms specifications. If multiple synonyms
            conditions match, all matching synonyms controls in the list
            will execute. Maximum number of specifications is 100.

            Can only be set if
            [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
            is
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_SEARCH].
        oneway_synonyms_control_ids (MutableSequence[str]):
            Condition oneway synonyms specifications. If multiple oneway
            synonyms conditions match, all matching oneway synonyms
            controls in the list will execute. Maximum number of
            specifications is 100.

            Can only be set if
            [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
            is
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_SEARCH].
        dissociate_control_ids (MutableSequence[str]):
            Condition do not associate specifications. If multiple do
            not associate conditions match, all matching do not
            associate controls in the list will execute. Order does not
            matter. Maximum number of specifications is 100.

            Can only be set if
            [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
            is
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_SEARCH].
        replacement_control_ids (MutableSequence[str]):
            Condition replacement specifications. Applied according to
            the order in the list. A previously replaced term can not be
            re-replaced. Maximum number of specifications is 100.

            Can only be set if
            [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
            is
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_SEARCH].
        ignore_control_ids (MutableSequence[str]):
            Condition ignore specifications. If multiple
            ignore conditions match, all matching ignore
            controls in the list will execute.
            Order does not matter.
            Maximum number of specifications is 100.
    """

    class MediaConfig(proto.Message):
        r"""Specifies the configurations needed for Media Discovery. Currently
        we support:

        - ``demote_content_watched``: Threshold for watched content
          demotion. Customers can specify if using watched content demotion
          or use viewed detail page. Using the content watched demotion,
          customers need to specify the watched minutes or percentage
          exceeds the threshold, the content will be demoted in the
          recommendation result.
        - ``promote_fresh_content``: cutoff days for fresh content
          promotion. Customers can specify if using content freshness
          promotion. If the content was published within the cutoff days,
          the content will be promoted in the recommendation result. Can
          only be set if
          [SolutionType][google.cloud.discoveryengine.v1alpha.SolutionType]
          is
          [SOLUTION_TYPE_RECOMMENDATION][google.cloud.discoveryengine.v1alpha.SolutionType.SOLUTION_TYPE_RECOMMENDATION].

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            content_watched_percentage_threshold (float):
                Specifies the content watched percentage threshold for
                demotion. Threshold value must be between [0, 1.0]
                inclusive.

                This field is a member of `oneof`_ ``demote_content_watched``.
            content_watched_seconds_threshold (float):
                Specifies the content watched minutes
                threshold for demotion.

                This field is a member of `oneof`_ ``demote_content_watched``.
            demotion_event_type (str):
                Specifies the event type used for demoting recommendation
                result. Currently supported values:

                - ``view-item``: Item viewed.
                - ``media-play``: Start/resume watching a video, playing a
                  song, etc.
                - ``media-complete``: Finished or stopped midway through a
                  video, song, etc.

                If unset, watch history demotion will not be applied.
                Content freshness demotion will still be applied.
            content_freshness_cutoff_days (int):
                Specifies the content freshness used for
                recommendation result. Contents will be demoted
                if contents were published for more than content
                freshness cutoff days.
        """

        content_watched_percentage_threshold: float = proto.Field(
            proto.FLOAT,
            number=2,
            oneof="demote_content_watched",
        )
        content_watched_seconds_threshold: float = proto.Field(
            proto.FLOAT,
            number=5,
            oneof="demote_content_watched",
        )
        demotion_event_type: str = proto.Field(
            proto.STRING,
            number=1,
        )
        content_freshness_cutoff_days: int = proto.Field(
            proto.INT32,
            number=4,
        )

    class GenericConfig(proto.Message):
        r"""Specifies the configurations needed for Generic Discovery.Currently
        we support:

        - ``content_search_spec``: configuration for generic content search.

        Attributes:
            content_search_spec (google.cloud.discoveryengine_v1alpha.types.SearchRequest.ContentSearchSpec):
                Specifies the expected behavior of content
                search. Only valid for content-search enabled
                data store.
        """

        content_search_spec: search_service.SearchRequest.ContentSearchSpec = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                message=search_service.SearchRequest.ContentSearchSpec,
            )
        )

    media_config: MediaConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="vertical_config",
        message=MediaConfig,
    )
    generic_config: GenericConfig = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="vertical_config",
        message=GenericConfig,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    solution_type: common.SolutionType = proto.Field(
        proto.ENUM,
        number=3,
        enum=common.SolutionType,
    )
    model_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    diversity_level: str = proto.Field(
        proto.STRING,
        number=5,
    )
    embedding_config: common.EmbeddingConfig = proto.Field(
        proto.MESSAGE,
        number=20,
        message=common.EmbeddingConfig,
    )
    ranking_expression: str = proto.Field(
        proto.STRING,
        number=21,
    )
    guided_search_spec: common.GuidedSearchSpec = proto.Field(
        proto.MESSAGE,
        number=22,
        message=common.GuidedSearchSpec,
    )
    custom_fine_tuning_spec: common.CustomFineTuningSpec = proto.Field(
        proto.MESSAGE,
        number=24,
        message=common.CustomFineTuningSpec,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    filter_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )
    boost_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=12,
    )
    redirect_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=14,
    )
    synonyms_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=15,
    )
    oneway_synonyms_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=16,
    )
    dissociate_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=17,
    )
    replacement_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=18,
    )
    ignore_control_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=19,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
