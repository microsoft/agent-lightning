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
        "Condition",
        "Control",
    },
)


class Condition(proto.Message):
    r"""Defines circumstances to be checked before allowing a
    behavior

    Attributes:
        query_terms (MutableSequence[google.cloud.discoveryengine_v1.types.Condition.QueryTerm]):
            Search only A list of terms to match the query on. Cannot be
            set when
            [Condition.query_regex][google.cloud.discoveryengine.v1.Condition.query_regex]
            is set.

            Maximum of 10 query terms.
        active_time_range (MutableSequence[google.cloud.discoveryengine_v1.types.Condition.TimeRange]):
            Range of time(s) specifying when condition is
            active.
            Maximum of 10 time ranges.
        query_regex (str):
            Optional. Query regex to match the whole search query.
            Cannot be set when
            [Condition.query_terms][google.cloud.discoveryengine.v1.Condition.query_terms]
            is set. Only supported for Basic Site Search promotion
            serving controls.
    """

    class QueryTerm(proto.Message):
        r"""Matcher for search request query

        Attributes:
            value (str):
                The specific query value to match against

                Must be lowercase, must be UTF-8. Can have at most 3 space
                separated terms if full_match is true. Cannot be an empty
                string. Maximum length of 5000 characters.
            full_match (bool):
                Whether the search query needs to exactly
                match the query term.
        """

        value: str = proto.Field(
            proto.STRING,
            number=1,
        )
        full_match: bool = proto.Field(
            proto.BOOL,
            number=2,
        )

    class TimeRange(proto.Message):
        r"""Used for time-dependent conditions.

        Attributes:
            start_time (google.protobuf.timestamp_pb2.Timestamp):
                Start of time range.

                Range is inclusive.
            end_time (google.protobuf.timestamp_pb2.Timestamp):
                End of time range.

                Range is inclusive.
                Must be in the future.
        """

        start_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )
        end_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=2,
            message=timestamp_pb2.Timestamp,
        )

    query_terms: MutableSequence[QueryTerm] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=QueryTerm,
    )
    active_time_range: MutableSequence[TimeRange] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=TimeRange,
    )
    query_regex: str = proto.Field(
        proto.STRING,
        number=4,
    )


class Control(proto.Message):
    r"""Defines a conditioned behavior to employ during serving. Must be
    attached to a
    [ServingConfig][google.cloud.discoveryengine.v1.ServingConfig] to be
    considered at serving time. Permitted actions dependent on
    ``SolutionType``.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        boost_action (google.cloud.discoveryengine_v1.types.Control.BoostAction):
            Defines a boost-type control

            This field is a member of `oneof`_ ``action``.
        filter_action (google.cloud.discoveryengine_v1.types.Control.FilterAction):
            Defines a filter-type control
            Currently not supported by Recommendation

            This field is a member of `oneof`_ ``action``.
        redirect_action (google.cloud.discoveryengine_v1.types.Control.RedirectAction):
            Defines a redirect-type control.

            This field is a member of `oneof`_ ``action``.
        synonyms_action (google.cloud.discoveryengine_v1.types.Control.SynonymsAction):
            Treats a group of terms as synonyms of one
            another.

            This field is a member of `oneof`_ ``action``.
        promote_action (google.cloud.discoveryengine_v1.types.Control.PromoteAction):
            Promote certain links based on predefined
            trigger queries.

            This field is a member of `oneof`_ ``action``.
        name (str):
            Immutable. Fully qualified name
            ``projects/*/locations/global/dataStore/*/controls/*``
        display_name (str):
            Required. Human readable name. The identifier
            used in UI views.
            Must be UTF-8 encoded string. Length limit is
            128 characters. Otherwise an INVALID ARGUMENT
            error is thrown.
        associated_serving_config_ids (MutableSequence[str]):
            Output only. List of all
            [ServingConfig][google.cloud.discoveryengine.v1.ServingConfig]
            IDs this control is attached to. May take up to 10 minutes
            to update after changes.
        solution_type (google.cloud.discoveryengine_v1.types.SolutionType):
            Required. Immutable. What solution the
            control belongs to.
            Must be compatible with vertical of resource.
            Otherwise an INVALID ARGUMENT error is thrown.
        use_cases (MutableSequence[google.cloud.discoveryengine_v1.types.SearchUseCase]):
            Specifies the use case for the control. Affects what
            condition fields can be set. Only applies to
            [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_SEARCH].
            Currently only allow one use case per control. Must be set
            when solution_type is
            [SolutionType.SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_SEARCH].
        conditions (MutableSequence[google.cloud.discoveryengine_v1.types.Condition]):
            Determines when the associated action will
            trigger.
            Omit to always apply the action.
            Currently only a single condition may be
            specified. Otherwise an INVALID ARGUMENT error
            is thrown.
    """

    class BoostAction(proto.Message):
        r"""Adjusts order of products in returned list.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            fixed_boost (float):
                Optional. Strength of the boost, which should be in [-1, 1].
                Negative boost means demotion. Default is 0.0 (No-op).

                This field is a member of `oneof`_ ``boost_spec``.
            interpolation_boost_spec (google.cloud.discoveryengine_v1.types.Control.BoostAction.InterpolationBoostSpec):
                Optional. Complex specification for custom
                ranking based on customer defined attribute
                value.

                This field is a member of `oneof`_ ``boost_spec``.
            boost (float):
                Strength of the boost, which should be in [-1, 1]. Negative
                boost means demotion. Default is 0.0 (No-op).
            filter (str):
                Required. Specifies which products to apply
                the boost to.
                If no filter is provided all products will be
                boosted (No-op). Syntax documentation:

                https://cloud.google.com/retail/docs/filter-and-order
                Maximum length is 5000 characters.
                Otherwise an INVALID ARGUMENT error is thrown.
            data_store (str):
                Required. Specifies which data store's documents can be
                boosted by this control. Full data store name e.g.
                projects/123/locations/global/collections/default_collection/dataStores/default_data_store
        """

        class InterpolationBoostSpec(proto.Message):
            r"""Specification for custom ranking based on customer specified
            attribute value. It provides more controls for customized
            ranking than the simple (condition, boost) combination above.

            Attributes:
                field_name (str):
                    Optional. The name of the field whose value
                    will be used to determine the boost amount.
                attribute_type (google.cloud.discoveryengine_v1.types.Control.BoostAction.InterpolationBoostSpec.AttributeType):
                    Optional. The attribute type to be used to determine the
                    boost amount. The attribute value can be derived from the
                    field value of the specified field_name. In the case of
                    numerical it is straightforward i.e. attribute_value =
                    numerical_field_value. In the case of freshness however,
                    attribute_value = (time.now() - datetime_field_value).
                interpolation_type (google.cloud.discoveryengine_v1.types.Control.BoostAction.InterpolationBoostSpec.InterpolationType):
                    Optional. The interpolation type to be
                    applied to connect the control points listed
                    below.
                control_points (MutableSequence[google.cloud.discoveryengine_v1.types.Control.BoostAction.InterpolationBoostSpec.ControlPoint]):
                    Optional. The control points used to define the curve. The
                    monotonic function (defined through the interpolation_type
                    above) passes through the control points listed here.
            """

            class AttributeType(proto.Enum):
                r"""The attribute(or function) for which the custom ranking is to
                be applied.

                Values:
                    ATTRIBUTE_TYPE_UNSPECIFIED (0):
                        Unspecified AttributeType.
                    NUMERICAL (1):
                        The value of the numerical field will be used to dynamically
                        update the boost amount. In this case, the attribute_value
                        (the x value) of the control point will be the actual value
                        of the numerical field for which the boost_amount is
                        specified.
                    FRESHNESS (2):
                        For the freshness use case the attribute value will be the
                        duration between the current time and the date in the
                        datetime field specified. The value must be formatted as an
                        XSD ``dayTimeDuration`` value (a restricted subset of an ISO
                        8601 duration value). The pattern for this is:
                        ``[nD][T[nH][nM][nS]]``. For example, ``5D``, ``3DT12H30M``,
                        ``T24H``.
                """
                ATTRIBUTE_TYPE_UNSPECIFIED = 0
                NUMERICAL = 1
                FRESHNESS = 2

            class InterpolationType(proto.Enum):
                r"""The interpolation type to be applied. Default will be linear
                (Piecewise Linear).

                Values:
                    INTERPOLATION_TYPE_UNSPECIFIED (0):
                        Interpolation type is unspecified. In this
                        case, it defaults to Linear.
                    LINEAR (1):
                        Piecewise linear interpolation will be
                        applied.
                """
                INTERPOLATION_TYPE_UNSPECIFIED = 0
                LINEAR = 1

            class ControlPoint(proto.Message):
                r"""The control points used to define the curve. The curve
                defined through these control points can only be monotonically
                increasing or decreasing(constant values are acceptable).

                Attributes:
                    attribute_value (str):
                        Optional. Can be one of:

                        1. The numerical field value.
                        2. The duration spec for freshness: The value must be
                           formatted as an XSD ``dayTimeDuration`` value (a
                           restricted subset of an ISO 8601 duration value). The
                           pattern for this is: ``[nD][T[nH][nM][nS]]``.
                    boost_amount (float):
                        Optional. The value between -1 to 1 by which to boost the
                        score if the attribute_value evaluates to the value
                        specified above.
                """

                attribute_value: str = proto.Field(
                    proto.STRING,
                    number=1,
                )
                boost_amount: float = proto.Field(
                    proto.FLOAT,
                    number=2,
                )

            field_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            attribute_type: "Control.BoostAction.InterpolationBoostSpec.AttributeType" = proto.Field(
                proto.ENUM,
                number=2,
                enum="Control.BoostAction.InterpolationBoostSpec.AttributeType",
            )
            interpolation_type: "Control.BoostAction.InterpolationBoostSpec.InterpolationType" = proto.Field(
                proto.ENUM,
                number=3,
                enum="Control.BoostAction.InterpolationBoostSpec.InterpolationType",
            )
            control_points: MutableSequence[
                "Control.BoostAction.InterpolationBoostSpec.ControlPoint"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=4,
                message="Control.BoostAction.InterpolationBoostSpec.ControlPoint",
            )

        fixed_boost: float = proto.Field(
            proto.FLOAT,
            number=4,
            oneof="boost_spec",
        )
        interpolation_boost_spec: "Control.BoostAction.InterpolationBoostSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=5,
                oneof="boost_spec",
                message="Control.BoostAction.InterpolationBoostSpec",
            )
        )
        boost: float = proto.Field(
            proto.FLOAT,
            number=1,
        )
        filter: str = proto.Field(
            proto.STRING,
            number=2,
        )
        data_store: str = proto.Field(
            proto.STRING,
            number=3,
        )

    class FilterAction(proto.Message):
        r"""Specified which products may be included in results.
        Uses same filter as boost.

        Attributes:
            filter (str):
                Required. A filter to apply on the matching
                condition results.
                Required
                Syntax documentation:

                https://cloud.google.com/retail/docs/filter-and-order
                Maximum length is 5000 characters. Otherwise an
                INVALID ARGUMENT error is thrown.
            data_store (str):
                Required. Specifies which data store's documents can be
                filtered by this control. Full data store name e.g.
                projects/123/locations/global/collections/default_collection/dataStores/default_data_store
        """

        filter: str = proto.Field(
            proto.STRING,
            number=1,
        )
        data_store: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class RedirectAction(proto.Message):
        r"""Redirects a shopper to the provided URI.

        Attributes:
            redirect_uri (str):
                Required. The URI to which the shopper will
                be redirected.
                Required.
                URI must have length equal or less than 2000
                characters. Otherwise an INVALID ARGUMENT error
                is thrown.
        """

        redirect_uri: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class SynonymsAction(proto.Message):
        r"""Creates a set of terms that will act as synonyms of one
        another.
        Example: "happy" will also be considered as "glad", "glad" will
        also be considered as "happy".

        Attributes:
            synonyms (MutableSequence[str]):
                Defines a set of synonyms.
                Can specify up to 100 synonyms.
                Must specify at least 2 synonyms. Otherwise an
                INVALID ARGUMENT error is thrown.
        """

        synonyms: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    class PromoteAction(proto.Message):
        r"""Promote certain links based on some trigger queries.

        Example: Promote shoe store link when searching for ``shoe``
        keyword. The link can be outside of associated data store.

        Attributes:
            data_store (str):
                Required. Data store with which this
                promotion is attached to.
            search_link_promotion (google.cloud.discoveryengine_v1.types.SearchLinkPromotion):
                Required. Promotion attached to this action.
        """

        data_store: str = proto.Field(
            proto.STRING,
            number=1,
        )
        search_link_promotion: common.SearchLinkPromotion = proto.Field(
            proto.MESSAGE,
            number=2,
            message=common.SearchLinkPromotion,
        )

    boost_action: BoostAction = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="action",
        message=BoostAction,
    )
    filter_action: FilterAction = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="action",
        message=FilterAction,
    )
    redirect_action: RedirectAction = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="action",
        message=RedirectAction,
    )
    synonyms_action: SynonymsAction = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="action",
        message=SynonymsAction,
    )
    promote_action: PromoteAction = proto.Field(
        proto.MESSAGE,
        number=15,
        oneof="action",
        message=PromoteAction,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    associated_serving_config_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    solution_type: common.SolutionType = proto.Field(
        proto.ENUM,
        number=4,
        enum=common.SolutionType,
    )
    use_cases: MutableSequence[common.SearchUseCase] = proto.RepeatedField(
        proto.ENUM,
        number=8,
        enum=common.SearchUseCase,
    )
    conditions: MutableSequence["Condition"] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message="Condition",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
