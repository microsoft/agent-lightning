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

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from google.type import timeofday_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.monitoring_v3.types import mutation_record as gm_mutation_record
from google.cloud.monitoring_v3.types import common

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "AlertPolicy",
    },
)


class AlertPolicy(proto.Message):
    r"""A description of the conditions under which some aspect of your
    system is considered to be "unhealthy" and the ways to notify people
    or services about this state. For an overview of alerting policies,
    see `Introduction to
    Alerting <https://cloud.google.com/monitoring/alerts/>`__.

    Attributes:
        name (str):
            Identifier. Required if the policy exists. The resource name
            for this policy. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/alertPolicies/[ALERT_POLICY_ID]

            ``[ALERT_POLICY_ID]`` is assigned by Cloud Monitoring when
            the policy is created. When calling the
            [alertPolicies.create][google.monitoring.v3.AlertPolicyService.CreateAlertPolicy]
            method, do not include the ``name`` field in the alerting
            policy passed as part of the request.
        display_name (str):
            A short name or phrase used to identify the policy in
            dashboards, notifications, and incidents. To avoid
            confusion, don't use the same display name for multiple
            policies in the same project. The name is limited to 512
            Unicode characters.

            The convention for the display_name of a
            PrometheusQueryLanguageCondition is "{rule group
            name}/{alert name}", where the {rule group name} and {alert
            name} should be taken from the corresponding Prometheus
            configuration file. This convention is not enforced. In any
            case the display_name is not a unique key of the
            AlertPolicy.
        documentation (google.cloud.monitoring_v3.types.AlertPolicy.Documentation):
            Documentation that is included with
            notifications and incidents related to this
            policy. Best practice is for the documentation
            to include information to help responders
            understand, mitigate, escalate, and correct the
            underlying problems detected by the alerting
            policy. Notification channels that have limited
            capacity might not show this documentation.
        user_labels (MutableMapping[str, str]):
            User-supplied key/value data to be used for organizing and
            identifying the ``AlertPolicy`` objects.

            The field can contain up to 64 entries. Each key and value
            is limited to 63 Unicode characters or 128 bytes, whichever
            is smaller. Labels and values can contain only lowercase
            letters, numerals, underscores, and dashes. Keys must begin
            with a letter.

            Note that Prometheus {alert name} is a `valid Prometheus
            label
            names <https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels>`__,
            whereas Prometheus {rule group} is an unrestricted UTF-8
            string. This means that they cannot be stored as-is in user
            labels, because they may contain characters that are not
            allowed in user-label values.
        conditions (MutableSequence[google.cloud.monitoring_v3.types.AlertPolicy.Condition]):
            A list of conditions for the policy. The conditions are
            combined by AND or OR according to the ``combiner`` field.
            If the combined conditions evaluate to true, then an
            incident is created. A policy can have from one to six
            conditions. If ``condition_time_series_query_language`` is
            present, it must be the only ``condition``. If
            ``condition_monitoring_query_language`` is present, it must
            be the only ``condition``.
        combiner (google.cloud.monitoring_v3.types.AlertPolicy.ConditionCombinerType):
            How to combine the results of multiple conditions to
            determine if an incident should be opened. If
            ``condition_time_series_query_language`` is present, this
            must be ``COMBINE_UNSPECIFIED``.
        enabled (google.protobuf.wrappers_pb2.BoolValue):
            Whether or not the policy is enabled. On
            write, the default interpretation if unset is
            that the policy is enabled. On read, clients
            should not make any assumption about the state
            if it has not been populated. The field should
            always be populated on List and Get operations,
            unless a field projection has been specified
            that strips it out.
        validity (google.rpc.status_pb2.Status):
            Read-only description of how the alerting
            policy is invalid. This field is only set when
            the alerting policy is invalid. An invalid
            alerting policy will not generate incidents.
        notification_channels (MutableSequence[str]):
            Identifies the notification channels to which notifications
            should be sent when incidents are opened or closed or when
            new violations occur on an already opened incident. Each
            element of this array corresponds to the ``name`` field in
            each of the
            [``NotificationChannel``][google.monitoring.v3.NotificationChannel]
            objects that are returned from the
            [``ListNotificationChannels``]
            [google.monitoring.v3.NotificationChannelService.ListNotificationChannels]
            method. The format of the entries in this field is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]
        creation_record (google.cloud.monitoring_v3.types.MutationRecord):
            A read-only record of the creation of the
            alerting policy. If provided in a call to create
            or update, this field will be ignored.
        mutation_record (google.cloud.monitoring_v3.types.MutationRecord):
            A read-only record of the most recent change
            to the alerting policy. If provided in a call to
            create or update, this field will be ignored.
        alert_strategy (google.cloud.monitoring_v3.types.AlertPolicy.AlertStrategy):
            Control over how this alerting policy's
            notification channels are notified.
        severity (google.cloud.monitoring_v3.types.AlertPolicy.Severity):
            Optional. The severity of an alerting policy
            indicates how important incidents generated by
            that policy are. The severity level will be
            displayed on the Incident detail page and in
            notifications.
    """

    class ConditionCombinerType(proto.Enum):
        r"""Operators for combining conditions.

        Values:
            COMBINE_UNSPECIFIED (0):
                An unspecified combiner.
            AND (1):
                Combine conditions using the logical ``AND`` operator. An
                incident is created only if all the conditions are met
                simultaneously. This combiner is satisfied if all conditions
                are met, even if they are met on completely different
                resources.
            OR (2):
                Combine conditions using the logical ``OR`` operator. An
                incident is created if any of the listed conditions is met.
            AND_WITH_MATCHING_RESOURCE (3):
                Combine conditions using logical ``AND`` operator, but
                unlike the regular ``AND`` option, an incident is created
                only if all conditions are met simultaneously on at least
                one resource.
        """
        COMBINE_UNSPECIFIED = 0
        AND = 1
        OR = 2
        AND_WITH_MATCHING_RESOURCE = 3

    class Severity(proto.Enum):
        r"""An enumeration of possible severity level for an alerting
        policy.

        Values:
            SEVERITY_UNSPECIFIED (0):
                No severity is specified. This is the default
                value.
            CRITICAL (1):
                This is the highest severity level. Use this
                if the problem could cause significant damage or
                downtime.
            ERROR (2):
                This is the medium severity level. Use this
                if the problem could cause minor damage or
                downtime.
            WARNING (3):
                This is the lowest severity level. Use this
                if the problem is not causing any damage or
                downtime, but could potentially lead to a
                problem in the future.
        """
        SEVERITY_UNSPECIFIED = 0
        CRITICAL = 1
        ERROR = 2
        WARNING = 3

    class Documentation(proto.Message):
        r"""Documentation that is included in the notifications and
        incidents pertaining to this policy.

        Attributes:
            content (str):
                The body of the documentation, interpreted according to
                ``mime_type``. The content may not exceed 8,192 Unicode
                characters and may not exceed more than 10,240 bytes when
                encoded in UTF-8 format, whichever is smaller. This text can
                be `templatized by using
                variables <https://cloud.google.com/monitoring/alerts/doc-variables#doc-vars>`__.
            mime_type (str):
                The format of the ``content`` field. Presently, only the
                value ``"text/markdown"`` is supported. See
                `Markdown <https://en.wikipedia.org/wiki/Markdown>`__ for
                more information.
            subject (str):
                Optional. The subject line of the notification. The subject
                line may not exceed 10,240 bytes. In notifications generated
                by this policy, the contents of the subject line after
                variable expansion will be truncated to 255 bytes or shorter
                at the latest UTF-8 character boundary. The 255-byte limit
                is recommended by `this
                thread <https://stackoverflow.com/questions/1592291/what-is-the-email-subject-length-limit>`__.
                It is both the limit imposed by some third-party ticketing
                products and it is common to define textual fields in
                databases as VARCHAR(255).

                The contents of the subject line can be `templatized by
                using
                variables <https://cloud.google.com/monitoring/alerts/doc-variables#doc-vars>`__.
                If this field is missing or empty, a default subject line
                will be generated.
            links (MutableSequence[google.cloud.monitoring_v3.types.AlertPolicy.Documentation.Link]):
                Optional. Links to content such as playbooks,
                repositories, and other resources. This field
                can contain up to 3 entries.
        """

        class Link(proto.Message):
            r"""Links to content such as playbooks, repositories, and other
            resources.

            Attributes:
                display_name (str):
                    A short display name for the link. The
                    display name must not be empty or exceed 63
                    characters. Example: "playbook".
                url (str):
                    The url of a webpage. A url can be templatized by using
                    variables in the path or the query parameters. The total
                    length of a URL should not exceed 2083 characters before and
                    after variable expansion. Example:
                    "https://my_domain.com/playbook?name=${resource.name}".
            """

            display_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            url: str = proto.Field(
                proto.STRING,
                number=2,
            )

        content: str = proto.Field(
            proto.STRING,
            number=1,
        )
        mime_type: str = proto.Field(
            proto.STRING,
            number=2,
        )
        subject: str = proto.Field(
            proto.STRING,
            number=3,
        )
        links: MutableSequence["AlertPolicy.Documentation.Link"] = proto.RepeatedField(
            proto.MESSAGE,
            number=4,
            message="AlertPolicy.Documentation.Link",
        )

    class Condition(proto.Message):
        r"""A condition is a true/false test that determines when an
        alerting policy should open an incident. If a condition
        evaluates to true, it signifies that something is wrong.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            name (str):
                Required if the condition exists. The unique resource name
                for this condition. Its format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/alertPolicies/[POLICY_ID]/conditions/[CONDITION_ID]

                ``[CONDITION_ID]`` is assigned by Cloud Monitoring when the
                condition is created as part of a new or updated alerting
                policy.

                When calling the
                [alertPolicies.create][google.monitoring.v3.AlertPolicyService.CreateAlertPolicy]
                method, do not include the ``name`` field in the conditions
                of the requested alerting policy. Cloud Monitoring creates
                the condition identifiers and includes them in the new
                policy.

                When calling the
                [alertPolicies.update][google.monitoring.v3.AlertPolicyService.UpdateAlertPolicy]
                method to update a policy, including a condition ``name``
                causes the existing condition to be updated. Conditions
                without names are added to the updated policy. Existing
                conditions are deleted if they are not updated.

                Best practice is to preserve ``[CONDITION_ID]`` if you make
                only small changes, such as those to condition thresholds,
                durations, or trigger values. Otherwise, treat the change as
                a new condition and let the existing condition be deleted.
            display_name (str):
                A short name or phrase used to identify the
                condition in dashboards, notifications, and
                incidents. To avoid confusion, don't use the
                same display name for multiple conditions in the
                same policy.
            condition_threshold (google.cloud.monitoring_v3.types.AlertPolicy.Condition.MetricThreshold):
                A condition that compares a time series
                against a threshold.

                This field is a member of `oneof`_ ``condition``.
            condition_absent (google.cloud.monitoring_v3.types.AlertPolicy.Condition.MetricAbsence):
                A condition that checks that a time series
                continues to receive new data points.

                This field is a member of `oneof`_ ``condition``.
            condition_matched_log (google.cloud.monitoring_v3.types.AlertPolicy.Condition.LogMatch):
                A condition that checks for log messages
                matching given constraints. If set, no other
                conditions can be present.

                This field is a member of `oneof`_ ``condition``.
            condition_monitoring_query_language (google.cloud.monitoring_v3.types.AlertPolicy.Condition.MonitoringQueryLanguageCondition):
                A condition that uses the Monitoring Query
                Language to define alerts.

                This field is a member of `oneof`_ ``condition``.
            condition_prometheus_query_language (google.cloud.monitoring_v3.types.AlertPolicy.Condition.PrometheusQueryLanguageCondition):
                A condition that uses the Prometheus query
                language to define alerts.

                This field is a member of `oneof`_ ``condition``.
            condition_sql (google.cloud.monitoring_v3.types.AlertPolicy.Condition.SqlCondition):
                A condition that periodically evaluates a SQL
                query result.

                This field is a member of `oneof`_ ``condition``.
        """

        class EvaluationMissingData(proto.Enum):
            r"""A condition control that determines how metric-threshold
            conditions are evaluated when data stops arriving.
            This control doesn't affect metric-absence policies.

            Values:
                EVALUATION_MISSING_DATA_UNSPECIFIED (0):
                    An unspecified evaluation missing data option. Equivalent to
                    EVALUATION_MISSING_DATA_NO_OP.
                EVALUATION_MISSING_DATA_INACTIVE (1):
                    If there is no data to evaluate the
                    condition, then evaluate the condition as false.
                EVALUATION_MISSING_DATA_ACTIVE (2):
                    If there is no data to evaluate the
                    condition, then evaluate the condition as true.
                EVALUATION_MISSING_DATA_NO_OP (3):
                    Do not evaluate the condition to any value if
                    there is no data.
            """
            EVALUATION_MISSING_DATA_UNSPECIFIED = 0
            EVALUATION_MISSING_DATA_INACTIVE = 1
            EVALUATION_MISSING_DATA_ACTIVE = 2
            EVALUATION_MISSING_DATA_NO_OP = 3

        class Trigger(proto.Message):
            r"""Specifies how many time series must fail a predicate to trigger a
            condition. If not specified, then a ``{count: 1}`` trigger is used.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                count (int):
                    The absolute number of time series that must
                    fail the predicate for the condition to be
                    triggered.

                    This field is a member of `oneof`_ ``type``.
                percent (float):
                    The percentage of time series that must fail
                    the predicate for the condition to be triggered.

                    This field is a member of `oneof`_ ``type``.
            """

            count: int = proto.Field(
                proto.INT32,
                number=1,
                oneof="type",
            )
            percent: float = proto.Field(
                proto.DOUBLE,
                number=2,
                oneof="type",
            )

        class MetricThreshold(proto.Message):
            r"""A condition type that compares a collection of time series
            against a threshold.

            Attributes:
                filter (str):
                    Required. A
                    `filter <https://cloud.google.com/monitoring/api/v3/filters>`__
                    that identifies which time series should be compared with
                    the threshold.

                    The filter is similar to the one that is specified in the
                    ```ListTimeSeries``
                    request <https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/list>`__
                    (that call is useful to verify the time series that will be
                    retrieved / processed). The filter must specify the metric
                    type and the resource type. Optionally, it can specify
                    resource labels and metric labels. This field must not
                    exceed 2048 Unicode characters in length.
                aggregations (MutableSequence[google.cloud.monitoring_v3.types.Aggregation]):
                    Specifies the alignment of data points in individual time
                    series as well as how to combine the retrieved time series
                    together (such as when aggregating multiple streams on each
                    resource to a single stream for each resource or when
                    aggregating streams across all members of a group of
                    resources). Multiple aggregations are applied in the order
                    specified.

                    This field is similar to the one in the ```ListTimeSeries``
                    request <https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/list>`__.
                    It is advisable to use the ``ListTimeSeries`` method when
                    debugging this field.
                denominator_filter (str):
                    A
                    `filter <https://cloud.google.com/monitoring/api/v3/filters>`__
                    that identifies a time series that should be used as the
                    denominator of a ratio that will be compared with the
                    threshold. If a ``denominator_filter`` is specified, the
                    time series specified by the ``filter`` field will be used
                    as the numerator.

                    The filter must specify the metric type and optionally may
                    contain restrictions on resource type, resource labels, and
                    metric labels. This field may not exceed 2048 Unicode
                    characters in length.
                denominator_aggregations (MutableSequence[google.cloud.monitoring_v3.types.Aggregation]):
                    Specifies the alignment of data points in individual time
                    series selected by ``denominatorFilter`` as well as how to
                    combine the retrieved time series together (such as when
                    aggregating multiple streams on each resource to a single
                    stream for each resource or when aggregating streams across
                    all members of a group of resources).

                    When computing ratios, the ``aggregations`` and
                    ``denominator_aggregations`` fields must use the same
                    alignment period and produce time series that have the same
                    periodicity and labels.
                forecast_options (google.cloud.monitoring_v3.types.AlertPolicy.Condition.MetricThreshold.ForecastOptions):
                    When this field is present, the ``MetricThreshold``
                    condition forecasts whether the time series is predicted to
                    violate the threshold within the ``forecast_horizon``. When
                    this field is not set, the ``MetricThreshold`` tests the
                    current value of the timeseries against the threshold.
                comparison (google.cloud.monitoring_v3.types.ComparisonType):
                    The comparison to apply between the time series (indicated
                    by ``filter`` and ``aggregation``) and the threshold
                    (indicated by ``threshold_value``). The comparison is
                    applied on each time series, with the time series on the
                    left-hand side and the threshold on the right-hand side.

                    Only ``COMPARISON_LT`` and ``COMPARISON_GT`` are supported
                    currently.
                threshold_value (float):
                    A value against which to compare the time
                    series.
                duration (google.protobuf.duration_pb2.Duration):
                    The amount of time that a time series must violate the
                    threshold to be considered failing. Currently, only values
                    that are a multiple of a minute--e.g., 0, 60, 120, or 300
                    seconds--are supported. If an invalid value is given, an
                    error will be returned. When choosing a duration, it is
                    useful to keep in mind the frequency of the underlying time
                    series data (which may also be affected by any alignments
                    specified in the ``aggregations`` field); a good duration is
                    long enough so that a single outlier does not generate
                    spurious alerts, but short enough that unhealthy states are
                    detected and alerted on quickly.
                trigger (google.cloud.monitoring_v3.types.AlertPolicy.Condition.Trigger):
                    The number/percent of time series for which the comparison
                    must hold in order for the condition to trigger. If
                    unspecified, then the condition will trigger if the
                    comparison is true for any of the time series that have been
                    identified by ``filter`` and ``aggregations``, or by the
                    ratio, if ``denominator_filter`` and
                    ``denominator_aggregations`` are specified.
                evaluation_missing_data (google.cloud.monitoring_v3.types.AlertPolicy.Condition.EvaluationMissingData):
                    A condition control that determines how metric-threshold
                    conditions are evaluated when data stops arriving. To use
                    this control, the value of the ``duration`` field must be
                    greater than or equal to 60 seconds.
            """

            class ForecastOptions(proto.Message):
                r"""Options used when forecasting the time series and testing
                the predicted value against the threshold.

                Attributes:
                    forecast_horizon (google.protobuf.duration_pb2.Duration):
                        Required. The length of time into the future to forecast
                        whether a time series will violate the threshold. If the
                        predicted value is found to violate the threshold, and the
                        violation is observed in all forecasts made for the
                        configured ``duration``, then the time series is considered
                        to be failing. The forecast horizon can range from 1 hour to
                        60 hours.
                """

                forecast_horizon: duration_pb2.Duration = proto.Field(
                    proto.MESSAGE,
                    number=1,
                    message=duration_pb2.Duration,
                )

            filter: str = proto.Field(
                proto.STRING,
                number=2,
            )
            aggregations: MutableSequence[common.Aggregation] = proto.RepeatedField(
                proto.MESSAGE,
                number=8,
                message=common.Aggregation,
            )
            denominator_filter: str = proto.Field(
                proto.STRING,
                number=9,
            )
            denominator_aggregations: MutableSequence[
                common.Aggregation
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=10,
                message=common.Aggregation,
            )
            forecast_options: "AlertPolicy.Condition.MetricThreshold.ForecastOptions" = proto.Field(
                proto.MESSAGE,
                number=12,
                message="AlertPolicy.Condition.MetricThreshold.ForecastOptions",
            )
            comparison: common.ComparisonType = proto.Field(
                proto.ENUM,
                number=4,
                enum=common.ComparisonType,
            )
            threshold_value: float = proto.Field(
                proto.DOUBLE,
                number=5,
            )
            duration: duration_pb2.Duration = proto.Field(
                proto.MESSAGE,
                number=6,
                message=duration_pb2.Duration,
            )
            trigger: "AlertPolicy.Condition.Trigger" = proto.Field(
                proto.MESSAGE,
                number=7,
                message="AlertPolicy.Condition.Trigger",
            )
            evaluation_missing_data: "AlertPolicy.Condition.EvaluationMissingData" = (
                proto.Field(
                    proto.ENUM,
                    number=11,
                    enum="AlertPolicy.Condition.EvaluationMissingData",
                )
            )

        class MetricAbsence(proto.Message):
            r"""A condition type that checks that monitored resources are reporting
            data. The configuration defines a metric and a set of monitored
            resources. The predicate is considered in violation when a time
            series for the specified metric of a monitored resource does not
            include any data in the specified ``duration``.

            Attributes:
                filter (str):
                    Required. A
                    `filter <https://cloud.google.com/monitoring/api/v3/filters>`__
                    that identifies which time series should be compared with
                    the threshold.

                    The filter is similar to the one that is specified in the
                    ```ListTimeSeries``
                    request <https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/list>`__
                    (that call is useful to verify the time series that will be
                    retrieved / processed). The filter must specify the metric
                    type and the resource type. Optionally, it can specify
                    resource labels and metric labels. This field must not
                    exceed 2048 Unicode characters in length.
                aggregations (MutableSequence[google.cloud.monitoring_v3.types.Aggregation]):
                    Specifies the alignment of data points in individual time
                    series as well as how to combine the retrieved time series
                    together (such as when aggregating multiple streams on each
                    resource to a single stream for each resource or when
                    aggregating streams across all members of a group of
                    resources). Multiple aggregations are applied in the order
                    specified.

                    This field is similar to the one in the ```ListTimeSeries``
                    request <https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/list>`__.
                    It is advisable to use the ``ListTimeSeries`` method when
                    debugging this field.
                duration (google.protobuf.duration_pb2.Duration):
                    The amount of time that a time series must fail to report
                    new data to be considered failing. The minimum value of this
                    field is 120 seconds. Larger values that are a multiple of a
                    minute--for example, 240 or 300 seconds--are supported. If
                    an invalid value is given, an error will be returned. The
                    ``Duration.nanos`` field is ignored.
                trigger (google.cloud.monitoring_v3.types.AlertPolicy.Condition.Trigger):
                    The number/percent of time series for which the comparison
                    must hold in order for the condition to trigger. If
                    unspecified, then the condition will trigger if the
                    comparison is true for any of the time series that have been
                    identified by ``filter`` and ``aggregations``.
            """

            filter: str = proto.Field(
                proto.STRING,
                number=1,
            )
            aggregations: MutableSequence[common.Aggregation] = proto.RepeatedField(
                proto.MESSAGE,
                number=5,
                message=common.Aggregation,
            )
            duration: duration_pb2.Duration = proto.Field(
                proto.MESSAGE,
                number=2,
                message=duration_pb2.Duration,
            )
            trigger: "AlertPolicy.Condition.Trigger" = proto.Field(
                proto.MESSAGE,
                number=3,
                message="AlertPolicy.Condition.Trigger",
            )

        class LogMatch(proto.Message):
            r"""A condition type that checks whether a log message in the `scoping
            project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            satisfies the given filter. Logs from other projects in the metrics
            scope are not evaluated.

            Attributes:
                filter (str):
                    Required. A logs-based filter. See `Advanced Logs
                    Queries <https://cloud.google.com/logging/docs/view/advanced-queries>`__
                    for how this filter should be constructed.
                label_extractors (MutableMapping[str, str]):
                    Optional. A map from a label key to an extractor expression,
                    which is used to extract the value for this label key. Each
                    entry in this map is a specification for how data should be
                    extracted from log entries that match ``filter``. Each
                    combination of extracted values is treated as a separate
                    rule for the purposes of triggering notifications. Label
                    keys and corresponding values can be used in notifications
                    generated by this condition.

                    Please see `the documentation on logs-based metric
                    ``valueExtractor``\ s <https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.metrics#LogMetric.FIELDS.value_extractor>`__
                    for syntax and examples.
            """

            filter: str = proto.Field(
                proto.STRING,
                number=1,
            )
            label_extractors: MutableMapping[str, str] = proto.MapField(
                proto.STRING,
                proto.STRING,
                number=2,
            )

        class MonitoringQueryLanguageCondition(proto.Message):
            r"""A condition type that allows alerting policies to be defined using
            `Monitoring Query
            Language <https://cloud.google.com/monitoring/mql>`__.

            Attributes:
                query (str):
                    `Monitoring Query
                    Language <https://cloud.google.com/monitoring/mql>`__ query
                    that outputs a boolean stream.
                duration (google.protobuf.duration_pb2.Duration):
                    The amount of time that a time series must violate the
                    threshold to be considered failing. Currently, only values
                    that are a multiple of a minute--e.g., 0, 60, 120, or 300
                    seconds--are supported. If an invalid value is given, an
                    error will be returned. When choosing a duration, it is
                    useful to keep in mind the frequency of the underlying time
                    series data (which may also be affected by any alignments
                    specified in the ``aggregations`` field); a good duration is
                    long enough so that a single outlier does not generate
                    spurious alerts, but short enough that unhealthy states are
                    detected and alerted on quickly.
                trigger (google.cloud.monitoring_v3.types.AlertPolicy.Condition.Trigger):
                    The number/percent of time series for which the comparison
                    must hold in order for the condition to trigger. If
                    unspecified, then the condition will trigger if the
                    comparison is true for any of the time series that have been
                    identified by ``filter`` and ``aggregations``, or by the
                    ratio, if ``denominator_filter`` and
                    ``denominator_aggregations`` are specified.
                evaluation_missing_data (google.cloud.monitoring_v3.types.AlertPolicy.Condition.EvaluationMissingData):
                    A condition control that determines how
                    metric-threshold conditions are evaluated when
                    data stops arriving.
            """

            query: str = proto.Field(
                proto.STRING,
                number=1,
            )
            duration: duration_pb2.Duration = proto.Field(
                proto.MESSAGE,
                number=2,
                message=duration_pb2.Duration,
            )
            trigger: "AlertPolicy.Condition.Trigger" = proto.Field(
                proto.MESSAGE,
                number=3,
                message="AlertPolicy.Condition.Trigger",
            )
            evaluation_missing_data: "AlertPolicy.Condition.EvaluationMissingData" = (
                proto.Field(
                    proto.ENUM,
                    number=4,
                    enum="AlertPolicy.Condition.EvaluationMissingData",
                )
            )

        class PrometheusQueryLanguageCondition(proto.Message):
            r"""A condition type that allows alerting policies to be defined using
            `Prometheus Query Language
            (PromQL) <https://prometheus.io/docs/prometheus/latest/querying/basics/>`__.

            The PrometheusQueryLanguageCondition message contains information
            from a Prometheus alerting rule and its associated rule group.

            A Prometheus alerting rule is described
            `here <https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>`__.
            The semantics of a Prometheus alerting rule is described
            `here <https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/#rule>`__.

            A Prometheus rule group is described
            `here <https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/>`__.
            The semantics of a Prometheus rule group is described
            `here <https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/#rule_group>`__.

            Because Cloud Alerting has no representation of a Prometheus rule
            group resource, we must embed the information of the parent rule
            group inside each of the conditions that refer to it. We must also
            update the contents of all Prometheus alerts in case the information
            of their rule group changes.

            The PrometheusQueryLanguageCondition protocol buffer combines the
            information of the corresponding rule group and alerting rule. The
            structure of the PrometheusQueryLanguageCondition protocol buffer
            does NOT mimic the structure of the Prometheus rule group and
            alerting rule YAML declarations. The
            PrometheusQueryLanguageCondition protocol buffer may change in the
            future to support future rule group and/or alerting rule features.
            There are no new such features at the present time (2023-06-26).

            Attributes:
                query (str):
                    Required. The PromQL expression to evaluate.
                    Every evaluation cycle this expression is
                    evaluated at the current time, and all resultant
                    time series become pending/firing alerts. This
                    field must not be empty.
                duration (google.protobuf.duration_pb2.Duration):
                    Optional. Alerts are considered firing once
                    their PromQL expression was evaluated to be
                    "true" for this long. Alerts whose PromQL
                    expression was not evaluated to be "true" for
                    long enough are considered pending.
                    Must be a non-negative duration or missing.
                    This field is optional. Its default value is
                    zero.
                evaluation_interval (google.protobuf.duration_pb2.Duration):
                    Optional. How often this rule should be
                    evaluated. Must be a positive multiple of 30
                    seconds or missing. This field is optional. Its
                    default value is 30 seconds. If this
                    PrometheusQueryLanguageCondition was generated
                    from a Prometheus alerting rule, then this value
                    should be taken from the enclosing rule group.
                labels (MutableMapping[str, str]):
                    Optional. Labels to add to or overwrite in the PromQL query
                    result. Label names `must be
                    valid <https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels>`__.
                    Label values can be `templatized by using
                    variables <https://cloud.google.com/monitoring/alerts/doc-variables#doc-vars>`__.
                    The only available variable names are the names of the
                    labels in the PromQL result, including "**name**" and
                    "value". "labels" may be empty.
                rule_group (str):
                    Optional. The rule group name of this alert
                    in the corresponding Prometheus configuration
                    file.

                    Some external tools may require this field to be
                    populated correctly in order to refer to the
                    original Prometheus configuration file. The rule
                    group name and the alert name are necessary to
                    update the relevant AlertPolicies in case the
                    definition of the rule group changes in the
                    future.

                    This field is optional. If this field is not
                    empty, then it must contain a valid UTF-8
                    string.
                    This field may not exceed 2048 Unicode
                    characters in length.
                alert_rule (str):
                    Optional. The alerting rule name of this alert in the
                    corresponding Prometheus configuration file.

                    Some external tools may require this field to be populated
                    correctly in order to refer to the original Prometheus
                    configuration file. The rule group name and the alert name
                    are necessary to update the relevant AlertPolicies in case
                    the definition of the rule group changes in the future.

                    This field is optional. If this field is not empty, then it
                    must be a `valid Prometheus label
                    name <https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels>`__.
                    This field may not exceed 2048 Unicode characters in length.
                disable_metric_validation (bool):
                    Optional. Whether to disable metric existence validation for
                    this condition.

                    This allows alerting policies to be defined on metrics that
                    do not yet exist, improving advanced customer workflows such
                    as configuring alerting policies using Terraform.

                    Users with the ``monitoring.alertPolicyViewer`` role are
                    able to see the name of the non-existent metric in the
                    alerting policy condition.
            """

            query: str = proto.Field(
                proto.STRING,
                number=1,
            )
            duration: duration_pb2.Duration = proto.Field(
                proto.MESSAGE,
                number=2,
                message=duration_pb2.Duration,
            )
            evaluation_interval: duration_pb2.Duration = proto.Field(
                proto.MESSAGE,
                number=3,
                message=duration_pb2.Duration,
            )
            labels: MutableMapping[str, str] = proto.MapField(
                proto.STRING,
                proto.STRING,
                number=4,
            )
            rule_group: str = proto.Field(
                proto.STRING,
                number=5,
            )
            alert_rule: str = proto.Field(
                proto.STRING,
                number=6,
            )
            disable_metric_validation: bool = proto.Field(
                proto.BOOL,
                number=7,
            )

        class SqlCondition(proto.Message):
            r"""A condition that allows alerting policies to be defined using
            GoogleSQL. SQL conditions examine a sliding window of logs using
            GoogleSQL. Alert policies with SQL conditions may incur
            additional billing.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                query (str):
                    Required. The Log Analytics SQL query to run, as a string.
                    The query must conform to the required shape. Specifically,
                    the query must not try to filter the input by time. A filter
                    will automatically be applied to filter the input so that
                    the query receives all rows received since the last time the
                    query was run.

                    For example, the following query extracts all log entries
                    containing an HTTP request:

                    ::

                        SELECT
                          timestamp, log_name, severity, http_request, resource, labels
                        FROM
                          my-project.global._Default._AllLogs
                        WHERE
                          http_request IS NOT NULL
                minutes (google.cloud.monitoring_v3.types.AlertPolicy.Condition.SqlCondition.Minutes):
                    Schedule the query to execute every so many
                    minutes.

                    This field is a member of `oneof`_ ``schedule``.
                hourly (google.cloud.monitoring_v3.types.AlertPolicy.Condition.SqlCondition.Hourly):
                    Schedule the query to execute every so many
                    hours.

                    This field is a member of `oneof`_ ``schedule``.
                daily (google.cloud.monitoring_v3.types.AlertPolicy.Condition.SqlCondition.Daily):
                    Schedule the query to execute every so many
                    days.

                    This field is a member of `oneof`_ ``schedule``.
                row_count_test (google.cloud.monitoring_v3.types.AlertPolicy.Condition.SqlCondition.RowCountTest):
                    Test the row count against a threshold.

                    This field is a member of `oneof`_ ``evaluate``.
                boolean_test (google.cloud.monitoring_v3.types.AlertPolicy.Condition.SqlCondition.BooleanTest):
                    Test the boolean value in the indicated
                    column.

                    This field is a member of `oneof`_ ``evaluate``.
            """

            class Minutes(proto.Message):
                r"""Used to schedule the query to run every so many minutes.

                Attributes:
                    periodicity (int):
                        Required. Number of minutes between runs. The
                        interval must be greater than or equal to 5
                        minutes and less than or equal to 1440 minutes.
                """

                periodicity: int = proto.Field(
                    proto.INT32,
                    number=1,
                )

            class Hourly(proto.Message):
                r"""Used to schedule the query to run every so many hours.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    periodicity (int):
                        Required. The number of hours between runs.
                        Must be greater than or equal to 1 hour and less
                        than or equal to 48 hours.
                    minute_offset (int):
                        Optional. The number of minutes after the
                        hour (in UTC) to run the query. Must be greater
                        than or equal to 0 minutes and less than or
                        equal to 59 minutes.  If left unspecified, then
                        an arbitrary offset is used.

                        This field is a member of `oneof`_ ``_minute_offset``.
                """

                periodicity: int = proto.Field(
                    proto.INT32,
                    number=1,
                )
                minute_offset: int = proto.Field(
                    proto.INT32,
                    number=2,
                    optional=True,
                )

            class Daily(proto.Message):
                r"""Used to schedule the query to run every so many days.

                Attributes:
                    periodicity (int):
                        Required. The number of days between runs.
                        Must be greater than or equal to 1 day and less
                        than or equal to 31 days.
                    execution_time (google.type.timeofday_pb2.TimeOfDay):
                        Optional. The time of day (in UTC) at which
                        the query should run. If left unspecified, the
                        server picks an arbitrary time of day and runs
                        the query at the same time each day.
                """

                periodicity: int = proto.Field(
                    proto.INT32,
                    number=1,
                )
                execution_time: timeofday_pb2.TimeOfDay = proto.Field(
                    proto.MESSAGE,
                    number=2,
                    message=timeofday_pb2.TimeOfDay,
                )

            class RowCountTest(proto.Message):
                r"""A test that checks if the number of rows in the result set
                violates some threshold.

                Attributes:
                    comparison (google.cloud.monitoring_v3.types.ComparisonType):
                        Required. The comparison to apply between the
                        number of rows returned by the query and the
                        threshold.
                    threshold (int):
                        Required. The value against which to compare
                        the row count.
                """

                comparison: common.ComparisonType = proto.Field(
                    proto.ENUM,
                    number=1,
                    enum=common.ComparisonType,
                )
                threshold: int = proto.Field(
                    proto.INT64,
                    number=2,
                )

            class BooleanTest(proto.Message):
                r"""A test that uses an alerting result in a boolean column
                produced by the SQL query.

                Attributes:
                    column (str):
                        Required. The name of the column containing
                        the boolean value. If the value in a row is
                        NULL, that row is ignored.
                """

                column: str = proto.Field(
                    proto.STRING,
                    number=1,
                )

            query: str = proto.Field(
                proto.STRING,
                number=1,
            )
            minutes: "AlertPolicy.Condition.SqlCondition.Minutes" = proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="schedule",
                message="AlertPolicy.Condition.SqlCondition.Minutes",
            )
            hourly: "AlertPolicy.Condition.SqlCondition.Hourly" = proto.Field(
                proto.MESSAGE,
                number=3,
                oneof="schedule",
                message="AlertPolicy.Condition.SqlCondition.Hourly",
            )
            daily: "AlertPolicy.Condition.SqlCondition.Daily" = proto.Field(
                proto.MESSAGE,
                number=4,
                oneof="schedule",
                message="AlertPolicy.Condition.SqlCondition.Daily",
            )
            row_count_test: "AlertPolicy.Condition.SqlCondition.RowCountTest" = (
                proto.Field(
                    proto.MESSAGE,
                    number=5,
                    oneof="evaluate",
                    message="AlertPolicy.Condition.SqlCondition.RowCountTest",
                )
            )
            boolean_test: "AlertPolicy.Condition.SqlCondition.BooleanTest" = (
                proto.Field(
                    proto.MESSAGE,
                    number=6,
                    oneof="evaluate",
                    message="AlertPolicy.Condition.SqlCondition.BooleanTest",
                )
            )

        name: str = proto.Field(
            proto.STRING,
            number=12,
        )
        display_name: str = proto.Field(
            proto.STRING,
            number=6,
        )
        condition_threshold: "AlertPolicy.Condition.MetricThreshold" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="condition",
            message="AlertPolicy.Condition.MetricThreshold",
        )
        condition_absent: "AlertPolicy.Condition.MetricAbsence" = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="condition",
            message="AlertPolicy.Condition.MetricAbsence",
        )
        condition_matched_log: "AlertPolicy.Condition.LogMatch" = proto.Field(
            proto.MESSAGE,
            number=20,
            oneof="condition",
            message="AlertPolicy.Condition.LogMatch",
        )
        condition_monitoring_query_language: "AlertPolicy.Condition.MonitoringQueryLanguageCondition" = proto.Field(
            proto.MESSAGE,
            number=19,
            oneof="condition",
            message="AlertPolicy.Condition.MonitoringQueryLanguageCondition",
        )
        condition_prometheus_query_language: "AlertPolicy.Condition.PrometheusQueryLanguageCondition" = proto.Field(
            proto.MESSAGE,
            number=21,
            oneof="condition",
            message="AlertPolicy.Condition.PrometheusQueryLanguageCondition",
        )
        condition_sql: "AlertPolicy.Condition.SqlCondition" = proto.Field(
            proto.MESSAGE,
            number=22,
            oneof="condition",
            message="AlertPolicy.Condition.SqlCondition",
        )

    class AlertStrategy(proto.Message):
        r"""Control over how the notification channels in
        ``notification_channels`` are notified when this alert fires.

        Attributes:
            notification_rate_limit (google.cloud.monitoring_v3.types.AlertPolicy.AlertStrategy.NotificationRateLimit):
                Required for log-based alerting policies, i.e. policies with
                a ``LogMatch`` condition.

                This limit is not implemented for alerting policies that do
                not have a LogMatch condition.
            notification_prompts (MutableSequence[google.cloud.monitoring_v3.types.AlertPolicy.AlertStrategy.NotificationPrompt]):
                For log-based alert policies, the notification prompts is
                always [OPENED]. For non log-based alert policies, the
                notification prompts can be [OPENED] or [OPENED, CLOSED].
            auto_close (google.protobuf.duration_pb2.Duration):
                If an alerting policy that was active has no
                data for this long, any open incidents will
                close
            notification_channel_strategy (MutableSequence[google.cloud.monitoring_v3.types.AlertPolicy.AlertStrategy.NotificationChannelStrategy]):
                Control how notifications will be sent out,
                on a per-channel basis.
        """

        class NotificationPrompt(proto.Enum):
            r"""Control when notifications will be sent out.

            Values:
                NOTIFICATION_PROMPT_UNSPECIFIED (0):
                    No strategy specified. Treated as error.
                OPENED (1):
                    Notify when an incident is opened.
                CLOSED (3):
                    Notify when an incident is closed.
            """
            NOTIFICATION_PROMPT_UNSPECIFIED = 0
            OPENED = 1
            CLOSED = 3

        class NotificationRateLimit(proto.Message):
            r"""Control over the rate of notifications sent to this alerting
            policy's notification channels.

            Attributes:
                period (google.protobuf.duration_pb2.Duration):
                    Not more than one notification per ``period``.
            """

            period: duration_pb2.Duration = proto.Field(
                proto.MESSAGE,
                number=1,
                message=duration_pb2.Duration,
            )

        class NotificationChannelStrategy(proto.Message):
            r"""Control over how the notification channels in
            ``notification_channels`` are notified when this alert fires, on a
            per-channel basis.

            Attributes:
                notification_channel_names (MutableSequence[str]):
                    The full REST resource name for the notification channels
                    that these settings apply to. Each of these correspond to
                    the name field in one of the NotificationChannel objects
                    referenced in the notification_channels field of this
                    AlertPolicy. The format is:

                    ::

                        projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]
                renotify_interval (google.protobuf.duration_pb2.Duration):
                    The frequency at which to send reminder
                    notifications for open incidents.
            """

            notification_channel_names: MutableSequence[str] = proto.RepeatedField(
                proto.STRING,
                number=1,
            )
            renotify_interval: duration_pb2.Duration = proto.Field(
                proto.MESSAGE,
                number=2,
                message=duration_pb2.Duration,
            )

        notification_rate_limit: "AlertPolicy.AlertStrategy.NotificationRateLimit" = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                message="AlertPolicy.AlertStrategy.NotificationRateLimit",
            )
        )
        notification_prompts: MutableSequence[
            "AlertPolicy.AlertStrategy.NotificationPrompt"
        ] = proto.RepeatedField(
            proto.ENUM,
            number=2,
            enum="AlertPolicy.AlertStrategy.NotificationPrompt",
        )
        auto_close: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=3,
            message=duration_pb2.Duration,
        )
        notification_channel_strategy: MutableSequence[
            "AlertPolicy.AlertStrategy.NotificationChannelStrategy"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=4,
            message="AlertPolicy.AlertStrategy.NotificationChannelStrategy",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    documentation: Documentation = proto.Field(
        proto.MESSAGE,
        number=13,
        message=Documentation,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=16,
    )
    conditions: MutableSequence[Condition] = proto.RepeatedField(
        proto.MESSAGE,
        number=12,
        message=Condition,
    )
    combiner: ConditionCombinerType = proto.Field(
        proto.ENUM,
        number=6,
        enum=ConditionCombinerType,
    )
    enabled: wrappers_pb2.BoolValue = proto.Field(
        proto.MESSAGE,
        number=17,
        message=wrappers_pb2.BoolValue,
    )
    validity: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=18,
        message=status_pb2.Status,
    )
    notification_channels: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=14,
    )
    creation_record: gm_mutation_record.MutationRecord = proto.Field(
        proto.MESSAGE,
        number=10,
        message=gm_mutation_record.MutationRecord,
    )
    mutation_record: gm_mutation_record.MutationRecord = proto.Field(
        proto.MESSAGE,
        number=11,
        message=gm_mutation_record.MutationRecord,
    )
    alert_strategy: AlertStrategy = proto.Field(
        proto.MESSAGE,
        number=21,
        message=AlertStrategy,
    )
    severity: Severity = proto.Field(
        proto.ENUM,
        number=22,
        enum=Severity,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
