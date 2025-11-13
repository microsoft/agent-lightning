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

from google.api import distribution_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "ComparisonType",
        "ServiceTier",
        "TypedValue",
        "TimeInterval",
        "Aggregation",
    },
)


class ComparisonType(proto.Enum):
    r"""Specifies an ordering relationship on two arguments, called ``left``
    and ``right``.

    Values:
        COMPARISON_UNSPECIFIED (0):
            No ordering relationship is specified.
        COMPARISON_GT (1):
            True if the left argument is greater than the
            right argument.
        COMPARISON_GE (2):
            True if the left argument is greater than or
            equal to the right argument.
        COMPARISON_LT (3):
            True if the left argument is less than the
            right argument.
        COMPARISON_LE (4):
            True if the left argument is less than or
            equal to the right argument.
        COMPARISON_EQ (5):
            True if the left argument is equal to the
            right argument.
        COMPARISON_NE (6):
            True if the left argument is not equal to the
            right argument.
    """
    COMPARISON_UNSPECIFIED = 0
    COMPARISON_GT = 1
    COMPARISON_GE = 2
    COMPARISON_LT = 3
    COMPARISON_LE = 4
    COMPARISON_EQ = 5
    COMPARISON_NE = 6


class ServiceTier(proto.Enum):
    r"""The tier of service for a Metrics Scope. Please see the `service
    tiers
    documentation <https://cloud.google.com/monitoring/workspaces/tiers>`__
    for more details.

    Values:
        SERVICE_TIER_UNSPECIFIED (0):
            An invalid sentinel value, used to indicate
            that a tier has not been provided explicitly.
        SERVICE_TIER_BASIC (1):
            The Cloud Monitoring Basic tier, a free tier of service that
            provides basic features, a moderate allotment of logs, and
            access to built-in metrics. A number of features are not
            available in this tier. For more details, see `the service
            tiers
            documentation <https://cloud.google.com/monitoring/workspaces/tiers>`__.
        SERVICE_TIER_PREMIUM (2):
            The Cloud Monitoring Premium tier, a higher, more expensive
            tier of service that provides access to all Cloud Monitoring
            features, lets you use Cloud Monitoring with AWS accounts,
            and has a larger allotments for logs and metrics. For more
            details, see `the service tiers
            documentation <https://cloud.google.com/monitoring/workspaces/tiers>`__.
    """
    _pb_options = {"deprecated": True}
    SERVICE_TIER_UNSPECIFIED = 0
    SERVICE_TIER_BASIC = 1
    SERVICE_TIER_PREMIUM = 2


class TypedValue(proto.Message):
    r"""A single strongly-typed value.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        bool_value (bool):
            A Boolean value: ``true`` or ``false``.

            This field is a member of `oneof`_ ``value``.
        int64_value (int):
            A 64-bit integer. Its range is approximately
            &plusmn;9.2x10<sup>18</sup>.

            This field is a member of `oneof`_ ``value``.
        double_value (float):
            A 64-bit double-precision floating-point
            number. Its magnitude is approximately
            &plusmn;10<sup>&plusmn;300</sup> and it has 16
            significant digits of precision.

            This field is a member of `oneof`_ ``value``.
        string_value (str):
            A variable-length string value.

            This field is a member of `oneof`_ ``value``.
        distribution_value (google.api.distribution_pb2.Distribution):
            A distribution value.

            This field is a member of `oneof`_ ``value``.
    """

    bool_value: bool = proto.Field(
        proto.BOOL,
        number=1,
        oneof="value",
    )
    int64_value: int = proto.Field(
        proto.INT64,
        number=2,
        oneof="value",
    )
    double_value: float = proto.Field(
        proto.DOUBLE,
        number=3,
        oneof="value",
    )
    string_value: str = proto.Field(
        proto.STRING,
        number=4,
        oneof="value",
    )
    distribution_value: distribution_pb2.Distribution = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="value",
        message=distribution_pb2.Distribution,
    )


class TimeInterval(proto.Message):
    r"""Describes a time interval:

    - Reads: A half-open time interval. It includes the end time but
      excludes the start time: ``(startTime, endTime]``. The start time
      must be specified, must be earlier than the end time, and should
      be no older than the data retention period for the metric.
    - Writes: A closed time interval. It extends from the start time to
      the end time, and includes both: ``[startTime, endTime]``. Valid
      time intervals depend on the
      ```MetricKind`` <https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.metricDescriptors#MetricKind>`__
      of the metric value. The end time must not be earlier than the
      start time, and the end time must not be more than 25 hours in the
      past or more than five minutes in the future.

      - For ``GAUGE`` metrics, the ``startTime`` value is technically
        optional; if no value is specified, the start time defaults to
        the value of the end time, and the interval represents a single
        point in time. If both start and end times are specified, they
        must be identical. Such an interval is valid only for ``GAUGE``
        metrics, which are point-in-time measurements. The end time of a
        new interval must be at least a millisecond after the end time
        of the previous interval.
      - For ``DELTA`` metrics, the start time and end time must specify
        a non-zero interval, with subsequent points specifying
        contiguous and non-overlapping intervals. For ``DELTA`` metrics,
        the start time of the next interval must be at least a
        millisecond after the end time of the previous interval.
      - For ``CUMULATIVE`` metrics, the start time and end time must
        specify a non-zero interval, with subsequent points specifying
        the same start time and increasing end times, until an event
        resets the cumulative value to zero and sets a new start time
        for the following points. The new start time must be at least a
        millisecond after the end time of the previous interval.
      - The start time of a new interval must be at least a millisecond
        after the end time of the previous interval because intervals
        are closed. If the start time of a new interval is the same as
        the end time of the previous interval, then data written at the
        new start time could overwrite data written at the previous end
        time.

    Attributes:
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Required. The end of the time interval.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Optional. The beginning of the time interval.
            The default value for the start time is the end
            time. The start time must not be later than the
            end time.
    """

    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )


class Aggregation(proto.Message):
    r"""Describes how to combine multiple time series to provide a different
    view of the data. Aggregation of time series is done in two steps.
    First, each time series in the set is *aligned* to the same time
    interval boundaries, then the set of time series is optionally
    *reduced* in number.

    Alignment consists of applying the ``per_series_aligner`` operation
    to each time series after its data has been divided into regular
    ``alignment_period`` time intervals. This process takes *all* of the
    data points in an alignment period, applies a mathematical
    transformation such as averaging, minimum, maximum, delta, etc., and
    converts them into a single data point per period.

    Reduction is when the aligned and transformed time series can
    optionally be combined, reducing the number of time series through
    similar mathematical transformations. Reduction involves applying a
    ``cross_series_reducer`` to all the time series, optionally sorting
    the time series into subsets with ``group_by_fields``, and applying
    the reducer to each subset.

    The raw time series data can contain a huge amount of information
    from multiple sources. Alignment and reduction transforms this mass
    of data into a more manageable and representative collection of
    data, for example "the 95% latency across the average of all tasks
    in a cluster". This representative data can be more easily graphed
    and comprehended, and the individual time series data is still
    available for later drilldown. For more details, see `Filtering and
    aggregation <https://cloud.google.com/monitoring/api/v3/aggregation>`__.

    Attributes:
        alignment_period (google.protobuf.duration_pb2.Duration):
            The ``alignment_period`` specifies a time interval, in
            seconds, that is used to divide the data in all the [time
            series][google.monitoring.v3.TimeSeries] into consistent
            blocks of time. This will be done before the per-series
            aligner can be applied to the data.

            The value must be at least 60 seconds. If a per-series
            aligner other than ``ALIGN_NONE`` is specified, this field
            is required or an error is returned. If no per-series
            aligner is specified, or the aligner ``ALIGN_NONE`` is
            specified, then this field is ignored.

            The maximum value of the ``alignment_period`` is 104 weeks
            (2 years) for charts, and 90,000 seconds (25 hours) for
            alerting policies.
        per_series_aligner (google.cloud.monitoring_v3.types.Aggregation.Aligner):
            An ``Aligner`` describes how to bring the data points in a
            single time series into temporal alignment. Except for
            ``ALIGN_NONE``, all alignments cause all the data points in
            an ``alignment_period`` to be mathematically grouped
            together, resulting in a single data point for each
            ``alignment_period`` with end timestamp at the end of the
            period.

            Not all alignment operations may be applied to all time
            series. The valid choices depend on the ``metric_kind`` and
            ``value_type`` of the original time series. Alignment can
            change the ``metric_kind`` or the ``value_type`` of the time
            series.

            Time series data must be aligned in order to perform
            cross-time series reduction. If ``cross_series_reducer`` is
            specified, then ``per_series_aligner`` must be specified and
            not equal to ``ALIGN_NONE`` and ``alignment_period`` must be
            specified; otherwise, an error is returned.
        cross_series_reducer (google.cloud.monitoring_v3.types.Aggregation.Reducer):
            The reduction operation to be used to combine time series
            into a single time series, where the value of each data
            point in the resulting series is a function of all the
            already aligned values in the input time series.

            Not all reducer operations can be applied to all time
            series. The valid choices depend on the ``metric_kind`` and
            the ``value_type`` of the original time series. Reduction
            can yield a time series with a different ``metric_kind`` or
            ``value_type`` than the input time series.

            Time series data must first be aligned (see
            ``per_series_aligner``) in order to perform cross-time
            series reduction. If ``cross_series_reducer`` is specified,
            then ``per_series_aligner`` must be specified, and must not
            be ``ALIGN_NONE``. An ``alignment_period`` must also be
            specified; otherwise, an error is returned.
        group_by_fields (MutableSequence[str]):
            The set of fields to preserve when ``cross_series_reducer``
            is specified. The ``group_by_fields`` determine how the time
            series are partitioned into subsets prior to applying the
            aggregation operation. Each subset contains time series that
            have the same value for each of the grouping fields. Each
            individual time series is a member of exactly one subset.
            The ``cross_series_reducer`` is applied to each subset of
            time series. It is not possible to reduce across different
            resource types, so this field implicitly contains
            ``resource.type``. Fields not specified in
            ``group_by_fields`` are aggregated away. If
            ``group_by_fields`` is not specified and all the time series
            have the same resource type, then the time series are
            aggregated into a single output time series. If
            ``cross_series_reducer`` is not defined, this field is
            ignored.
    """

    class Aligner(proto.Enum):
        r"""The ``Aligner`` specifies the operation that will be applied to the
        data points in each alignment period in a time series. Except for
        ``ALIGN_NONE``, which specifies that no operation be applied, each
        alignment operation replaces the set of data values in each
        alignment period with a single value: the result of applying the
        operation to the data values. An aligned time series has a single
        data value at the end of each ``alignment_period``.

        An alignment operation can change the data type of the values, too.
        For example, if you apply a counting operation to boolean values,
        the data ``value_type`` in the original time series is ``BOOLEAN``,
        but the ``value_type`` in the aligned result is ``INT64``.

        Values:
            ALIGN_NONE (0):
                No alignment. Raw data is returned. Not valid if
                cross-series reduction is requested. The ``value_type`` of
                the result is the same as the ``value_type`` of the input.
            ALIGN_DELTA (1):
                Align and convert to
                [DELTA][google.api.MetricDescriptor.MetricKind.DELTA]. The
                output is ``delta = y1 - y0``.

                This alignment is valid for
                [CUMULATIVE][google.api.MetricDescriptor.MetricKind.CUMULATIVE]
                and ``DELTA`` metrics. If the selected alignment period
                results in periods with no data, then the aligned value for
                such a period is created by interpolation. The
                ``value_type`` of the aligned result is the same as the
                ``value_type`` of the input.
            ALIGN_RATE (2):
                Align and convert to a rate. The result is computed as
                ``rate = (y1 - y0)/(t1 - t0)``, or "delta over time". Think
                of this aligner as providing the slope of the line that
                passes through the value at the start and at the end of the
                ``alignment_period``.

                This aligner is valid for ``CUMULATIVE`` and ``DELTA``
                metrics with numeric values. If the selected alignment
                period results in periods with no data, then the aligned
                value for such a period is created by interpolation. The
                output is a ``GAUGE`` metric with ``value_type`` ``DOUBLE``.

                If, by "rate", you mean "percentage change", see the
                ``ALIGN_PERCENT_CHANGE`` aligner instead.
            ALIGN_INTERPOLATE (3):
                Align by interpolating between adjacent points around the
                alignment period boundary. This aligner is valid for
                ``GAUGE`` metrics with numeric values. The ``value_type`` of
                the aligned result is the same as the ``value_type`` of the
                input.
            ALIGN_NEXT_OLDER (4):
                Align by moving the most recent data point before the end of
                the alignment period to the boundary at the end of the
                alignment period. This aligner is valid for ``GAUGE``
                metrics. The ``value_type`` of the aligned result is the
                same as the ``value_type`` of the input.
            ALIGN_MIN (10):
                Align the time series by returning the minimum value in each
                alignment period. This aligner is valid for ``GAUGE`` and
                ``DELTA`` metrics with numeric values. The ``value_type`` of
                the aligned result is the same as the ``value_type`` of the
                input.
            ALIGN_MAX (11):
                Align the time series by returning the maximum value in each
                alignment period. This aligner is valid for ``GAUGE`` and
                ``DELTA`` metrics with numeric values. The ``value_type`` of
                the aligned result is the same as the ``value_type`` of the
                input.
            ALIGN_MEAN (12):
                Align the time series by returning the mean value in each
                alignment period. This aligner is valid for ``GAUGE`` and
                ``DELTA`` metrics with numeric values. The ``value_type`` of
                the aligned result is ``DOUBLE``.
            ALIGN_COUNT (13):
                Align the time series by returning the number of values in
                each alignment period. This aligner is valid for ``GAUGE``
                and ``DELTA`` metrics with numeric or Boolean values. The
                ``value_type`` of the aligned result is ``INT64``.
            ALIGN_SUM (14):
                Align the time series by returning the sum of the values in
                each alignment period. This aligner is valid for ``GAUGE``
                and ``DELTA`` metrics with numeric and distribution values.
                The ``value_type`` of the aligned result is the same as the
                ``value_type`` of the input.
            ALIGN_STDDEV (15):
                Align the time series by returning the standard deviation of
                the values in each alignment period. This aligner is valid
                for ``GAUGE`` and ``DELTA`` metrics with numeric values. The
                ``value_type`` of the output is ``DOUBLE``.
            ALIGN_COUNT_TRUE (16):
                Align the time series by returning the number of ``True``
                values in each alignment period. This aligner is valid for
                ``GAUGE`` metrics with Boolean values. The ``value_type`` of
                the output is ``INT64``.
            ALIGN_COUNT_FALSE (24):
                Align the time series by returning the number of ``False``
                values in each alignment period. This aligner is valid for
                ``GAUGE`` metrics with Boolean values. The ``value_type`` of
                the output is ``INT64``.
            ALIGN_FRACTION_TRUE (17):
                Align the time series by returning the ratio of the number
                of ``True`` values to the total number of values in each
                alignment period. This aligner is valid for ``GAUGE``
                metrics with Boolean values. The output value is in the
                range [0.0, 1.0] and has ``value_type`` ``DOUBLE``.
            ALIGN_PERCENTILE_99 (18):
                Align the time series by using `percentile
                aggregation <https://en.wikipedia.org/wiki/Percentile>`__.
                The resulting data point in each alignment period is the
                99th percentile of all data points in the period. This
                aligner is valid for ``GAUGE`` and ``DELTA`` metrics with
                distribution values. The output is a ``GAUGE`` metric with
                ``value_type`` ``DOUBLE``.
            ALIGN_PERCENTILE_95 (19):
                Align the time series by using `percentile
                aggregation <https://en.wikipedia.org/wiki/Percentile>`__.
                The resulting data point in each alignment period is the
                95th percentile of all data points in the period. This
                aligner is valid for ``GAUGE`` and ``DELTA`` metrics with
                distribution values. The output is a ``GAUGE`` metric with
                ``value_type`` ``DOUBLE``.
            ALIGN_PERCENTILE_50 (20):
                Align the time series by using `percentile
                aggregation <https://en.wikipedia.org/wiki/Percentile>`__.
                The resulting data point in each alignment period is the
                50th percentile of all data points in the period. This
                aligner is valid for ``GAUGE`` and ``DELTA`` metrics with
                distribution values. The output is a ``GAUGE`` metric with
                ``value_type`` ``DOUBLE``.
            ALIGN_PERCENTILE_05 (21):
                Align the time series by using `percentile
                aggregation <https://en.wikipedia.org/wiki/Percentile>`__.
                The resulting data point in each alignment period is the 5th
                percentile of all data points in the period. This aligner is
                valid for ``GAUGE`` and ``DELTA`` metrics with distribution
                values. The output is a ``GAUGE`` metric with ``value_type``
                ``DOUBLE``.
            ALIGN_PERCENT_CHANGE (23):
                Align and convert to a percentage change. This aligner is
                valid for ``GAUGE`` and ``DELTA`` metrics with numeric
                values. This alignment returns
                ``((current - previous)/previous) * 100``, where the value
                of ``previous`` is determined based on the
                ``alignment_period``.

                If the values of ``current`` and ``previous`` are both 0,
                then the returned value is 0. If only ``previous`` is 0, the
                returned value is infinity.

                A 10-minute moving mean is computed at each point of the
                alignment period prior to the above calculation to smooth
                the metric and prevent false positives from very short-lived
                spikes. The moving mean is only applicable for data whose
                values are ``>= 0``. Any values ``< 0`` are treated as a
                missing datapoint, and are ignored. While ``DELTA`` metrics
                are accepted by this alignment, special care should be taken
                that the values for the metric will always be positive. The
                output is a ``GAUGE`` metric with ``value_type`` ``DOUBLE``.
        """
        ALIGN_NONE = 0
        ALIGN_DELTA = 1
        ALIGN_RATE = 2
        ALIGN_INTERPOLATE = 3
        ALIGN_NEXT_OLDER = 4
        ALIGN_MIN = 10
        ALIGN_MAX = 11
        ALIGN_MEAN = 12
        ALIGN_COUNT = 13
        ALIGN_SUM = 14
        ALIGN_STDDEV = 15
        ALIGN_COUNT_TRUE = 16
        ALIGN_COUNT_FALSE = 24
        ALIGN_FRACTION_TRUE = 17
        ALIGN_PERCENTILE_99 = 18
        ALIGN_PERCENTILE_95 = 19
        ALIGN_PERCENTILE_50 = 20
        ALIGN_PERCENTILE_05 = 21
        ALIGN_PERCENT_CHANGE = 23

    class Reducer(proto.Enum):
        r"""A Reducer operation describes how to aggregate data points
        from multiple time series into a single time series, where the
        value of each data point in the resulting series is a function
        of all the already aligned values in the input time series.

        Values:
            REDUCE_NONE (0):
                No cross-time series reduction. The output of the
                ``Aligner`` is returned.
            REDUCE_MEAN (1):
                Reduce by computing the mean value across time series for
                each alignment period. This reducer is valid for
                [DELTA][google.api.MetricDescriptor.MetricKind.DELTA] and
                [GAUGE][google.api.MetricDescriptor.MetricKind.GAUGE]
                metrics with numeric or distribution values. The
                ``value_type`` of the output is
                [DOUBLE][google.api.MetricDescriptor.ValueType.DOUBLE].
            REDUCE_MIN (2):
                Reduce by computing the minimum value across time series for
                each alignment period. This reducer is valid for ``DELTA``
                and ``GAUGE`` metrics with numeric values. The
                ``value_type`` of the output is the same as the
                ``value_type`` of the input.
            REDUCE_MAX (3):
                Reduce by computing the maximum value across time series for
                each alignment period. This reducer is valid for ``DELTA``
                and ``GAUGE`` metrics with numeric values. The
                ``value_type`` of the output is the same as the
                ``value_type`` of the input.
            REDUCE_SUM (4):
                Reduce by computing the sum across time series for each
                alignment period. This reducer is valid for ``DELTA`` and
                ``GAUGE`` metrics with numeric and distribution values. The
                ``value_type`` of the output is the same as the
                ``value_type`` of the input.
            REDUCE_STDDEV (5):
                Reduce by computing the standard deviation across time
                series for each alignment period. This reducer is valid for
                ``DELTA`` and ``GAUGE`` metrics with numeric or distribution
                values. The ``value_type`` of the output is ``DOUBLE``.
            REDUCE_COUNT (6):
                Reduce by computing the number of data points across time
                series for each alignment period. This reducer is valid for
                ``DELTA`` and ``GAUGE`` metrics of numeric, Boolean,
                distribution, and string ``value_type``. The ``value_type``
                of the output is ``INT64``.
            REDUCE_COUNT_TRUE (7):
                Reduce by computing the number of ``True``-valued data
                points across time series for each alignment period. This
                reducer is valid for ``DELTA`` and ``GAUGE`` metrics of
                Boolean ``value_type``. The ``value_type`` of the output is
                ``INT64``.
            REDUCE_COUNT_FALSE (15):
                Reduce by computing the number of ``False``-valued data
                points across time series for each alignment period. This
                reducer is valid for ``DELTA`` and ``GAUGE`` metrics of
                Boolean ``value_type``. The ``value_type`` of the output is
                ``INT64``.
            REDUCE_FRACTION_TRUE (8):
                Reduce by computing the ratio of the number of
                ``True``-valued data points to the total number of data
                points for each alignment period. This reducer is valid for
                ``DELTA`` and ``GAUGE`` metrics of Boolean ``value_type``.
                The output value is in the range [0.0, 1.0] and has
                ``value_type`` ``DOUBLE``.
            REDUCE_PERCENTILE_99 (9):
                Reduce by computing the `99th
                percentile <https://en.wikipedia.org/wiki/Percentile>`__ of
                data points across time series for each alignment period.
                This reducer is valid for ``GAUGE`` and ``DELTA`` metrics of
                numeric and distribution type. The value of the output is
                ``DOUBLE``.
            REDUCE_PERCENTILE_95 (10):
                Reduce by computing the `95th
                percentile <https://en.wikipedia.org/wiki/Percentile>`__ of
                data points across time series for each alignment period.
                This reducer is valid for ``GAUGE`` and ``DELTA`` metrics of
                numeric and distribution type. The value of the output is
                ``DOUBLE``.
            REDUCE_PERCENTILE_50 (11):
                Reduce by computing the `50th
                percentile <https://en.wikipedia.org/wiki/Percentile>`__ of
                data points across time series for each alignment period.
                This reducer is valid for ``GAUGE`` and ``DELTA`` metrics of
                numeric and distribution type. The value of the output is
                ``DOUBLE``.
            REDUCE_PERCENTILE_05 (12):
                Reduce by computing the `5th
                percentile <https://en.wikipedia.org/wiki/Percentile>`__ of
                data points across time series for each alignment period.
                This reducer is valid for ``GAUGE`` and ``DELTA`` metrics of
                numeric and distribution type. The value of the output is
                ``DOUBLE``.
        """
        REDUCE_NONE = 0
        REDUCE_MEAN = 1
        REDUCE_MIN = 2
        REDUCE_MAX = 3
        REDUCE_SUM = 4
        REDUCE_STDDEV = 5
        REDUCE_COUNT = 6
        REDUCE_COUNT_TRUE = 7
        REDUCE_COUNT_FALSE = 15
        REDUCE_FRACTION_TRUE = 8
        REDUCE_PERCENTILE_99 = 9
        REDUCE_PERCENTILE_95 = 10
        REDUCE_PERCENTILE_50 = 11
        REDUCE_PERCENTILE_05 = 12

    alignment_period: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )
    per_series_aligner: Aligner = proto.Field(
        proto.ENUM,
        number=2,
        enum=Aligner,
    )
    cross_series_reducer: Reducer = proto.Field(
        proto.ENUM,
        number=4,
        enum=Reducer,
    )
    group_by_fields: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
