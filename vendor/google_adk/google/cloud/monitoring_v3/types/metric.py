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

from google.api import label_pb2  # type: ignore
from google.api import metric_pb2  # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.monitoring_v3.types import common

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "Point",
        "TimeSeries",
        "TimeSeriesDescriptor",
        "TimeSeriesData",
        "LabelValue",
        "QueryError",
        "TextLocator",
    },
)


class Point(proto.Message):
    r"""A single data point in a time series.

    Attributes:
        interval (google.cloud.monitoring_v3.types.TimeInterval):
            The time interval to which the data point applies. For
            ``GAUGE`` metrics, the start time is optional, but if it is
            supplied, it must equal the end time. For ``DELTA`` metrics,
            the start and end time should specify a non-zero interval,
            with subsequent points specifying contiguous and
            non-overlapping intervals. For ``CUMULATIVE`` metrics, the
            start and end time should specify a non-zero interval, with
            subsequent points specifying the same start time and
            increasing end times, until an event resets the cumulative
            value to zero and sets a new start time for the following
            points.
        value (google.cloud.monitoring_v3.types.TypedValue):
            The value of the data point.
    """

    interval: common.TimeInterval = proto.Field(
        proto.MESSAGE,
        number=1,
        message=common.TimeInterval,
    )
    value: common.TypedValue = proto.Field(
        proto.MESSAGE,
        number=2,
        message=common.TypedValue,
    )


class TimeSeries(proto.Message):
    r"""A collection of data points that describes the time-varying
    values of a metric. A time series is identified by a combination
    of a fully-specified monitored resource and a fully-specified
    metric. This type is used for both listing and creating time
    series.

    Attributes:
        metric (google.api.metric_pb2.Metric):
            The associated metric. A fully-specified
            metric used to identify the time series.
        resource (google.api.monitored_resource_pb2.MonitoredResource):
            The associated monitored resource. Custom metrics can use
            only certain monitored resource types in their time series
            data. For more information, see `Monitored resources for
            custom
            metrics <https://cloud.google.com/monitoring/custom-metrics/creating-metrics#custom-metric-resources>`__.
        metadata (google.api.monitored_resource_pb2.MonitoredResourceMetadata):
            Output only. The associated monitored
            resource metadata. When reading a time series,
            this field will include metadata labels that are
            explicitly named in the reduction. When creating
            a time series, this field is ignored.
        metric_kind (google.api.metric_pb2.MetricKind):
            The metric kind of the time series. When listing time
            series, this metric kind might be different from the metric
            kind of the associated metric if this time series is an
            alignment or reduction of other time series.

            When creating a time series, this field is optional. If
            present, it must be the same as the metric kind of the
            associated metric. If the associated metric's descriptor
            must be auto-created, then this field specifies the metric
            kind of the new descriptor and must be either ``GAUGE`` (the
            default) or ``CUMULATIVE``.
        value_type (google.api.metric_pb2.ValueType):
            The value type of the time series. When listing time series,
            this value type might be different from the value type of
            the associated metric if this time series is an alignment or
            reduction of other time series.

            When creating a time series, this field is optional. If
            present, it must be the same as the type of the data in the
            ``points`` field.
        points (MutableSequence[google.cloud.monitoring_v3.types.Point]):
            The data points of this time series. When listing time
            series, points are returned in reverse time order.

            When creating a time series, this field must contain exactly
            one point and the point's type must be the same as the value
            type of the associated metric. If the associated metric's
            descriptor must be auto-created, then the value type of the
            descriptor is determined by the point's type, which must be
            ``BOOL``, ``INT64``, ``DOUBLE``, or ``DISTRIBUTION``.
        unit (str):
            The units in which the metric value is reported. It is only
            applicable if the ``value_type`` is ``INT64``, ``DOUBLE``,
            or ``DISTRIBUTION``. The ``unit`` defines the representation
            of the stored metric values. This field can only be changed
            through CreateTimeSeries when it is empty.
        description (str):
            Input only. A detailed description of the time series that
            will be associated with the
            [google.api.MetricDescriptor][google.api.MetricDescriptor]
            for the metric. Once set, this field cannot be changed
            through CreateTimeSeries.
    """

    metric: metric_pb2.Metric = proto.Field(
        proto.MESSAGE,
        number=1,
        message=metric_pb2.Metric,
    )
    resource: monitored_resource_pb2.MonitoredResource = proto.Field(
        proto.MESSAGE,
        number=2,
        message=monitored_resource_pb2.MonitoredResource,
    )
    metadata: monitored_resource_pb2.MonitoredResourceMetadata = proto.Field(
        proto.MESSAGE,
        number=7,
        message=monitored_resource_pb2.MonitoredResourceMetadata,
    )
    metric_kind: metric_pb2.MetricDescriptor.MetricKind = proto.Field(
        proto.ENUM,
        number=3,
        enum=metric_pb2.MetricDescriptor.MetricKind,
    )
    value_type: metric_pb2.MetricDescriptor.ValueType = proto.Field(
        proto.ENUM,
        number=4,
        enum=metric_pb2.MetricDescriptor.ValueType,
    )
    points: MutableSequence["Point"] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message="Point",
    )
    unit: str = proto.Field(
        proto.STRING,
        number=8,
    )
    description: str = proto.Field(
        proto.STRING,
        number=9,
    )


class TimeSeriesDescriptor(proto.Message):
    r"""A descriptor for the labels and points in a time series.

    Attributes:
        label_descriptors (MutableSequence[google.api.label_pb2.LabelDescriptor]):
            Descriptors for the labels.
        point_descriptors (MutableSequence[google.cloud.monitoring_v3.types.TimeSeriesDescriptor.ValueDescriptor]):
            Descriptors for the point data value columns.
    """

    class ValueDescriptor(proto.Message):
        r"""A descriptor for the value columns in a data point.

        Attributes:
            key (str):
                The value key.
            value_type (google.api.metric_pb2.ValueType):
                The value type.
            metric_kind (google.api.metric_pb2.MetricKind):
                The value stream kind.
            unit (str):
                The unit in which ``time_series`` point values are reported.
                ``unit`` follows the UCUM format for units as seen in
                https://unitsofmeasure.org/ucum.html. ``unit`` is only valid
                if ``value_type`` is INTEGER, DOUBLE, DISTRIBUTION.
        """

        key: str = proto.Field(
            proto.STRING,
            number=1,
        )
        value_type: metric_pb2.MetricDescriptor.ValueType = proto.Field(
            proto.ENUM,
            number=2,
            enum=metric_pb2.MetricDescriptor.ValueType,
        )
        metric_kind: metric_pb2.MetricDescriptor.MetricKind = proto.Field(
            proto.ENUM,
            number=3,
            enum=metric_pb2.MetricDescriptor.MetricKind,
        )
        unit: str = proto.Field(
            proto.STRING,
            number=4,
        )

    label_descriptors: MutableSequence[label_pb2.LabelDescriptor] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=label_pb2.LabelDescriptor,
    )
    point_descriptors: MutableSequence[ValueDescriptor] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message=ValueDescriptor,
    )


class TimeSeriesData(proto.Message):
    r"""Represents the values of a time series associated with a
    TimeSeriesDescriptor.

    Attributes:
        label_values (MutableSequence[google.cloud.monitoring_v3.types.LabelValue]):
            The values of the labels in the time series identifier,
            given in the same order as the ``label_descriptors`` field
            of the TimeSeriesDescriptor associated with this object.
            Each value must have a value of the type given in the
            corresponding entry of ``label_descriptors``.
        point_data (MutableSequence[google.cloud.monitoring_v3.types.TimeSeriesData.PointData]):
            The points in the time series.
    """

    class PointData(proto.Message):
        r"""A point's value columns and time interval. Each point has one or
        more point values corresponding to the entries in
        ``point_descriptors`` field in the TimeSeriesDescriptor associated
        with this object.

        Attributes:
            values (MutableSequence[google.cloud.monitoring_v3.types.TypedValue]):
                The values that make up the point.
            time_interval (google.cloud.monitoring_v3.types.TimeInterval):
                The time interval associated with the point.
        """

        values: MutableSequence[common.TypedValue] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=common.TypedValue,
        )
        time_interval: common.TimeInterval = proto.Field(
            proto.MESSAGE,
            number=2,
            message=common.TimeInterval,
        )

    label_values: MutableSequence["LabelValue"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="LabelValue",
    )
    point_data: MutableSequence[PointData] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=PointData,
    )


class LabelValue(proto.Message):
    r"""A label value.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        bool_value (bool):
            A bool label value.

            This field is a member of `oneof`_ ``value``.
        int64_value (int):
            An int64 label value.

            This field is a member of `oneof`_ ``value``.
        string_value (str):
            A string label value.

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
    string_value: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="value",
    )


class QueryError(proto.Message):
    r"""An error associated with a query in the time series query
    language format.

    Attributes:
        locator (google.cloud.monitoring_v3.types.TextLocator):
            The location of the time series query
            language text that this error applies to.
        message (str):
            The error message.
    """

    locator: "TextLocator" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TextLocator",
    )
    message: str = proto.Field(
        proto.STRING,
        number=2,
    )


class TextLocator(proto.Message):
    r"""A locator for text. Indicates a particular part of the text of a
    request or of an object referenced in the request.

    For example, suppose the request field ``text`` contains:

    text: "The quick brown fox jumps over the lazy dog."

    Then the locator:

    source: "text" start_position { line: 1 column: 17 } end_position {
    line: 1 column: 19 }

    refers to the part of the text: "fox".

    Attributes:
        source (str):
            The source of the text. The source may be a field in the
            request, in which case its format is the format of the
            google.rpc.BadRequest.FieldViolation.field field in
            https://cloud.google.com/apis/design/errors#error_details.
            It may also be be a source other than the request field
            (e.g. a macro definition referenced in the text of the
            query), in which case this is the name of the source (e.g.
            the macro name).
        start_position (google.cloud.monitoring_v3.types.TextLocator.Position):
            The position of the first byte within the
            text.
        end_position (google.cloud.monitoring_v3.types.TextLocator.Position):
            The position of the last byte within the
            text.
        nested_locator (google.cloud.monitoring_v3.types.TextLocator):
            If ``source``, ``start_position``, and ``end_position``
            describe a call on some object (e.g. a macro in the time
            series query language text) and a location is to be
            designated in that object's text, ``nested_locator``
            identifies the location within that object.
        nesting_reason (str):
            When ``nested_locator`` is set, this field gives the reason
            for the nesting. Usually, the reason is a macro invocation.
            In that case, the macro name (including the leading '@')
            signals the location of the macro call in the text and a
            macro argument name (including the leading '$') signals the
            location of the macro argument inside the macro body that
            got substituted away.
    """

    class Position(proto.Message):
        r"""The position of a byte within the text.

        Attributes:
            line (int):
                The line, starting with 1, where the byte is
                positioned.
            column (int):
                The column within the line, starting with 1,
                where the byte is positioned. This is a byte
                index even though the text is UTF-8.
        """

        line: int = proto.Field(
            proto.INT32,
            number=1,
        )
        column: int = proto.Field(
            proto.INT32,
            number=2,
        )

    source: str = proto.Field(
        proto.STRING,
        number=1,
    )
    start_position: Position = proto.Field(
        proto.MESSAGE,
        number=2,
        message=Position,
    )
    end_position: Position = proto.Field(
        proto.MESSAGE,
        number=3,
        message=Position,
    )
    nested_locator: "TextLocator" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="TextLocator",
    )
    nesting_reason: str = proto.Field(
        proto.STRING,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
