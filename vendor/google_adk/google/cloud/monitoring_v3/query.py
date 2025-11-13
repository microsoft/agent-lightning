# Copyright 2016 Google LLC
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

"""Time series query for the `Google Stackdriver Monitoring API (V3)`_.

.. _Google Stackdriver Monitoring API (V3):
    https://cloud.google.com/monitoring/api/ref_v3/rest/v3/\
    projects.timeSeries/list
"""

import copy
import datetime

import google.cloud.monitoring_v3 as monitoring_v3
from google.cloud.monitoring_v3 import _dataframe, types

_UTCNOW = datetime.datetime.utcnow  # To be replaced by tests.


class Query(object):
    """Query object for retrieving metric data.

    :type client: :class:`google.cloud.monitoring_v3.gapic.
        metric_service_client.MetricServiceClient`
    :param client: The client to use.

    :type project: str
    :param project: The project ID or number.

    :type metric_type: str
    :param metric_type: The metric type name. The default value is
        :data:`Query.DEFAULT_METRIC_TYPE
        <google.cloud.monitoring.query.Query.DEFAULT_METRIC_TYPE>`,
        but please note that this default value is provided only for
        demonstration purposes and is subject to change. See the
        `supported metrics`_.

    :type end_time: :class:`datetime.datetime`
    :param end_time: (Optional) The end time (inclusive) of the time interval
        for which results should be returned, as a datetime object.
        The default is the start of the current minute.

        The start time (exclusive) is determined by combining the
        values of  ``days``, ``hours``, and ``minutes``, and
        subtracting the resulting duration from the end time.

        It is also allowed to omit the end time and duration here,
        in which case
        :meth:`~google.cloud.monitoring.query.Query.select_interval`
        must be called before the query is executed.

    :type days: int
    :param days: The number of days in the time interval.

    :type hours: int
    :param hours: The number of hours in the time interval.

    :type minutes: int
    :param minutes: The number of minutes in the time interval.

    :raises: :exc:`ValueError` if ``end_time`` is specified but
        ``days``, ``hours``, and ``minutes`` are all zero.
        If you really want to specify a point in time, use
        :meth:`~google.cloud.monitoring.query.Query.select_interval`.

    .. _supported metrics: https://cloud.google.com/monitoring/api/metrics
    """

    DEFAULT_METRIC_TYPE = "compute.googleapis.com/instance/cpu/utilization"

    def __init__(
        self,
        client,
        project,
        metric_type=DEFAULT_METRIC_TYPE,
        end_time=None,
        days=0,
        hours=0,
        minutes=0,
    ):
        start_time = None
        if days or hours or minutes:
            if end_time is None:
                end_time = _UTCNOW().replace(second=0, microsecond=0)
            start_time = end_time - datetime.timedelta(
                days=days, hours=hours, minutes=minutes
            )
        elif end_time is not None:
            raise ValueError("Non-zero duration required for time interval.")

        self._client = client
        self._project_path = f"projects/{project}"
        self._end_time = end_time
        self._start_time = start_time
        self._filter = _Filter(metric_type)

        self._per_series_aligner = 0
        self._alignment_period_seconds = 0
        self._cross_series_reducer = 0
        self._group_by_fields = ()

    def __iter__(self):
        return self.iter()

    @property
    def metric_type(self):
        """The metric type name."""
        return self._filter.metric_type

    @property
    def filter(self):
        """The filter string.

        This is constructed from the metric type, the resource type, and
        selectors for the group ID, monitored projects, resource labels,
        and metric labels.
        """
        return str(self._filter)

    def select_interval(self, end_time, start_time=None):
        """Copy the query and set the query time interval.

        Example::

            import datetime

            now = datetime.datetime.utcnow()
            query = query.select_interval(
                end_time=now,
                start_time=now - datetime.timedelta(minutes=5))

        As a convenience, you can alternatively specify the end time and
        an interval duration when you create the query initially.

        :type end_time: :class:`datetime.datetime`
        :param end_time: The end time (inclusive) of the time interval
            for which results should be returned, as a datetime object.

        :type start_time: :class:`datetime.datetime`
        :param start_time:
            (Optional) The start time (exclusive) of the time interval
            for which results should be returned, as a datetime object.
            If not specified, the interval is a point in time.

        :rtype: :class:`Query`
        :returns: The new query object.
        """
        new_query = copy.deepcopy(self)
        new_query._end_time = end_time
        new_query._start_time = start_time
        return new_query

    def select_group(self, group_id):
        """Copy the query and add filtering by group.

        Example::

            query = query.select_group('1234567')

        :type group_id: str
        :param group_id: The ID of a group to filter by.

        :rtype: :class:`Query`
        :returns: The new query object.
        """
        new_query = copy.deepcopy(self)
        new_query._filter.group_id = group_id
        return new_query

    def select_projects(self, *args):
        """Copy the query and add filtering by monitored projects.

        This is only useful if the target project represents a Stackdriver
        account containing the specified monitored projects.

        Examples::

            query = query.select_projects('project-1')
            query = query.select_projects('project-1', 'project-2')

        :type args: tuple
        :param args: Project IDs limiting the resources to be included
            in the query.

        :rtype: :class:`Query`
        :returns: The new query object.
        """
        new_query = copy.deepcopy(self)
        new_query._filter.projects = args
        return new_query

    def select_resources(self, *args, **kwargs):
        """Copy the query and add filtering by resource labels.

        See more documentation at: https://cloud.google.com/monitoring/api/v3/filters#comparisons.

        Examples::

            query = query.select_resources(zone='us-central1-a')
            query = query.select_resources(zone_prefix='europe-')
            query = query.select_resources(resource_type='gce_instance')

        A keyword argument ``<label>=<value>`` ordinarily generates a filter
        expression of the form::

            resource.label.<label> = "<value>"

        However, by adding ``"_prefix"`` or ``"_suffix"`` to the keyword,
        you can specify a partial match.

        ``<label>_prefix=<value>`` generates::

            resource.label.<label> = starts_with("<value>")

        ``<label>_suffix=<value>`` generates::

            resource.label.<label> = ends_with("<value>")

        As a special case, ``"resource_type"`` is treated as a special
        pseudo-label corresponding to the filter object ``resource.type``.
        For example, ``resource_type=<value>`` generates::

            resource.type = "<value>"

        See the `defined resource types`_.

        .. note::

            The label ``"instance_name"`` is a metric label,
            not a resource label. You would filter on it using
            ``select_metrics(instance_name=...)``.

        :type args: tuple
        :param args: Raw filter expression strings to include in the
            conjunction. If just one is provided and no keyword arguments
            are provided, it can be a disjunction.

        :param kwargs: Label filters to include in the conjunction as
            described above.

        :rtype: :class:`Query`
        :returns: The new query object.

        .. _defined resource types:
            https://cloud.google.com/monitoring/api/v3/monitored-resources
        """
        new_query = copy.deepcopy(self)
        new_query._filter.select_resources(*args, **kwargs)
        return new_query

    def select_metrics(self, *args, **kwargs):
        """Copy the query and add filtering by metric labels.

        Examples::

            query = query.select_metrics(instance_name='myinstance')
            query = query.select_metrics(instance_name_prefix='mycluster-')

        A keyword argument ``<label>=<value>`` ordinarily generates a filter
        expression of the form::

            metric.label.<label> = "<value>"

        However, by adding ``"_notequal"`` to the keyword, you can inequality:

        ``<label>_notequal=<value>`` generates::

            metric.label.<label> != <value>

        By adding ``"_prefix"`` or ``"_suffix"`` to the keyword, you can specify
        a partial match.

        ``<label>_prefix=<value>`` generates::

            metric.label.<label> = starts_with("<value>")

        ``<label>_suffix=<value>`` generates::

            metric.label.<label> = ends_with("<value>")

        If the label's value type is ``INT64``, a similar notation can be
        used to express inequalities:

        ``<label>_less=<value>`` generates::

            metric.label.<label> < <value>

        ``<label>_lessequal=<value>`` generates::

            metric.label.<label> <= <value>

        ``<label>_greater=<value>`` generates::

            metric.label.<label> > <value>

        ``<label>_greaterequal=<value>`` generates::

            metric.label.<label> >= <value>

        :type args: tuple
        :param args: Raw filter expression strings to include in the
            conjunction. If just one is provided and no keyword arguments
            are provided, it can be a disjunction.

        :param kwargs: Label filters to include in the conjunction as
            described above.

        :rtype: :class:`Query`
        :returns: The new query object.
        """
        new_query = copy.deepcopy(self)
        new_query._filter.select_metrics(*args, **kwargs)
        return new_query

    def align(self, per_series_aligner, seconds=0, minutes=0, hours=0):
        """Copy the query and add temporal alignment.

        If ``per_series_aligner`` is not :data:`Aligner.ALIGN_NONE`, each time
        series will contain data points only on the period boundaries.

        Example::

            from google.cloud import monitoring
            query = query.align(
                monitoring.Aggregation.Aligner.ALIGN_MEAN, minutes=5)

        It is also possible to specify the aligner as a literal string::

            query = query.align('ALIGN_MEAN', minutes=5)

        :type per_series_aligner: str or
            :class:`~google.cloud.monitoring_v3.Aggregation.Aligner`
        :param per_series_aligner: The approach to be used to align
            individual time series. For example: :data:`Aligner.ALIGN_MEAN`.
            See
            :class:`~google.cloud.monitoring_v3.Aggregation.Aligner`
            and the descriptions of the `supported aligners`_.

        :type seconds: int
        :param seconds: The number of seconds in the alignment period.

        :type minutes: int
        :param minutes: The number of minutes in the alignment period.

        :type hours: int
        :param hours: The number of hours in the alignment period.

        :rtype: :class:`Query`
        :returns: The new query object.

        .. _supported aligners:
            https://cloud.google.com/monitoring/api/ref_v3/rest/v3/\
            projects.timeSeries/list#Aligner
        """
        new_query = copy.deepcopy(self)
        new_query._per_series_aligner = per_series_aligner
        new_query._alignment_period_seconds = seconds + 60 * (minutes + 60 * hours)
        return new_query

    def reduce(self, cross_series_reducer, *group_by_fields):
        """Copy the query and add cross-series reduction.

        Cross-series reduction combines time series by aggregating their
        data points.

        For example, you could request an aggregated time series for each
        combination of project and zone as follows::

            from google.cloud import monitoring
            query = query.reduce(monitoring.Aggregation.Reducer.REDUCE_MEAN,
                                 'resource.project_id', 'resource.zone')

        :type cross_series_reducer: str or
            :class:`~google.cloud.monitoring_v3.Aggregation.Reducer`
        :param cross_series_reducer:
            The approach to be used to combine time series. For example:
            :data:`Reducer.REDUCE_MEAN`. See
            :class:`~google.cloud.monitoring_v3.Aggregation.Reducer`
            and the descriptions of the `supported reducers`_.

        :type group_by_fields: strs
        :param group_by_fields:
            Fields to be preserved by the reduction. For example, specifying
            just ``"resource.zone"`` will result in one time series per zone.
            The default is to aggregate all of the time series into just one.

        :rtype: :class:`Query`
        :returns: The new query object.

        .. _supported reducers:
            https://cloud.google.com/monitoring/api/ref_v3/rest/v3/\
            projects.timeSeries/list#Reducer
        """
        new_query = copy.deepcopy(self)
        new_query._cross_series_reducer = cross_series_reducer
        new_query._group_by_fields = group_by_fields
        return new_query

    def iter(self, headers_only=False, page_size=None):
        """Yield all time series objects selected by the query.

        The generator returned iterates over
        :class:`~google.cloud.monitoring_v3.types.TimeSeries` objects
        containing points ordered from oldest to newest.

        Note that the :class:`Query` object itself is an iterable, such that
        the following are equivalent::

            for timeseries in query:
                ...

            for timeseries in query.iter():
                ...

        :type headers_only: bool
        :param headers_only:
             Whether to omit the point data from the time series objects.

        :type page_size: int
        :param page_size:
            (Optional) The maximum number of points in each page of results
            from this request. Non-positive values are ignored. Defaults
            to a sensible value set by the API.

        :raises: :exc:`ValueError` if the query time interval has not been
            specified.
        """
        if self._end_time is None:
            raise ValueError("Query time interval not specified.")

        params = self._build_query_params(headers_only, page_size)

        request = monitoring_v3.ListTimeSeriesRequest(**params)
        for ts in self._client.list_time_series(request):
            yield ts

    def _build_query_params(self, headers_only=False, page_size=None):
        """Return key-value pairs for the list_time_series API call.

        :type headers_only: bool
        :param headers_only:
             Whether to omit the point data from the
             :class:`~google.cloud.monitoring_v3.types.TimeSeries` objects.

        :type page_size: int
        :param page_size:
            (Optional) The maximum number of points in each page of results
            from this request. Non-positive values are ignored. Defaults
            to a sensible value set by the API.
        """
        params = {
            "name": self._project_path,
            "filter": self.filter,
            "interval": types.TimeInterval(
                start_time=self._start_time, end_time=self._end_time
            ),
        }

        if (
            self._per_series_aligner
            or self._alignment_period_seconds
            or self._cross_series_reducer
            or self._group_by_fields
        ):
            params["aggregation"] = types.Aggregation(
                per_series_aligner=self._per_series_aligner,
                cross_series_reducer=self._cross_series_reducer,
                group_by_fields=self._group_by_fields,
                alignment_period={"seconds": self._alignment_period_seconds},
            )

        tsv = monitoring_v3.ListTimeSeriesRequest.TimeSeriesView
        params["view"] = tsv.HEADERS if headers_only else tsv.FULL

        if page_size is not None:
            params["page_size"] = page_size

        return params

    def as_dataframe(self, label=None, labels=None):
        """Return all the selected time series as a :mod:`pandas` dataframe.

        .. note::

            Use of this method requires that you have :mod:`pandas` installed.

        Examples::

            # Generate a dataframe with a multi-level column header including
            # the resource type and all available resource and metric labels.
            # This can be useful for seeing what labels are available.
            dataframe = query.as_dataframe()

            # Generate a dataframe using a particular label for the column
            # names.
            dataframe = query.as_dataframe(label='instance_name')

            # Generate a dataframe with a multi-level column header.
            dataframe = query.as_dataframe(labels=['zone', 'instance_name'])

            # Generate a dataframe with a multi-level column header, assuming
            # the metric is issued by more than one type of resource.
            dataframe = query.as_dataframe(
                labels=['resource_type', 'instance_id'])

        :type label: str
        :param label:
            (Optional) The label name to use for the dataframe header.
            This can be the name of a resource label or metric label
            (e.g., ``"instance_name"``), or the string ``"resource_type"``.

        :type labels: list of strings, or None
        :param labels: A list or tuple of label names to use for the dataframe
            header. If more than one label name is provided, the resulting
            dataframe will have a multi-level column header. Providing values
            for both ``label`` and ``labels`` is an error.

        :rtype: :class:`pandas.DataFrame`
        :returns: A dataframe where each column represents one time series.
        """
        return _dataframe._build_dataframe(self, label, labels)

    def __deepcopy__(self, memo):
        """Create a deepcopy of the query object.

        The `client` attribute is copied by reference only.

        :type memo: dict
        :param memo: the memo dict to avoid excess copying in case  the object
            is referenced from its member.

        :rtype: :class:`Query`
        :returns: The new query object.
        """
        new_query = copy.copy(self)
        new_query._filter = copy.deepcopy(self._filter, memo)
        return new_query


class _Filter(object):
    """Helper for assembling a filter string."""

    def __init__(self, metric_type):
        self.metric_type = metric_type
        self.group_id = None
        self.projects = ()
        self.resource_label_filter = None
        self.metric_label_filter = None

    def select_resources(self, *args, **kwargs):
        """Select by resource labels.

        See :meth:`Query.select_resources`.
        """
        self.resource_label_filter = _build_label_filter("resource", *args, **kwargs)

    def select_metrics(self, *args, **kwargs):
        """Select by metric labels.

        See :meth:`Query.select_metrics`.
        """
        self.metric_label_filter = _build_label_filter("metric", *args, **kwargs)

    def __str__(self):
        filters = ['metric.type = "{type}"'.format(type=self.metric_type)]
        if self.group_id is not None:
            filters.append('group.id = "{id}"'.format(id=self.group_id))
        if self.projects:
            filters.append(
                " OR ".join(
                    'project = "{project}"'.format(project=project)
                    for project in self.projects
                )
            )
        if self.resource_label_filter:
            filters.append(self.resource_label_filter)
        if self.metric_label_filter:
            filters.append(self.metric_label_filter)

        # Parentheses are never actually required, because OR binds more
        # tightly than AND in the Monitoring API's filter syntax.
        return " AND ".join(filters)


def _build_label_filter(category, *args, **kwargs):
    """Construct a filter string to filter on metric or resource labels."""
    terms = list(args)
    for key, value in kwargs.items():
        if value is None:
            continue

        suffix = None
        if key.endswith(
            (
                "_prefix",
                "_suffix",
                "_greater",
                "_greaterequal",
                "_less",
                "_lessequal",
                "_notequal",
            )
        ):
            key, suffix = key.rsplit("_", 1)

        if category == "resource" and key == "resource_type":
            key = "resource.type"
        else:
            key = ".".join((category, "label", key))

        if suffix == "prefix":
            term = '{key} = starts_with("{value}")'
        elif suffix == "suffix":
            term = '{key} = ends_with("{value}")'
        elif suffix == "greater":
            term = "{key} > {value}"
        elif suffix == "greaterequal":
            term = "{key} >= {value}"
        elif suffix == "less":
            term = "{key} < {value}"
        elif suffix == "lessequal":
            term = "{key} <= {value}"
        elif suffix == "notequal":
            term = "{key} != {value}"
        else:
            term = '{key} = "{value}"'

        terms.append(term.format(key=key, value=value))

    return " AND ".join(sorted(terms))
