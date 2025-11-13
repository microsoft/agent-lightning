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

import datetime
import re

from typing import Optional, List, Tuple, Union, TYPE_CHECKING
from google.auth import credentials as auth_credentials
from vertexai.resources.preview.feature_store import (
    FeatureGroup,
    Feature,
)
from google.cloud.aiplatform import initializer, __version__

from . import _offline_store_impl as impl


if TYPE_CHECKING:
    try:
        import bigframes
    except ImportError:
        bigframes = None

    try:
        import pandas as pd
    except ImportError:
        pd = None


def _try_import_bigframes():
    """Try to import `bigframes` and return it if successful - otherwise raise an import error."""
    try:
        import bigframes
        import bigframes.pandas

        return bigframes
    except ImportError as exc:
        raise ImportError(
            "`bigframes` is not installed but required for this functionality."
        ) from exc


def _get_feature_group_from_feature(
    feature: Feature, credentials: auth_credentials.Credentials
):
    """Given a feature, return the feature group resource."""
    result = re.fullmatch(
        r"projects/(?P<project>.+)/locations/(?P<location>.+)/featureGroups/(?P<feature_group>.+)/features/.+",
        feature.resource_name,
    )

    if not result:
        raise ValueError("Couldn't find feature group in feature.")

    project = feature.project
    location = feature.location
    feature_group = result.group("feature_group")

    return FeatureGroup(
        feature_group, project=project, location=location, credentials=credentials
    )


def _extract_feature_from_str_repr(
    str_feature: str, credentials: auth_credentials.Credentials
) -> Tuple[FeatureGroup, Feature]:
    """Given a feature in string representation, return the feature and feature group."""
    # TODO: compile expr + place it in a constant
    result = re.fullmatch(
        r"((?P<project>.*)\.)?(?P<feature_group>.*)\.(?P<feature>.*)",
        str_feature,
    )
    if not result:
        raise ValueError(
            f"Feature '{str_feature}' is a string but not in expected format 'feature_group.feature' or 'project.feature_group.feature'."
        )

    feature_group = FeatureGroup(
        result.group("feature_group"),
        project=result.group("project"),  # None if no match.
        credentials=credentials,
    )
    feature = feature_group.get_feature(result.group("feature"))

    return (feature_group, feature)


def _feature_to_data_source(
    feature_group: FeatureGroup, feature: Feature
) -> impl.DataSource:
    qualifying_name = f"{feature_group.name}__{feature.name}"
    gbq_column = feature.version_column_name
    assert gbq_column

    column_name = feature.name
    assert column_name

    timestamp_column = "feature_timestamp"

    # TODO: Expose entity_id_columns as a property in FeatureGroup
    entity_id_columns = feature_group._gca_resource.big_query.entity_id_columns
    assert entity_id_columns

    bq_uri = feature_group._gca_resource.big_query.big_query_source.input_uri
    assert bq_uri

    fully_qualified_table = bq_uri.lstrip("bq://")
    assert fully_qualified_table

    query = (
        f"SELECT\n"
        f'  {", ".join(entity_id_columns)},\n'
        f"  {gbq_column} AS {column_name},\n"
        f"  {timestamp_column}\n"
        f"FROM {fully_qualified_table}"
    )

    return impl.DataSource(
        qualifying_name=qualifying_name,
        sql=query,
        data_columns=[column_name],
        # TODO: this will be parameterized in the future
        timestamp_column=timestamp_column,
        entity_id_columns=entity_id_columns,
    )


class _DataFrameToBigQueryDataFramesConverter:
    @classmethod
    def to_bigquery_dataframe(
        cls, df: "pd.DataFrame", session: "Optional[bigframes.session.Session]" = None
    ) -> "bigframes.pandas.DataFrame":
        bigframes = _try_import_bigframes()
        return bigframes.pandas.DataFrame(data=df, session=session)


def fetch_historical_feature_values(
    entity_df: "bigframes.pandas.DataFrame",
    # TODO: Add support for FeatureView | FeatureGroup | bigframes.pandas.DataFrame
    features: List[Union[str, Feature]],
    # TODO: Add support for feature_age_threshold
    feature_age_threshold: Optional[datetime.timedelta] = None,
    dry_run: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> "Union[bigframes.pandas.DataFrame, None]":
    """Fetch historical data at the timestamp specified for each entity.

    This runs a Point-In-Time Lookup (PITL) query in BigQuery across all
    features and returns the historical feature values. Feature data will be
    joined by matching their entity_id_column(s) with corresponding columns in
    the entity data frame.

    Args:
      entity_df:
        An entity DataFrame where one/multiple columns have entity ID.
        One column should have a timestamp (used for feature lookup). Other
        columns may have feature data. Entity IDs may be repeated with
        different timestamp values (in the timestamp column) to lookup data for
        entities at different points in time.
      features:
        Feature data will be joined with the entity data frame.
         * If `str` is given use `project.feature_group.feature` as the format.
          `project_id.feature_group_id.feature_id` may be used if features are
          in another project.
         * If `FeatureView` is given, the *sources* of the FeatureView will be
           used - but data will be read from the backing BigQuery table.
      feature_age_threshold:
        How far back from the timestamp to look for features values. If no
        feature values are found, empty/null value will be populated.
      dry_run:
        Build the Point-In-Time Lookup (PITL) query but don't run it. The PITL
        query will be printed to stdout.
      project:
        The project to use for feature lookup and running the Point-In-Time
        Lookup (PITL) query in BigQuery. If unset, the project set in
        aiplatform.init will be used.
      location:
        The location to use for feature lookup and running the Point-In-Time
        Lookup (PITL) query in BigQuery. If unset, the project set in
        aiplatform.init will be used.
      credentials:
        Custom credentials to use for feature lookup and running the
        Point-In-Time Lookup (PITL) query in BigQuery. Overrides credentials
        set in aiplatform.init.

    Returns:
      A `bigframes.pandas.DataFrame` with the historical feature values. `None`
      if in `dry_run` mode.
    """

    bigframes = _try_import_bigframes()
    project = project or initializer.global_config.project
    location = location or initializer.global_config.location
    credentials = credentials or initializer.global_config.credentials
    application_name = (
        f"vertexai-offline-store/{__version__}+fetch-historical-feature-values"
    )
    session_options = bigframes.BigQueryOptions(
        credentials=credentials,
        project=project,
        location=location,
        application_name=application_name,
    )
    session = bigframes.connect(session_options)

    if feature_age_threshold is not None:
        raise NotImplementedError("feature_age_threshold is not yet supported.")

    if not features:
        raise ValueError("Please specify a non-empty list of features.")

    # Convert to bigframe if needed.
    if not isinstance(entity_df, bigframes.pandas.DataFrame):
        entity_df = _DataFrameToBigQueryDataFramesConverter.to_bigquery_dataframe(
            df=entity_df,
            session=session,
        )

    # Ensure one timestamp column is present in the entity DataFrame.
    ts_cols = entity_df.select_dtypes(include=["datetime"]).columns
    if len(ts_cols) > 1:
        # TODO: Support multiple timestamp columns by specifying feature_timestamp column in an override.
        raise ValueError(
            'Multiple timestamp columns ("datetime" dtype) found in entity DataFrame. '
            "Only one timestamp column is allowed. "
            f"Timestamp columns: {', '.join([col for col in ts_cols])}"
        )
    elif len(ts_cols) == 0:
        raise ValueError(
            'No timestamp column ("datetime" dtype) found in entity DataFrame.'
        )
    entity_df_ts_col = ts_cols[0]
    entity_df_non_ts_cols = [c for c in entity_df.columns if c != entity_df_ts_col]
    entity_data_source = impl.DataSource(
        qualifying_name="entity_df",
        sql=entity_df.sql,
        data_columns=entity_df_non_ts_cols,
        timestamp_column=entity_df_ts_col,
    )

    feature_data: List[impl.DataSource] = []
    for feature in features:
        if isinstance(feature, Feature):
            feature_group = _get_feature_group_from_feature(feature, credentials)
            feature_data.append(_feature_to_data_source(feature_group, feature))
        elif isinstance(feature, str):
            feature_group, feature = _extract_feature_from_str_repr(
                feature, credentials
            )
            feature_data.append(_feature_to_data_source(feature_group, feature))
        else:
            raise ValueError(
                f"Unsupported feature type {type(feature)} found in feature list. Feature: {feature}"
            )

    # TODO: Verify `feature_data`.
    #  * Ensure that qualifying_names are not interfering.
    #  * Ensure that feature names are not interfering.
    #  * Ensure that entity id columns of all features are present in the entity DF.

    query = impl.render_pitl_query(
        entity_data=entity_data_source,
        feature_data=feature_data,
    )

    if dry_run:
        print("--- Dry run mode: PITL QUERY BEGIN ---")
        print(query)
        print("--- Dry run mode: PITL QUERY END ---")
        return None

    return session.read_gbq_query(
        query,
        index_col=bigframes.enums.DefaultIndexKind.NULL,
    )
