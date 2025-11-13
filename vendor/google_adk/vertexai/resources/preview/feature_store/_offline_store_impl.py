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

import textwrap
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DataSource:
    """An object to represent a data source - both entity DataFrame and any feature data.

    Contains helpers for use with SQL templating.
    """

    def __init__(
        self,
        qualifying_name: str,
        sql: str,
        data_columns: List[str],
        timestamp_column: str,
        entity_id_columns: Optional[List[str]] = None,
    ):
        """Initialize DataSource object.

        Args:
            qualifying_name:
                A unique name used to qualify the data in the PITL query.
            sql:
                SQL query representing the data_source.
            data_columns:
                Columns other than entity ID column(s) and timestamp column.
            timestamp_column:
                The column that holds feature timestamp data.
            entity_id_columns:
                The column(s) that holds entity IDs. Shouldn't be populated for
                entity_df.
        """
        self.qualifying_name = qualifying_name
        self._sql = sql
        self.data_columns = data_columns
        self.timestamp_column = timestamp_column
        self.entity_id_columns = entity_id_columns

    def copy_with_pitl_suffix(self) -> "DataSource":
        import copy

        data_source = copy.copy(self)
        data_source.qualifying_name += "_pitl"
        return data_source

    @property
    def sql(self):
        return self._sql

    @property
    def comma_separated_qualified_data_columns(self):
        return ", ".join(
            [self.qualifying_name + "." + col for col in self.data_columns]
        )

    @property
    def comma_separated_name_qualified_all_non_timestamp_columns(self):
        """Same as `comma_separated_qualified_data_columns` but including entity ID column."""
        all_columns = self.data_columns.copy()
        if self.entity_id_columns:
            all_columns += self.entity_id_columns
        return ", ".join([self.qualifying_name + "." + col for col in all_columns])

    @property
    def qualified_timestamp_column(self) -> str:
        """Returns name qualified timestamp column e.g. `name.feature_timestamp`."""
        return f"{self.qualifying_name}.{self.timestamp_column}"


def _generate_eid_check(entity_data: DataSource, feature: DataSource):
    """Generate equality check for entity columns of feature against matching columns in entity_data."""
    e_cols = set(entity_data.data_columns)
    f_cols = feature.entity_id_columns
    assert f_cols

    equal_statements = []
    for col in f_cols:
        if col not in e_cols:
            raise ValueError(
                f"Feature entity ID column '{col}' should be a column in the entity DataFrame."
            )
        equal_statements.append(
            f"{entity_data.qualifying_name}.{col} = {feature.qualifying_name}.{col}"
        )

    statement = " AND\n".join(equal_statements)

    return statement


# Args:
#   textwrap: Module
#   generate_eid_check: function (above)
#   entity_data: DataSource
#   feature_data: List[DataSource]
_PITL_QUERY_TEMPLATE_RAW = """WITH
  {{ entity_data.qualifying_name }}_without_row_num AS (
{{ textwrap.indent(entity_data.sql, ' ' * 4) }}
  ),
  {{ entity_data.qualifying_name }} AS (
    SELECT *, ROW_NUMBER() OVER() AS row_num,
    FROM entity_df_without_row_num
  ),

  # Features
  {% for feature_data_elem in feature_data %}
  {{ feature_data_elem.qualifying_name }} AS (
{{ textwrap.indent(feature_data_elem.sql, ' ' * 4) }}
  ),
  {% endfor %}

  # Features with PITL
  {% for feature_data_elem in feature_data %}
  {{ feature_data_elem.qualifying_name }}_pitl AS (
    SELECT
      {{ entity_data.qualifying_name }}.row_num,
      {{ feature_data_elem.comma_separated_qualified_data_columns }},
    FROM {{ entity_data.qualifying_name }}
    LEFT JOIN {{ feature_data_elem.qualifying_name }}
    ON (
{{ textwrap.indent(generate_eid_check(entity_data, feature_data_elem) + ' AND', ' ' * 6) }}
      CAST({{ feature_data_elem.qualified_timestamp_column }} AS TIMESTAMP) <= CAST({{ entity_data.qualified_timestamp_column }} AS TIMESTAMP)
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY {{ entity_data.qualifying_name }}.row_num ORDER BY {{ feature_data_elem.qualified_timestamp_column }} DESC) = 1
  ){{ ',' if not loop.last else '' }}
  {% endfor %}


SELECT
  {{ entity_data.comma_separated_name_qualified_all_non_timestamp_columns }},
  {% for feature_data_elem in feature_data %}
  {% set feature_pitl = feature_data_elem.copy_with_pitl_suffix() %}
  {{ feature_pitl.comma_separated_qualified_data_columns }},
  {% endfor %}
  {{ entity_data.qualified_timestamp_column }}

FROM {{ entity_data.qualifying_name }}
{% for feature_data_elem in feature_data %}
JOIN {{ feature_data_elem.qualifying_name }}_pitl USING (row_num)
{% endfor %}
"""


def pitl_query_template():
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(
            "`Jinja2` is not installed but required for this functionality."
        ) from exc

    return jinja2.Environment(
        loader=jinja2.BaseLoader, lstrip_blocks=True, trim_blocks=True
    ).from_string(_PITL_QUERY_TEMPLATE_RAW)


def render_pitl_query(entity_data: DataSource, feature_data: List[DataSource]):
    """Return the PITL query jinja template.

    The args for the query are as follows:
      textwrap: The python textwrap module.
      entity_data[DataSource]: The entity data(frame) as SQL source.
      feature_data[List[DataSource]]:
    """
    return pitl_query_template().render(
        textwrap=textwrap,
        generate_eid_check=_generate_eid_check,
        entity_data=entity_data,
        feature_data=feature_data,
    )
