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

import datetime
from typing import Any, Dict, Optional

from google.api_core.datetime_helpers import DatetimeWithNanoseconds

from google.cloud.bigtable.data.exceptions import ParameterTypeInferenceFailed
from google.cloud.bigtable.data.execute_query.metadata import SqlType
from google.cloud.bigtable.data.execute_query.values import ExecuteQueryValueType
from google.cloud.bigtable_v2.types.data import Value


def _format_execute_query_params(
    params: Optional[Dict[str, ExecuteQueryValueType]],
    parameter_types: Optional[Dict[str, SqlType.Type]],
) -> Dict[str, Value]:
    """
    Takes a dictionary of param_name -> param_value and optionally parameter types.
    If the parameters types are not provided, this function tries to infer them.

    Args:
        params (Optional[Dict[str, ExecuteQueryValueType]]): mapping from parameter names
        like they appear in query (without @ at the beginning) to their values.
        Only values of type ExecuteQueryValueType are permitted.
        parameter_types (Optional[Dict[str, SqlType.Type]]): mapping of parameter names
        to their types.

    Raises:
        ValueError: raised when parameter types cannot be inferred and were not
        provided explicitly.

    Returns:
         dictionary prasable to a protobuf represenging parameters as defined
         in ExecuteQueryRequest.params
    """
    if not params:
        return {}
    parameter_types = parameter_types or {}

    result_values = {}
    for key, value in params.items():
        user_provided_type = parameter_types.get(key)
        try:
            if user_provided_type:
                if not isinstance(user_provided_type, SqlType.Type):
                    raise ValueError(
                        f"Parameter type for {key} should be provided as an instance of SqlType.Type subclass."
                    )
                param_type = user_provided_type
            else:
                param_type = _detect_type(value)

            value_pb_dict = _convert_value_to_pb_value_dict(value, param_type)
        except ValueError as err:
            raise ValueError(f"Error when parsing parameter {key}") from err
        result_values[key] = value_pb_dict

    return result_values


def _to_param_types(
    params: Optional[Dict[str, ExecuteQueryValueType]],
    param_types: Optional[Dict[str, SqlType.Type]],
) -> Dict[str, Dict[str, Any]]:
    """
    Takes the params and user supplied types and creates a param_type dict for the PrepareQuery api

    Args:
        params: Dict of param name to param value
        param_types: Dict of param name to param type for params with types that cannot be inferred

    Returns:
        Dict containing the param name and type for each parameter
    """
    if params is None:
        return {}
    formatted_types = {}
    for param_key, param_value in params.items():
        if param_types and param_key in param_types:
            formatted_types[param_key] = param_types[param_key]._to_type_pb_dict()
        else:
            formatted_types[param_key] = _detect_type(param_value)._to_type_pb_dict()
    return formatted_types


def _convert_value_to_pb_value_dict(
    value: ExecuteQueryValueType, param_type: SqlType.Type
) -> Any:
    """
    Takes a value and converts it to a dictionary parsable to a protobuf.

    Args:
        value (ExecuteQueryValueType): value
        param_type (SqlType.Type): object describing which ExecuteQuery type the value represents.

    Returns:
        dictionary parsable to a protobuf.
    """
    # type field will be set only in top-level Value.
    value_dict = param_type._to_value_pb_dict(value)
    value_dict["type_"] = param_type._to_type_pb_dict()
    return value_dict


_TYPES_TO_TYPE_DICTS = [
    (bytes, SqlType.Bytes()),
    (str, SqlType.String()),
    (bool, SqlType.Bool()),
    (int, SqlType.Int64()),
    (DatetimeWithNanoseconds, SqlType.Timestamp()),
    (datetime.datetime, SqlType.Timestamp()),
    (datetime.date, SqlType.Date()),
]


def _detect_type(value: ExecuteQueryValueType) -> SqlType.Type:
    """
    Infers the ExecuteQuery type based on value. Raises error if type is amiguous.
    raises ParameterTypeInferenceFailed if not possible.
    """
    if value is None:
        raise ParameterTypeInferenceFailed(
            "Cannot infer type of None, please provide the type manually."
        )

    if isinstance(value, list):
        raise ParameterTypeInferenceFailed(
            "Cannot infer type of ARRAY parameters, please provide the type manually."
        )

    if isinstance(value, float):
        raise ParameterTypeInferenceFailed(
            "Cannot infer type of float, must specify either FLOAT32 or FLOAT64 type manually."
        )

    for field_type, type_dict in _TYPES_TO_TYPE_DICTS:
        if isinstance(value, field_type):
            return type_dict

    raise ParameterTypeInferenceFailed(
        f"Cannot infer type of {type(value).__name__}, please provide the type manually."
    )
