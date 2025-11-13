# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

import json
import logging
import os
from typing import Dict, List, MutableSequence, Optional
from google.cloud import bigquery
from google.cloud.aiplatform.compat.types import (
    model_monitor_v1beta1 as model_monitor,
)

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import tensorflow as tf
except ImportError:
    tf = None


class FieldSchema:
    """Field Schema.

    The class identifies the data type of a single feature,
    which combines together to form the Schema for different fields in
    ModelMonitoringSchema.

    Attributes:
        name (str):
            Required. Field name.
        data_type (str):
            Required. Supported data types are: ``float``, ``integer``
            ``boolean``, ``string``, ``categorical``.
        repeated (bool):
            Optional. Describes if the schema field is an array of given data
            type.
    """

    def __init__(
        self,
        name: str,
        data_type: str,
        repeated: Optional[bool] = False,
    ):
        self.name = name
        self.data_type = data_type
        self.repeated = repeated

    def _as_proto(self) -> model_monitor.ModelMonitoringSchema.FieldSchema:
        """Converts ModelMonitoringSchema.FieldSchema to a proto message.

        Returns:
           The GAPIC representation of the model monitoring field schema.
        """
        return model_monitor.ModelMonitoringSchema.FieldSchema(
            name=self.name,
            data_type=self.data_type,
            repeated=self.repeated,
        )


class ModelMonitoringSchema:
    """Initializer for ModelMonitoringSchema.

    Args:
        feature_fields (MutableSequence[FieldSchema]):
            Required. Feature names of the model. Vertex AI will try to match
            the features from your dataset as follows:
            * For 'csv' files, the header names are required, and we will
              extract thecorresponding feature values when the header names
              align with the feature names.
            * For 'jsonl' files, we will extract the corresponding feature
              values if the key names match the feature names. Note: Nested
              features are not supported, so please ensure your features are
              flattened. Ensure the feature values are scalar or an array of
              scalars.
            * For 'bigquery' dataset, we will extract the corresponding feature
              values if the column names match the feature names.
              Note: The column type can be a scalar or an array of scalars.
              STRUCT or JSON types are not supported. You may use SQL queries to
              select or aggregate the relevant features from your original
              table. However, ensure that the 'schema' of the query results
              meets our requirements.
            * For the Vertex AI Endpoint Request Response Logging table or
              Vertex AI Batch Prediction Job results. If the prediction
              instance format is an array, ensure that the sequence in
              ``feature_fields`` matches the order of features in the prediction
              instance. We will match the feature with the array in the order
              specified in ``feature_fields``.
        prediction_fields (MutableSequence[FieldSchema]):
            Optional. Prediction output names of the model. The requirements are
            the same as the ``feature_fields``.
            For AutoML Tables, the prediction output name presented in schema
            will be: `predicted_{target_column}`, the `target_column` is the one
            you specified when you train the model.
            For Prediction output drift analysis:
            * AutoML Classification, the distribution of the argmax label will
              be analyzed.
            * AutoML Regression, the distribution of the value will be analyzed.
        ground_truth_fields (MutableSequence[FieldSchema]):
            Optional. Target /ground truth names of the model.
    """

    def __init__(
        self,
        feature_fields: MutableSequence[FieldSchema],
        ground_truth_fields: Optional[MutableSequence[FieldSchema]] = None,
        prediction_fields: Optional[MutableSequence[FieldSchema]] = None,
    ):
        self.feature_fields = feature_fields
        self.prediction_fields = prediction_fields
        self.ground_truth_fields = ground_truth_fields

    def _as_proto(self) -> model_monitor.ModelMonitoringSchema:
        """Converts ModelMonitoringSchema to a proto message.

        Returns:
           The GAPIC representation of the model monitoring schema.
        """
        user_feature_fields = list()
        user_prediction_fields = list()
        user_ground_truth_fields = list()
        for field in self.feature_fields:
            user_feature_fields.append(field._as_proto())
        if self.prediction_fields:
            for field in self.prediction_fields:
                user_prediction_fields.append(field._as_proto())
        if self.ground_truth_fields:
            for field in self.ground_truth_fields:
                user_ground_truth_fields.append(field._as_proto())
        return model_monitor.ModelMonitoringSchema(
            feature_fields=user_feature_fields,
            prediction_fields=(
                user_prediction_fields if self.prediction_fields else None
            ),
            ground_truth_fields=(
                user_ground_truth_fields if self.ground_truth_fields else None
            ),
        )

    def to_json(self, output_dir: Optional[str] = None) -> str:
        """Transform ModelMonitoringSchema to json format.

        Args:
            output_dir (str):
                Optional. The output directory that the transformed json file
                would be put into.
        """
        result = model_monitor.ModelMonitoringSchema.to_json(self._as_proto())
        if output_dir:
            result_path = os.path.join(output_dir, "model_monitoring_schema.json")
            with tf.io.gfile.GFile(result_path, "w") as f:
                json.dump(result, f)
                f.close()
            logging.info("Transformed schema to json file: %s", result_path)
        return result


def _check_duplicate(
    field: str,
    feature_fields: Optional[List[str]] = None,
    ground_truth_fields: Optional[List[str]] = None,
    prediction_fields: Optional[List[str]] = None,
) -> bool:
    """Check if a field appears in two field lists."""
    feature = True
    ground_truth = True
    prediction = True
    if not feature_fields or field not in feature_fields:
        feature = False
    if not ground_truth_fields or field not in ground_truth_fields:
        ground_truth = False
    if not prediction_fields or field not in prediction_fields:
        prediction = False
    return feature if (feature == ground_truth) else prediction


def _transform_schema_pandas(
    dataset: Dict[str, str],
    feature_fields: Optional[List[str]] = None,
    ground_truth_fields: Optional[List[str]] = None,
    prediction_fields: Optional[List[str]] = None,
) -> ModelMonitoringSchema:
    """Transforms the pandas schema to model monitoring schema."""
    ground_truth_fields_list = list()
    prediction_fields_list = list()
    feature_fields_list = list()
    pandas_integer_types = ["integer", "Int32", "Int64", "UInt32", "UInt64"]
    pandas_string_types = [
        "string",
        "bytes",
        "date",
        "time",
        "datetime64",
        "datetime",
        "mixed-integer",
        "inteval",
        "Interval",
    ]
    pandas_float_types = [
        "floating",
        "decimal",
        "mixed-integer-float",
        "Float32",
        "Float64",
    ]
    for field in dataset:
        infer_type = dataset[field]
        if infer_type in pandas_string_types:
            data_type = "string"
        elif infer_type in pandas_integer_types:
            data_type = "integer"
        elif infer_type in pandas_float_types:
            data_type = "float"
        elif infer_type == "boolean":
            data_type = "boolean"
        elif infer_type == "categorical" or infer_type == "category":
            data_type = "categorical"
        else:
            raise ValueError(f"Unsupported data type: {infer_type}")
        if _check_duplicate(
            field, feature_fields, ground_truth_fields, prediction_fields
        ):
            raise ValueError(f"The field {field} specified in two or more field lists")
        if ground_truth_fields and field in ground_truth_fields:
            ground_truth_fields_list.append(
                FieldSchema(
                    name=field,
                    data_type=data_type,
                )
            )
        elif prediction_fields and field in prediction_fields:
            prediction_fields_list.append(
                FieldSchema(
                    name=field,
                    data_type=data_type,
                )
            )
        elif (feature_fields and field in feature_fields) or not feature_fields:
            feature_fields_list.append(
                FieldSchema(
                    name=field,
                    data_type=data_type,
                )
            )
    return ModelMonitoringSchema(
        ground_truth_fields=ground_truth_fields_list if ground_truth_fields else None,
        prediction_fields=prediction_fields_list if prediction_fields else None,
        feature_fields=feature_fields_list,
    )


def transform_schema_from_bigquery(
    feature_fields: Optional[List[str]] = None,
    ground_truth_fields: Optional[List[str]] = None,
    prediction_fields: Optional[List[str]] = None,
    table: Optional[str] = None,
    query: Optional[str] = None,
) -> ModelMonitoringSchema:
    """Transform the existing dataset to ModelMonitoringSchema as model monitor
    could accept.

    Args:
        feature_fields (List[str]):
            Optional. The input feature fields for given dataset.
            By default all features we find would be the input features.
        ground_truth_fields (List[str]):
            Optional. The ground truth fields for given dataset.
            By default all features we find would be the input features.
        prediction_fields (List[str]):
            Optional. The prediction output field for given dataset.
            By default all features we find would be the input features.
        table (str):
            Optional. The BigQuery table uri.
        query (str):
            Optional. The BigQuery query.
    """
    ground_truth_fields_list = list()
    prediction_fields_list = list()
    feature_fields_list = list()
    bq_string_types = [
        "STRING",
        "BYTES",
        "DATE",
        "TIME",
        "GEOGRAPHY",
        "DATETIME",
        "JSON",
        "INTEVAL",
        "RANGE",
    ]
    bq_integer_types = ["INTEGER", "INT64", "TIMESTAMP"]
    bq_float_types = ["FLOAT", "DOUBLE", "FLOAT64", "NUMERIC", "BIGNUMERIC"]
    if table:
        if table.startswith("bq://"):
            table = table[len("bq://") :]
        try:
            client = bigquery.Client()
            table = client.get_table(table)
            bq_schema = table.schema
        except Exception as e:
            raise ValueError("Failed to get table from bq address provided.") from e
    elif query:
        try:
            client = bigquery.Client()
            bq_schema = client.query(
                query=query, job_config=bigquery.job.QueryJobConfig(dry_run=True)
            ).schema
        except Exception as e:
            raise ValueError("Failed to get query from bq address provided.") from e
    else:
        raise ValueError("Either table or query must be provided.")
    for field in bq_schema:
        if field.field_type in bq_string_types:
            data_type = "string"
        elif field.field_type in bq_integer_types:
            data_type = "integer"
        elif field.field_type in bq_float_types:
            data_type = "float"
        elif field.field_type == "BOOLEAN" or field.field_type == "BOOL":
            data_type = "boolean"
        else:
            raise ValueError(f"Unsupported data type: {field.field_type}")
        if _check_duplicate(
            field.name, feature_fields, ground_truth_fields, prediction_fields
        ):
            raise ValueError(
                f"The field {field.name} specified in two or more field lists"
            )
        if ground_truth_fields and field.name in ground_truth_fields:
            ground_truth_fields_list.append(
                FieldSchema(
                    name=field.name,
                    data_type=data_type,
                    repeated=True if field.mode == "REPEATED" else False,
                )
            )
        elif prediction_fields and field.name in prediction_fields:
            prediction_fields_list.append(
                FieldSchema(
                    name=field.name,
                    data_type=data_type,
                    repeated=True if field.mode == "REPEATED" else False,
                )
            )
        elif (feature_fields and field.name in feature_fields) or not feature_fields:
            feature_fields_list.append(
                FieldSchema(
                    name=field.name,
                    data_type=data_type,
                    repeated=True if field.mode == "REPEATED" else False,
                )
            )
    return ModelMonitoringSchema(
        ground_truth_fields=ground_truth_fields_list if ground_truth_fields else None,
        prediction_fields=prediction_fields_list if prediction_fields else None,
        feature_fields=feature_fields_list,
    )


def transform_schema_from_csv(
    file_path: str,
    feature_fields: Optional[List[str]] = None,
    ground_truth_fields: Optional[List[str]] = None,
    prediction_fields: Optional[List[str]] = None,
) -> ModelMonitoringSchema:
    """Transform the existing dataset to ModelMonitoringSchema as model monitor could accept.

    Args:
        file_path (str):
            Required. The dataset file path.
        feature_fields (List[str]):
            Optional. The input feature fields for given dataset.
            By default all features we find would be the input features.
        ground_truth_fields (List[str]):
            Optional. The ground truth fields for given dataset.
            By default all features we find would be the input features.
        prediction_fields (List[str]):s
            Optional. The prediction output field for given dataset.
            By default all features we find would be the input features.
    """
    with tf.io.gfile.GFile(file_path, "r") as f:
        input_dataset = pd.read_csv(f)
        dict_dataset = dict()
        for field in input_dataset.columns:
            dict_dataset[field] = input_dataset.convert_dtypes().dtypes[field]
        monitoring_schema = _transform_schema_pandas(
            dict_dataset, feature_fields, ground_truth_fields, prediction_fields
        )
        f.close()
    return monitoring_schema


def transform_schema_from_json(
    file_path: str,
    feature_fields: Optional[List[str]] = None,
    ground_truth_fields: Optional[List[str]] = None,
    prediction_fields: Optional[List[str]] = None,
) -> ModelMonitoringSchema:
    """Transform the existing dataset to ModelMonitoringSchema as model monitor
    could accept.

    Args:
        file_path (str):
            Required. The dataset file path.
        feature_fields (List[str]):
            Optional. The input feature fields for given dataset.
            By default all features we find would be the input features.
        ground_truth_fields (List[str]):
            Optional. The ground truth fields for given dataset.
            By default all features we find would be the input features.
        prediction_fields (List[str]):
            Optional. The prediction output field for given dataset.
            By default all features we find would be the input features.
    """
    with tf.io.gfile.GFile(file_path, "r") as f:
        input_dataset = pd.read_json(f, lines=True)
        dict_dataset = dict()
        for field in input_dataset.columns:
            dict_dataset[field] = input_dataset.convert_dtypes().dtypes[field]
        monitoring_schema = _transform_schema_pandas(
            dict_dataset, feature_fields, ground_truth_fields, prediction_fields
        )
        f.close()
    return monitoring_schema
