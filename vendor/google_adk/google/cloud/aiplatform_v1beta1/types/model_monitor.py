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

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import explanation
from google.cloud.aiplatform_v1beta1.types import model_monitoring_spec
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ModelMonitor",
        "ModelMonitoringSchema",
    },
)


class ModelMonitor(proto.Message):
    r"""Vertex AI Model Monitoring Service serves as a central hub
    for the analysis and visualization of data quality and
    performance related to models. ModelMonitor stands as a top
    level resource for overseeing your model monitoring tasks.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tabular_objective (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveSpec.TabularObjective):
            Optional default tabular model monitoring
            objective.

            This field is a member of `oneof`_ ``default_objective``.
        name (str):
            Immutable. Resource name of the ModelMonitor. Format:
            ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}``.
        display_name (str):
            The display name of the ModelMonitor.
            The name can be up to 128 characters long and
            can consist of any UTF-8.
        model_monitoring_target (google.cloud.aiplatform_v1beta1.types.ModelMonitor.ModelMonitoringTarget):
            The entity that is subject to analysis.
            Currently only models in Vertex AI Model
            Registry are supported. If you want to analyze
            the model which is outside the Vertex AI, you
            could register a model in Vertex AI Model
            Registry using just a display name.
        training_dataset (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput):
            Optional training dataset used to train the
            model. It can serve as a reference dataset to
            identify changes in production.
        notification_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringNotificationSpec):
            Optional default notification spec, it can be
            overridden in the ModelMonitoringJob
            notification spec.
        output_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringOutputSpec):
            Optional default monitoring metrics/logs
            export spec, it can be overridden in the
            ModelMonitoringJob output spec. If not
            specified, a default Google Cloud Storage bucket
            will be created under your project.
        explanation_spec (google.cloud.aiplatform_v1beta1.types.ExplanationSpec):
            Optional model explanation spec. It is used
            for feature attribution monitoring.
        model_monitoring_schema (google.cloud.aiplatform_v1beta1.types.ModelMonitoringSchema):
            Monitoring Schema is to specify the model's
            features, prediction outputs and ground truth
            properties. It is used to extract pertinent data
            from the dataset and to process features based
            on their properties. Make sure that the schema
            aligns with your dataset, if it does not, we
            will be unable to extract data from the dataset.
            It is required for most models, but optional for
            Vertex AI AutoML Tables unless the schem
            information is not available.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key spec for a
            ModelMonitor. If set, this ModelMonitor and all
            sub-resources of this ModelMonitor will be
            secured by this key.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this ModelMonitor
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this ModelMonitor
            was updated most recently.
        satisfies_pzs (bool):
            Output only. Reserved for future use.
        satisfies_pzi (bool):
            Output only. Reserved for future use.
    """

    class ModelMonitoringTarget(proto.Message):
        r"""The monitoring target refers to the entity that is subject to
        analysis. e.g. Vertex AI Model version.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            vertex_model (google.cloud.aiplatform_v1beta1.types.ModelMonitor.ModelMonitoringTarget.VertexModelSource):
                Model in Vertex AI Model Registry.

                This field is a member of `oneof`_ ``source``.
        """

        class VertexModelSource(proto.Message):
            r"""Model in Vertex AI Model Registry.

            Attributes:
                model (str):
                    Model resource name. Format:

                    projects/{project}/locations/{location}/models/{model}.
                model_version_id (str):
                    Model version id.
            """

            model: str = proto.Field(
                proto.STRING,
                number=1,
            )
            model_version_id: str = proto.Field(
                proto.STRING,
                number=2,
            )

        vertex_model: "ModelMonitor.ModelMonitoringTarget.VertexModelSource" = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="source",
                message="ModelMonitor.ModelMonitoringTarget.VertexModelSource",
            )
        )

    tabular_objective: (
        model_monitoring_spec.ModelMonitoringObjectiveSpec.TabularObjective
    ) = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="default_objective",
        message=model_monitoring_spec.ModelMonitoringObjectiveSpec.TabularObjective,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    model_monitoring_target: ModelMonitoringTarget = proto.Field(
        proto.MESSAGE,
        number=3,
        message=ModelMonitoringTarget,
    )
    training_dataset: model_monitoring_spec.ModelMonitoringInput = proto.Field(
        proto.MESSAGE,
        number=10,
        message=model_monitoring_spec.ModelMonitoringInput,
    )
    notification_spec: model_monitoring_spec.ModelMonitoringNotificationSpec = (
        proto.Field(
            proto.MESSAGE,
            number=12,
            message=model_monitoring_spec.ModelMonitoringNotificationSpec,
        )
    )
    output_spec: model_monitoring_spec.ModelMonitoringOutputSpec = proto.Field(
        proto.MESSAGE,
        number=13,
        message=model_monitoring_spec.ModelMonitoringOutputSpec,
    )
    explanation_spec: explanation.ExplanationSpec = proto.Field(
        proto.MESSAGE,
        number=16,
        message=explanation.ExplanationSpec,
    )
    model_monitoring_schema: "ModelMonitoringSchema" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="ModelMonitoringSchema",
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=5,
        message=gca_encryption_spec.EncryptionSpec,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    satisfies_pzs: bool = proto.Field(
        proto.BOOL,
        number=17,
    )
    satisfies_pzi: bool = proto.Field(
        proto.BOOL,
        number=18,
    )


class ModelMonitoringSchema(proto.Message):
    r"""The Model Monitoring Schema definition.

    Attributes:
        feature_fields (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringSchema.FieldSchema]):
            Feature names of the model. Vertex AI will try to match the
            features from your dataset as follows:

            - For 'csv' files, the header names are required, and we
              will extract the corresponding feature values when the
              header names align with the feature names.
            - For 'jsonl' files, we will extract the corresponding
              feature values if the key names match the feature names.
              Note: Nested features are not supported, so please ensure
              your features are flattened. Ensure the feature values are
              scalar or an array of scalars.
            - For 'bigquery' dataset, we will extract the corresponding
              feature values if the column names match the feature
              names. Note: The column type can be a scalar or an array
              of scalars. STRUCT or JSON types are not supported. You
              may use SQL queries to select or aggregate the relevant
              features from your original table. However, ensure that
              the 'schema' of the query results meets our requirements.
            - For the Vertex AI Endpoint Request Response Logging table
              or Vertex AI Batch Prediction Job results. If the
              [instance_type][google.cloud.aiplatform.v1beta1.ModelMonitoringSchema.instance_type]
              is an array, ensure that the sequence in
              [feature_fields][google.cloud.aiplatform.v1beta1.ModelMonitoringSchema.feature_fields]
              matches the order of features in the prediction instance.
              We will match the feature with the array in the order
              specified in [feature_fields].
        prediction_fields (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringSchema.FieldSchema]):
            Prediction output names of the model. The requirements are
            the same as the
            [feature_fields][google.cloud.aiplatform.v1beta1.ModelMonitoringSchema.feature_fields].
            For AutoML Tables, the prediction output name presented in
            schema will be: ``predicted_{target_column}``, the
            ``target_column`` is the one you specified when you train
            the model. For Prediction output drift analysis:

            - AutoML Classification, the distribution of the argmax
              label will be analyzed.
            - AutoML Regression, the distribution of the value will be
              analyzed.
        ground_truth_fields (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringSchema.FieldSchema]):
            Target /ground truth names of the model.
    """

    class FieldSchema(proto.Message):
        r"""Schema field definition.

        Attributes:
            name (str):
                Field name.
            data_type (str):
                Supported data types are: ``float`` ``integer`` ``boolean``
                ``string`` ``categorical``
            repeated (bool):
                Describes if the schema field is an array of
                given data type.
        """

        name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        data_type: str = proto.Field(
            proto.STRING,
            number=2,
        )
        repeated: bool = proto.Field(
            proto.BOOL,
            number=3,
        )

    feature_fields: MutableSequence[FieldSchema] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=FieldSchema,
    )
    prediction_fields: MutableSequence[FieldSchema] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=FieldSchema,
    )
    ground_truth_fields: MutableSequence[FieldSchema] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=FieldSchema,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
