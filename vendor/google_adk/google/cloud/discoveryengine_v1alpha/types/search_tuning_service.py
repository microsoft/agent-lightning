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
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1alpha.types import (
    custom_tuning_model,
    import_config,
)

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "ListCustomModelsRequest",
        "ListCustomModelsResponse",
        "TrainCustomModelRequest",
        "TrainCustomModelResponse",
        "TrainCustomModelMetadata",
    },
)


class ListCustomModelsRequest(proto.Message):
    r"""Request message for
    [SearchTuningService.ListCustomModels][google.cloud.discoveryengine.v1alpha.SearchTuningService.ListCustomModels]
    method.

    Attributes:
        data_store (str):
            Required. The resource name of the parent Data Store, such
            as
            ``projects/*/locations/global/collections/default_collection/dataStores/default_data_store``.
            This field is used to identify the data store where to fetch
            the models from.
    """

    data_store: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListCustomModelsResponse(proto.Message):
    r"""Response message for
    [SearchTuningService.ListCustomModels][google.cloud.discoveryengine.v1alpha.SearchTuningService.ListCustomModels]
    method.

    Attributes:
        models (MutableSequence[google.cloud.discoveryengine_v1alpha.types.CustomTuningModel]):
            List of custom tuning models.
    """

    models: MutableSequence[
        custom_tuning_model.CustomTuningModel
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=custom_tuning_model.CustomTuningModel,
    )


class TrainCustomModelRequest(proto.Message):
    r"""Request message for
    [SearchTuningService.TrainCustomModel][google.cloud.discoveryengine.v1alpha.SearchTuningService.TrainCustomModel]
    method.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_training_input (google.cloud.discoveryengine_v1alpha.types.TrainCustomModelRequest.GcsTrainingInput):
            Cloud Storage training input.

            This field is a member of `oneof`_ ``training_input``.
        data_store (str):
            Required. The resource name of the Data Store, such as
            ``projects/*/locations/global/collections/default_collection/dataStores/default_data_store``.
            This field is used to identify the data store where to train
            the models.
        model_type (str):
            Model to be trained. Supported values are:

            - **search-tuning**: Fine tuning the search system based on
              data provided.
        error_config (google.cloud.discoveryengine_v1alpha.types.ImportErrorConfig):
            The desired location of errors incurred
            during the data ingestion and training.
        model_id (str):
            If not provided, a UUID will be generated.
    """

    class GcsTrainingInput(proto.Message):
        r"""Cloud Storage training data input.

        Attributes:
            corpus_data_path (str):
                The Cloud Storage corpus data which could be associated in
                train data. The data path format is
                ``gs://<bucket_to_data>/<jsonl_file_name>``. A newline
                delimited jsonl/ndjson file.

                For search-tuning model, each line should have the \_id,
                title and text. Example:
                ``{"_id": "doc1", title: "relevant doc", "text": "relevant text"}``
            query_data_path (str):
                The gcs query data which could be associated in train data.
                The data path format is
                ``gs://<bucket_to_data>/<jsonl_file_name>``. A newline
                delimited jsonl/ndjson file.

                For search-tuning model, each line should have the \_id and
                text. Example: {"\_id": "query1", "text": "example query"}
            train_data_path (str):
                Cloud Storage training data path whose format should be
                ``gs://<bucket_to_data>/<tsv_file_name>``. The file should
                be in tsv format. Each line should have the doc_id and
                query_id and score (number).

                For search-tuning model, it should have the query-id
                corpus-id score as tsv file header. The score should be a
                number in ``[0, inf+)``. The larger the number is, the more
                relevant the pair is. Example:

                - ``query-id\tcorpus-id\tscore``
                - ``query1\tdoc1\t1``
            test_data_path (str):
                Cloud Storage test data. Same format as train_data_path. If
                not provided, a random 80/20 train/test split will be
                performed on train_data_path.
        """

        corpus_data_path: str = proto.Field(
            proto.STRING,
            number=1,
        )
        query_data_path: str = proto.Field(
            proto.STRING,
            number=2,
        )
        train_data_path: str = proto.Field(
            proto.STRING,
            number=3,
        )
        test_data_path: str = proto.Field(
            proto.STRING,
            number=4,
        )

    gcs_training_input: GcsTrainingInput = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="training_input",
        message=GcsTrainingInput,
    )
    data_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    model_type: str = proto.Field(
        proto.STRING,
        number=3,
    )
    error_config: import_config.ImportErrorConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        message=import_config.ImportErrorConfig,
    )
    model_id: str = proto.Field(
        proto.STRING,
        number=5,
    )


class TrainCustomModelResponse(proto.Message):
    r"""Response of the
    [TrainCustomModelRequest][google.cloud.discoveryengine.v1alpha.TrainCustomModelRequest].
    This message is returned by the
    google.longrunning.Operations.response field.

    Attributes:
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the data.
        error_config (google.cloud.discoveryengine_v1alpha.types.ImportErrorConfig):
            Echoes the destination for the complete
            errors in the request if set.
        model_status (str):
            The trained model status. Possible values are:

            - **bad-data**: The training data quality is bad.
            - **no-improvement**: Tuning didn't improve performance.
              Won't deploy.
            - **in-progress**: Model training job creation is in
              progress.
            - **training**: Model is actively training.
            - **evaluating**: The model is evaluating trained metrics.
            - **indexing**: The model trained metrics are indexing.
            - **ready**: The model is ready for serving.
        metrics (MutableMapping[str, float]):
            The metrics of the trained model.
        model_name (str):
            Fully qualified name of the
            CustomTuningModel.
    """

    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    error_config: import_config.ImportErrorConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        message=import_config.ImportErrorConfig,
    )
    model_status: str = proto.Field(
        proto.STRING,
        number=3,
    )
    metrics: MutableMapping[str, float] = proto.MapField(
        proto.STRING,
        proto.DOUBLE,
        number=4,
    )
    model_name: str = proto.Field(
        proto.STRING,
        number=5,
    )


class TrainCustomModelMetadata(proto.Message):
    r"""Metadata related to the progress of the TrainCustomModel
    operation. This is returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
