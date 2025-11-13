# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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

from typing import Dict, Optional, Sequence, Tuple, Union

from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import datasets
from google.cloud.aiplatform.datasets import _datasources
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import schema
from google.cloud.aiplatform import utils


class TextDataset(datasets._Dataset):
    """A managed text dataset resource for Vertex AI.

    Use this class to work with a managed text dataset. To create a managed
    text dataset, you need a datasource file in CSV format and a schema file in
    YAML format. A schema is optional for a custom model. The CSV file and the
    schema are accessed in Cloud Storage buckets.

    Use text data for the following objectives:

    * Classification. For more information, see
    [Prepare text training data for classification](https://cloud.google.com/vertex-ai/docs/text-data/classification/prepare-data).
    * Entity extraction. For more information, see
    [Prepare text training data for entity extraction](https://cloud.google.com/vertex-ai/docs/text-data/entity-extraction/prepare-data).
    * Sentiment analysis. For more information, see
    [Prepare text training data for sentiment analysis](Prepare text training data for sentiment analysis).

    The following code shows you how to create and import a text dataset with
    a CSV datasource file and a YAML schema file. The schema file you use
    depends on whether your text dataset is used for single-label
    classification, multi-label classification, or object detection.

    ```py
    my_dataset = aiplatform.TextDataset.create(
        display_name="my-text-dataset",
        gcs_source=['gs://path/to/my/text-dataset.csv'],
        import_schema_uri=['gs://path/to/my/schema.yaml'],
    )
    ```
    """

    _supported_metadata_schema_uris: Optional[Tuple[str]] = (
        schema.dataset.metadata.text,
    )

    @classmethod
    def create(
        cls,
        display_name: Optional[str] = None,
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        import_schema_uri: Optional[str] = None,
        data_item_labels: Optional[Dict] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "TextDataset":
        """Creates a new text dataset.

        Optionally imports data into this dataset when a source and
        `import_schema_uri` are passed in. The following is an example of how
        this method is used:

        ```py
        ds = aiplatform.TextDataset.create(
                display_name='my-dataset',
                gcs_source='gs://my-bucket/dataset.csv',
                import_schema_uri=aiplatform.schema.dataset.ioformat.text.multi_label_classification
            )
        ```

        Args:
            display_name (str):
                Optional. The user-defined name of the dataset. The name must
                contain 128 or fewer UTF-8 characters.
            gcs_source (Union[str, Sequence[str]]):
                Optional. The URI to one or more Google Cloud Storage buckets
                that contain your datasets. For example, `str:
                "gs://bucket/file.csv"` or `Sequence[str]:
                ["gs://bucket/file1.csv", "gs://bucket/file2.csv"]`.
            import_schema_uri (str):
                Optional. A URI for a YAML file stored in Cloud Storage that
                describes the import schema used to validate the
                dataset. The schema is an
                [OpenAPI 3.0.2 Schema](https://tinyurl.com/y538mdwt) object.
            data_item_labels (Dict):
                Optional. A dictionary of label information. Each dictionary
                item contains a label and a label key. Each item in the dataset
                includes one dictionary of label information. If a data item is
                added or merged into a dataset, and that data item contains an
                image that's identical to an image that’s already in the
                dataset, then the data items are merged. If two identical labels
                are detected during the merge, each with a different label key,
                then one of the label and label key dictionary items is randomly
                chosen to be into the merged data item. Data items are
                compared using their binary data (bytes), not on their content.
                If annotation labels are referenced in a schema specified by the
                `import_schema_url` parameter, then the labels in the
                `data_item_labels` dictionary are overriden by the annotations.
            project (str):
                Optional. The name of the Google Cloud project to which this
                `TextDataset` is uploaded. This overrides the project that
                was set by `aiplatform.init`.
            location (str):
                Optional. The Google Cloud region where this dataset is uploaded. This
                region overrides the region that was set by `aiplatform.init`.
            credentials (auth_credentials.Credentials):
                Optional. The credentials that are used to upload the `TextDataset`.
                These credentials override the credentials set by
                `aiplatform.init`.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings that contain metadata that's sent with the request.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your
                Vertex AI Tensorboards. The maximum length of a key and of a
                value is 64 unicode characters. Labels and keys can contain only
                lowercase letters, numeric characters, underscores, and dashes.
                International characters are allowed. No more than 64 user
                labels can be associated with one Tensorboard (system labels are
                excluded). For more information and examples of using labels, see
                [Using labels to organize Google Cloud Platform resources](https://goo.gl/xmQnxf).
                System reserved label keys are prefixed with
                `aiplatform.googleapis.com/` and are immutable.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key that's used to protect the dataset. The
                format of the key is
                `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key`.
                The key needs to be in the same region as where the compute
                resource is created.

                If `encryption_spec_key_name` is set, this `TextDataset` and
                all of its sub-resources are secured by this key.

                This `encryption_spec_key_name` overrides the
                `encryption_spec_key_name` set by `aiplatform.init`.
            sync (bool):
                If `true`, the `create` method creates a text dataset
                synchronously. If `false`, the `create` method creates a text
                dataset asynchronously.
            create_request_timeout (float):
                Optional. The number of seconds for the timeout of the create
                request.

        Returns:
            text_dataset (TextDataset):
                An instantiated representation of the managed `TextDataset`
                resource.
        """
        if not display_name:
            display_name = cls._generate_display_name()
        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        metadata_schema_uri = schema.dataset.metadata.text

        datasource = _datasources.create_datasource(
            metadata_schema_uri=metadata_schema_uri,
            import_schema_uri=import_schema_uri,
            gcs_source=gcs_source,
            data_item_labels=data_item_labels,
        )

        return cls._create_and_import(
            api_client=api_client,
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            display_name=display_name,
            metadata_schema_uri=metadata_schema_uri,
            datasource=datasource,
            project=project or initializer.global_config.project,
            location=location or initializer.global_config.location,
            credentials=credentials or initializer.global_config.credentials,
            request_metadata=request_metadata,
            labels=labels,
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
            sync=sync,
            create_request_timeout=create_request_timeout,
        )
