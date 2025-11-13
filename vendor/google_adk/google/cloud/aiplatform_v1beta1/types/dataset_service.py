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

from google.cloud.aiplatform_v1beta1.types import annotation
from google.cloud.aiplatform_v1beta1.types import content
from google.cloud.aiplatform_v1beta1.types import data_item as gca_data_item
from google.cloud.aiplatform_v1beta1.types import dataset as gca_dataset
from google.cloud.aiplatform_v1beta1.types import dataset_version as gca_dataset_version
from google.cloud.aiplatform_v1beta1.types import operation
from google.cloud.aiplatform_v1beta1.types import saved_query as gca_saved_query
from google.cloud.aiplatform_v1beta1.types import tool
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateDatasetRequest",
        "CreateDatasetOperationMetadata",
        "GetDatasetRequest",
        "UpdateDatasetRequest",
        "UpdateDatasetVersionRequest",
        "ListDatasetsRequest",
        "ListDatasetsResponse",
        "DeleteDatasetRequest",
        "ImportDataRequest",
        "ImportDataResponse",
        "ImportDataOperationMetadata",
        "ExportDataRequest",
        "ExportDataResponse",
        "ExportDataOperationMetadata",
        "CreateDatasetVersionRequest",
        "CreateDatasetVersionOperationMetadata",
        "DeleteDatasetVersionRequest",
        "GetDatasetVersionRequest",
        "ListDatasetVersionsRequest",
        "ListDatasetVersionsResponse",
        "RestoreDatasetVersionRequest",
        "RestoreDatasetVersionOperationMetadata",
        "ListDataItemsRequest",
        "ListDataItemsResponse",
        "SearchDataItemsRequest",
        "SearchDataItemsResponse",
        "DataItemView",
        "ListSavedQueriesRequest",
        "ListSavedQueriesResponse",
        "DeleteSavedQueryRequest",
        "GetAnnotationSpecRequest",
        "ListAnnotationsRequest",
        "ListAnnotationsResponse",
        "AssessDataRequest",
        "AssessDataResponse",
        "AssessDataOperationMetadata",
        "GeminiTemplateConfig",
        "GeminiRequestReadConfig",
        "GeminiExample",
        "AssembleDataRequest",
        "AssembleDataResponse",
        "AssembleDataOperationMetadata",
    },
)


class CreateDatasetRequest(proto.Message):
    r"""Request message for
    [DatasetService.CreateDataset][google.cloud.aiplatform.v1beta1.DatasetService.CreateDataset].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            Dataset in. Format:
            ``projects/{project}/locations/{location}``
        dataset (google.cloud.aiplatform_v1beta1.types.Dataset):
            Required. The Dataset to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    dataset: gca_dataset.Dataset = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_dataset.Dataset,
    )


class CreateDatasetOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [DatasetService.CreateDataset][google.cloud.aiplatform.v1beta1.DatasetService.CreateDataset].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetDatasetRequest(proto.Message):
    r"""Request message for
    [DatasetService.GetDataset][google.cloud.aiplatform.v1beta1.DatasetService.GetDataset].

    Attributes:
        name (str):
            Required. The name of the Dataset resource.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateDatasetRequest(proto.Message):
    r"""Request message for
    [DatasetService.UpdateDataset][google.cloud.aiplatform.v1beta1.DatasetService.UpdateDataset].

    Attributes:
        dataset (google.cloud.aiplatform_v1beta1.types.Dataset):
            Required. The Dataset which replaces the
            resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The update mask applies to the resource. For the
            ``FieldMask`` definition, see
            [google.protobuf.FieldMask][google.protobuf.FieldMask].
            Updatable fields:

            - ``display_name``
            - ``description``
            - ``labels``
    """

    dataset: gca_dataset.Dataset = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_dataset.Dataset,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateDatasetVersionRequest(proto.Message):
    r"""Request message for
    [DatasetService.UpdateDatasetVersion][google.cloud.aiplatform.v1beta1.DatasetService.UpdateDatasetVersion].

    Attributes:
        dataset_version (google.cloud.aiplatform_v1beta1.types.DatasetVersion):
            Required. The DatasetVersion which replaces
            the resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The update mask applies to the resource. For the
            ``FieldMask`` definition, see
            [google.protobuf.FieldMask][google.protobuf.FieldMask].
            Updatable fields:

            - ``display_name``
    """

    dataset_version: gca_dataset_version.DatasetVersion = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_dataset_version.DatasetVersion,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class ListDatasetsRequest(proto.Message):
    r"""Request message for
    [DatasetService.ListDatasets][google.cloud.aiplatform.v1beta1.DatasetService.ListDatasets].

    Attributes:
        parent (str):
            Required. The name of the Dataset's parent resource. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            An expression for filtering the results of the request. For
            field names both snake_case and camelCase are supported.

            - ``display_name``: supports = and !=
            - ``metadata_schema_uri``: supports = and !=
            - ``labels`` supports general map functions that is:

              - ``labels.key=value`` - key:value equality
              - \`labels.key:\* or labels:key - key existence
              - A key including a space must be quoted.
                ``labels."a key"``.

            Some examples:

            - ``displayName="myDisplayName"``
            - ``labels.myKey="myValue"``
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order. Use "desc" after a field name for
            descending. Supported fields:

            - ``display_name``
            - ``create_time``
            - ``update_time``
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListDatasetsResponse(proto.Message):
    r"""Response message for
    [DatasetService.ListDatasets][google.cloud.aiplatform.v1beta1.DatasetService.ListDatasets].

    Attributes:
        datasets (MutableSequence[google.cloud.aiplatform_v1beta1.types.Dataset]):
            A list of Datasets that matches the specified
            filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    datasets: MutableSequence[gca_dataset.Dataset] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_dataset.Dataset,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteDatasetRequest(proto.Message):
    r"""Request message for
    [DatasetService.DeleteDataset][google.cloud.aiplatform.v1beta1.DatasetService.DeleteDataset].

    Attributes:
        name (str):
            Required. The resource name of the Dataset to delete.
            Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ImportDataRequest(proto.Message):
    r"""Request message for
    [DatasetService.ImportData][google.cloud.aiplatform.v1beta1.DatasetService.ImportData].

    Attributes:
        name (str):
            Required. The name of the Dataset resource. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        import_configs (MutableSequence[google.cloud.aiplatform_v1beta1.types.ImportDataConfig]):
            Required. The desired input locations. The
            contents of all input locations will be imported
            in one batch.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    import_configs: MutableSequence[gca_dataset.ImportDataConfig] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gca_dataset.ImportDataConfig,
    )


class ImportDataResponse(proto.Message):
    r"""Response message for
    [DatasetService.ImportData][google.cloud.aiplatform.v1beta1.DatasetService.ImportData].

    """


class ImportDataOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [DatasetService.ImportData][google.cloud.aiplatform.v1beta1.DatasetService.ImportData].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class ExportDataRequest(proto.Message):
    r"""Request message for
    [DatasetService.ExportData][google.cloud.aiplatform.v1beta1.DatasetService.ExportData].

    Attributes:
        name (str):
            Required. The name of the Dataset resource. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        export_config (google.cloud.aiplatform_v1beta1.types.ExportDataConfig):
            Required. The desired output location.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    export_config: gca_dataset.ExportDataConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_dataset.ExportDataConfig,
    )


class ExportDataResponse(proto.Message):
    r"""Response message for
    [DatasetService.ExportData][google.cloud.aiplatform.v1beta1.DatasetService.ExportData].

    Attributes:
        exported_files (MutableSequence[str]):
            All of the files that are exported in this export operation.
            For custom code training export, only three (training,
            validation and test) Cloud Storage paths in wildcard format
            are populated (for example, gs://.../training-\*).
    """

    exported_files: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )


class ExportDataOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [DatasetService.ExportData][google.cloud.aiplatform.v1beta1.DatasetService.ExportData].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
        gcs_output_directory (str):
            A Google Cloud Storage directory which path
            ends with '/'. The exported data is stored in
            the directory.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    gcs_output_directory: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateDatasetVersionRequest(proto.Message):
    r"""Request message for
    [DatasetService.CreateDatasetVersion][google.cloud.aiplatform.v1beta1.DatasetService.CreateDatasetVersion].

    Attributes:
        parent (str):
            Required. The name of the Dataset resource. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        dataset_version (google.cloud.aiplatform_v1beta1.types.DatasetVersion):
            Required. The version to be created. The same
            CMEK policies with the original Dataset will be
            applied the dataset version. So here we don't
            need to specify the EncryptionSpecType here.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    dataset_version: gca_dataset_version.DatasetVersion = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_dataset_version.DatasetVersion,
    )


class CreateDatasetVersionOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [DatasetService.CreateDatasetVersion][google.cloud.aiplatform.v1beta1.DatasetService.CreateDatasetVersion].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class DeleteDatasetVersionRequest(proto.Message):
    r"""Request message for
    [DatasetService.DeleteDatasetVersion][google.cloud.aiplatform.v1beta1.DatasetService.DeleteDatasetVersion].

    Attributes:
        name (str):
            Required. The resource name of the Dataset version to
            delete. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}/datasetVersions/{dataset_version}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetDatasetVersionRequest(proto.Message):
    r"""Request message for
    [DatasetService.GetDatasetVersion][google.cloud.aiplatform.v1beta1.DatasetService.GetDatasetVersion].

    Attributes:
        name (str):
            Required. The resource name of the Dataset version to
            delete. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}/datasetVersions/{dataset_version}``
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class ListDatasetVersionsRequest(proto.Message):
    r"""Request message for
    [DatasetService.ListDatasetVersions][google.cloud.aiplatform.v1beta1.DatasetService.ListDatasetVersions].

    Attributes:
        parent (str):
            Required. The resource name of the Dataset to list
            DatasetVersions from. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        filter (str):
            Optional. The standard list filter.
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Mask specifying which fields to
            read.
        order_by (str):
            Optional. A comma-separated list of fields to
            order by, sorted in ascending order. Use "desc"
            after a field name for descending.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListDatasetVersionsResponse(proto.Message):
    r"""Response message for
    [DatasetService.ListDatasetVersions][google.cloud.aiplatform.v1beta1.DatasetService.ListDatasetVersions].

    Attributes:
        dataset_versions (MutableSequence[google.cloud.aiplatform_v1beta1.types.DatasetVersion]):
            A list of DatasetVersions that matches the
            specified filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    dataset_versions: MutableSequence[gca_dataset_version.DatasetVersion] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=gca_dataset_version.DatasetVersion,
        )
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class RestoreDatasetVersionRequest(proto.Message):
    r"""Request message for
    [DatasetService.RestoreDatasetVersion][google.cloud.aiplatform.v1beta1.DatasetService.RestoreDatasetVersion].

    Attributes:
        name (str):
            Required. The name of the DatasetVersion resource. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}/datasetVersions/{dataset_version}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class RestoreDatasetVersionOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [DatasetService.RestoreDatasetVersion][google.cloud.aiplatform.v1beta1.DatasetService.RestoreDatasetVersion].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class ListDataItemsRequest(proto.Message):
    r"""Request message for
    [DatasetService.ListDataItems][google.cloud.aiplatform.v1beta1.DatasetService.ListDataItems].

    Attributes:
        parent (str):
            Required. The resource name of the Dataset to list DataItems
            from. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        filter (str):
            The standard list filter.
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
        order_by (str):
            A comma-separated list of fields to order by,
            sorted in ascending order. Use "desc" after a
            field name for descending.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListDataItemsResponse(proto.Message):
    r"""Response message for
    [DatasetService.ListDataItems][google.cloud.aiplatform.v1beta1.DatasetService.ListDataItems].

    Attributes:
        data_items (MutableSequence[google.cloud.aiplatform_v1beta1.types.DataItem]):
            A list of DataItems that matches the
            specified filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    data_items: MutableSequence[gca_data_item.DataItem] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_data_item.DataItem,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SearchDataItemsRequest(proto.Message):
    r"""Request message for
    [DatasetService.SearchDataItems][google.cloud.aiplatform.v1beta1.DatasetService.SearchDataItems].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        order_by_data_item (str):
            A comma-separated list of data item fields to
            order by, sorted in ascending order. Use "desc"
            after a field name for descending.

            This field is a member of `oneof`_ ``order``.
        order_by_annotation (google.cloud.aiplatform_v1beta1.types.SearchDataItemsRequest.OrderByAnnotation):
            Expression that allows ranking results based
            on annotation's property.

            This field is a member of `oneof`_ ``order``.
        dataset (str):
            Required. The resource name of the Dataset from which to
            search DataItems. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        saved_query (str):
            The resource name of a SavedQuery(annotation set in UI).
            Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}/savedQueries/{saved_query}``
            All of the search will be done in the context of this
            SavedQuery.
        data_labeling_job (str):
            The resource name of a DataLabelingJob. Format:
            ``projects/{project}/locations/{location}/dataLabelingJobs/{data_labeling_job}``
            If this field is set, all of the search will be done in the
            context of this DataLabelingJob.
        data_item_filter (str):
            An expression for filtering the DataItem that will be
            returned.

            - ``data_item_id`` - for = or !=.
            - ``labeled`` - for = or !=.
            - ``has_annotation(ANNOTATION_SPEC_ID)`` - true only for
              DataItem that have at least one annotation with
              annotation_spec_id = ``ANNOTATION_SPEC_ID`` in the context
              of SavedQuery or DataLabelingJob.

            For example:

            - ``data_item=1``
            - ``has_annotation(5)``
        annotations_filter (str):
            An expression for filtering the Annotations that will be
            returned per DataItem.

            - ``annotation_spec_id`` - for = or !=.
        annotation_filters (MutableSequence[str]):
            An expression that specifies what Annotations will be
            returned per DataItem. Annotations satisfied either of the
            conditions will be returned.

            - ``annotation_spec_id`` - for = or !=. Must specify
              ``saved_query_id=`` - saved query id that annotations
              should belong to.
        field_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields of
            [DataItemView][google.cloud.aiplatform.v1beta1.DataItemView]
            to read.
        annotations_limit (int):
            If set, only up to this many of Annotations
            will be returned per DataItemView. The maximum
            value is 1000. If not set, the maximum value
            will be used.
        page_size (int):
            Requested page size. Server may return fewer
            results than requested. Default and maximum page
            size is 100.
        order_by (str):
            A comma-separated list of fields to order by,
            sorted in ascending order. Use "desc" after a
            field name for descending.
        page_token (str):
            A token identifying a page of results for the server to
            return Typically obtained via
            [SearchDataItemsResponse.next_page_token][google.cloud.aiplatform.v1beta1.SearchDataItemsResponse.next_page_token]
            of the previous
            [DatasetService.SearchDataItems][google.cloud.aiplatform.v1beta1.DatasetService.SearchDataItems]
            call.
    """

    class OrderByAnnotation(proto.Message):
        r"""Expression that allows ranking results based on annotation's
        property.

        Attributes:
            saved_query (str):
                Required. Saved query of the Annotation. Only
                Annotations belong to this saved query will be
                considered for ordering.
            order_by (str):
                A comma-separated list of annotation fields to order by,
                sorted in ascending order. Use "desc" after a field name for
                descending. Must also specify saved_query.
        """

        saved_query: str = proto.Field(
            proto.STRING,
            number=1,
        )
        order_by: str = proto.Field(
            proto.STRING,
            number=2,
        )

    order_by_data_item: str = proto.Field(
        proto.STRING,
        number=12,
        oneof="order",
    )
    order_by_annotation: OrderByAnnotation = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="order",
        message=OrderByAnnotation,
    )
    dataset: str = proto.Field(
        proto.STRING,
        number=1,
    )
    saved_query: str = proto.Field(
        proto.STRING,
        number=2,
    )
    data_labeling_job: str = proto.Field(
        proto.STRING,
        number=3,
    )
    data_item_filter: str = proto.Field(
        proto.STRING,
        number=4,
    )
    annotations_filter: str = proto.Field(
        proto.STRING,
        number=5,
    )
    annotation_filters: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )
    field_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=6,
        message=field_mask_pb2.FieldMask,
    )
    annotations_limit: int = proto.Field(
        proto.INT32,
        number=7,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=8,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=9,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=10,
    )


class SearchDataItemsResponse(proto.Message):
    r"""Response message for
    [DatasetService.SearchDataItems][google.cloud.aiplatform.v1beta1.DatasetService.SearchDataItems].

    Attributes:
        data_item_views (MutableSequence[google.cloud.aiplatform_v1beta1.types.DataItemView]):
            The DataItemViews read.
        next_page_token (str):
            A token to retrieve next page of results. Pass to
            [SearchDataItemsRequest.page_token][google.cloud.aiplatform.v1beta1.SearchDataItemsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    data_item_views: MutableSequence["DataItemView"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="DataItemView",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DataItemView(proto.Message):
    r"""A container for a single DataItem and Annotations on it.

    Attributes:
        data_item (google.cloud.aiplatform_v1beta1.types.DataItem):
            The DataItem.
        annotations (MutableSequence[google.cloud.aiplatform_v1beta1.types.Annotation]):
            The Annotations on the DataItem. If too many Annotations
            should be returned for the DataItem, this field will be
            truncated per annotations_limit in request. If it was, then
            the has_truncated_annotations will be set to true.
        has_truncated_annotations (bool):
            True if and only if the Annotations field has been
            truncated. It happens if more Annotations for this DataItem
            met the request's annotation_filter than are allowed to be
            returned by annotations_limit. Note that if Annotations
            field is not being returned due to field mask, then this
            field will not be set to true no matter how many Annotations
            are there.
    """

    data_item: gca_data_item.DataItem = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_data_item.DataItem,
    )
    annotations: MutableSequence[annotation.Annotation] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=annotation.Annotation,
    )
    has_truncated_annotations: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class ListSavedQueriesRequest(proto.Message):
    r"""Request message for
    [DatasetService.ListSavedQueries][google.cloud.aiplatform.v1beta1.DatasetService.ListSavedQueries].

    Attributes:
        parent (str):
            Required. The resource name of the Dataset to list
            SavedQueries from. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        filter (str):
            The standard list filter.
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
        order_by (str):
            A comma-separated list of fields to order by,
            sorted in ascending order. Use "desc" after a
            field name for descending.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListSavedQueriesResponse(proto.Message):
    r"""Response message for
    [DatasetService.ListSavedQueries][google.cloud.aiplatform.v1beta1.DatasetService.ListSavedQueries].

    Attributes:
        saved_queries (MutableSequence[google.cloud.aiplatform_v1beta1.types.SavedQuery]):
            A list of SavedQueries that match the
            specified filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    saved_queries: MutableSequence[gca_saved_query.SavedQuery] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_saved_query.SavedQuery,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteSavedQueryRequest(proto.Message):
    r"""Request message for
    [DatasetService.DeleteSavedQuery][google.cloud.aiplatform.v1beta1.DatasetService.DeleteSavedQuery].

    Attributes:
        name (str):
            Required. The resource name of the SavedQuery to delete.
            Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}/savedQueries/{saved_query}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetAnnotationSpecRequest(proto.Message):
    r"""Request message for
    [DatasetService.GetAnnotationSpec][google.cloud.aiplatform.v1beta1.DatasetService.GetAnnotationSpec].

    Attributes:
        name (str):
            Required. The name of the AnnotationSpec resource. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}/annotationSpecs/{annotation_spec}``
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class ListAnnotationsRequest(proto.Message):
    r"""Request message for
    [DatasetService.ListAnnotations][google.cloud.aiplatform.v1beta1.DatasetService.ListAnnotations].

    Attributes:
        parent (str):
            Required. The resource name of the DataItem to list
            Annotations from. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}/dataItems/{data_item}``
        filter (str):
            The standard list filter.
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
        order_by (str):
            A comma-separated list of fields to order by,
            sorted in ascending order. Use "desc" after a
            field name for descending.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListAnnotationsResponse(proto.Message):
    r"""Response message for
    [DatasetService.ListAnnotations][google.cloud.aiplatform.v1beta1.DatasetService.ListAnnotations].

    Attributes:
        annotations (MutableSequence[google.cloud.aiplatform_v1beta1.types.Annotation]):
            A list of Annotations that matches the
            specified filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    annotations: MutableSequence[annotation.Annotation] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=annotation.Annotation,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class AssessDataRequest(proto.Message):
    r"""Request message for
    [DatasetService.AssessData][google.cloud.aiplatform.v1beta1.DatasetService.AssessData].
    Used only for MULTIMODAL datasets.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tuning_validation_assessment_config (google.cloud.aiplatform_v1beta1.types.AssessDataRequest.TuningValidationAssessmentConfig):
            Optional. Configuration for the tuning
            validation assessment.

            This field is a member of `oneof`_ ``assessment_config``.
        tuning_resource_usage_assessment_config (google.cloud.aiplatform_v1beta1.types.AssessDataRequest.TuningResourceUsageAssessmentConfig):
            Optional. Configuration for the tuning
            resource usage assessment.

            This field is a member of `oneof`_ ``assessment_config``.
        batch_prediction_validation_assessment_config (google.cloud.aiplatform_v1beta1.types.AssessDataRequest.BatchPredictionValidationAssessmentConfig):
            Optional. Configuration for the batch
            prediction validation assessment.

            This field is a member of `oneof`_ ``assessment_config``.
        batch_prediction_resource_usage_assessment_config (google.cloud.aiplatform_v1beta1.types.AssessDataRequest.BatchPredictionResourceUsageAssessmentConfig):
            Optional. Configuration for the batch
            prediction resource usage assessment.

            This field is a member of `oneof`_ ``assessment_config``.
        name (str):
            Required. The name of the Dataset resource. Used only for
            MULTIMODAL datasets. Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        gemini_request_read_config (google.cloud.aiplatform_v1beta1.types.GeminiRequestReadConfig):
            Optional. The Gemini request read config for
            the dataset.
    """

    class TuningValidationAssessmentConfig(proto.Message):
        r"""Configuration for the tuning validation assessment.

        Attributes:
            model_name (str):
                Required. The name of the model used for
                tuning.
            dataset_usage (google.cloud.aiplatform_v1beta1.types.AssessDataRequest.TuningValidationAssessmentConfig.DatasetUsage):
                Required. The dataset usage (e.g.
                training/validation).
        """

        class DatasetUsage(proto.Enum):
            r"""The dataset usage (e.g. training/validation).

            Values:
                DATASET_USAGE_UNSPECIFIED (0):
                    Default value. Should not be used.
                SFT_TRAINING (1):
                    Supervised fine-tuning training dataset.
                SFT_VALIDATION (2):
                    Supervised fine-tuning validation dataset.
            """

            DATASET_USAGE_UNSPECIFIED = 0
            SFT_TRAINING = 1
            SFT_VALIDATION = 2

        model_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        dataset_usage: (
            "AssessDataRequest.TuningValidationAssessmentConfig.DatasetUsage"
        ) = proto.Field(
            proto.ENUM,
            number=2,
            enum="AssessDataRequest.TuningValidationAssessmentConfig.DatasetUsage",
        )

    class TuningResourceUsageAssessmentConfig(proto.Message):
        r"""Configuration for the tuning resource usage assessment.

        Attributes:
            model_name (str):
                Required. The name of the model used for
                tuning.
        """

        model_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class BatchPredictionValidationAssessmentConfig(proto.Message):
        r"""Configuration for the batch prediction validation assessment.

        Attributes:
            model_name (str):
                Required. The name of the model used for
                batch prediction.
        """

        model_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class BatchPredictionResourceUsageAssessmentConfig(proto.Message):
        r"""Configuration for the batch prediction resource usage
        assessment.

        Attributes:
            model_name (str):
                Required. The name of the model used for
                batch prediction.
        """

        model_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    tuning_validation_assessment_config: TuningValidationAssessmentConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="assessment_config",
        message=TuningValidationAssessmentConfig,
    )
    tuning_resource_usage_assessment_config: TuningResourceUsageAssessmentConfig = (
        proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="assessment_config",
            message=TuningResourceUsageAssessmentConfig,
        )
    )
    batch_prediction_validation_assessment_config: (
        BatchPredictionValidationAssessmentConfig
    ) = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="assessment_config",
        message=BatchPredictionValidationAssessmentConfig,
    )
    batch_prediction_resource_usage_assessment_config: (
        BatchPredictionResourceUsageAssessmentConfig
    ) = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="assessment_config",
        message=BatchPredictionResourceUsageAssessmentConfig,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    gemini_request_read_config: "GeminiRequestReadConfig" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="GeminiRequestReadConfig",
    )


class AssessDataResponse(proto.Message):
    r"""Response message for
    [DatasetService.AssessData][google.cloud.aiplatform.v1beta1.DatasetService.AssessData].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tuning_validation_assessment_result (google.cloud.aiplatform_v1beta1.types.AssessDataResponse.TuningValidationAssessmentResult):
            Optional. The result of the tuning validation
            assessment.

            This field is a member of `oneof`_ ``assessment_result``.
        tuning_resource_usage_assessment_result (google.cloud.aiplatform_v1beta1.types.AssessDataResponse.TuningResourceUsageAssessmentResult):
            Optional. The result of the tuning resource
            usage assessment.

            This field is a member of `oneof`_ ``assessment_result``.
        batch_prediction_validation_assessment_result (google.cloud.aiplatform_v1beta1.types.AssessDataResponse.BatchPredictionValidationAssessmentResult):
            Optional. The result of the batch prediction
            validation assessment.

            This field is a member of `oneof`_ ``assessment_result``.
        batch_prediction_resource_usage_assessment_result (google.cloud.aiplatform_v1beta1.types.AssessDataResponse.BatchPredictionResourceUsageAssessmentResult):
            Optional. The result of the batch prediction
            resource usage assessment.

            This field is a member of `oneof`_ ``assessment_result``.
    """

    class TuningValidationAssessmentResult(proto.Message):
        r"""The result of the tuning validation assessment.

        Attributes:
            errors (MutableSequence[str]):
                Optional. A list containing the first
                validation errors.
        """

        errors: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    class TuningResourceUsageAssessmentResult(proto.Message):
        r"""The result of the tuning resource usage assessment.

        Attributes:
            token_count (int):
                Number of tokens in the tuning dataset.
            billable_character_count (int):
                Number of billable tokens in the tuning
                dataset.
        """

        token_count: int = proto.Field(
            proto.INT64,
            number=1,
        )
        billable_character_count: int = proto.Field(
            proto.INT64,
            number=2,
        )

    class BatchPredictionValidationAssessmentResult(proto.Message):
        r"""The result of the batch prediction validation assessment."""

    class BatchPredictionResourceUsageAssessmentResult(proto.Message):
        r"""The result of the batch prediction resource usage assessment.

        Attributes:
            token_count (int):
                Number of tokens in the batch prediction
                dataset.
            audio_token_count (int):
                Number of audio tokens in the batch
                prediction dataset.
        """

        token_count: int = proto.Field(
            proto.INT64,
            number=1,
        )
        audio_token_count: int = proto.Field(
            proto.INT64,
            number=2,
        )

    tuning_validation_assessment_result: TuningValidationAssessmentResult = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="assessment_result",
        message=TuningValidationAssessmentResult,
    )
    tuning_resource_usage_assessment_result: TuningResourceUsageAssessmentResult = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="assessment_result",
            message=TuningResourceUsageAssessmentResult,
        )
    )
    batch_prediction_validation_assessment_result: (
        BatchPredictionValidationAssessmentResult
    ) = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="assessment_result",
        message=BatchPredictionValidationAssessmentResult,
    )
    batch_prediction_resource_usage_assessment_result: (
        BatchPredictionResourceUsageAssessmentResult
    ) = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="assessment_result",
        message=BatchPredictionResourceUsageAssessmentResult,
    )


class AssessDataOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [DatasetService.AssessData][google.cloud.aiplatform.v1beta1.DatasetService.AssessData].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GeminiTemplateConfig(proto.Message):
    r"""Template configuration to create Gemini examples from a
    multimodal dataset.

    Attributes:
        gemini_example (google.cloud.aiplatform_v1beta1.types.GeminiExample):
            Required. The template that will be used for
            assembling the request to use for downstream
            applications.
        field_mapping (MutableMapping[str, str]):
            Required. Map of template parameters to the
            columns in the dataset table.
    """

    gemini_example: "GeminiExample" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="GeminiExample",
    )
    field_mapping: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=2,
    )


class GeminiRequestReadConfig(proto.Message):
    r"""Configuration for how to read Gemini requests from a
    multimodal dataset.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        template_config (google.cloud.aiplatform_v1beta1.types.GeminiTemplateConfig):
            Gemini request template with placeholders.

            This field is a member of `oneof`_ ``read_config``.
        assembled_request_column_name (str):
            Optional. Column name in the dataset table
            that contains already fully assembled Gemini
            requests.

            This field is a member of `oneof`_ ``read_config``.
    """

    template_config: "GeminiTemplateConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="read_config",
        message="GeminiTemplateConfig",
    )
    assembled_request_column_name: str = proto.Field(
        proto.STRING,
        number=4,
        oneof="read_config",
    )


class GeminiExample(proto.Message):
    r"""Format for Gemini examples used for Vertex Multimodal
    datasets.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        model (str):
            Optional. The fully qualified name of the publisher model or
            tuned model endpoint to use.

            Publisher model format:
            ``projects/{project}/locations/{location}/publishers/*/models/*``

            Tuned model endpoint format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
            Required. The content of the current
            conversation with the model.
            For single-turn queries, this is a single
            instance. For multi-turn queries, this is a
            repeated field that contains conversation
            history + latest request.
        system_instruction (google.cloud.aiplatform_v1beta1.types.Content):
            Optional. The user provided system
            instructions for the model. Note: only text
            should be used in parts and content in each part
            will be in a separate paragraph.

            This field is a member of `oneof`_ ``_system_instruction``.
        cached_content (str):
            Optional. The name of the cached content used as context to
            serve the prediction. Note: only used in explicit caching,
            where users can have control over caching (e.g. what content
            to cache) and enjoy guaranteed cost savings. Format:
            ``projects/{project}/locations/{location}/cachedContents/{cachedContent}``
        tools (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tool]):
            Optional. A list of ``Tools`` the model may use to generate
            the next response.

            A ``Tool`` is a piece of code that enables the system to
            interact with external systems to perform an action, or set
            of actions, outside of knowledge and scope of the model.
        tool_config (google.cloud.aiplatform_v1beta1.types.ToolConfig):
            Optional. Tool config. This config is shared
            for all tools provided in the request.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata for the request. It is used for billing
            and reporting only.

            Label keys and values can be no longer than 63
            characters (Unicode codepoints) and can only
            contain lowercase letters, numeric characters,
            underscores, and dashes. International
            characters are allowed. Label values are
            optional. Label keys must start with a letter.
        safety_settings (MutableSequence[google.cloud.aiplatform_v1beta1.types.SafetySetting]):
            Optional. Per request settings for blocking
            unsafe content. Enforced on
            GenerateContentResponse.candidates.
        generation_config (google.cloud.aiplatform_v1beta1.types.GenerationConfig):
            Optional. Generation config.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    contents: MutableSequence[content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=content.Content,
    )
    system_instruction: content.Content = proto.Field(
        proto.MESSAGE,
        number=8,
        optional=True,
        message=content.Content,
    )
    cached_content: str = proto.Field(
        proto.STRING,
        number=9,
    )
    tools: MutableSequence[tool.Tool] = proto.RepeatedField(
        proto.MESSAGE,
        number=6,
        message=tool.Tool,
    )
    tool_config: tool.ToolConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        message=tool.ToolConfig,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=10,
    )
    safety_settings: MutableSequence[content.SafetySetting] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=content.SafetySetting,
    )
    generation_config: content.GenerationConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        message=content.GenerationConfig,
    )


class AssembleDataRequest(proto.Message):
    r"""Request message for
    [DatasetService.AssembleData][google.cloud.aiplatform.v1beta1.DatasetService.AssembleData].
    Used only for MULTIMODAL datasets.

    Attributes:
        name (str):
            Required. The name of the Dataset resource (used only for
            MULTIMODAL datasets). Format:
            ``projects/{project}/locations/{location}/datasets/{dataset}``
        gemini_request_read_config (google.cloud.aiplatform_v1beta1.types.GeminiRequestReadConfig):
            Optional. The read config for the dataset.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    gemini_request_read_config: "GeminiRequestReadConfig" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="GeminiRequestReadConfig",
    )


class AssembleDataResponse(proto.Message):
    r"""Response message for
    [DatasetService.AssembleData][google.cloud.aiplatform.v1beta1.DatasetService.AssembleData].

    Attributes:
        bigquery_destination (str):
            Destination BigQuery table path containing
            the assembled data as a single column.
    """

    bigquery_destination: str = proto.Field(
        proto.STRING,
        number=1,
    )


class AssembleDataOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [DatasetService.AssembleData][google.cloud.aiplatform.v1beta1.DatasetService.AssembleData].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
