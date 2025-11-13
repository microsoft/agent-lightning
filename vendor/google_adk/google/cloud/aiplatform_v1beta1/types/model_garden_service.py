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

from google.cloud.aiplatform_v1beta1.types import io
from google.cloud.aiplatform_v1beta1.types import machine_resources
from google.cloud.aiplatform_v1beta1.types import model as gca_model
from google.cloud.aiplatform_v1beta1.types import operation
from google.cloud.aiplatform_v1beta1.types import publisher_model as gca_publisher_model
from google.cloud.aiplatform_v1beta1.types import service_networking


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "PublisherModelView",
        "GetPublisherModelRequest",
        "ListPublisherModelsRequest",
        "ListPublisherModelsResponse",
        "DeployRequest",
        "DeployPublisherModelRequest",
        "DeployResponse",
        "DeployPublisherModelResponse",
        "DeployOperationMetadata",
        "DeployPublisherModelOperationMetadata",
        "ExportPublisherModelResponse",
        "ExportPublisherModelOperationMetadata",
        "ExportPublisherModelRequest",
        "CheckPublisherModelEulaAcceptanceRequest",
        "AcceptPublisherModelEulaRequest",
        "PublisherModelEulaAcceptance",
    },
)


class PublisherModelView(proto.Enum):
    r"""View enumeration of PublisherModel.

    Values:
        PUBLISHER_MODEL_VIEW_UNSPECIFIED (0):
            The default / unset value. The API will
            default to the BASIC view.
        PUBLISHER_MODEL_VIEW_BASIC (1):
            Include basic metadata about the publisher
            model, but not the full contents.
        PUBLISHER_MODEL_VIEW_FULL (2):
            Include everything.
        PUBLISHER_MODEL_VERSION_VIEW_BASIC (3):
            Include: VersionId, ModelVersionExternalName,
            and SupportedActions.
    """

    PUBLISHER_MODEL_VIEW_UNSPECIFIED = 0
    PUBLISHER_MODEL_VIEW_BASIC = 1
    PUBLISHER_MODEL_VIEW_FULL = 2
    PUBLISHER_MODEL_VERSION_VIEW_BASIC = 3


class GetPublisherModelRequest(proto.Message):
    r"""Request message for
    [ModelGardenService.GetPublisherModel][google.cloud.aiplatform.v1beta1.ModelGardenService.GetPublisherModel]

    Attributes:
        name (str):
            Required. The name of the PublisherModel resource. Format:
            ``publishers/{publisher}/models/{publisher_model}``
        language_code (str):
            Optional. The IETF BCP-47 language code
            representing the language in which the publisher
            model's text information should be written in.
        view (google.cloud.aiplatform_v1beta1.types.PublisherModelView):
            Optional. PublisherModel view specifying
            which fields to read.
        is_hugging_face_model (bool):
            Optional. Boolean indicates whether the
            requested model is a Hugging Face model.
        hugging_face_token (str):
            Optional. Token used to access Hugging Face
            gated models.
        include_equivalent_model_garden_model_deployment_configs (bool):
            Optional. Whether to cnclude the deployment
            configs from the equivalent Model Garden model
            if the requested model is a Hugging Face model.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    language_code: str = proto.Field(
        proto.STRING,
        number=2,
    )
    view: "PublisherModelView" = proto.Field(
        proto.ENUM,
        number=3,
        enum="PublisherModelView",
    )
    is_hugging_face_model: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    hugging_face_token: str = proto.Field(
        proto.STRING,
        number=6,
    )
    include_equivalent_model_garden_model_deployment_configs: bool = proto.Field(
        proto.BOOL,
        number=7,
    )


class ListPublisherModelsRequest(proto.Message):
    r"""Request message for
    [ModelGardenService.ListPublisherModels][google.cloud.aiplatform.v1beta1.ModelGardenService.ListPublisherModels].

    Attributes:
        parent (str):
            Required. The name of the Publisher from which to list the
            PublisherModels. Format: ``publishers/{publisher}``
        filter (str):
            Optional. The standard list filter.
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token. Typically obtained
            via
            [ListPublisherModelsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListPublisherModelsResponse.next_page_token]
            of the previous
            [ModelGardenService.ListPublisherModels][google.cloud.aiplatform.v1beta1.ModelGardenService.ListPublisherModels]
            call.
        view (google.cloud.aiplatform_v1beta1.types.PublisherModelView):
            Optional. PublisherModel view specifying
            which fields to read.
        order_by (str):
            Optional. A comma-separated list of fields to
            order by, sorted in ascending order. Use "desc"
            after a field name for descending.
        language_code (str):
            Optional. The IETF BCP-47 language code
            representing the language in which the publisher
            models' text information should be written in.
            If not set, by default English (en).
        list_all_versions (bool):
            Optional. List all publisher model versions
            if the flag is set to true.
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
    view: "PublisherModelView" = proto.Field(
        proto.ENUM,
        number=5,
        enum="PublisherModelView",
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )
    language_code: str = proto.Field(
        proto.STRING,
        number=7,
    )
    list_all_versions: bool = proto.Field(
        proto.BOOL,
        number=8,
    )


class ListPublisherModelsResponse(proto.Message):
    r"""Response message for
    [ModelGardenService.ListPublisherModels][google.cloud.aiplatform.v1beta1.ModelGardenService.ListPublisherModels].

    Attributes:
        publisher_models (MutableSequence[google.cloud.aiplatform_v1beta1.types.PublisherModel]):
            List of PublisherModels in the requested
            page.
        next_page_token (str):
            A token to retrieve next page of results. Pass to
            [ListPublisherModels.page_token][] to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    publisher_models: MutableSequence[gca_publisher_model.PublisherModel] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=gca_publisher_model.PublisherModel,
        )
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeployRequest(proto.Message):
    r"""Request message for
    [ModelGardenService.Deploy][google.cloud.aiplatform.v1beta1.ModelGardenService.Deploy].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        publisher_model_name (str):
            The Model Garden model to deploy. Format:
            ``publishers/{publisher}/models/{publisher_model}@{version_id}``,
            or
            ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}@001``.

            This field is a member of `oneof`_ ``artifacts``.
        hugging_face_model_id (str):
            The Hugging Face model to deploy. Format: Hugging Face model
            ID like ``google/gemma-2-2b-it``.

            This field is a member of `oneof`_ ``artifacts``.
        custom_model (google.cloud.aiplatform_v1beta1.types.DeployRequest.CustomModel):
            The custom model to deploy from a Google
            Cloud Storage URI.

            This field is a member of `oneof`_ ``artifacts``.
        destination (str):
            Required. The resource name of the Location to deploy the
            model in. Format:
            ``projects/{project}/locations/{location}``
        model_config (google.cloud.aiplatform_v1beta1.types.DeployRequest.ModelConfig):
            Optional. The model config to use for the
            deployment. If not specified, the default model
            config will be used.
        endpoint_config (google.cloud.aiplatform_v1beta1.types.DeployRequest.EndpointConfig):
            Optional. The endpoint config to use for the
            deployment. If not specified, the default
            endpoint config will be used.
        deploy_config (google.cloud.aiplatform_v1beta1.types.DeployRequest.DeployConfig):
            Optional. The deploy config to use for the
            deployment. If not specified, the default deploy
            config will be used.
    """

    class CustomModel(proto.Message):
        r"""The custom model to deploy from model weights in a Google
        Cloud Storage URI or Model Registry model.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            gcs_uri (str):
                Immutable. The Google Cloud Storage URI of
                the custom model, storing weights and config
                files (which can be used to infer the base
                model).

                This field is a member of `oneof`_ ``model_source``.
        """

        gcs_uri: str = proto.Field(
            proto.STRING,
            number=2,
            oneof="model_source",
        )

    class ModelConfig(proto.Message):
        r"""The model config to use for the deployment.

        Attributes:
            accept_eula (bool):
                Optional. Whether the user accepts the End
                User License Agreement (EULA) for the model.
            hugging_face_access_token (str):
                Optional. The Hugging Face read access token
                used to access the model artifacts of gated
                models.
            hugging_face_cache_enabled (bool):
                Optional. If true, the model will deploy with
                a cached version instead of directly downloading
                the model artifacts from Hugging Face. This is
                suitable for VPC-SC users with limited internet
                access.
            model_display_name (str):
                Optional. The user-specified display name of
                the uploaded model. If not set, a default name
                will be used.
            container_spec (google.cloud.aiplatform_v1beta1.types.ModelContainerSpec):
                Optional. The specification of the container
                that is to be used when deploying. If not set,
                the default container spec will be used.
            model_user_id (str):
                Optional. The ID to use for the uploaded Model, which will
                become the final component of the model resource name. When
                not provided, Vertex AI will generate a value for this ID.
                When Model Registry model is provided, this field will be
                ignored.

                This value may be up to 63 characters, and valid characters
                are ``[a-z0-9_-]``. The first character cannot be a number
                or hyphen.
        """

        accept_eula: bool = proto.Field(
            proto.BOOL,
            number=1,
        )
        hugging_face_access_token: str = proto.Field(
            proto.STRING,
            number=2,
        )
        hugging_face_cache_enabled: bool = proto.Field(
            proto.BOOL,
            number=3,
        )
        model_display_name: str = proto.Field(
            proto.STRING,
            number=4,
        )
        container_spec: gca_model.ModelContainerSpec = proto.Field(
            proto.MESSAGE,
            number=5,
            message=gca_model.ModelContainerSpec,
        )
        model_user_id: str = proto.Field(
            proto.STRING,
            number=6,
        )

    class EndpointConfig(proto.Message):
        r"""The endpoint config to use for the deployment.

        Attributes:
            endpoint_display_name (str):
                Optional. The user-specified display name of
                the endpoint. If not set, a default name will be
                used.
            dedicated_endpoint_enabled (bool):
                Optional. Deprecated. Use dedicated_endpoint_disabled
                instead. If true, the endpoint will be exposed through a
                dedicated DNS [Endpoint.dedicated_endpoint_dns]. Your
                request to the dedicated DNS will be isolated from other
                users' traffic and will have better performance and
                reliability. Note: Once you enabled dedicated endpoint, you
                won't be able to send request to the shared DNS
                {region}-aiplatform.googleapis.com. The limitations will be
                removed soon.
            dedicated_endpoint_disabled (bool):
                Optional. By default, if dedicated endpoint is enabled and
                private service connect config is not set, the endpoint will
                be exposed through a dedicated DNS
                [Endpoint.dedicated_endpoint_dns]. If private service
                connect config is set, the endpoint will be exposed through
                private service connect. Your request to the dedicated DNS
                will be isolated from other users' traffic and will have
                better performance and reliability. Note: Once you enabled
                dedicated endpoint, you won't be able to send request to the
                shared DNS {region}-aiplatform.googleapis.com. The
                limitations will be removed soon.

                If this field is set to true, the dedicated endpoint will be
                disabled and the deployed model will be exposed through the
                shared DNS {region}-aiplatform.googleapis.com.
            private_service_connect_config (google.cloud.aiplatform_v1beta1.types.PrivateServiceConnectConfig):
                Optional. Configuration for private service
                connect. If set, the endpoint will be exposed
                through private service connect.
            endpoint_user_id (str):
                Optional. Immutable. The ID to use for endpoint, which will
                become the final component of the endpoint resource name. If
                not provided, Vertex AI will generate a value for this ID.

                If the first character is a letter, this value may be up to
                63 characters, and valid characters are ``[a-z0-9-]``. The
                last character must be a letter or number.

                If the first character is a number, this value may be up to
                9 characters, and valid characters are ``[0-9]`` with no
                leading zeros.

                When using HTTP/JSON, this field is populated based on a
                query string argument, such as ``?endpoint_id=12345``. This
                is the fallback for fields that are not included in either
                the URI or the body.
        """

        endpoint_display_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        dedicated_endpoint_enabled: bool = proto.Field(
            proto.BOOL,
            number=2,
        )
        dedicated_endpoint_disabled: bool = proto.Field(
            proto.BOOL,
            number=4,
        )
        private_service_connect_config: (
            service_networking.PrivateServiceConnectConfig
        ) = proto.Field(
            proto.MESSAGE,
            number=5,
            message=service_networking.PrivateServiceConnectConfig,
        )
        endpoint_user_id: str = proto.Field(
            proto.STRING,
            number=3,
        )

    class DeployConfig(proto.Message):
        r"""The deploy config to use for the deployment.

        Attributes:
            dedicated_resources (google.cloud.aiplatform_v1beta1.types.DedicatedResources):
                Optional. The dedicated resources to use for
                the endpoint. If not set, the default resources
                will be used.
            fast_tryout_enabled (bool):
                Optional. If true, enable the QMT fast tryout
                feature for this model if possible.
            system_labels (MutableMapping[str, str]):
                Optional. System labels for Model Garden
                deployments. These labels are managed by Google
                and for tracking purposes only.
        """

        dedicated_resources: machine_resources.DedicatedResources = proto.Field(
            proto.MESSAGE,
            number=1,
            message=machine_resources.DedicatedResources,
        )
        fast_tryout_enabled: bool = proto.Field(
            proto.BOOL,
            number=2,
        )
        system_labels: MutableMapping[str, str] = proto.MapField(
            proto.STRING,
            proto.STRING,
            number=3,
        )

    publisher_model_name: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="artifacts",
    )
    hugging_face_model_id: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="artifacts",
    )
    custom_model: CustomModel = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="artifacts",
        message=CustomModel,
    )
    destination: str = proto.Field(
        proto.STRING,
        number=4,
    )
    model_config: ModelConfig = proto.Field(
        proto.MESSAGE,
        number=5,
        message=ModelConfig,
    )
    endpoint_config: EndpointConfig = proto.Field(
        proto.MESSAGE,
        number=6,
        message=EndpointConfig,
    )
    deploy_config: DeployConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        message=DeployConfig,
    )


class DeployPublisherModelRequest(proto.Message):
    r"""Request message for
    [ModelGardenService.DeployPublisherModel][google.cloud.aiplatform.v1beta1.ModelGardenService.DeployPublisherModel].

    Attributes:
        model (str):
            Required. The model to deploy. Format:

            1. ``publishers/{publisher}/models/{publisher_model}@{version_id}``,
               or
               ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}@001``.
            2. Hugging Face model ID like ``google/gemma-2-2b-it``.
            3. Custom model Google Cloud Storage URI like
               ``gs://bucket``.
            4. Custom model zip file like ``https://example.com/a.zip``.
        destination (str):
            Required. The resource name of the Location to deploy the
            model in. Format:
            ``projects/{project}/locations/{location}``
        endpoint_display_name (str):
            Optional. The user-specified display name of
            the endpoint. If not set, a default name will be
            used.
        dedicated_resources (google.cloud.aiplatform_v1beta1.types.DedicatedResources):
            Optional. The dedicated resources to use for
            the endpoint. If not set, the default resources
            will be used.
        model_display_name (str):
            Optional. The user-specified display name of
            the uploaded model. If not set, a default name
            will be used.
        hugging_face_access_token (str):
            Optional. The Hugging Face read access token
            used to access the model artifacts of gated
            models.
        accept_eula (bool):
            Optional. Whether the user accepts the End
            User License Agreement (EULA) for the model.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    destination: str = proto.Field(
        proto.STRING,
        number=2,
    )
    endpoint_display_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    dedicated_resources: machine_resources.DedicatedResources = proto.Field(
        proto.MESSAGE,
        number=4,
        message=machine_resources.DedicatedResources,
    )
    model_display_name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    hugging_face_access_token: str = proto.Field(
        proto.STRING,
        number=6,
    )
    accept_eula: bool = proto.Field(
        proto.BOOL,
        number=7,
    )


class DeployResponse(proto.Message):
    r"""Response message for
    [ModelGardenService.Deploy][google.cloud.aiplatform.v1beta1.ModelGardenService.Deploy].

    Attributes:
        publisher_model (str):
            Output only. The name of the PublisherModel resource.
            Format:
            ``publishers/{publisher}/models/{publisher_model}@{version_id}``,
            or
            ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}@001``
        endpoint (str):
            Output only. The name of the Endpoint created. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        model (str):
            Output only. The name of the Model created. Format:
            ``projects/{project}/locations/{location}/models/{model}``
    """

    publisher_model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    endpoint: str = proto.Field(
        proto.STRING,
        number=2,
    )
    model: str = proto.Field(
        proto.STRING,
        number=3,
    )


class DeployPublisherModelResponse(proto.Message):
    r"""Response message for
    [ModelGardenService.DeployPublisherModel][google.cloud.aiplatform.v1beta1.ModelGardenService.DeployPublisherModel].

    Attributes:
        publisher_model (str):
            Output only. The name of the PublisherModel resource.
            Format:
            ``publishers/{publisher}/models/{publisher_model}@{version_id}``,
            or
            ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}@001``
        endpoint (str):
            Output only. The name of the Endpoint created. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        model (str):
            Output only. The name of the Model created. Format:
            ``projects/{project}/locations/{location}/models/{model}``
    """

    publisher_model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    endpoint: str = proto.Field(
        proto.STRING,
        number=2,
    )
    model: str = proto.Field(
        proto.STRING,
        number=3,
    )


class DeployOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [ModelGardenService.Deploy][google.cloud.aiplatform.v1beta1.ModelGardenService.Deploy].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
        publisher_model (str):
            Output only. The name of the model resource.
        destination (str):
            Output only. The resource name of the Location to deploy the
            model in. Format:
            ``projects/{project}/locations/{location}``
        project_number (int):
            Output only. The project number where the
            deploy model request is sent.
        model_id (str):
            Output only. The model id to be used at query
            time.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    publisher_model: str = proto.Field(
        proto.STRING,
        number=2,
    )
    destination: str = proto.Field(
        proto.STRING,
        number=3,
    )
    project_number: int = proto.Field(
        proto.INT64,
        number=4,
    )
    model_id: str = proto.Field(
        proto.STRING,
        number=5,
    )


class DeployPublisherModelOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [ModelGardenService.DeployPublisherModel][google.cloud.aiplatform.v1beta1.ModelGardenService.DeployPublisherModel].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
        publisher_model (str):
            Output only. The name of the PublisherModel resource.
            Format:
            ``publishers/{publisher}/models/{publisher_model}@{version_id}``,
            or
            ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}@001``
        destination (str):
            Output only. The resource name of the Location to deploy the
            model in. Format:
            ``projects/{project}/locations/{location}``
        project_number (int):
            Output only. The project number where the
            deploy model request is sent.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    publisher_model: str = proto.Field(
        proto.STRING,
        number=2,
    )
    destination: str = proto.Field(
        proto.STRING,
        number=3,
    )
    project_number: int = proto.Field(
        proto.INT64,
        number=4,
    )


class ExportPublisherModelResponse(proto.Message):
    r"""Response message for
    [ModelGardenService.ExportPublisherModel][google.cloud.aiplatform.v1beta1.ModelGardenService.ExportPublisherModel].

    Attributes:
        publisher_model (str):
            The name of the PublisherModel resource. Format:
            ``publishers/{publisher}/models/{publisher_model}@{version_id}``
        destination_uri (str):
            The destination uri of the model weights.
    """

    publisher_model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    destination_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ExportPublisherModelOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [ModelGardenService.ExportPublisherModel][google.cloud.aiplatform.v1beta1.ModelGardenService.ExportPublisherModel].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class ExportPublisherModelRequest(proto.Message):
    r"""Request message for
    [ModelGardenService.ExportPublisherModel][google.cloud.aiplatform.v1beta1.ModelGardenService.ExportPublisherModel].

    Attributes:
        name (str):
            Required. The name of the PublisherModel resource. Format:
            ``publishers/{publisher}/models/{publisher_model}@{version_id}``,
            or
            ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}@001``
        destination (google.cloud.aiplatform_v1beta1.types.GcsDestination):
            Required. The target where we are exporting
            the model weights to
        parent (str):
            Required. The Location to export the model weights from
            Format: ``projects/{project}/locations/{location}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    destination: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=2,
        message=io.GcsDestination,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CheckPublisherModelEulaAcceptanceRequest(proto.Message):
    r"""Request message for [ModelGardenService.CheckPublisherModelEula][].

    Attributes:
        parent (str):
            Required. The project requesting access for named model. The
            format is ``projects/{project}``.
        publisher_model (str):
            Required. The name of the PublisherModel resource. Format:
            ``publishers/{publisher}/models/{publisher_model}``, or
            ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}``
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    publisher_model: str = proto.Field(
        proto.STRING,
        number=2,
    )


class AcceptPublisherModelEulaRequest(proto.Message):
    r"""Request message for
    [ModelGardenService.AcceptPublisherModelEula][google.cloud.aiplatform.v1beta1.ModelGardenService.AcceptPublisherModelEula].

    Attributes:
        parent (str):
            Required. The project requesting access for named model. The
            format is ``projects/{project}``.
        publisher_model (str):
            Required. The name of the PublisherModel resource. Format:
            ``publishers/{publisher}/models/{publisher_model}``, or
            ``publishers/hf-{hugging-face-author}/models/{hugging-face-model-name}``
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    publisher_model: str = proto.Field(
        proto.STRING,
        number=2,
    )


class PublisherModelEulaAcceptance(proto.Message):
    r"""Response message for
    [ModelGardenService.UpdatePublisherModelEula][].

    Attributes:
        project_number (int):
            The project number requesting access for
            named model.
        publisher_model (str):
            The publisher model resource name.
        publisher_model_eula_acked (bool):
            The EULA content acceptance status.
    """

    project_number: int = proto.Field(
        proto.INT64,
        number=1,
    )
    publisher_model: str = proto.Field(
        proto.STRING,
        number=2,
    )
    publisher_model_eula_acked: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
