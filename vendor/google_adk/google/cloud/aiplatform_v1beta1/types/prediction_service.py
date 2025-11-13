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

from google.api import httpbody_pb2  # type: ignore
from google.cloud.aiplatform_v1beta1.types import content
from google.cloud.aiplatform_v1beta1.types import explanation
from google.cloud.aiplatform_v1beta1.types import tool
from google.cloud.aiplatform_v1beta1.types import types
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "PredictRequest",
        "PredictResponse",
        "RawPredictRequest",
        "StreamRawPredictRequest",
        "DirectPredictRequest",
        "DirectPredictResponse",
        "DirectRawPredictRequest",
        "DirectRawPredictResponse",
        "StreamDirectPredictRequest",
        "StreamDirectPredictResponse",
        "StreamDirectRawPredictRequest",
        "StreamDirectRawPredictResponse",
        "StreamingPredictRequest",
        "StreamingPredictResponse",
        "StreamingRawPredictRequest",
        "StreamingRawPredictResponse",
        "ExplainRequest",
        "ExplainResponse",
        "CountTokensRequest",
        "CountTokensResponse",
        "GenerateContentRequest",
        "GenerateContentResponse",
        "ChatCompletionsRequest",
        "PredictLongRunningResponse",
        "PredictLongRunningMetadata",
        "GenerateVideoResponse",
    },
)


class PredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.Predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        instances (MutableSequence[google.protobuf.struct_pb2.Value]):
            Required. The instances that are the input to the prediction
            call. A DeployedModel may have an upper limit on the number
            of instances it supports per request, and when it is
            exceeded the prediction call errors in case of AutoML
            Models, or, in case of customer created Models, the
            behaviour is as documented by that Model. The schema of any
            single instance may be specified via Endpoint's
            DeployedModels'
            [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
            [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
            [instance_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.instance_schema_uri].
        parameters (google.protobuf.struct_pb2.Value):
            The parameters that govern the prediction. The schema of the
            parameters may be specified via Endpoint's DeployedModels'
            [Model's
            ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
            [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
            [parameters_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.parameters_schema_uri].
        labels (MutableMapping[str, str]):
            Optional. The user labels for Imagen billing
            usage only. Only Imagen supports labels. For
            other use cases, it will be ignored.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    instances: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Value,
    )
    parameters: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Value,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=4,
    )


class PredictResponse(proto.Message):
    r"""Response message for
    [PredictionService.Predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict].

    Attributes:
        predictions (MutableSequence[google.protobuf.struct_pb2.Value]):
            The predictions that are the output of the predictions call.
            The schema of any single prediction may be specified via
            Endpoint's DeployedModels' [Model's
            ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
            [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
            [prediction_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.prediction_schema_uri].
        deployed_model_id (str):
            ID of the Endpoint's DeployedModel that
            served this prediction.
        model (str):
            Output only. The resource name of the Model
            which is deployed as the DeployedModel that this
            prediction hits.
        model_version_id (str):
            Output only. The version ID of the Model
            which is deployed as the DeployedModel that this
            prediction hits.
        model_display_name (str):
            Output only. The [display
            name][google.cloud.aiplatform.v1beta1.Model.display_name] of
            the Model which is deployed as the DeployedModel that this
            prediction hits.
        metadata (google.protobuf.struct_pb2.Value):
            Output only. Request-level metadata returned
            by the model. The metadata type will be
            dependent upon the model implementation.
    """

    predictions: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=struct_pb2.Value,
    )
    deployed_model_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    model: str = proto.Field(
        proto.STRING,
        number=3,
    )
    model_version_id: str = proto.Field(
        proto.STRING,
        number=5,
    )
    model_display_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    metadata: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=6,
        message=struct_pb2.Value,
    )


class RawPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.RawPredict][google.cloud.aiplatform.v1beta1.PredictionService.RawPredict].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        http_body (google.api.httpbody_pb2.HttpBody):
            The prediction input. Supports HTTP headers and arbitrary
            data payload.

            A
            [DeployedModel][google.cloud.aiplatform.v1beta1.DeployedModel]
            may have an upper limit on the number of instances it
            supports per request. When this limit it is exceeded for an
            AutoML model, the
            [RawPredict][google.cloud.aiplatform.v1beta1.PredictionService.RawPredict]
            method returns an error. When this limit is exceeded for a
            custom-trained model, the behavior varies depending on the
            model.

            You can specify the schema for each instance in the
            [predict_schemata.instance_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.instance_schema_uri]
            field when you create a
            [Model][google.cloud.aiplatform.v1beta1.Model]. This schema
            applies when you deploy the ``Model`` as a ``DeployedModel``
            to an [Endpoint][google.cloud.aiplatform.v1beta1.Endpoint]
            and use the ``RawPredict`` method.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    http_body: httpbody_pb2.HttpBody = proto.Field(
        proto.MESSAGE,
        number=2,
        message=httpbody_pb2.HttpBody,
    )


class StreamRawPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.StreamRawPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamRawPredict].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        http_body (google.api.httpbody_pb2.HttpBody):
            The prediction input. Supports HTTP headers
            and arbitrary data payload.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    http_body: httpbody_pb2.HttpBody = proto.Field(
        proto.MESSAGE,
        number=2,
        message=httpbody_pb2.HttpBody,
    )


class DirectPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.DirectPredict][google.cloud.aiplatform.v1beta1.PredictionService.DirectPredict].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        inputs (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tensor]):
            The prediction input.
        parameters (google.cloud.aiplatform_v1beta1.types.Tensor):
            The parameters that govern the prediction.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    inputs: MutableSequence[types.Tensor] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=types.Tensor,
    )
    parameters: types.Tensor = proto.Field(
        proto.MESSAGE,
        number=3,
        message=types.Tensor,
    )


class DirectPredictResponse(proto.Message):
    r"""Response message for
    [PredictionService.DirectPredict][google.cloud.aiplatform.v1beta1.PredictionService.DirectPredict].

    Attributes:
        outputs (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tensor]):
            The prediction output.
        parameters (google.cloud.aiplatform_v1beta1.types.Tensor):
            The parameters that govern the prediction.
    """

    outputs: MutableSequence[types.Tensor] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=types.Tensor,
    )
    parameters: types.Tensor = proto.Field(
        proto.MESSAGE,
        number=2,
        message=types.Tensor,
    )


class DirectRawPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.DirectRawPredict][google.cloud.aiplatform.v1beta1.PredictionService.DirectRawPredict].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        method_name (str):
            Fully qualified name of the API method being invoked to
            perform predictions.

            Format: ``/namespace.Service/Method/`` Example:
            ``/tensorflow.serving.PredictionService/Predict``
        input (bytes):
            The prediction input.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    method_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    input: bytes = proto.Field(
        proto.BYTES,
        number=3,
    )


class DirectRawPredictResponse(proto.Message):
    r"""Response message for
    [PredictionService.DirectRawPredict][google.cloud.aiplatform.v1beta1.PredictionService.DirectRawPredict].

    Attributes:
        output (bytes):
            The prediction output.
    """

    output: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )


class StreamDirectPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.StreamDirectPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamDirectPredict].

    The first message must contain
    [endpoint][google.cloud.aiplatform.v1beta1.StreamDirectPredictRequest.endpoint]
    field and optionally [input][]. The subsequent messages must contain
    [input][].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        inputs (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tensor]):
            Optional. The prediction input.
        parameters (google.cloud.aiplatform_v1beta1.types.Tensor):
            Optional. The parameters that govern the
            prediction.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    inputs: MutableSequence[types.Tensor] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=types.Tensor,
    )
    parameters: types.Tensor = proto.Field(
        proto.MESSAGE,
        number=3,
        message=types.Tensor,
    )


class StreamDirectPredictResponse(proto.Message):
    r"""Response message for
    [PredictionService.StreamDirectPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamDirectPredict].

    Attributes:
        outputs (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tensor]):
            The prediction output.
        parameters (google.cloud.aiplatform_v1beta1.types.Tensor):
            The parameters that govern the prediction.
    """

    outputs: MutableSequence[types.Tensor] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=types.Tensor,
    )
    parameters: types.Tensor = proto.Field(
        proto.MESSAGE,
        number=2,
        message=types.Tensor,
    )


class StreamDirectRawPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.StreamDirectRawPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamDirectRawPredict].

    The first message must contain
    [endpoint][google.cloud.aiplatform.v1beta1.StreamDirectRawPredictRequest.endpoint]
    and
    [method_name][google.cloud.aiplatform.v1beta1.StreamDirectRawPredictRequest.method_name]
    fields and optionally
    [input][google.cloud.aiplatform.v1beta1.StreamDirectRawPredictRequest.input].
    The subsequent messages must contain
    [input][google.cloud.aiplatform.v1beta1.StreamDirectRawPredictRequest.input].
    [method_name][google.cloud.aiplatform.v1beta1.StreamDirectRawPredictRequest.method_name]
    in the subsequent messages have no effect.

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        method_name (str):
            Optional. Fully qualified name of the API method being
            invoked to perform predictions.

            Format: ``/namespace.Service/Method/`` Example:
            ``/tensorflow.serving.PredictionService/Predict``
        input (bytes):
            Optional. The prediction input.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    method_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    input: bytes = proto.Field(
        proto.BYTES,
        number=3,
    )


class StreamDirectRawPredictResponse(proto.Message):
    r"""Response message for
    [PredictionService.StreamDirectRawPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamDirectRawPredict].

    Attributes:
        output (bytes):
            The prediction output.
    """

    output: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )


class StreamingPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.StreamingPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamingPredict].

    The first message must contain
    [endpoint][google.cloud.aiplatform.v1beta1.StreamingPredictRequest.endpoint]
    field and optionally [input][]. The subsequent messages must contain
    [input][].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        inputs (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tensor]):
            The prediction input.
        parameters (google.cloud.aiplatform_v1beta1.types.Tensor):
            The parameters that govern the prediction.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    inputs: MutableSequence[types.Tensor] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=types.Tensor,
    )
    parameters: types.Tensor = proto.Field(
        proto.MESSAGE,
        number=3,
        message=types.Tensor,
    )


class StreamingPredictResponse(proto.Message):
    r"""Response message for
    [PredictionService.StreamingPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamingPredict].

    Attributes:
        outputs (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tensor]):
            The prediction output.
        parameters (google.cloud.aiplatform_v1beta1.types.Tensor):
            The parameters that govern the prediction.
    """

    outputs: MutableSequence[types.Tensor] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=types.Tensor,
    )
    parameters: types.Tensor = proto.Field(
        proto.MESSAGE,
        number=2,
        message=types.Tensor,
    )


class StreamingRawPredictRequest(proto.Message):
    r"""Request message for
    [PredictionService.StreamingRawPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamingRawPredict].

    The first message must contain
    [endpoint][google.cloud.aiplatform.v1beta1.StreamingRawPredictRequest.endpoint]
    and
    [method_name][google.cloud.aiplatform.v1beta1.StreamingRawPredictRequest.method_name]
    fields and optionally
    [input][google.cloud.aiplatform.v1beta1.StreamingRawPredictRequest.input].
    The subsequent messages must contain
    [input][google.cloud.aiplatform.v1beta1.StreamingRawPredictRequest.input].
    [method_name][google.cloud.aiplatform.v1beta1.StreamingRawPredictRequest.method_name]
    in the subsequent messages have no effect.

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        method_name (str):
            Fully qualified name of the API method being invoked to
            perform predictions.

            Format: ``/namespace.Service/Method/`` Example:
            ``/tensorflow.serving.PredictionService/Predict``
        input (bytes):
            The prediction input.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    method_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    input: bytes = proto.Field(
        proto.BYTES,
        number=3,
    )


class StreamingRawPredictResponse(proto.Message):
    r"""Response message for
    [PredictionService.StreamingRawPredict][google.cloud.aiplatform.v1beta1.PredictionService.StreamingRawPredict].

    Attributes:
        output (bytes):
            The prediction output.
    """

    output: bytes = proto.Field(
        proto.BYTES,
        number=1,
    )


class ExplainRequest(proto.Message):
    r"""Request message for
    [PredictionService.Explain][google.cloud.aiplatform.v1beta1.PredictionService.Explain].

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to serve the
            explanation. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        instances (MutableSequence[google.protobuf.struct_pb2.Value]):
            Required. The instances that are the input to the
            explanation call. A DeployedModel may have an upper limit on
            the number of instances it supports per request, and when it
            is exceeded the explanation call errors in case of AutoML
            Models, or, in case of customer created Models, the
            behaviour is as documented by that Model. The schema of any
            single instance may be specified via Endpoint's
            DeployedModels'
            [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
            [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
            [instance_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.instance_schema_uri].
        parameters (google.protobuf.struct_pb2.Value):
            The parameters that govern the prediction. The schema of the
            parameters may be specified via Endpoint's DeployedModels'
            [Model's
            ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
            [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
            [parameters_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.parameters_schema_uri].
        explanation_spec_override (google.cloud.aiplatform_v1beta1.types.ExplanationSpecOverride):
            If specified, overrides the
            [explanation_spec][google.cloud.aiplatform.v1beta1.DeployedModel.explanation_spec]
            of the DeployedModel. Can be used for explaining prediction
            results with different configurations, such as:

            - Explaining top-5 predictions results as opposed to top-1;
            - Increasing path count or step count of the attribution
              methods to reduce approximate errors;
            - Using different baselines for explaining the prediction
              results.
        concurrent_explanation_spec_override (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ExplanationSpecOverride]):
            Optional. This field is the same as the one above, but
            supports multiple explanations to occur in parallel. The key
            can be any string. Each override will be run against the
            model, then its explanations will be grouped together.

            Note - these explanations are run **In Addition** to the
            default Explanation in the deployed model.
        deployed_model_id (str):
            If specified, this ExplainRequest will be served by the
            chosen DeployedModel, overriding
            [Endpoint.traffic_split][google.cloud.aiplatform.v1beta1.Endpoint.traffic_split].
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    instances: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Value,
    )
    parameters: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=4,
        message=struct_pb2.Value,
    )
    explanation_spec_override: explanation.ExplanationSpecOverride = proto.Field(
        proto.MESSAGE,
        number=5,
        message=explanation.ExplanationSpecOverride,
    )
    concurrent_explanation_spec_override: MutableMapping[
        str, explanation.ExplanationSpecOverride
    ] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=6,
        message=explanation.ExplanationSpecOverride,
    )
    deployed_model_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ExplainResponse(proto.Message):
    r"""Response message for
    [PredictionService.Explain][google.cloud.aiplatform.v1beta1.PredictionService.Explain].

    Attributes:
        explanations (MutableSequence[google.cloud.aiplatform_v1beta1.types.Explanation]):
            The explanations of the Model's
            [PredictResponse.predictions][google.cloud.aiplatform.v1beta1.PredictResponse.predictions].

            It has the same number of elements as
            [instances][google.cloud.aiplatform.v1beta1.ExplainRequest.instances]
            to be explained.
        concurrent_explanations (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ExplainResponse.ConcurrentExplanation]):
            This field stores the results of the
            explanations run in parallel with The default
            explanation strategy/method.
        deployed_model_id (str):
            ID of the Endpoint's DeployedModel that
            served this explanation.
        predictions (MutableSequence[google.protobuf.struct_pb2.Value]):
            The predictions that are the output of the predictions call.
            Same as
            [PredictResponse.predictions][google.cloud.aiplatform.v1beta1.PredictResponse.predictions].
    """

    class ConcurrentExplanation(proto.Message):
        r"""This message is a wrapper grouping Concurrent Explanations.

        Attributes:
            explanations (MutableSequence[google.cloud.aiplatform_v1beta1.types.Explanation]):
                The explanations of the Model's
                [PredictResponse.predictions][google.cloud.aiplatform.v1beta1.PredictResponse.predictions].

                It has the same number of elements as
                [instances][google.cloud.aiplatform.v1beta1.ExplainRequest.instances]
                to be explained.
        """

        explanations: MutableSequence[explanation.Explanation] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=explanation.Explanation,
        )

    explanations: MutableSequence[explanation.Explanation] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=explanation.Explanation,
    )
    concurrent_explanations: MutableMapping[str, ConcurrentExplanation] = (
        proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=4,
            message=ConcurrentExplanation,
        )
    )
    deployed_model_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    predictions: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Value,
    )


class CountTokensRequest(proto.Message):
    r"""Request message for
    [PredictionService.CountTokens][google.cloud.aiplatform.v1beta1.PredictionService.CountTokens].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested to perform
            token counting. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        model (str):
            Optional. The name of the publisher model requested to serve
            the prediction. Format:
            ``projects/{project}/locations/{location}/publishers/*/models/*``
        instances (MutableSequence[google.protobuf.struct_pb2.Value]):
            Optional. The instances that are the input to
            token counting call. Schema is identical to the
            prediction schema of the underlying model.
        contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
            Optional. Input content.
        system_instruction (google.cloud.aiplatform_v1beta1.types.Content):
            Optional. The user provided system
            instructions for the model. Note: only text
            should be used in parts and content in each part
            will be in a separate paragraph.

            This field is a member of `oneof`_ ``_system_instruction``.
        tools (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tool]):
            Optional. A list of ``Tools`` the model may use to generate
            the next response.

            A ``Tool`` is a piece of code that enables the system to
            interact with external systems to perform an action, or set
            of actions, outside of knowledge and scope of the model.
        generation_config (google.cloud.aiplatform_v1beta1.types.GenerationConfig):
            Optional. Generation config that the model
            will use to generate the response.

            This field is a member of `oneof`_ ``_generation_config``.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    model: str = proto.Field(
        proto.STRING,
        number=3,
    )
    instances: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Value,
    )
    contents: MutableSequence[content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=content.Content,
    )
    system_instruction: content.Content = proto.Field(
        proto.MESSAGE,
        number=5,
        optional=True,
        message=content.Content,
    )
    tools: MutableSequence[tool.Tool] = proto.RepeatedField(
        proto.MESSAGE,
        number=6,
        message=tool.Tool,
    )
    generation_config: content.GenerationConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        optional=True,
        message=content.GenerationConfig,
    )


class CountTokensResponse(proto.Message):
    r"""Response message for
    [PredictionService.CountTokens][google.cloud.aiplatform.v1beta1.PredictionService.CountTokens].

    Attributes:
        total_tokens (int):
            The total number of tokens counted across all
            instances from the request.
        total_billable_characters (int):
            The total number of billable characters
            counted across all instances from the request.
        prompt_tokens_details (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModalityTokenCount]):
            Output only. List of modalities that were
            processed in the request input.
    """

    total_tokens: int = proto.Field(
        proto.INT32,
        number=1,
    )
    total_billable_characters: int = proto.Field(
        proto.INT32,
        number=2,
    )
    prompt_tokens_details: MutableSequence[content.ModalityTokenCount] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=3,
            message=content.ModalityTokenCount,
        )
    )


class GenerateContentRequest(proto.Message):
    r"""Request message for [PredictionService.GenerateContent].

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        model (str):
            Required. The fully qualified name of the publisher model or
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
        model_armor_config (google.cloud.aiplatform_v1beta1.types.ModelArmorConfig):
            Optional. Settings for prompt and response sanitization
            using the Model Armor service. If supplied, safety_settings
            must not be supplied.
        generation_config (google.cloud.aiplatform_v1beta1.types.GenerationConfig):
            Optional. Generation config.
    """

    model: str = proto.Field(
        proto.STRING,
        number=5,
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
    model_armor_config: content.ModelArmorConfig = proto.Field(
        proto.MESSAGE,
        number=11,
        message=content.ModelArmorConfig,
    )
    generation_config: content.GenerationConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        message=content.GenerationConfig,
    )


class GenerateContentResponse(proto.Message):
    r"""Response message for [PredictionService.GenerateContent].

    Attributes:
        candidates (MutableSequence[google.cloud.aiplatform_v1beta1.types.Candidate]):
            Output only. Generated candidates.
        model_version (str):
            Output only. The model version used to
            generate the response.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the request is
            made to the server.
        response_id (str):
            Output only. response_id is used to identify each response.
            It is the encoding of the event_id.
        prompt_feedback (google.cloud.aiplatform_v1beta1.types.GenerateContentResponse.PromptFeedback):
            Output only. Content filter results for a
            prompt sent in the request. Note: Sent only in
            the first stream chunk. Only happens when no
            candidates were generated due to content
            violations.
        usage_metadata (google.cloud.aiplatform_v1beta1.types.GenerateContentResponse.UsageMetadata):
            Usage metadata about the response(s).
    """

    class PromptFeedback(proto.Message):
        r"""Content filter results for a prompt sent in the request.

        Attributes:
            block_reason (google.cloud.aiplatform_v1beta1.types.GenerateContentResponse.PromptFeedback.BlockedReason):
                Output only. Blocked reason.
            safety_ratings (MutableSequence[google.cloud.aiplatform_v1beta1.types.SafetyRating]):
                Output only. Safety ratings.
            block_reason_message (str):
                Output only. A readable block reason message.
        """

        class BlockedReason(proto.Enum):
            r"""Blocked reason enumeration.

            Values:
                BLOCKED_REASON_UNSPECIFIED (0):
                    Unspecified blocked reason.
                SAFETY (1):
                    Candidates blocked due to safety.
                OTHER (2):
                    Candidates blocked due to other reason.
                BLOCKLIST (3):
                    Candidates blocked due to the terms which are
                    included from the terminology blocklist.
                PROHIBITED_CONTENT (4):
                    Candidates blocked due to prohibited content.
                MODEL_ARMOR (5):
                    The user prompt was blocked by Model Armor.
            """

            BLOCKED_REASON_UNSPECIFIED = 0
            SAFETY = 1
            OTHER = 2
            BLOCKLIST = 3
            PROHIBITED_CONTENT = 4
            MODEL_ARMOR = 5

        block_reason: "GenerateContentResponse.PromptFeedback.BlockedReason" = (
            proto.Field(
                proto.ENUM,
                number=1,
                enum="GenerateContentResponse.PromptFeedback.BlockedReason",
            )
        )
        safety_ratings: MutableSequence[content.SafetyRating] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message=content.SafetyRating,
        )
        block_reason_message: str = proto.Field(
            proto.STRING,
            number=3,
        )

    class UsageMetadata(proto.Message):
        r"""Usage metadata about response(s).

        Attributes:
            prompt_token_count (int):
                Number of tokens in the request. When ``cached_content`` is
                set, this is still the total effective prompt size meaning
                this includes the number of tokens in the cached content.
            candidates_token_count (int):
                Number of tokens in the response(s).
            thoughts_token_count (int):
                Output only. Number of tokens present in
                thoughts output.
            total_token_count (int):
                Total token count for prompt and response
                candidates.
            cached_content_token_count (int):
                Output only. Number of tokens in the cached
                part in the input (the cached content).
            prompt_tokens_details (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModalityTokenCount]):
                Output only. List of modalities that were
                processed in the request input.
            cache_tokens_details (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModalityTokenCount]):
                Output only. List of modalities of the cached
                content in the request input.
            candidates_tokens_details (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModalityTokenCount]):
                Output only. List of modalities that were
                returned in the response.
        """

        prompt_token_count: int = proto.Field(
            proto.INT32,
            number=1,
        )
        candidates_token_count: int = proto.Field(
            proto.INT32,
            number=2,
        )
        thoughts_token_count: int = proto.Field(
            proto.INT32,
            number=14,
        )
        total_token_count: int = proto.Field(
            proto.INT32,
            number=3,
        )
        cached_content_token_count: int = proto.Field(
            proto.INT32,
            number=5,
        )
        prompt_tokens_details: MutableSequence[content.ModalityTokenCount] = (
            proto.RepeatedField(
                proto.MESSAGE,
                number=9,
                message=content.ModalityTokenCount,
            )
        )
        cache_tokens_details: MutableSequence[content.ModalityTokenCount] = (
            proto.RepeatedField(
                proto.MESSAGE,
                number=10,
                message=content.ModalityTokenCount,
            )
        )
        candidates_tokens_details: MutableSequence[content.ModalityTokenCount] = (
            proto.RepeatedField(
                proto.MESSAGE,
                number=11,
                message=content.ModalityTokenCount,
            )
        )

    candidates: MutableSequence[content.Candidate] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=content.Candidate,
    )
    model_version: str = proto.Field(
        proto.STRING,
        number=11,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
    )
    response_id: str = proto.Field(
        proto.STRING,
        number=13,
    )
    prompt_feedback: PromptFeedback = proto.Field(
        proto.MESSAGE,
        number=3,
        message=PromptFeedback,
    )
    usage_metadata: UsageMetadata = proto.Field(
        proto.MESSAGE,
        number=4,
        message=UsageMetadata,
    )


class ChatCompletionsRequest(proto.Message):
    r"""Request message for [PredictionService.ChatCompletions]

    Attributes:
        endpoint (str):
            Required. The name of the endpoint requested to serve the
            prediction. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        http_body (google.api.httpbody_pb2.HttpBody):
            Optional. The prediction input. Supports HTTP
            headers and arbitrary data payload.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    http_body: httpbody_pb2.HttpBody = proto.Field(
        proto.MESSAGE,
        number=2,
        message=httpbody_pb2.HttpBody,
    )


class PredictLongRunningResponse(proto.Message):
    r"""Response message for [PredictionService.PredictLongRunning]

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        generate_video_response (google.cloud.aiplatform_v1beta1.types.GenerateVideoResponse):
            The response of the video generation
            prediction.

            This field is a member of `oneof`_ ``response``.
    """

    generate_video_response: "GenerateVideoResponse" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="response",
        message="GenerateVideoResponse",
    )


class PredictLongRunningMetadata(proto.Message):
    r"""Metadata for PredictLongRunning long running operations."""


class GenerateVideoResponse(proto.Message):
    r"""Generate video response.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        generated_samples (MutableSequence[str]):
            The cloud storage uris of the generated
            videos.
        rai_media_filtered_count (int):
            Returns if any videos were filtered due to
            RAI policies.

            This field is a member of `oneof`_ ``_rai_media_filtered_count``.
        rai_media_filtered_reasons (MutableSequence[str]):
            Returns rai failure reasons if any.
    """

    generated_samples: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    rai_media_filtered_count: int = proto.Field(
        proto.INT32,
        number=2,
        optional=True,
    )
    rai_media_filtered_reasons: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
