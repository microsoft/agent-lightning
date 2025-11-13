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

from google.cloud.aiplatform_v1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1.types import env_var
from google.cloud.aiplatform_v1.types import service_networking
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "ReasoningEngineSpec",
        "ReasoningEngine",
    },
)


class ReasoningEngineSpec(proto.Message):
    r"""ReasoningEngine configurations

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        source_code_spec (google.cloud.aiplatform_v1.types.ReasoningEngineSpec.SourceCodeSpec):
            Deploy from source code files with a defined
            entrypoint.

            This field is a member of `oneof`_ ``deployment_source``.
        service_account (str):
            Optional. The service account that the
            Reasoning Engine artifact runs as. It should
            have "roles/storage.objectViewer" for reading
            the user project's Cloud Storage and
            "roles/aiplatform.user" for using Vertex
            extensions. If not specified, the Vertex AI
            Reasoning Engine Service Agent in the project
            will be used.

            This field is a member of `oneof`_ ``_service_account``.
        package_spec (google.cloud.aiplatform_v1.types.ReasoningEngineSpec.PackageSpec):
            Optional. User provided package spec of the ReasoningEngine.
            Ignored when users directly specify a deployment image
            through ``deployment_spec.first_party_image_override``, but
            keeping the field_behavior to avoid introducing breaking
            changes. The ``deployment_source`` field should not be set
            if ``package_spec`` is specified.
        deployment_spec (google.cloud.aiplatform_v1.types.ReasoningEngineSpec.DeploymentSpec):
            Optional. The specification of a Reasoning
            Engine deployment.
        class_methods (MutableSequence[google.protobuf.struct_pb2.Struct]):
            Optional. Declarations for object class
            methods in OpenAPI specification format.
        agent_framework (str):
            Optional. The OSS agent framework used to
            develop the agent. Currently supported values:
            "google-adk", "langchain", "langgraph", "ag2",
            "llama-index", "custom".
    """

    class PackageSpec(proto.Message):
        r"""User-provided package specification, containing pickled
        object and package requirements.

        Attributes:
            pickle_object_gcs_uri (str):
                Optional. The Cloud Storage URI of the
                pickled python object.
            dependency_files_gcs_uri (str):
                Optional. The Cloud Storage URI of the
                dependency files in tar.gz format.
            requirements_gcs_uri (str):
                Optional. The Cloud Storage URI of the ``requirements.txt``
                file
            python_version (str):
                Optional. The Python version. Currently
                support 3.8, 3.9, 3.10, 3.11. If not specified,
                default value is 3.10.
        """

        pickle_object_gcs_uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        dependency_files_gcs_uri: str = proto.Field(
            proto.STRING,
            number=2,
        )
        requirements_gcs_uri: str = proto.Field(
            proto.STRING,
            number=3,
        )
        python_version: str = proto.Field(
            proto.STRING,
            number=4,
        )

    class DeploymentSpec(proto.Message):
        r"""The specification of a Reasoning Engine deployment.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            env (MutableSequence[google.cloud.aiplatform_v1.types.EnvVar]):
                Optional. Environment variables to be set
                with the Reasoning Engine deployment. The
                environment variables can be updated through the
                UpdateReasoningEngine API.
            secret_env (MutableSequence[google.cloud.aiplatform_v1.types.SecretEnvVar]):
                Optional. Environment variables where the
                value is a secret in Cloud Secret Manager.
                To use this feature, add 'Secret Manager Secret
                Accessor' role
                (roles/secretmanager.secretAccessor) to AI
                Platform Reasoning Engine Service Agent.
            psc_interface_config (google.cloud.aiplatform_v1.types.PscInterfaceConfig):
                Optional. Configuration for PSC-I.
            min_instances (int):
                Optional. The minimum number of application instances that
                will be kept running at all times. Defaults to 1. Range: [0,
                10].

                This field is a member of `oneof`_ ``_min_instances``.
            max_instances (int):
                Optional. The maximum number of application instances that
                can be launched to handle increased traffic. Defaults to
                100. Range: [1, 1000].

                If VPC-SC or PSC-I is enabled, the acceptable range is [1,
                100].

                This field is a member of `oneof`_ ``_max_instances``.
            resource_limits (MutableMapping[str, str]):
                Optional. Resource limits for each container. Only 'cpu' and
                'memory' keys are supported. Defaults to {"cpu": "4",
                "memory": "4Gi"}.

                - The only supported values for CPU are '1', '2', '4', '6'
                  and '8'. For more information, go to
                  https://cloud.google.com/run/docs/configuring/cpu.
                - The only supported values for memory are '1Gi', '2Gi', ...
                  '32 Gi'.
                - For required cpu on different memory values, go to
                  https://cloud.google.com/run/docs/configuring/memory-limits
            container_concurrency (int):
                Optional. Concurrency for each container and agent server.
                Recommended value: 2 \* cpu + 1. Defaults to 9.

                This field is a member of `oneof`_ ``_container_concurrency``.
        """

        env: MutableSequence[env_var.EnvVar] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=env_var.EnvVar,
        )
        secret_env: MutableSequence[env_var.SecretEnvVar] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message=env_var.SecretEnvVar,
        )
        psc_interface_config: service_networking.PscInterfaceConfig = proto.Field(
            proto.MESSAGE,
            number=4,
            message=service_networking.PscInterfaceConfig,
        )
        min_instances: int = proto.Field(
            proto.INT32,
            number=5,
            optional=True,
        )
        max_instances: int = proto.Field(
            proto.INT32,
            number=6,
            optional=True,
        )
        resource_limits: MutableMapping[str, str] = proto.MapField(
            proto.STRING,
            proto.STRING,
            number=7,
        )
        container_concurrency: int = proto.Field(
            proto.INT32,
            number=8,
            optional=True,
        )

    class SourceCodeSpec(proto.Message):
        r"""Specification for deploying from source code.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            inline_source (google.cloud.aiplatform_v1.types.ReasoningEngineSpec.SourceCodeSpec.InlineSource):
                Source code is provided directly in the
                request.

                This field is a member of `oneof`_ ``source``.
            python_spec (google.cloud.aiplatform_v1.types.ReasoningEngineSpec.SourceCodeSpec.PythonSpec):
                Configuration for a Python application.

                This field is a member of `oneof`_ ``language_spec``.
        """

        class InlineSource(proto.Message):
            r"""Specifies source code provided as a byte stream.

            Attributes:
                source_archive (bytes):
                    Required. Input only. The application source
                    code archive, provided as a compressed tarball
                    (.tar.gz) file.
            """

            source_archive: bytes = proto.Field(
                proto.BYTES,
                number=1,
            )

        class PythonSpec(proto.Message):
            r"""Specification for running a Python application from source.

            Attributes:
                version (str):
                    Optional. The version of Python to use.
                    Support version includes 3.9, 3.10, 3.11, 3.12,
                    3.13. If not specified, default value is 3.10.
                entrypoint_module (str):
                    Optional. The Python module to load as the
                    entrypoint, specified as a fully qualified
                    module name. For example: path.to.agent. If not
                    specified, defaults to "agent".

                    The project root will be added to Python
                    sys.path, allowing imports to be specified
                    relative to the root.
                entrypoint_object (str):
                    Optional. The name of the callable object within the
                    ``entrypoint_module`` to use as the application If not
                    specified, defaults to "root_agent".
                requirements_file (str):
                    Optional. The path to the requirements file,
                    relative to the source root. If not specified,
                    defaults to "requirements.txt".
            """

            version: str = proto.Field(
                proto.STRING,
                number=1,
            )
            entrypoint_module: str = proto.Field(
                proto.STRING,
                number=2,
            )
            entrypoint_object: str = proto.Field(
                proto.STRING,
                number=3,
            )
            requirements_file: str = proto.Field(
                proto.STRING,
                number=4,
            )

        inline_source: "ReasoningEngineSpec.SourceCodeSpec.InlineSource" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="source",
            message="ReasoningEngineSpec.SourceCodeSpec.InlineSource",
        )
        python_spec: "ReasoningEngineSpec.SourceCodeSpec.PythonSpec" = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="language_spec",
            message="ReasoningEngineSpec.SourceCodeSpec.PythonSpec",
        )

    source_code_spec: SourceCodeSpec = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="deployment_source",
        message=SourceCodeSpec,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    package_spec: PackageSpec = proto.Field(
        proto.MESSAGE,
        number=2,
        message=PackageSpec,
    )
    deployment_spec: DeploymentSpec = proto.Field(
        proto.MESSAGE,
        number=4,
        message=DeploymentSpec,
    )
    class_methods: MutableSequence[struct_pb2.Struct] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Struct,
    )
    agent_framework: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ReasoningEngine(proto.Message):
    r"""ReasoningEngine provides a customizable runtime for models to
    determine which actions to take and in which order.

    Attributes:
        name (str):
            Identifier. The resource name of the ReasoningEngine.
            Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        display_name (str):
            Required. The display name of the
            ReasoningEngine.
        description (str):
            Optional. The description of the
            ReasoningEngine.
        spec (google.cloud.aiplatform_v1.types.ReasoningEngineSpec):
            Optional. Configurations of the
            ReasoningEngine
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ReasoningEngine was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ReasoningEngine was most recently updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Customer-managed encryption key spec for a
            ReasoningEngine. If set, this ReasoningEngine
            and all sub-resources of this ReasoningEngine
            will be secured by this key.
        labels (MutableMapping[str, str]):
            Labels for the ReasoningEngine.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=7,
    )
    spec: "ReasoningEngineSpec" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="ReasoningEngineSpec",
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=6,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=11,
        message=gca_encryption_spec.EncryptionSpec,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=17,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
