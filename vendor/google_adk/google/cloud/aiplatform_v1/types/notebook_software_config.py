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

from google.cloud.aiplatform_v1.types import env_var


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "PostStartupScriptConfig",
        "ColabImage",
        "NotebookSoftwareConfig",
    },
)


class PostStartupScriptConfig(proto.Message):
    r"""Post startup script config.

    Attributes:
        post_startup_script (str):
            Optional. Post startup script to run after
            runtime is started.
        post_startup_script_url (str):
            Optional. Post startup script url to download. Example:
            ``gs://bucket/script.sh``
        post_startup_script_behavior (google.cloud.aiplatform_v1.types.PostStartupScriptConfig.PostStartupScriptBehavior):
            Optional. Post startup script behavior that
            defines download and execution behavior.
    """

    class PostStartupScriptBehavior(proto.Enum):
        r"""Represents a notebook runtime post startup script behavior.

        Values:
            POST_STARTUP_SCRIPT_BEHAVIOR_UNSPECIFIED (0):
                Unspecified post startup script behavior.
            RUN_ONCE (1):
                Run post startup script after runtime is
                started.
            RUN_EVERY_START (2):
                Run post startup script after runtime is
                stopped.
            DOWNLOAD_AND_RUN_EVERY_START (3):
                Download and run post startup script every
                time runtime is started.
        """

        POST_STARTUP_SCRIPT_BEHAVIOR_UNSPECIFIED = 0
        RUN_ONCE = 1
        RUN_EVERY_START = 2
        DOWNLOAD_AND_RUN_EVERY_START = 3

    post_startup_script: str = proto.Field(
        proto.STRING,
        number=1,
    )
    post_startup_script_url: str = proto.Field(
        proto.STRING,
        number=2,
    )
    post_startup_script_behavior: PostStartupScriptBehavior = proto.Field(
        proto.ENUM,
        number=3,
        enum=PostStartupScriptBehavior,
    )


class ColabImage(proto.Message):
    r"""Colab image of the runtime.

    Attributes:
        release_name (str):
            Optional. The release name of the
            NotebookRuntime Colab image, e.g. "py310". If
            not specified, detault to the latest release.
        description (str):
            Output only. A human-readable description of
            the specified colab image release, populated by
            the system. Example: "Python 3.10", "Latest -
            current Python 3.11".
    """

    release_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    description: str = proto.Field(
        proto.STRING,
        number=2,
    )


class NotebookSoftwareConfig(proto.Message):
    r"""Notebook Software Config. This is passed to the backend when
    user makes software configurations in UI.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        colab_image (google.cloud.aiplatform_v1.types.ColabImage):
            Optional. Google-managed NotebookRuntime
            colab image.

            This field is a member of `oneof`_ ``runtime_image``.
        env (MutableSequence[google.cloud.aiplatform_v1.types.EnvVar]):
            Optional. Environment variables to be passed
            to the container. Maximum limit is 100.
        post_startup_script_config (google.cloud.aiplatform_v1.types.PostStartupScriptConfig):
            Optional. Post startup script config.
    """

    colab_image: "ColabImage" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="runtime_image",
        message="ColabImage",
    )
    env: MutableSequence[env_var.EnvVar] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=env_var.EnvVar,
    )
    post_startup_script_config: "PostStartupScriptConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="PostStartupScriptConfig",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
