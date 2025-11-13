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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "DeploymentStage",
    },
)


class DeploymentStage(proto.Enum):
    r"""Stage field indicating the current progress of a deployment.

    Values:
        DEPLOYMENT_STAGE_UNSPECIFIED (0):
            Default value. This value is unused.
        STARTING_DEPLOYMENT (5):
            The deployment is initializing and setting up
            the environment.
        PREPARING_MODEL (6):
            The deployment is preparing the model assets.
        CREATING_SERVING_CLUSTER (7):
            The deployment is creating the underlying
            serving cluster.
        ADDING_NODES_TO_CLUSTER (8):
            The deployment is adding nodes to the serving
            cluster.
        GETTING_CONTAINER_IMAGE (9):
            The deployment is getting the container image
            for the model server.
        STARTING_MODEL_SERVER (3):
            The deployment is starting the model server.
        FINISHING_UP (4):
            The deployment is performing finalization
            steps.
        DEPLOYMENT_TERMINATED (10):
            The deployment has terminated.
    """

    DEPLOYMENT_STAGE_UNSPECIFIED = 0
    STARTING_DEPLOYMENT = 5
    PREPARING_MODEL = 6
    CREATING_SERVING_CLUSTER = 7
    ADDING_NODES_TO_CLUSTER = 8
    GETTING_CONTAINER_IMAGE = 9
    STARTING_MODEL_SERVER = 3
    FINISHING_UP = 4
    DEPLOYMENT_TERMINATED = 10


__all__ = tuple(sorted(__protobuf__.manifest))
