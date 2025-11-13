# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
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
"""Constants."""

# Required Names for model files are specified here
# https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts#framework-specific_requirements
_PICKLE_FILE_NAME = "model.pkl"
_PICKLE_EXTENTION = ".pkl"

_XGBOOST_VERSION = "1.6"
# TensorFlow 2.13 requires typing_extensions<4.6 and will cause errors in Ray.
# https://github.com/tensorflow/tensorflow/blob/v2.13.0/tensorflow/tools/pip_package/setup.py#L100
# 2.13 is the latest supported version of Vertex prebuilt prediction container.
# Set 2.12 as default here since 2.13 cause errors.
_TENSORFLOW_VERSION = "2.12"
