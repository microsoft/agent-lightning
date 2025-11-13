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

"""Classes and functions for working with Model Garden."""

# We just want to re-export certain classes
# pylint: disable=g-multiple-import,g-importing-member
from vertexai.model_garden import _model_garden

OpenModel = _model_garden.OpenModel
PartnerModel = _model_garden.PartnerModel
list_deployable_models = _model_garden.list_deployable_models

__all__ = ("OpenModel", "PartnerModel", "list_deployable_models")
