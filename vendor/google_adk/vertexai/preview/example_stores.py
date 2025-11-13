# Copyright 2024 Google LLC
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
"""Classes for working with example stores."""

# We just want to re-export certain classes
# pylint: disable=g-multiple-import,g-importing-member
from google.cloud.aiplatform_v1beta1 import types
from vertexai.example_stores._example_stores import (
    ContentsExample,
    ContentSearchKey,
    Example,
    ExampleStore,
    ExamplesArrayFilter,
    ExpectedContent,
    StoredContentsExample,
    StoredContentsExampleFilter,
    StoredContentsExampleParameters,
)

ArrayOperator = types.ExamplesArrayFilter.ArrayOperator
ExampleStoreConfig = types.ExampleStoreConfig

__all__ = (
    "ArrayOperator",
    "ContentsExample",
    "ContentSearchKey",
    "Example",
    "ExampleStore",
    "ExampleStoreConfig",
    "ExamplesArrayFilter",
    "ExpectedContent",
    "StoredContentsExample",
    "StoredContentsExampleFilter",
    "StoredContentsExampleParameters",
)
