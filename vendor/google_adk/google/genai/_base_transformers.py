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

"""Base transformers for Google GenAI SDK."""
import base64

# Some fields don't accept url safe base64 encoding.
# We shouldn't use this transformer if the backend adhere to Cloud Type
# format https://cloud.google.com/docs/discovery/type-format.
# TODO(b/389133914,b/390320301): Remove the hack after backend fix the issue.
def t_bytes(data: bytes) -> str:
  if not isinstance(data, bytes):
    return data
  return base64.b64encode(data).decode('ascii')
