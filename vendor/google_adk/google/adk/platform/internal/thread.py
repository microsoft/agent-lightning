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

from __future__ import annotations

import asyncio
from typing import Callable

from google3.learning.deepmind.python.threading import g3_executor
from google3.learning.deepmind.python.threading import g3_thread

# TODO(b/423882251): Switch to copybara replacements.
# Right now this doesn't work because the ADK runs unit and integration tests,
# outside google3, before copybara replacements.


def create_thread(target: Callable[..., None], *args, **kwargs):
  """Creates a thread."""
  return g3_thread.G3Thread(target=target, args=args, kwargs=kwargs)
