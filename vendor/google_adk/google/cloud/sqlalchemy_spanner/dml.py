# Copyright 2024 Google LLC All rights reserved.
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

from sqlalchemy import Insert, insert
from sqlalchemy.sql._typing import _DMLTableArgument


def insert_or_update(table: _DMLTableArgument) -> Insert:
    """Construct a Spanner-specific insert-or-update statement."""
    return insert(table).prefix_with("OR UPDATE")


def insert_or_ignore(table: _DMLTableArgument) -> Insert:
    """Construct a Spanner-specific insert-or-ignore statement."""
    return insert(table).prefix_with("OR IGNORE")
