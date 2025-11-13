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
import warnings

with warnings.catch_warnings(record=True) as import_warning:
    import google_crc32c  # type: ignore


class _CRC32C(object):
    """
    Wrapper around ``google_crc32c`` library
    """

    warn_emitted = False

    @classmethod
    def checksum(cls, val: bytearray) -> int:
        """
        Returns the crc32c checksum of the data.
        """
        if import_warning and not cls.warn_emitted:
            cls.warn_emitted = True
            warnings.warn(
                "Using pure python implementation of `google-crc32` for ExecuteQuery response "
                "validation. This is significantly slower than the c extension. If possible, "
                "run in an environment that supports the c extension.",
                RuntimeWarning,
            )
        memory_view = memoryview(val)
        return google_crc32c.value(bytes(memory_view))
