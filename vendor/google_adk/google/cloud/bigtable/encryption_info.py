# Copyright 2021 Google LLC
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

"""Class for encryption info for tables and backups."""

from google.cloud.bigtable.error import Status


class EncryptionInfo:
    """Encryption information for a given resource.

    If this resource is protected with customer managed encryption, the in-use Google
    Cloud Key Management Service (KMS) key versions will be specified along with their
    status.

    :type encryption_type: int
    :param encryption_type: See :class:`enums.EncryptionInfo.EncryptionType`

    :type encryption_status: google.cloud.bigtable.encryption.Status
    :param encryption_status: The encryption status.

    :type kms_key_version: str
    :param kms_key_version: The key version used for encryption.
    """

    @classmethod
    def _from_pb(cls, info_pb):
        return cls(
            info_pb.encryption_type,
            Status(info_pb.encryption_status),
            info_pb.kms_key_version,
        )

    def __init__(self, encryption_type, encryption_status, kms_key_version):
        self.encryption_type = encryption_type
        self.encryption_status = encryption_status
        self.kms_key_version = kms_key_version

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.encryption_type == other.encryption_type
            and self.encryption_status == other.encryption_status
            and self.kms_key_version == other.kms_key_version
        )

    def __ne__(self, other):
        return not self == other
