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

"""Class for error status."""


class Status:
    """A status, comprising a code and a message.

    See: `Cloud APIs Errors <https://cloud.google.com/apis/design/errors>`_

    This is a thin wrapper for ``google.rpc.status_pb2.Status``.

    :type status_pb: google.rpc.status_pb2.Status
    :param status_pb: The status protocol buffer.
    """

    def __init__(self, status_pb):
        self.status_pb = status_pb

    @property
    def code(self):
        """The status code.

        Values are defined in ``google.rpc.code_pb2.Code``.

        See: `google.rpc.Code
        <https://github.com/googleapis/googleapis/blob/main/google/rpc/code.proto>`_

        :rtype: int
        :returns: The status code.
        """
        return self.status_pb.code

    @property
    def message(self):
        """A human readable status message.

        :rypte: str
        :returns: The status message.
        """
        return self.status_pb.message

    def __repr__(self):
        return repr(self.status_pb)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.status_pb == other.status_pb
        return NotImplemented

    def __ne__(self, other):
        return not self == other
