# Copyright 2020 Google LLC
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

"""API to calculate checksums of SQL statements results."""

import hashlib
import pickle

from google.cloud.spanner_dbapi.exceptions import RetryAborted


class ResultsChecksum:
    """Cumulative checksum.

    Used to calculate a total checksum of all the results
    returned by operations executed within transaction.
    Includes methods for checksums comparison.
    These checksums are used while retrying an aborted
    transaction to check if the results of a retried transaction
    are equal to the results of the original transaction.
    """

    def __init__(self):
        self.checksum = hashlib.sha256()
        self.count = 0  # counter of consumed results

    def __len__(self):
        """Return the number of consumed results.

        :rtype: :class:`int`
        :returns: The number of results.
        """
        return self.count

    def __eq__(self, other):
        """Check if checksums are equal.

        :type other: :class:`google.cloud.spanner_dbapi.checksum.ResultsChecksum`
        :param other: Another checksum to compare with this one.
        """
        return self.checksum.digest() == other.checksum.digest()

    def consume_result(self, result):
        """Add the given result into the checksum.

        :type result: Union[int, list]
        :param result: Streamed row or row count from an UPDATE operation.
        """
        self.checksum.update(pickle.dumps(result))
        self.count += 1


def _compare_checksums(original, retried):
    from google.cloud.spanner_dbapi.transaction_helper import RETRY_ABORTED_ERROR

    """Compare the given checksums.

    Raise an error if the given checksums are not equal.

    :type original: :class:`~google.cloud.spanner_dbapi.checksum.ResultsChecksum`
    :param original: results checksum of the original transaction.

    :type retried: :class:`~google.cloud.spanner_dbapi.checksum.ResultsChecksum`
    :param retried: results checksum of the retried transaction.

    :raises: :exc:`google.cloud.spanner_dbapi.exceptions.RetryAborted` in case if checksums are not equal.
    """
    if retried != original:
        raise RetryAborted(RETRY_ABORTED_ERROR)
