# Copyright 2016 Google LLC All rights reserved.
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

"""Wrap representation of Spanner keys / ranges."""

from google.cloud.spanner_v1 import KeyRangePB
from google.cloud.spanner_v1 import KeySetPB

from google.cloud.spanner_v1._helpers import _make_list_value_pb
from google.cloud.spanner_v1._helpers import _make_list_value_pbs


class KeyRange(object):
    """Identify range of table rows via start / end points.

    Specify either a `start_open` or `start_closed` key, or defaults to
    `start_closed = []`.  Specify either an `end_open` or `end_closed` key,
    or defaults to `end_closed = []`.  However, at least one key has to be
    specified.  If no keys are specified, ValueError is raised.

    :type start_open: list of scalars
    :param start_open: keys identifying start of range (this key excluded)

    :type start_closed: list of scalars
    :param start_closed: keys identifying start of range (this key included)

    :type end_open: list of scalars
    :param end_open: keys identifying end of range (this key excluded)

    :type end_closed: list of scalars
    :param end_closed: keys identifying end of range (this key included)

    :raises ValueError: if no keys are specified
    """

    def __init__(
        self, start_open=None, start_closed=None, end_open=None, end_closed=None
    ):
        if not any([start_open, start_closed, end_open, end_closed]):
            raise ValueError("Must specify at least a start or end row.")

        if start_open and start_closed:
            raise ValueError("Specify one of 'start_open' / 'start_closed'.")
        elif start_open is None and start_closed is None:
            start_closed = []

        if end_open and end_closed:
            raise ValueError("Specify one of 'end_open' / 'end_closed'.")
        elif end_open is None and end_closed is None:
            end_closed = []

        self.start_open = start_open
        self.start_closed = start_closed
        self.end_open = end_open
        self.end_closed = end_closed

    def _to_pb(self):
        """Construct a KeyRange protobuf.

        :rtype: :class:`~google.cloud.spanner_v1.types.KeyRange`
        :returns: protobuf corresponding to this instance.
        """
        kwargs = {}

        if self.start_open is not None:
            kwargs["start_open"] = _make_list_value_pb(self.start_open)

        if self.start_closed is not None:
            kwargs["start_closed"] = _make_list_value_pb(self.start_closed)

        if self.end_open is not None:
            kwargs["end_open"] = _make_list_value_pb(self.end_open)

        if self.end_closed is not None:
            kwargs["end_closed"] = _make_list_value_pb(self.end_closed)

        return KeyRangePB(**kwargs)

    def _to_dict(self):
        """Return the state of the keyrange as a dict.

        :rtype: dict
        :returns: state of this instance.
        """
        mapping = {}

        if self.start_open:
            mapping["start_open"] = self.start_open

        if self.start_closed:
            mapping["start_closed"] = self.start_closed

        if self.end_open:
            mapping["end_open"] = self.end_open

        if self.end_closed:
            mapping["end_closed"] = self.end_closed

        return mapping

    def __eq__(self, other):
        """Compare by serialized state."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._to_dict() == other._to_dict()


class KeySet(object):
    """Identify table rows via keys / ranges.

    :type keys: list of list of scalars
    :param keys: keys identifying individual rows within a table.

    :type ranges: list of :class:`KeyRange`
    :param ranges: ranges identifying rows within a table.

    :type all_: boolean
    :param all_: if True, identify all rows within a table
    """

    def __init__(self, keys=(), ranges=(), all_=False):
        if all_ and (keys or ranges):
            raise ValueError("'all_' is exclusive of 'keys' / 'ranges'.")
        self.keys = list(keys)
        self.ranges = list(ranges)
        self.all_ = all_

    def _to_pb(self):
        """Construct a KeySet protobuf.

        :rtype: :class:`~google.cloud.spanner_v1.types.KeySet`
        :returns: protobuf corresponding to this instance.
        """
        if self.all_:
            return KeySetPB(all_=True)
        kwargs = {}

        if self.keys:
            kwargs["keys"] = _make_list_value_pbs(self.keys)

        if self.ranges:
            kwargs["ranges"] = [krange._to_pb() for krange in self.ranges]

        return KeySetPB(**kwargs)

    def _to_dict(self):
        """Return the state of the keyset as a dict.

        The result can be used to serialize the instance and reconstitute
        it later using :meth:`_from_dict`.

        :rtype: dict
        :returns: state of this instance.
        """
        if self.all_:
            return {"all": True}

        return {
            "keys": self.keys,
            "ranges": [keyrange._to_dict() for keyrange in self.ranges],
        }

    def __eq__(self, other):
        """Compare by serialized state."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._to_dict() == other._to_dict()

    @classmethod
    def _from_dict(cls, mapping):
        """Create an instance from the corresponding state mapping.

        :type mapping: dict
        :param mapping: the instance state.
        """
        if mapping.get("all"):
            return cls(all_=True)

        r_mappings = mapping.get("ranges", ())
        ranges = [KeyRange(**r_mapping) for r_mapping in r_mappings]

        return cls(keys=mapping.get("keys", ()), ranges=ranges)
