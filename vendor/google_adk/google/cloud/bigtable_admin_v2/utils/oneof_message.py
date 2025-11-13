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
#
import collections.abc
import proto


class OneofMessage(proto.Message):
    def _get_oneof_field_from_key(self, key):
        """Given a field name, return the corresponding oneof associated with it. If it doesn't exist, return None."""

        oneof_type = None

        try:
            oneof_type = self._meta.fields[key].oneof
        except KeyError:
            # Underscores may be appended to field names
            # that collide with python or proto-plus keywords.
            # In case a key only exists with a `_` suffix, coerce the key
            # to include the `_` suffix. It's not possible to
            # natively define the same field with a trailing underscore in protobuf.
            # See related issue
            # https://github.com/googleapis/python-api-core/issues/227
            if f"{key}_" in self._meta.fields:
                key = f"{key}_"
                oneof_type = self._meta.fields[key].oneof

        return oneof_type

    def __init__(
        self,
        mapping=None,
        *,
        ignore_unknown_fields=False,
        **kwargs,
    ):
        # We accept several things for `mapping`:
        #   * An instance of this class.
        #   * An instance of the underlying protobuf descriptor class.
        #   * A dict
        #   * Nothing (keyword arguments only).
        #
        #
        # Check for oneofs collisions in the parameters provided. Extract a set of
        # all fields that are set from the mappings + kwargs combined.
        mapping_fields = set(kwargs.keys())

        if mapping is None:
            pass
        elif isinstance(mapping, collections.abc.Mapping):
            mapping_fields.update(mapping.keys())
        elif isinstance(mapping, self._meta.pb):
            mapping_fields.update(field.name for field, _ in mapping.ListFields())
        elif isinstance(mapping, type(self)):
            mapping_fields.update(field.name for field, _ in mapping._pb.ListFields())
        else:
            # Sanity check: Did we get something not a map? Error if so.
            raise TypeError(
                "Invalid constructor input for %s: %r"
                % (
                    self.__class__.__name__,
                    mapping,
                )
            )

        oneofs = set()

        for field in mapping_fields:
            oneof_field = self._get_oneof_field_from_key(field)
            if oneof_field is not None:
                if oneof_field in oneofs:
                    raise ValueError(
                        "Invalid constructor input for %s: Multiple fields defined for oneof %s"
                        % (self.__class__.__name__, oneof_field)
                    )
                else:
                    oneofs.add(oneof_field)

        super().__init__(mapping, ignore_unknown_fields=ignore_unknown_fields, **kwargs)

    def __setattr__(self, key, value):
        # Oneof check: Only set the value of an existing oneof field
        # if the field being overridden is the same as the field already set
        # for the oneof.
        oneof = self._get_oneof_field_from_key(key)
        if (
            oneof is not None
            and self._pb.HasField(oneof)
            and self._pb.WhichOneof(oneof) != key
        ):
            raise ValueError(
                "Overriding the field set for oneof %s with a different field %s"
                % (oneof, key)
            )
        super().__setattr__(key, value)
