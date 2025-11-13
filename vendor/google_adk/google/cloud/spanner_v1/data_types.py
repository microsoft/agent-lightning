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

"""Custom data types for spanner."""

import json
import types
import re
from dataclasses import dataclass
from google.protobuf.message import Message
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper


class JsonObject(dict):
    """
    Provides functionality of JSON data type in Cloud Spanner
    API, mimicking simple `dict()` behaviour and making
    all the necessary conversions under the hood.
    """

    def __init__(self, *args, **kwargs):
        self._is_null = (args, kwargs) == ((), {}) or args == (None,)
        self._is_array = len(args) and isinstance(args[0], (list, tuple))
        self._is_scalar_value = len(args) == 1 and not isinstance(args[0], (list, dict))

        # if the JSON object is represented with an array,
        # the value is contained separately
        if self._is_array:
            self._array_value = args[0]
            return

        # If it's a scalar value, set _simple_value and return early
        if self._is_scalar_value:
            self._simple_value = args[0]
            return

        if len(args) and isinstance(args[0], JsonObject):
            self._is_array = args[0]._is_array
            self._is_scalar_value = args[0]._is_scalar_value
            if self._is_array:
                self._array_value = args[0]._array_value
            elif self._is_scalar_value:
                self._simple_value = args[0]._simple_value

        if not self._is_null:
            super(JsonObject, self).__init__(*args, **kwargs)

    def __repr__(self):
        if self._is_array:
            return str(self._array_value)

        if self._is_scalar_value:
            return str(self._simple_value)

        return super(JsonObject, self).__repr__()

    @classmethod
    def from_str(cls, str_repr):
        """Initiate an object from its `str` representation.

        Args:
            str_repr (str): JSON text representation.

        Returns:
            JsonObject: JSON object.
        """
        if str_repr == "null":
            return cls()

        return cls(json.loads(str_repr))

    def serialize(self):
        """Return the object text representation.

        Returns:
            str: JSON object text representation.
        """
        if self._is_null:
            return None

        if self._is_scalar_value:
            return json.dumps(self._simple_value)

        if self._is_array:
            return json.dumps(self._array_value, sort_keys=True, separators=(",", ":"))

        return json.dumps(self, sort_keys=True, separators=(",", ":"))


@dataclass
class Interval:
    """Represents a Spanner INTERVAL type.

    An interval is a combination of months, days and nanoseconds.
    Internally, Spanner supports Interval value with the following range of individual fields:
    months: [-120000, 120000]
    days: [-3660000, 3660000]
    nanoseconds: [-316224000000000000000, 316224000000000000000]
    """

    months: int = 0
    days: int = 0
    nanos: int = 0

    def __str__(self) -> str:
        """Returns the ISO8601 duration format string representation."""
        result = ["P"]

        # Handle years and months
        if self.months:
            is_negative = self.months < 0
            abs_months = abs(self.months)
            years, months = divmod(abs_months, 12)
            if years:
                result.append(f"{'-' if is_negative else ''}{years}Y")
            if months:
                result.append(f"{'-' if is_negative else ''}{months}M")

        # Handle days
        if self.days:
            result.append(f"{self.days}D")

        # Handle time components
        if self.nanos:
            result.append("T")
            nanos = abs(self.nanos)
            is_negative = self.nanos < 0

            # Convert to hours, minutes, seconds
            nanos_per_hour = 3600000000000
            hours, nanos = divmod(nanos, nanos_per_hour)
            if hours:
                if is_negative:
                    result.append("-")
                result.append(f"{hours}H")

            nanos_per_minute = 60000000000
            minutes, nanos = divmod(nanos, nanos_per_minute)
            if minutes:
                if is_negative:
                    result.append("-")
                result.append(f"{minutes}M")

            nanos_per_second = 1000000000
            seconds, nanos_fraction = divmod(nanos, nanos_per_second)

            if seconds or nanos_fraction:
                if is_negative:
                    result.append("-")
                if seconds:
                    result.append(str(seconds))
                elif nanos_fraction:
                    result.append("0")

                if nanos_fraction:
                    nano_str = f"{nanos_fraction:09d}"
                    trimmed = nano_str.rstrip("0")
                    if len(trimmed) <= 3:
                        while len(trimmed) < 3:
                            trimmed += "0"
                    elif len(trimmed) <= 6:
                        while len(trimmed) < 6:
                            trimmed += "0"
                    else:
                        while len(trimmed) < 9:
                            trimmed += "0"
                    result.append(f".{trimmed}")
                result.append("S")

        if len(result) == 1:
            result.append("0Y")  # Special case for zero interval

        return "".join(result)

    @classmethod
    def from_str(cls, s: str) -> "Interval":
        """Parse an ISO8601 duration format string into an Interval."""
        pattern = r"^P(-?\d+Y)?(-?\d+M)?(-?\d+D)?(T(-?\d+H)?(-?\d+M)?(-?((\d+([.,]\d{1,9})?)|([.,]\d{1,9}))S)?)?$"
        match = re.match(pattern, s)
        if not match or len(s) == 1:
            raise ValueError(f"Invalid interval format: {s}")

        parts = match.groups()
        if not any(parts[:3]) and not parts[3]:
            raise ValueError(
                f"Invalid interval format: at least one component (Y/M/D/H/M/S) is required: {s}"
            )

        if parts[3] == "T" and not any(parts[4:7]):
            raise ValueError(
                f"Invalid interval format: time designator 'T' present but no time components specified: {s}"
            )

        def parse_num(s: str, suffix: str) -> int:
            if not s:
                return 0
            return int(s.rstrip(suffix))

        years = parse_num(parts[0], "Y")
        months = parse_num(parts[1], "M")
        total_months = years * 12 + months

        days = parse_num(parts[2], "D")

        nanos = 0
        if parts[3]:  # Has time component
            # Convert hours to nanoseconds
            hours = parse_num(parts[4], "H")
            nanos += hours * 3600000000000

            # Convert minutes to nanoseconds
            minutes = parse_num(parts[5], "M")
            nanos += minutes * 60000000000

            # Handle seconds and fractional seconds
            if parts[6]:
                seconds = parts[6].rstrip("S")
                if "," in seconds:
                    seconds = seconds.replace(",", ".")

                if "." in seconds:
                    sec_parts = seconds.split(".")
                    whole_seconds = sec_parts[0] if sec_parts[0] else "0"
                    nanos += int(whole_seconds) * 1000000000
                    frac = sec_parts[1][:9].ljust(9, "0")
                    frac_nanos = int(frac)
                    if seconds.startswith("-"):
                        frac_nanos = -frac_nanos
                    nanos += frac_nanos
                else:
                    nanos += int(seconds) * 1000000000

        return cls(months=total_months, days=days, nanos=nanos)


def _proto_message(bytes_val, proto_message_object):
    """Helper for :func:`get_proto_message`.
    parses serialized protocol buffer bytes data into proto message.

    Args:
        bytes_val (bytes): bytes object.
        proto_message_object (Message): Message object for parsing

    Returns:
        Message: parses serialized protocol buffer data into this message.

    Raises:
        ValueError: if the input proto_message_object is not of type Message
    """
    if isinstance(bytes_val, types.NoneType):
        return None

    if not isinstance(bytes_val, bytes):
        raise ValueError("Expected input bytes_val to be a string")

    proto_message = proto_message_object.__deepcopy__()
    proto_message.ParseFromString(bytes_val)
    return proto_message


def _proto_enum(int_val, proto_enum_object):
    """Helper for :func:`get_proto_enum`.
    parses int value into string containing the name of an enum value.

    Args:
        int_val (int): integer value.
        proto_enum_object (EnumTypeWrapper): Enum object.

    Returns:
        str: string containing the name of an enum value.

    Raises:
        ValueError: if the input proto_enum_object is not of type EnumTypeWrapper
    """
    if isinstance(int_val, types.NoneType):
        return None

    if not isinstance(int_val, int):
        raise ValueError("Expected input int_val to be a integer")

    return proto_enum_object.Name(int_val)


def get_proto_message(bytes_string, proto_message_object):
    """parses serialized protocol buffer bytes' data or its list into proto message or list of proto message.

    Args:
        bytes_string (bytes or list[bytes]): bytes object.
        proto_message_object (Message): Message object for parsing

    Returns:
        Message or list[Message]: parses serialized protocol buffer data into this message.

    Raises:
        ValueError: if the input proto_message_object is not of type Message
    """
    if isinstance(bytes_string, types.NoneType):
        return None

    if not isinstance(proto_message_object, Message):
        raise ValueError("Input proto_message_object should be of type Message")

    if not isinstance(bytes_string, (bytes, list)):
        raise ValueError(
            "Expected input bytes_string to be a string or list of strings"
        )

    if isinstance(bytes_string, list):
        return [_proto_message(item, proto_message_object) for item in bytes_string]

    return _proto_message(bytes_string, proto_message_object)


def get_proto_enum(int_value, proto_enum_object):
    """parses int or list of int values into enum or list of enum values.

    Args:
        int_value (int or list[int]): list of integer value.
        proto_enum_object (EnumTypeWrapper): Enum object.

    Returns:
        str or list[str]: list of strings containing the name of enum value.

    Raises:
        ValueError: if the input int_list is not of type list
    """
    if isinstance(int_value, types.NoneType):
        return None

    if not isinstance(proto_enum_object, EnumTypeWrapper):
        raise ValueError("Input proto_enum_object should be of type EnumTypeWrapper")

    if not isinstance(int_value, (int, list)):
        raise ValueError("Expected input int_value to be a integer or list of integers")

    if isinstance(int_value, list):
        return [_proto_enum(item, proto_enum_object) for item in int_value]

    return _proto_enum(int_value, proto_enum_object)
