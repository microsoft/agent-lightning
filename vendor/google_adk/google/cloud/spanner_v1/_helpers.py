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

"""Helper functions for Cloud Spanner."""

import datetime
import decimal
import math
import time
import base64
import threading

from google.protobuf.struct_pb2 import ListValue
from google.protobuf.struct_pb2 import Value
from google.protobuf.message import Message
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from google.api_core import datetime_helpers
from google.api_core.exceptions import Aborted
from google.cloud._helpers import _date_from_iso8601_date
from google.cloud.spanner_v1 import TypeCode
from google.cloud.spanner_v1 import ExecuteSqlRequest
from google.cloud.spanner_v1 import JsonObject, Interval
from google.cloud.spanner_v1 import TransactionOptions
from google.cloud.spanner_v1.request_id_header import with_request_id
from google.rpc.error_details_pb2 import RetryInfo

try:
    from opentelemetry.propagate import inject
    from opentelemetry.propagators.textmap import Setter

    HAS_OPENTELEMETRY_INSTALLED = True
except ImportError:
    HAS_OPENTELEMETRY_INSTALLED = False
from typing import List, Tuple
import random

# Validation error messages
NUMERIC_MAX_SCALE_ERR_MSG = (
    "Max scale for a numeric is 9. The requested numeric has scale {}"
)
NUMERIC_MAX_PRECISION_ERR_MSG = (
    "Max precision for the whole component of a numeric is 29. The requested "
    + "numeric has a whole component with precision {}"
)


if HAS_OPENTELEMETRY_INSTALLED:

    class OpenTelemetryContextSetter(Setter):
        """
        Used by Open Telemetry for context propagation.
        """

        def set(self, carrier: List[Tuple[str, str]], key: str, value: str) -> None:
            """
            Injects trace context into Spanner metadata

            Args:
                carrier(PubsubMessage): The Pub/Sub message which is the carrier of Open Telemetry
                data.
                key(str): The key for which the Open Telemetry context data needs to be set.
                value(str): The Open Telemetry context value to be set.

            Returns:
                None
            """
            carrier.append((key, value))


def _try_to_coerce_bytes(bytestring):
    """Try to coerce a byte string into the right thing based on Python
    version and whether or not it is base64 encoded.

    Return a text string or raise ValueError.
    """
    # Attempt to coerce using google.protobuf.Value, which will expect
    # something that is utf-8 (and base64 consistently is).
    try:
        Value(string_value=bytestring)
        return bytestring
    except ValueError:
        raise ValueError(
            "Received a bytes that is not base64 encoded. "
            "Ensure that you either send a Unicode string or a "
            "base64-encoded bytes."
        )


def _merge_query_options(base, merge):
    """Merge higher precedence QueryOptions with current QueryOptions.

    :type base:
        :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryOptions`
        or :class:`dict` or None
    :param base: The current QueryOptions that is intended for use.

    :type merge:
        :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryOptions`
        or :class:`dict` or None
    :param merge:
        The QueryOptions that have a higher priority than base. These options
        should overwrite the fields in base.

    :rtype:
        :class:`~google.cloud.spanner_v1.types.ExecuteSqlRequest.QueryOptions`
        or None
    :returns:
        QueryOptions object formed by merging the two given QueryOptions.
        If the resultant object only has empty fields, returns None.
    """
    combined = base or ExecuteSqlRequest.QueryOptions()
    if type(combined) is dict:
        combined = ExecuteSqlRequest.QueryOptions(
            optimizer_version=combined.get("optimizer_version", ""),
            optimizer_statistics_package=combined.get(
                "optimizer_statistics_package", ""
            ),
        )
    merge = merge or ExecuteSqlRequest.QueryOptions()
    if type(merge) is dict:
        merge = ExecuteSqlRequest.QueryOptions(
            optimizer_version=merge.get("optimizer_version", ""),
            optimizer_statistics_package=merge.get("optimizer_statistics_package", ""),
        )
    type(combined).pb(combined).MergeFrom(type(merge).pb(merge))
    if not combined.optimizer_version and not combined.optimizer_statistics_package:
        return None
    return combined


def _assert_numeric_precision_and_scale(value):
    """
    Asserts that input numeric field is within Spanner supported range.

    Spanner supports fixed 38 digits of precision and 9 digits of scale.
    This number can be optionally prefixed with a plus or minus sign.
    Read more here: https://cloud.google.com/spanner/docs/data-types#numeric_type

    :type value: decimal.Decimal
    :param value: The value to check for Cloud Spanner compatibility.

    :raises NotSupportedError: If value is not within supported precision or scale of Spanner.
    """
    scale = value.as_tuple().exponent
    precision = len(value.as_tuple().digits)

    if scale < -9:
        raise ValueError(NUMERIC_MAX_SCALE_ERR_MSG.format(abs(scale)))
    if precision + scale > 29:
        raise ValueError(NUMERIC_MAX_PRECISION_ERR_MSG.format(precision + scale))


def _datetime_to_rfc3339(value):
    """Format the provided datatime in the RFC 3339 format.

    :type value: datetime.datetime
    :param value: value to format

    :rtype: str
    :returns: RFC 3339 formatted datetime string
    """
    # Convert to UTC and then drop the timezone so we can append "Z" in lieu of
    # allowing isoformat to append the "+00:00" zone offset.
    value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return value.isoformat(sep="T", timespec="microseconds") + "Z"


def _datetime_to_rfc3339_nanoseconds(value):
    """Format the provided datatime in the RFC 3339 format.

    :type value: datetime_helpers.DatetimeWithNanoseconds
    :param value: value to format

    :rtype: str
    :returns: RFC 3339 formatted datetime string
    """

    if value.nanosecond == 0:
        return _datetime_to_rfc3339(value)
    nanos = str(value.nanosecond).rjust(9, "0").rstrip("0")
    # Convert to UTC and then drop the timezone so we can append "Z" in lieu of
    # allowing isoformat to append the "+00:00" zone offset.
    value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return "{}.{}Z".format(value.isoformat(sep="T", timespec="seconds"), nanos)


def _make_value_pb(value):
    """Helper for :func:`_make_list_value_pbs`.

    :type value: scalar value
    :param value: value to convert

    :rtype: :class:`~google.protobuf.struct_pb2.Value`
    :returns: value protobufs
    :raises ValueError: if value is not of a known scalar type.
    """
    if value is None:
        return Value(null_value="NULL_VALUE")
    if isinstance(value, (list, tuple)):
        return Value(list_value=_make_list_value_pb(value))
    if isinstance(value, bool):
        return Value(bool_value=value)
    if isinstance(value, int):
        return Value(string_value=str(value))
    if isinstance(value, float):
        if math.isnan(value):
            return Value(string_value="NaN")
        if math.isinf(value):
            if value > 0:
                return Value(string_value="Infinity")
            else:
                return Value(string_value="-Infinity")
        return Value(number_value=value)
    if isinstance(value, datetime_helpers.DatetimeWithNanoseconds):
        return Value(string_value=_datetime_to_rfc3339_nanoseconds(value))
    if isinstance(value, datetime.datetime):
        return Value(string_value=_datetime_to_rfc3339(value))
    if isinstance(value, datetime.date):
        return Value(string_value=value.isoformat())
    if isinstance(value, bytes):
        value = _try_to_coerce_bytes(value)
        return Value(string_value=value)
    if isinstance(value, str):
        return Value(string_value=value)
    if isinstance(value, ListValue):
        return Value(list_value=value)
    if isinstance(value, decimal.Decimal):
        _assert_numeric_precision_and_scale(value)
        return Value(string_value=str(value))
    if isinstance(value, JsonObject):
        value = value.serialize()
        if value is None:
            return Value(null_value="NULL_VALUE")
        else:
            return Value(string_value=value)
    if isinstance(value, Message):
        value = value.SerializeToString()
        if value is None:
            return Value(null_value="NULL_VALUE")
        else:
            return Value(string_value=base64.b64encode(value))
    if isinstance(value, Interval):
        return Value(string_value=str(value))

    raise ValueError("Unknown type: %s" % (value,))


def _make_list_value_pb(values):
    """Construct of ListValue protobufs.

    :type values: list of scalar
    :param values: Row data

    :rtype: :class:`~google.protobuf.struct_pb2.ListValue`
    :returns: protobuf
    """
    return ListValue(values=[_make_value_pb(value) for value in values])


def _make_list_value_pbs(values):
    """Construct a sequence of ListValue protobufs.

    :type values: list of list of scalar
    :param values: Row data

    :rtype: list of :class:`~google.protobuf.struct_pb2.ListValue`
    :returns: sequence of protobufs
    """
    return [_make_list_value_pb(row) for row in values]


def _parse_value_pb(value_pb, field_type, field_name, column_info=None):
    """Convert a Value protobuf to cell data.

    :type value_pb: :class:`~google.protobuf.struct_pb2.Value`
    :param value_pb: protobuf to convert

    :type field_type: :class:`~google.cloud.spanner_v1.types.Type`
    :param field_type: type code for the value

    :type field_name: str
    :param field_name: column name

    :type column_info: dict
    :param column_info: (Optional) dict of column name and column information.
            An object where column names as keys and custom objects as corresponding
            values for deserialization. It's specifically useful for data types like
            protobuf where deserialization logic is on user-specific code. When provided,
            the custom object enables deserialization of backend-received column data.
            If not provided, data remains serialized as bytes for Proto Messages and
            integer for Proto Enums.

    :rtype: varies on field_type
    :returns: value extracted from value_pb
    :raises ValueError: if unknown type is passed
    """
    decoder = _get_type_decoder(field_type, field_name, column_info)
    return _parse_nullable(value_pb, decoder)


def _get_type_decoder(field_type, field_name, column_info=None):
    """Returns a function that converts a Value protobuf to cell data.

    :type field_type: :class:`~google.cloud.spanner_v1.types.Type`
    :param field_type: type code for the value

    :type field_name: str
    :param field_name: column name

    :type column_info: dict
    :param column_info: (Optional) dict of column name and column information.
            An object where column names as keys and custom objects as corresponding
            values for deserialization. It's specifically useful for data types like
            protobuf where deserialization logic is on user-specific code. When provided,
            the custom object enables deserialization of backend-received column data.
            If not provided, data remains serialized as bytes for Proto Messages and
            integer for Proto Enums.

    :rtype: a function that takes a single protobuf value as an input argument
    :returns: a function that can be used to extract a value from a protobuf value
    :raises ValueError: if unknown type is passed
    """

    type_code = field_type.code
    if type_code == TypeCode.STRING:
        return _parse_string
    elif type_code == TypeCode.BYTES:
        return _parse_bytes
    elif type_code == TypeCode.BOOL:
        return _parse_bool
    elif type_code == TypeCode.INT64:
        return _parse_int64
    elif type_code == TypeCode.FLOAT64:
        return _parse_float
    elif type_code == TypeCode.FLOAT32:
        return _parse_float
    elif type_code == TypeCode.DATE:
        return _parse_date
    elif type_code == TypeCode.TIMESTAMP:
        return _parse_timestamp
    elif type_code == TypeCode.NUMERIC:
        return _parse_numeric
    elif type_code == TypeCode.JSON:
        return _parse_json
    elif type_code == TypeCode.PROTO:
        return lambda value_pb: _parse_proto(value_pb, column_info, field_name)
    elif type_code == TypeCode.ENUM:
        return lambda value_pb: _parse_proto_enum(value_pb, column_info, field_name)
    elif type_code == TypeCode.ARRAY:
        element_decoder = _get_type_decoder(
            field_type.array_element_type, field_name, column_info
        )
        return lambda value_pb: _parse_array(value_pb, element_decoder)
    elif type_code == TypeCode.STRUCT:
        element_decoders = [
            _get_type_decoder(item_field.type_, field_name, column_info)
            for item_field in field_type.struct_type.fields
        ]
        return lambda value_pb: _parse_struct(value_pb, element_decoders)
    elif type_code == TypeCode.INTERVAL:
        return _parse_interval
    else:
        raise ValueError("Unknown type: %s" % (field_type,))


def _parse_list_value_pbs(rows, row_type):
    """Convert a list of ListValue protobufs into a list of list of cell data.

    :type rows: list of :class:`~google.protobuf.struct_pb2.ListValue`
    :param rows: row data returned from a read/query

    :type row_type: :class:`~google.cloud.spanner_v1.types.StructType`
    :param row_type: row schema specification

    :rtype: list of list of cell data
    :returns: data for the rows, coerced into appropriate types
    """
    result = []
    for row in rows:
        row_data = []
        for value_pb, field in zip(row.values, row_type.fields):
            row_data.append(_parse_value_pb(value_pb, field.type_, field.name))
        result.append(row_data)
    return result


def _parse_string(value_pb) -> str:
    return value_pb.string_value


def _parse_bytes(value_pb):
    return value_pb.string_value.encode("utf8")


def _parse_bool(value_pb) -> bool:
    return value_pb.bool_value


def _parse_int64(value_pb) -> int:
    return int(value_pb.string_value)


def _parse_float(value_pb) -> float:
    if value_pb.HasField("string_value"):
        return float(value_pb.string_value)
    else:
        return value_pb.number_value


def _parse_date(value_pb):
    return _date_from_iso8601_date(value_pb.string_value)


def _parse_timestamp(value_pb):
    DatetimeWithNanoseconds = datetime_helpers.DatetimeWithNanoseconds
    return DatetimeWithNanoseconds.from_rfc3339(value_pb.string_value)


def _parse_numeric(value_pb):
    return decimal.Decimal(value_pb.string_value)


def _parse_json(value_pb):
    return JsonObject.from_str(value_pb.string_value)


def _parse_proto(value_pb, column_info, field_name):
    bytes_value = base64.b64decode(value_pb.string_value)
    if column_info is not None and column_info.get(field_name) is not None:
        default_proto_message = column_info.get(field_name)
        if isinstance(default_proto_message, Message):
            proto_message = type(default_proto_message)()
            proto_message.ParseFromString(bytes_value)
            return proto_message
    return bytes_value


def _parse_proto_enum(value_pb, column_info, field_name):
    int_value = int(value_pb.string_value)
    if column_info is not None and column_info.get(field_name) is not None:
        proto_enum = column_info.get(field_name)
        if isinstance(proto_enum, EnumTypeWrapper):
            return proto_enum.Name(int_value)
    return int_value


def _parse_array(value_pb, element_decoder) -> []:
    return [
        _parse_nullable(item_pb, element_decoder)
        for item_pb in value_pb.list_value.values
    ]


def _parse_struct(value_pb, element_decoders):
    return [
        _parse_nullable(item_pb, element_decoders[i])
        for (i, item_pb) in enumerate(value_pb.list_value.values)
    ]


def _parse_nullable(value_pb, decoder):
    if value_pb.HasField("null_value"):
        return None
    else:
        return decoder(value_pb)


def _parse_interval(value_pb):
    """Parse a Value protobuf containing an interval."""
    if hasattr(value_pb, "string_value"):
        return Interval.from_str(value_pb.string_value)
    return Interval.from_str(value_pb)


class _SessionWrapper(object):
    """Base class for objects wrapping a session.

    :type session: :class:`~google.cloud.spanner_v1.session.Session`
    :param session: the session used to perform the commit
    """

    def __init__(self, session):
        self._session = session


def _metadata_with_prefix(prefix, **kw):
    """Create RPC metadata containing a prefix.

    Args:
        prefix (str): appropriate resource path.

    Returns:
        List[Tuple[str, str]]: RPC metadata with supplied prefix
    """
    return [("google-cloud-resource-prefix", prefix)]


def _retry_on_aborted_exception(
    func,
    deadline,
    default_retry_delay=None,
):
    """
    Handles retry logic for Aborted exceptions, considering the deadline.
    """
    attempts = 0
    while True:
        try:
            attempts += 1
            return func()
        except Aborted as exc:
            _delay_until_retry(
                exc,
                deadline=deadline,
                attempts=attempts,
                default_retry_delay=default_retry_delay,
            )
            continue


def _retry(
    func,
    retry_count=5,
    delay=2,
    allowed_exceptions=None,
    before_next_retry=None,
):
    """
    Retry a function with a specified number of retries, delay between retries, and list of allowed exceptions.

    Args:
        func: The function to be retried.
        retry_count: The maximum number of times to retry the function.
        delay: The delay in seconds between retries.
        allowed_exceptions: A tuple of exceptions that are allowed to occur without triggering a retry.
                            Passing allowed_exceptions as None will lead to retrying for all exceptions.

    Returns:
        The result of the function if it is successful, or raises the last exception if all retries fail.
    """
    retries = 0
    while retries <= retry_count:
        if retries > 0 and before_next_retry:
            before_next_retry(retries, delay)

        try:
            return func()
        except Exception as exc:
            if (
                allowed_exceptions is None or exc.__class__ in allowed_exceptions
            ) and retries < retry_count:
                if (
                    allowed_exceptions is not None
                    and allowed_exceptions[exc.__class__] is not None
                ):
                    allowed_exceptions[exc.__class__](exc)
                time.sleep(delay)
                delay = delay * 2
                retries = retries + 1
            else:
                raise exc


def _check_rst_stream_error(exc):
    resumable_error = (
        any(
            resumable_message in exc.message
            for resumable_message in (
                "RST_STREAM",
                "Received unexpected EOS on DATA frame from server",
            )
        ),
    )
    if not resumable_error:
        raise


def _metadata_with_leader_aware_routing(value, **kw):
    """Create RPC metadata containing a leader aware routing header

    Args:
        value (bool): header value

    Returns:
        List[Tuple[str, str]]: RPC metadata with leader aware routing header
    """
    return ("x-goog-spanner-route-to-leader", str(value).lower())


def _metadata_with_span_context(metadata: List[Tuple[str, str]], **kw) -> None:
    """
    Appends metadata with end to end tracing header and OpenTelemetry span context .

    Args:
        metadata (list[tuple[str, str]]): The metadata carrier where the OpenTelemetry context
                                          should be injected.
    Returns:
        None
    """
    if HAS_OPENTELEMETRY_INSTALLED and metadata is not None:
        metadata.append(("x-goog-spanner-end-to-end-tracing", "true"))
        inject(setter=OpenTelemetryContextSetter(), carrier=metadata)


def _delay_until_retry(exc, deadline, attempts, default_retry_delay=None):
    """Helper for :meth:`Session.run_in_transaction`.

    Detect retryable abort, and impose server-supplied delay.

    :type exc: :class:`google.api_core.exceptions.Aborted`
    :param exc: exception for aborted transaction

    :type deadline: float
    :param deadline: maximum timestamp to continue retrying the transaction.

    :type attempts: int
    :param attempts: number of call retries
    """

    cause = exc.errors[0]
    now = time.time()
    if now >= deadline:
        raise

    delay = _get_retry_delay(cause, attempts, default_retry_delay=default_retry_delay)
    if delay is not None:
        if now + delay > deadline:
            raise

        time.sleep(delay)


def _get_retry_delay(cause, attempts, default_retry_delay=None):
    """Helper for :func:`_delay_until_retry`.

    :type exc: :class:`grpc.Call`
    :param exc: exception for aborted transaction

    :rtype: float
    :returns: seconds to wait before retrying the transaction.

    :type attempts: int
    :param attempts: number of call retries
    """
    if hasattr(cause, "trailing_metadata"):
        metadata = dict(cause.trailing_metadata())
    else:
        metadata = {}
    retry_info_pb = metadata.get("google.rpc.retryinfo-bin")
    if retry_info_pb is not None:
        retry_info = RetryInfo()
        retry_info.ParseFromString(retry_info_pb)
        nanos = retry_info.retry_delay.nanos
        return retry_info.retry_delay.seconds + nanos / 1.0e9
    if default_retry_delay is not None:
        return default_retry_delay

    return 2**attempts + random.random()


class AtomicCounter:
    def __init__(self, start_value=0):
        self.__lock = threading.Lock()
        self.__value = start_value

    @property
    def value(self):
        with self.__lock:
            return self.__value

    def increment(self, n=1):
        with self.__lock:
            self.__value += n
            return self.__value

    def __iadd__(self, n):
        """
        Defines the inplace += operator result.
        """
        with self.__lock:
            self.__value += n
            return self

    def __add__(self, n):
        """
        Defines the result of invoking: value = AtomicCounter + addable
        """
        with self.__lock:
            n += self.__value
            return n

    def __radd__(self, n):
        """
        Defines the result of invoking: value = addable + AtomicCounter
        """
        return self.__add__(n)

    def reset(self):
        with self.__lock:
            self.__value = 0


def _metadata_with_request_id(*args, **kwargs):
    return with_request_id(*args, **kwargs)


def _merge_Transaction_Options(
    defaultTransactionOptions: TransactionOptions,
    mergeTransactionOptions: TransactionOptions,
) -> TransactionOptions:
    """Merges two TransactionOptions objects.

    - Values from `mergeTransactionOptions` take precedence if set.
    - Values from `defaultTransactionOptions` are used only if missing.

    Args:
        defaultTransactionOptions (TransactionOptions): The default transaction options (fallback values).
        mergeTransactionOptions (TransactionOptions): The main transaction options (overrides when set).

    Returns:
        TransactionOptions: A merged TransactionOptions object.
    """

    if defaultTransactionOptions is None:
        return mergeTransactionOptions

    if mergeTransactionOptions is None:
        return defaultTransactionOptions

    merged_pb = TransactionOptions()._pb  # Create a new protobuf object

    # Merge defaultTransactionOptions first
    merged_pb.MergeFrom(defaultTransactionOptions._pb)

    # Merge transactionOptions, ensuring it overrides default values
    merged_pb.MergeFrom(mergeTransactionOptions._pb)

    # Convert protobuf object back into a TransactionOptions instance
    return TransactionOptions(merged_pb)
