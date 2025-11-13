# Copyright 2017 Google LLC All rights reserved.
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

"""Types exported from this package."""

from google.cloud.spanner_v1 import Type
from google.cloud.spanner_v1 import TypeAnnotationCode
from google.cloud.spanner_v1 import TypeCode
from google.cloud.spanner_v1 import StructType
from google.protobuf.message import Message
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper


# Scalar parameter types
STRING = Type(code=TypeCode.STRING)
BYTES = Type(code=TypeCode.BYTES)
BOOL = Type(code=TypeCode.BOOL)
INT64 = Type(code=TypeCode.INT64)
FLOAT64 = Type(code=TypeCode.FLOAT64)
FLOAT32 = Type(code=TypeCode.FLOAT32)
DATE = Type(code=TypeCode.DATE)
TIMESTAMP = Type(code=TypeCode.TIMESTAMP)
NUMERIC = Type(code=TypeCode.NUMERIC)
JSON = Type(code=TypeCode.JSON)
PG_NUMERIC = Type(code=TypeCode.NUMERIC, type_annotation=TypeAnnotationCode.PG_NUMERIC)
PG_JSONB = Type(code=TypeCode.JSON, type_annotation=TypeAnnotationCode.PG_JSONB)
PG_OID = Type(code=TypeCode.INT64, type_annotation=TypeAnnotationCode.PG_OID)
INTERVAL = Type(code=TypeCode.INTERVAL)


def Array(element_type):
    """Construct an array parameter type description protobuf.

    :type element_type: :class:`~google.cloud.spanner_v1.types.Type`
    :param element_type: the type of elements of the array

    :rtype: :class:`google.cloud.spanner_v1.types.Type`
    :returns: the appropriate array-type protobuf
    """
    return Type(code=TypeCode.ARRAY, array_element_type=element_type)


def StructField(name, field_type):
    """Construct a field description protobuf.

    :type name: str
    :param name: the name of the field

    :type field_type: :class:`google.cloud.spanner_v1.types.Type`
    :param field_type: the type of the field

    :rtype: :class:`google.cloud.spanner_v1.types.StructType.Field`
    :returns: the appropriate struct-field-type protobuf
    """
    return StructType.Field(name=name, type_=field_type)


def Struct(fields):
    """Construct a struct parameter type description protobuf.

    :type fields: list of :class:`google.cloud.spanner_v1.types.StructType.Field`
    :param fields: the fields of the struct

    :rtype: :class:`type_pb2.Type`
    :returns: the appropriate struct-type protobuf
    """
    return Type(code=TypeCode.STRUCT, struct_type=StructType(fields=fields))


def ProtoMessage(proto_message_object):
    """Construct a proto message type description protobuf.

    :type proto_message_object: :class:`google.protobuf.message.Message`
    :param proto_message_object: the proto message instance

    :rtype: :class:`type_pb2.Type`
    :returns: the appropriate proto-message-type protobuf
    """
    if not isinstance(proto_message_object, Message):
        raise ValueError("Expected input object of type Proto Message.")
    return Type(
        code=TypeCode.PROTO, proto_type_fqn=proto_message_object.DESCRIPTOR.full_name
    )


def ProtoEnum(proto_enum_object):
    """Construct a proto enum type description protobuf.

    :type proto_enum_object: :class:`google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper`
    :param proto_enum_object: the proto enum instance

    :rtype: :class:`type_pb2.Type`
    :returns: the appropriate proto-enum-type protobuf
    """
    if not isinstance(proto_enum_object, EnumTypeWrapper):
        raise ValueError("Expected input object of type Proto Enum")
    return Type(
        code=TypeCode.ENUM, proto_type_fqn=proto_enum_object.DESCRIPTOR.full_name
    )
