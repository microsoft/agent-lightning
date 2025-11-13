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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.speech_v1p1beta1.types import resource

__protobuf__ = proto.module(
    package="google.cloud.speech.v1p1beta1",
    manifest={
        "CreatePhraseSetRequest",
        "UpdatePhraseSetRequest",
        "GetPhraseSetRequest",
        "ListPhraseSetRequest",
        "ListPhraseSetResponse",
        "DeletePhraseSetRequest",
        "CreateCustomClassRequest",
        "UpdateCustomClassRequest",
        "GetCustomClassRequest",
        "ListCustomClassesRequest",
        "ListCustomClassesResponse",
        "DeleteCustomClassRequest",
    },
)


class CreatePhraseSetRequest(proto.Message):
    r"""Message sent by the client for the ``CreatePhraseSet`` method.

    Attributes:
        parent (str):
            Required. The parent resource where this phrase set will be
            created. Format:

            ``projects/{project}/locations/{location}``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
        phrase_set_id (str):
            Required. The ID to use for the phrase set,
            which will become the final component of the
            phrase set's resource name.

            This value should restrict to letters, numbers,
            and hyphens, with the first character a letter,
            the last a letter or a number, and be 4-63
            characters.
        phrase_set (google.cloud.speech_v1p1beta1.types.PhraseSet):
            Required. The phrase set to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    phrase_set_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    phrase_set: resource.PhraseSet = proto.Field(
        proto.MESSAGE,
        number=3,
        message=resource.PhraseSet,
    )


class UpdatePhraseSetRequest(proto.Message):
    r"""Message sent by the client for the ``UpdatePhraseSet`` method.

    Attributes:
        phrase_set (google.cloud.speech_v1p1beta1.types.PhraseSet):
            Required. The phrase set to update.

            The phrase set's ``name`` field is used to identify the set
            to be updated. Format:

            ``projects/{project}/locations/{location}/phraseSets/{phrase_set}``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            The list of fields to be updated.
    """

    phrase_set: resource.PhraseSet = proto.Field(
        proto.MESSAGE,
        number=1,
        message=resource.PhraseSet,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class GetPhraseSetRequest(proto.Message):
    r"""Message sent by the client for the ``GetPhraseSet`` method.

    Attributes:
        name (str):
            Required. The name of the phrase set to retrieve. Format:

            ``projects/{project}/locations/{location}/phraseSets/{phrase_set}``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListPhraseSetRequest(proto.Message):
    r"""Message sent by the client for the ``ListPhraseSet`` method.

    Attributes:
        parent (str):
            Required. The parent, which owns this collection of phrase
            set. Format:

            ``projects/{project}/locations/{location}``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
        page_size (int):
            The maximum number of phrase sets to return.
            The service may return fewer than this value. If
            unspecified, at most 50 phrase sets will be
            returned. The maximum value is 1000; values
            above 1000 will be coerced to 1000.
        page_token (str):
            A page token, received from a previous ``ListPhraseSet``
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``ListPhraseSet`` must match the call that provided the page
            token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListPhraseSetResponse(proto.Message):
    r"""Message returned to the client by the ``ListPhraseSet`` method.

    Attributes:
        phrase_sets (MutableSequence[google.cloud.speech_v1p1beta1.types.PhraseSet]):
            The phrase set.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    phrase_sets: MutableSequence[resource.PhraseSet] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=resource.PhraseSet,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeletePhraseSetRequest(proto.Message):
    r"""Message sent by the client for the ``DeletePhraseSet`` method.

    Attributes:
        name (str):
            Required. The name of the phrase set to delete. Format:

            ``projects/{project}/locations/{location}/phraseSets/{phrase_set}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateCustomClassRequest(proto.Message):
    r"""Message sent by the client for the ``CreateCustomClass`` method.

    Attributes:
        parent (str):
            Required. The parent resource where this custom class will
            be created. Format:

            ``projects/{project}/locations/{location}/customClasses``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
        custom_class_id (str):
            Required. The ID to use for the custom class,
            which will become the final component of the
            custom class' resource name.

            This value should restrict to letters, numbers,
            and hyphens, with the first character a letter,
            the last a letter or a number, and be 4-63
            characters.
        custom_class (google.cloud.speech_v1p1beta1.types.CustomClass):
            Required. The custom class to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    custom_class_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    custom_class: resource.CustomClass = proto.Field(
        proto.MESSAGE,
        number=3,
        message=resource.CustomClass,
    )


class UpdateCustomClassRequest(proto.Message):
    r"""Message sent by the client for the ``UpdateCustomClass`` method.

    Attributes:
        custom_class (google.cloud.speech_v1p1beta1.types.CustomClass):
            Required. The custom class to update.

            The custom class's ``name`` field is used to identify the
            custom class to be updated. Format:

            ``projects/{project}/locations/{location}/customClasses/{custom_class}``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            The list of fields to be updated.
    """

    custom_class: resource.CustomClass = proto.Field(
        proto.MESSAGE,
        number=1,
        message=resource.CustomClass,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class GetCustomClassRequest(proto.Message):
    r"""Message sent by the client for the ``GetCustomClass`` method.

    Attributes:
        name (str):
            Required. The name of the custom class to retrieve. Format:

            ``projects/{project}/locations/{location}/customClasses/{custom_class}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListCustomClassesRequest(proto.Message):
    r"""Message sent by the client for the ``ListCustomClasses`` method.

    Attributes:
        parent (str):
            Required. The parent, which owns this collection of custom
            classes. Format:

            ``projects/{project}/locations/{location}/customClasses``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
        page_size (int):
            The maximum number of custom classes to
            return. The service may return fewer than this
            value. If unspecified, at most 50 custom classes
            will be returned. The maximum value is 1000;
            values above 1000 will be coerced to 1000.
        page_token (str):
            A page token, received from a previous ``ListCustomClass``
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``ListCustomClass`` must match the call that provided the
            page token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListCustomClassesResponse(proto.Message):
    r"""Message returned to the client by the ``ListCustomClasses`` method.

    Attributes:
        custom_classes (MutableSequence[google.cloud.speech_v1p1beta1.types.CustomClass]):
            The custom classes.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    custom_classes: MutableSequence[resource.CustomClass] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=resource.CustomClass,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteCustomClassRequest(proto.Message):
    r"""Message sent by the client for the ``DeleteCustomClass`` method.

    Attributes:
        name (str):
            Required. The name of the custom class to delete. Format:

            ``projects/{project}/locations/{location}/customClasses/{custom_class}``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
