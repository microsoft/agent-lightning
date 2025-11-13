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

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import content as gca_content


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ContentsExample",
        "StoredContentsExample",
    },
)


class ContentsExample(proto.Message):
    r"""A single example of a conversation with the model.

    Attributes:
        contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
            Required. The content of the conversation
            with the model that resulted in the expected
            output.
        expected_contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.ContentsExample.ExpectedContent]):
            Required. The expected output for the given ``contents``. To
            represent multi-step reasoning, this is a repeated field
            that contains the iterative steps of the expected output.
    """

    class ExpectedContent(proto.Message):
        r"""A single step of the expected output.

        Attributes:
            content (google.cloud.aiplatform_v1beta1.types.Content):
                Required. A single step's content.
        """

        content: gca_content.Content = proto.Field(
            proto.MESSAGE,
            number=1,
            message=gca_content.Content,
        )

    contents: MutableSequence[gca_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_content.Content,
    )
    expected_contents: MutableSequence[ExpectedContent] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=ExpectedContent,
    )


class StoredContentsExample(proto.Message):
    r"""A ContentsExample to be used with GenerateContent alongside
    information required for storage and retrieval with Example
    Store.

    Attributes:
        search_key (str):
            Optional. (Optional) the search key used for retrieval. If
            not provided at upload-time, the search key will be
            generated from ``contents_example.contents`` using the
            method provided by ``search_key_generation_method``. The
            generated search key will be included in retrieved examples.
        contents_example (google.cloud.aiplatform_v1beta1.types.ContentsExample):
            Required. The example to be used with
            GenerateContent.
        search_key_generation_method (google.cloud.aiplatform_v1beta1.types.StoredContentsExample.SearchKeyGenerationMethod):
            Optional. The method used to generate the search key from
            ``contents_example.contents``. This is ignored when
            uploading an example if ``search_key`` is provided.
    """

    class SearchKeyGenerationMethod(proto.Message):
        r"""Options for generating the search key from the conversation
        history.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            last_entry (google.cloud.aiplatform_v1beta1.types.StoredContentsExample.SearchKeyGenerationMethod.LastEntry):
                Use only the last entry of the conversation history
                (``contents_example.contents``) as the search key.

                This field is a member of `oneof`_ ``method``.
        """

        class LastEntry(proto.Message):
            r"""Configuration for using only the last entry of the
            conversation history as the search key.

            """

        last_entry: "StoredContentsExample.SearchKeyGenerationMethod.LastEntry" = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="method",
                message="StoredContentsExample.SearchKeyGenerationMethod.LastEntry",
            )
        )

    search_key: str = proto.Field(
        proto.STRING,
        number=1,
    )
    contents_example: "ContentsExample" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ContentsExample",
    )
    search_key_generation_method: SearchKeyGenerationMethod = proto.Field(
        proto.MESSAGE,
        number=3,
        message=SearchKeyGenerationMethod,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
