# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
from typing import List, MutableSequence, Optional, TypedDict, Union

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils as aip_utils
from google.cloud.aiplatform_v1beta1 import types
from vertexai import generative_models
from vertexai.agent_engines import _utils

_LOGGER = base.Logger(__name__)
_DEFAULT_VERSION = "V0"
_ExampleLike = Union[
    types.ContentsExample,
    types.StoredContentsExample,
    types.Example,
]
_ExampleLikeDict = Union[
    "_ContentsExampleDict",
    "_StoredContentsExampleDict",
    "_ExampleDict",
]


# The set of *OrDict types are used to enforce the structure of examples that
# might contain non-dict objects like generative_models.Content. We need to keep
# track of them for the conversion logic into Example-like protos, because the
# non-dict objects might be nested deep inside the higher-level dicts.
_ContentOrDict = Union[generative_models.Content, "_ContentDict"]
_ExampleLikeOrDict = Union[
    "ContentsExample",
    "StoredContentsExample",
    "Example",
]


class _ContentDict(TypedDict, total=False):
    parts: Optional[List[TypedDict]]
    role: Optional[str]


class _ExpectedContentDict(TypedDict):
    content: _ContentDict


class _ContentsExampleDict(TypedDict):
    contents: List[_ContentDict]
    expected_contents: List[_ExpectedContentDict]


class _SearchKeyGenerationMethodDict(TypedDict):
    last_entry: TypedDict


class _StoredContentsExampleDict(TypedDict, total=False):
    search_key: Optional[str]
    contents_example: _ContentsExampleDict
    search_key_generation_method: Optional[_SearchKeyGenerationMethodDict]


class _ExampleDict(TypedDict, total=False):
    display_name: Optional[str]
    stored_contents_example: _StoredContentsExampleDict
    example_id: Optional[str]


# TypedDict is used to enforce the structure of the JSON objects that are used
# to represent examples that can be converted into Example-like protos.


class ExpectedContent(TypedDict):
    """A single step of the expected output."""

    content: _ContentOrDict
    """Required. The expected output for a single step's content."""


class ContentsExample(TypedDict):
    """A single example of a conversation with the model."""

    contents: List[_ContentOrDict]
    """Required. The content of the conversation with the model that resulted in the expected output."""

    expected_contents: List[ExpectedContent]
    """Required. The expected output for the given `contents`.

    To represent multi-step reasoning, this is a repeated field that contains
    the iterative steps of the expected output.
    """


class StoredContentsExample(TypedDict, total=False):
    """A single example to be used with GenerateContent.

    It augments a ContentsExample alongside information required for storage and
    retrieval with Example Store.
    """

    search_key: Optional[str]
    """Optional. A search key that can be used to retrieve the example."""

    contents_example: ContentsExample
    """Required. A chat history and its expected outcome to be used with GenerateContent."""

    search_key_generation_method: Optional[_SearchKeyGenerationMethodDict]
    """Optional. The method used to generate the search key from contents_example.contents.

    This is ignored if `search_key` is provided.
    """


class Example(TypedDict, total=False):
    """A single example to upload or read from the Example Store."""

    display_name: Optional[str]
    """"Optional. The display name for the example."""

    stored_contents_example: StoredContentsExample
    """Required. An example of chat history and its expected outcome to be used with GenerateContent."""

    example_id: Optional[str]
    """Optional. Unique identifier of an example.

    If not specified when upserting new examples, the example_id will be generated.
    """


class ContentSearchKey(TypedDict):
    """The chat history to use to generate the search key for retrieval."""

    contents: List[_ContentOrDict]
    """The conversation for generating a search key."""

    search_key_generation_method: _SearchKeyGenerationMethodDict
    """The method of generating a search key."""


class ExamplesArrayFilter(TypedDict):
    """Filters for examples' array metadata fields.

    An array field is example metadata where multiple values are attributed to a
    single example.
    """

    values: List[str]
    """The values by which to filter examples."""

    array_operator: types.ExamplesArrayFilter.ArrayOperator
    """The logic to use for filtering."""


class StoredContentsExampleParameters(TypedDict, total=False):
    """The filters that will be used to search StoredContentsExamples.

    If a field is unspecified, then no filtering for that field will be applied.
    """

    stored_contents_example_key: Optional[Union[str, ContentSearchKey]]
    """The key to use for retrieval."""

    function_names: Optional[ExamplesArrayFilter]
    """The function names for filtering."""


class StoredContentsExampleFilter(TypedDict, total=False):
    """The filters that will be used to remove or fetch StoredContentsExamples.

    If a field is unspecified, then no filtering for that field will be applied.
    """

    search_keys: Optional[List[str]]
    """The search keys for filtering.

    Only examples with one of the specified search keys are eligible to be returned.
    """

    function_names: Optional[ExamplesArrayFilter]
    """The function names for filtering."""


# The set of _isinstance_* functions are used to determine if a dictionary is
# an instance of a TypedDict defined in this package (since they are not
# natively supported by TypedDicts and we can avoid depending on pydantic).


def _isinstance_content_dict(obj: _utils.JsonDict) -> bool:
    return isinstance(obj, dict) and "parts" in obj


def _isinstance_content_or_dict(obj: _ContentOrDict) -> bool:
    return isinstance(obj, generative_models.Content) or _isinstance_content_dict(obj)


def _isinstance_expected_content_dict(obj: ExpectedContent) -> bool:
    return (
        isinstance(obj, dict)
        and "content" in obj
        and _isinstance_content_or_dict(obj["content"])
    )


def _isinstance_contents_example_dict(obj: ContentsExample) -> bool:
    return (
        isinstance(obj, dict)
        and "contents" in obj
        and all(_isinstance_content_or_dict(content) for content in obj["contents"])
        and (
            "expected_contents" in obj
            and all(
                _isinstance_expected_content_dict(expected_content)
                for expected_content in obj["expected_contents"]
            )
        )
        or (
            "expectedContents" in obj
            and all(
                _isinstance_expected_content_dict(expected_content)
                for expected_content in obj["expectedContents"]
            )
        )
    )


def _isinstance_stored_contents_example_dict(
    obj: StoredContentsExample,
) -> bool:
    return (
        isinstance(obj, dict)
        and (
            "contents_example" in obj
            and _isinstance_contents_example_dict(obj["contents_example"])
        )
        or (
            "contentsExample" in obj
            and _isinstance_contents_example_dict(obj["contentsExample"])
        )
    )


def _isinstance_example_dict(obj: Example) -> bool:
    return (
        isinstance(obj, dict)
        and (
            "stored_contents_example" in obj
            and _isinstance_stored_contents_example_dict(obj["stored_contents_example"])
        )
        or (
            "storedContentsExample" in obj
            and _isinstance_stored_contents_example_dict(obj["storedContentsExample"])
        )
    )


def _isinstance_content_search_key_dict(obj: ContentSearchKey) -> bool:
    return (
        isinstance(obj, dict)
        and "contents" in obj
        and all(_isinstance_content_or_dict(content) for content in obj["contents"])
        and "search_key_generation_method" in obj
        and isinstance(obj["search_key_generation_method"], dict)
    )


def _isinstance_stored_contents_example_parameters_str_key_dict(
    obj: StoredContentsExampleParameters,
) -> bool:
    return isinstance(obj, dict) and (
        "stored_contents_example_key" in obj
        and isinstance(obj["stored_contents_example_key"], str)
    )


def _isinstance_stored_contents_example_parameters_content_search_key_dict(
    obj: StoredContentsExampleParameters,
) -> bool:
    return isinstance(obj, dict) and (
        "stored_contents_example_key" in obj
        and _isinstance_content_search_key_dict(obj["stored_contents_example_key"])
    )


# We have a two step process of converting a dictionary to an Content-like proto
# because we need to handle the case where the dictionary contains a
# generative_models.Content object, which is not a dict.


def _coerce_to_dict(
    obj: Union[
        StoredContentsExampleParameters,
        _ExampleLikeOrDict,
        ExpectedContent,
        _ContentOrDict,
    ],
):
    if isinstance(obj, generative_models.Content):
        return obj.to_dict()
    elif _isinstance_content_dict(obj):
        return obj
    elif _isinstance_expected_content_dict(obj):
        return _ExpectedContentDict(content=_coerce_to_dict(obj["content"]))
    elif _isinstance_contents_example_dict(obj):
        if "expected_contents" in obj:
            expected_contents_dict = obj["expected_contents"]
        elif "expectedContents" in obj:
            expected_contents_dict = obj["expectedContents"]
        else:
            raise ValueError(f"Unsupported contents example: {obj}")
        return _ContentsExampleDict(
            contents=[_coerce_to_dict(content) for content in obj["contents"]],
            expected_contents=[
                _coerce_to_dict(content) for content in expected_contents_dict
            ],
        )
    elif _isinstance_stored_contents_example_dict(obj):
        if "contents_example" in obj:
            example_dict = _StoredContentsExampleDict(
                contents_example=_coerce_to_dict(obj["contents_example"]),
            )
        elif "contentsExample" in obj:
            example_dict = _StoredContentsExampleDict(
                contents_example=_coerce_to_dict(obj["contentsExample"]),
            )
        else:
            raise ValueError(f"Unsupported contents example: {obj}")
        for dict_key, pb_key in (
            ("search_key", "search_key"),
            ("search_key", "searchKey"),
            ("search_key_generation_method", "search_key_generation_method"),
            ("search_key_generation_method", "searchKeyGenerationMethod"),
        ):
            if pb_key in obj:
                example_dict[dict_key] = obj[pb_key]
        return example_dict
    elif _isinstance_example_dict(obj):
        if "stored_contents_example" in obj:
            example_dict = _ExampleDict(
                stored_contents_example=_coerce_to_dict(obj["stored_contents_example"]),
            )
        elif "storedContentsExample" in obj:
            example_dict = _ExampleDict(
                stored_contents_example=_coerce_to_dict(obj["storedContentsExample"]),
            )
        else:
            raise ValueError(f"Unsupported example: {obj}")
        for dict_key, pb_key in (
            ("display_name", "display_name"),
            ("display_name", "displayName"),
            ("example_id", "example_id"),
            ("example_id", "exampleId"),
        ):
            if pb_key in obj:
                example_dict[dict_key] = obj[pb_key]
        return example_dict
    elif _isinstance_content_search_key_dict(obj):
        return ContentSearchKey(
            contents=[_coerce_to_dict(content) for content in obj["contents"]],
            search_key_generation_method=obj["search_key_generation_method"],
        )
    elif _isinstance_stored_contents_example_parameters_str_key_dict(obj):
        parameters = {"search_key": obj["stored_contents_example_key"]}
        if "function_names" in obj:
            parameters["function_names"] = obj["function_names"]
        return parameters
    elif _isinstance_stored_contents_example_parameters_content_search_key_dict(obj):
        parameters = {
            "content_search_key": _coerce_to_dict(obj["stored_contents_example_key"]),
        }
        if "function_names" in obj:
            parameters["function_names"] = obj["function_names"]
        return parameters
    elif isinstance(obj, dict):
        raise TypeError(f"Unsupported example: {obj}")
    raise TypeError(f"Unsupported example type: {type(obj)}")


def _coerce_to_example(
    example: Union[_ExampleLike, _ExampleLikeOrDict],
) -> types.Example:
    if isinstance(example, types.ContentsExample):
        return types.Example(
            stored_contents_example=types.StoredContentsExample(
                contents_example=example,
                search_key_generation_method=types.StoredContentsExample.SearchKeyGenerationMethod(
                    last_entry=types.StoredContentsExample.SearchKeyGenerationMethod.LastEntry()
                ),
            ),
        )
    elif isinstance(example, types.StoredContentsExample):
        return types.Example(
            stored_contents_example=example,
        )
    elif isinstance(example, types.Example):
        return example
    elif isinstance(example, dict):
        example_dict = _coerce_to_dict(example)
        if _isinstance_contents_example_dict(example_dict):
            return _coerce_to_example(
                _utils.to_proto(example_dict, types.ContentsExample())
            )
        elif _isinstance_stored_contents_example_dict(example_dict):
            return _coerce_to_example(
                _utils.to_proto(example_dict, types.StoredContentsExample())
            )
        elif _isinstance_example_dict(example_dict):
            return _coerce_to_example(_utils.to_proto(example_dict, types.Example()))
        raise ValueError(f"Unsupported example: {example}")
    raise TypeError(f"Unsupported example type: {type(example)}")


class ExampleStore(base.VertexAiResourceNounWithFutureManager):
    """Represents a Vertex AI Example Store resource."""

    client_class = aip_utils.ExampleStoreClientWithOverride
    _resource_noun = "example_store"
    _getter_method = "get_example_store"
    _list_method = "list_example_stores"
    _delete_method = "delete_example_store"
    _parse_resource_name_method = "parse_example_store_path"
    _format_resource_name_method = "example_store_path"

    def __init__(self, example_store_name: str):
        """Retrieves an Example Store.

        Args:
            example_store_name (str):
                Required. A fully-qualified resource name or ID such as
                "projects/123/locations/us-central1/exampleStores/456" or
                "456" when project and location are initialized or passed.
        """
        super().__init__(resource_name=example_store_name)
        self._gca_resource = self._get_gca_resource(resource_name=example_store_name)

    @classmethod
    def create(
        cls,
        *,
        example_store_config: Union[_utils.JsonDict, types.ExampleStoreConfig],
        example_store_name: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ExampleStore":
        """Creates a new Example Store.

        Example Stores manage and retrieve examples to help with LLM reasoning.

        Args:
            example_store_config (Union[dict[str, Any], ExampleStoreConfig]):
                Required. The configuration of the Example Store. It includes
                the embedding model to be used for vector embedding (e.g.
                "textembedding-gecko@003", "text-embedding-004",
                "text-multilingual-embedding-002").
            example_store_name (str):
                Optional. A fully-qualified Example Store resource name or ID
                such as "projects/123/locations/us-central1/exampleStores/456"
                or "456" when project and location are initialized or passed. If
                specifying the Example Store ID, it should be 4-63 characters,
                valid characters are lowercase letters, numbers and hyphens
                ("-"), and it should start with a number or a lower-case letter.
                If not provided, Vertex AI will generate a value for this ID.
            display_name (str):
                Optional. The user-defined name of the Example Store.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
            description (str):
                Optional. The description of the Example Store.

        Returns:
            ExampleStore: The Example Store that was created.
        """
        sdk_resource = cls.__new__(cls)
        base.VertexAiResourceNounWithFutureManager.__init__(
            sdk_resource, resource_name=example_store_name
        )
        operation_future = sdk_resource.api_client.create_example_store(
            parent=initializer.global_config.common_location_path(
                project=initializer.global_config.project,
                location=initializer.global_config.location,
            ),
            example_store=types.ExampleStore(
                name=example_store_name,
                display_name=display_name or cls._generate_display_name(),
                description=description,
                example_store_config=_utils.to_proto(
                    example_store_config,
                    types.ExampleStoreConfig,
                ),
            ),
        )
        _LOGGER.log_create_with_lro(cls, operation_future)
        created_resource = operation_future.result()
        _LOGGER.log_create_complete(cls, created_resource, cls._resource_noun)
        sdk_resource._gca_resource = created_resource
        return sdk_resource

    @property
    def resource_name(self) -> str:
        """Fully-qualified resource name for the Example Store."""
        return self.gca_resource.name

    def upsert_examples(
        self,
        examples: MutableSequence[Union[_ExampleLike, _ExampleLikeOrDict]],
        overwrite: bool = False,
        **kwargs,
    ) -> _utils.JsonDict:
        """Upserts examples with the specified parameters.

        Args:
            examples (List[Union[dict, ContentsExample, StoredContentsExample, Example]]):
                Required. A list of examples to be created/updated. Each example
                can be either a ContentsExample, StoredContentsExample, Example,
                or a dictionary that can be converted to an Example.
            overwrite (bool):
                Optional. A flag for determining if examples can be overwritten.
                If `overwrite` is `True`, duplicates will be overwritten.
                If `overwrite` is `False`, duplicates will be rejected.
                It defaults to `False`.
            **kwargs:
                Optional. Any other arguments to pass to the underlying service.

        Returns:
            dict[str, Any]: A dictionary containing a list of results for
            creating or updating examples in the Example Store. Each
            UpsertResult is either a successfully created/updated example or a
            status with an error message.

        Raises:
            TypeError: If an example is not a StoredContentsExample, an Example,
            or a dictionary that can be converted to an Example.
            google.protobuf.json_format.ParseError: If an example is a
            dictionary that cannot be converted to an Example.
        """
        request = types.UpsertExamplesRequest(
            example_store=self.resource_name,
            examples=[_coerce_to_example(example) for example in examples],
            overwrite=overwrite,
        )
        response: types.UpsertExamplesResponse = self.api_client.upsert_examples(
            request, **kwargs
        )
        return _utils.to_dict(response)

    def search_examples(
        self,
        parameters: StoredContentsExampleParameters,
        *,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> _utils.JsonDict:
        """Searches examples with the specified parameters.

        Args:
            parameters (StoredContentsExampleParameters):
                Required. The parameters to use for searching examples.
            top_k (int):
                Optional. The number of similar examples to return. It defaults
                to a value of 3 if unspecified.
            **kwargs:
                Optional. Any other arguments to pass to the underlying service.

        Returns:
            dict[str, Any]: A dictionary containing a list of similar examples
            from Example Store.
        """
        request = types.SearchExamplesRequest(
            example_store=self.resource_name,
            stored_contents_example_parameters=_coerce_to_dict(parameters),
        )
        if top_k:
            request.top_k = top_k
        response: types.SearchExamplesResponse = self.api_client.search_examples(
            request, **kwargs
        )
        return _utils.to_dict(response)

    def fetch_examples(
        self,
        filter: Optional[StoredContentsExampleFilter] = None,
        *,
        example_ids: Optional[MutableSequence[str]] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        **kwargs,
    ) -> _utils.JsonDict:
        """Fetches examples that match the specified filter and/or example IDs.

        Args:
            filter (StoredContentsExampleFilter):
                Optional. The metadata filter to use for fetching examples. If
                unspecified, all examples in the Example Store will be returned.
            example_ids (MutableSequence[str]):
                Optional. Example IDs to fetch. If both filter and example_ids
                are specified, then both ID and metadata filtering will be
                applied.
            page_size (int):
                Optional. The maximum number of examples to return. The service
                may return fewer than this value. If unspecified, at most 100
                examples will be returned.
            page_token (str):
                Optional. The value returned from a previous call.
            **kwargs:
                Optional. Any other arguments to pass to the underlying service.

        Returns:
            dict[str, Any]: A dictionary, optionally with the following fields:
            *   "examples": A list of examples (if any) from Example Store that
                matches the metadata filter and/or example_ids.
            *   "nextPageToken": The next page's token (if any).
        """
        request = types.FetchExamplesRequest(
            example_store=self.resource_name,
            stored_contents_example_filter=filter,
            page_size=page_size,
            page_token=page_token,
            example_ids=example_ids,
        )
        response: types.FetchExamplesResponse = self.api_client.fetch_examples(
            request, **kwargs
        )
        return _utils.to_dict(response)

    def remove_examples(
        self,
        filter: Optional[StoredContentsExampleFilter] = None,
        *,
        example_ids: Optional[MutableSequence[str]] = None,
        **kwargs,
    ) -> _utils.JsonDict:
        """Removes examples that match the filter and/or example IDs.

        Warning: If nothing is specified, all examples in the Example Store will
        be removed. This action is irreversible.

        Args:
            filter (StoredContentsExampleFilter):
                Optional. The metadata filter to use for removing examples. If
                unspecified, all examples in the Example Store will be removed.
            example_ids (MutableSequence[str]):
                Optional. Example IDs to remove. If both filter and example_ids
                are specified, then both ID and metadata filtering will be
                applied.
            **kwargs:
                Optional. Any other arguments to pass to the underlying service.

        Returns:
            dict[str, Any]: A dictionary with the following key(s):
            *   "exampleIds": the IDs of the removed examples.
        """
        request = types.RemoveExamplesRequest(
            example_store=self.resource_name,
            stored_contents_example_filter=filter,
            example_ids=example_ids,
        )
        response: types.RemoveExamplesResponse = self.api_client.remove_examples(
            request, **kwargs
        )
        return _utils.to_dict(response)
