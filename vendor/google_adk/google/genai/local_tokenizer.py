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

"""[Experimental] Text Only Local Tokenizer."""

import logging
from typing import Any, Iterable
from typing import Optional, Union

from sentencepiece import sentencepiece_model_pb2

from . import _common
from . import _local_tokenizer_loader as loader
from . import _transformers as t
from . import types

logger = logging.getLogger("google_genai.local_tokenizer")

__all__ = [
    "_parse_hex_byte",
    "_token_str_to_bytes",
    "LocalTokenizer",
    "_TextsAccumulator",
]


class _TextsAccumulator:
  """Accumulates countable texts from `Content` and `Tool` objects.

  This class is responsible for traversing complex `Content` and `Tool`
  objects and extracting all the text content that should be included when
  calculating token counts.

  A key feature of this class is its ability to detect unsupported fields in
  `Content` objects. If a user provides a `Content` object with fields that
  this local tokenizer doesn't recognize (e.g., new fields added in a future
  API update), this class will log a warning.

  The detection mechanism for `Content` objects works by recursively building
  a "counted" version of the input object. This "counted" object only
  contains the data that was successfully processed and added to the text
  list for tokenization. After traversing the input, the original `Content`
  object is compared to the "counted" object. If they don't match, it
  signifies the presence of unsupported fields, and a warning is logged.
  """

  def __init__(self) -> None:
    self._texts: list[str] = []

  def get_texts(self) -> Iterable[str]:
    return self._texts

  def add_contents(self, contents: Iterable[types.Content]) -> None:
    for content in contents:
      self.add_content(content)

  def add_content(self, content: types.Content) -> None:
    counted_content = types.Content(parts=[], role=content.role)
    if content.parts:
      for part in content.parts:
        assert counted_content.parts is not None
        counted_part = types.Part()
        if part.file_data is not None or part.inline_data is not None:
          raise ValueError(
              "LocalTokenizers do not support non-text content types."
          )
        if part.video_metadata is not None:
          counted_part.video_metadata = part.video_metadata
        if part.function_call is not None:
          self.add_function_call(part.function_call)
          counted_part.function_call = part.function_call
        if part.function_response is not None:
          self.add_function_response(part.function_response)
          counted_part.function_response = part.function_response
        if part.text is not None:
          counted_part.text = part.text
          self._texts.append(part.text)
        counted_content.parts.append(counted_part)

    if content.model_dump(exclude_none=True) != counted_content.model_dump(
        exclude_none=True
    ):
      logger.warning(
          "Content contains unsupported types for token counting. Supported"
          f" fields {counted_content}. Got {content}."
      )

  def add_function_call(self, function_call: types.FunctionCall) -> None:
    """Processes a function call and adds relevant text to the accumulator.

    Args:
        function_call: The function call to process.
    """
    if function_call.name:
      self._texts.append(function_call.name)
    counted_function_call = types.FunctionCall(name=function_call.name)
    if function_call.args:
      counted_args = self._dict_traverse(function_call.args)
      counted_function_call.args = counted_args

  def add_tool(self, tool: types.Tool) -> types.Tool:
    counted_tool = types.Tool(function_declarations=[])
    if tool.function_declarations:
      for function_declaration in tool.function_declarations:
        counted_function_declaration = self._function_declaration_traverse(
            function_declaration
        )
        if counted_tool.function_declarations is None:
          counted_tool.function_declarations = []
        counted_tool.function_declarations.append(counted_function_declaration)

    return counted_tool

  def add_tools(self, tools: Iterable[types.Tool]) -> None:
    for tool in tools:
      self.add_tool(tool)

  def add_function_responses(
      self, function_responses: Iterable[types.FunctionResponse]
  ) -> None:
    for function_response in function_responses:
      self.add_function_response(function_response)

  def add_function_response(
      self, function_response: types.FunctionResponse
  ) -> None:
    counted_function_response = types.FunctionResponse()
    if function_response.name:
      self._texts.append(function_response.name)
      counted_function_response.name = function_response.name
    if function_response.response:
      counted_response = self._dict_traverse(function_response.response)
      counted_function_response.response = counted_response

  def _function_declaration_traverse(
      self, function_declaration: types.FunctionDeclaration
  ) -> types.FunctionDeclaration:
    counted_function_declaration = types.FunctionDeclaration()
    if function_declaration.name:
      self._texts.append(function_declaration.name)
      counted_function_declaration.name = function_declaration.name
    if function_declaration.description:
      self._texts.append(function_declaration.description)
      counted_function_declaration.description = (
          function_declaration.description
      )
    if function_declaration.parameters:
      counted_parameters = self.add_schema(function_declaration.parameters)
      counted_function_declaration.parameters = counted_parameters
    if function_declaration.response:
      counted_response = self.add_schema(function_declaration.response)
      counted_function_declaration.response = counted_response
    return counted_function_declaration

  def add_schema(self, schema: types.Schema) -> types.Schema:
    """Processes a schema and adds relevant text to the accumulator.

    Args:
        schema: The schema to process.

    Returns:
        The new schema object with only countable fields.
    """
    counted_schema = types.Schema()
    if schema.type:
      counted_schema.type = schema.type
    if schema.title:
      counted_schema.title = schema.title
    if schema.default is not None:
      counted_schema.default = schema.default
    if schema.format:
      self._texts.append(schema.format)
      counted_schema.format = schema.format
    if schema.description:
      self._texts.append(schema.description)
      counted_schema.description = schema.description
    if schema.enum:
      self._texts.extend(schema.enum)
      counted_schema.enum = schema.enum
    if schema.required:
      self._texts.extend(schema.required)
      counted_schema.required = schema.required
    if schema.property_ordering:
      counted_schema.property_ordering = schema.property_ordering
    if schema.items:
      counted_schema_items = self.add_schema(schema.items)
      counted_schema.items = counted_schema_items
    if schema.properties:
      d = {}
      for key, value in schema.properties.items():
        self._texts.append(key)
        counted_value = self.add_schema(value)
        d[key] = counted_value
      counted_schema.properties = d
    if schema.example:
      counted_schema_example = self._any_traverse(schema.example)
      counted_schema.example = counted_schema_example
    return counted_schema

  def _dict_traverse(self, d: dict[str, Any]) -> dict[str, Any]:
    """Processes a dict and adds relevant text to the accumulator.

    Args:
        d: The dict to process.

    Returns:
        The new dict object with only countable fields.
    """
    counted_dict = {}
    self._texts.extend(list(d.keys()))
    for key, val in d.items():
      counted_dict[key] = self._any_traverse(val)
    return counted_dict

  def _any_traverse(self, value: Any) -> Any:
    """Processes a value and adds relevant text to the accumulator.

    Args:
        value: The value to process.

    Returns:
        The new value with only countable fields.
    """
    if isinstance(value, str):
      self._texts.append(value)
      return value
    elif isinstance(value, dict):
      return self._dict_traverse(value)
    elif isinstance(value, list):
      return [self._any_traverse(item) for item in value]
    else:
      return value


def _token_str_to_bytes(
    token: str, type: sentencepiece_model_pb2.ModelProto.SentencePiece.Type
) -> bytes:
  if type == sentencepiece_model_pb2.ModelProto.SentencePiece.Type.BYTE:
    return _parse_hex_byte(token).to_bytes(length=1, byteorder="big")
  else:
    return token.replace("▁", " ").encode("utf-8")


def _parse_hex_byte(token: str) -> int:
  """Parses a hex byte string of the form '<0xXX>' and returns the integer value.

  Raises ValueError if the input is malformed or the byte value is invalid.
  """

  if len(token) != 6:
    raise ValueError(f"Invalid byte length: {token}")
  if not token.startswith("<0x") or not token.endswith(">"):
    raise ValueError(f"Invalid byte format: {token}")

  try:
    val = int(token[3:5], 16)  # Parse the hex part directly
  except ValueError:
    raise ValueError(f"Invalid hex value: {token}")

  if val >= 256:
    raise ValueError(f"Byte value out of range: {token}")

  return val


class LocalTokenizer:
  """[Experimental] Text Only Local Tokenizer.

  This class provides a local tokenizer for text only token counting.

  LIMITATIONS:
  - Only supports text based tokenization and no multimodal tokenization.
  - Forward compatibility depends on the open-source tokenizer models for future
  Gemini versions.
  - For token counting of tools and response schemas, the `LocalTokenizer` only
  supports `types.Tool` and `types.Schema` objects. Python functions or Pydantic
  models cannot be passed directly.
  """

  def __init__(self, model_name: str):
    self._tokenizer_name = loader.get_tokenizer_name(model_name)
    self._model_proto = loader.load_model_proto(self._tokenizer_name)
    self._tokenizer = loader.get_sentencepiece(self._tokenizer_name)

  @_common.experimental_warning(
      "The SDK's local tokenizer implementation is experimental and may change"
      " in the future. It only supports text based tokenization."
  )
  def count_tokens(
      self,
      contents: Union[types.ContentListUnion, types.ContentListUnionDict],
      *,
      config: Optional[types.CountTokensConfigOrDict] = None,
  ) -> types.CountTokensResult:
    """Counts the number of tokens in a given text.

    Args:
      contents: The contents to tokenize.
      config: The configuration for counting tokens.

    Returns:
      A `CountTokensResult` containing the total number of tokens.

    Usage:

    .. code-block:: python

      from google import genai
      tokenizer = genai.LocalTokenizer(model_name='gemini-2.0-flash-001')
      result = tokenizer.count_tokens("What is your name?")
      print(result)
      # total_tokens=5
    """
    processed_contents = t.t_contents(contents)
    text_accumulator = _TextsAccumulator()
    config = types.CountTokensConfig.model_validate(config or {})
    text_accumulator.add_contents(processed_contents)
    if config.tools:
      text_accumulator.add_tools(config.tools)
    if config.generation_config and config.generation_config.response_schema:
      text_accumulator.add_schema(config.generation_config.response_schema)
    if config.system_instruction:
      text_accumulator.add_contents(t.t_contents([config.system_instruction]))
    tokens_list = self._tokenizer.encode(list(text_accumulator.get_texts()))
    return types.CountTokensResult(
        total_tokens=sum(len(tokens) for tokens in tokens_list)
    )

  @_common.experimental_warning(
      "The SDK's local tokenizer implementation is experimental and may change"
      " in the future. It only supports text based tokenization."
  )
  def compute_tokens(
      self,
      contents: Union[types.ContentListUnion, types.ContentListUnionDict],
  ) -> types.ComputeTokensResult:
    """Computes the tokens ids and string pieces in the input.

    Args:
      contents: The contents to tokenize.

    Returns:
      A `ComputeTokensResult` containing the token information.

    Usage:

    .. code-block:: python

      from google import genai
      tokenizer = genai.LocalTokenizer(model_name='gemini-2.0-flash-001')
      result = tokenizer.compute_tokens("What is your name?")
      print(result)
      # tokens_info=[TokensInfo(token_ids=[279, 329, 1313, 2508, 13], tokens=[b' What', b' is', b' your', b' name', b'?'], role='user')]
    """
    processed_contents = t.t_contents(contents)
    text_accumulator = _TextsAccumulator()
    for content in processed_contents:
      text_accumulator.add_content(content)
    tokens_protos = self._tokenizer.EncodeAsImmutableProto(
        text_accumulator.get_texts()
    )

    roles = []
    for content in processed_contents:
      if content.parts:
        for _ in content.parts:
          roles.append(content.role)

    token_infos = []
    for tokens_proto, role in zip(tokens_protos, roles):
      token_infos.append(
          types.TokensInfo(
              token_ids=[piece.id for piece in tokens_proto.pieces],
              tokens=[
                  _token_str_to_bytes(
                      piece.piece, self._model_proto.pieces[piece.id].type
                  )
                  for piece in tokens_proto.pieces
              ],
              role=role,
          )
      )
    return types.ComputeTokensResult(tokens_info=token_infos)
