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

import requests
import uuid
import os
import tempfile
import hashlib
import dataclasses

import sentencepiece as spm
import functools
from sentencepiece import sentencepiece_model_pb2


@dataclasses.dataclass(frozen=True)
class _TokenizerConfig:
    model_url: str
    model_hash: str


_GEMMA_TOKENIZER = "google/gemma"

# SoT: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
_GEMINI_MODEL_NAMES = ["gemini-1.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
_GEMINI_STABLE_MODEL_NAMES = [
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
]

_TOKENIZERS = {
    _GEMMA_TOKENIZER: _TokenizerConfig(
        model_url="https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model",
        model_hash="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    )
}


def _load_file(file_url_path: str) -> bytes:
    """Loads file bytes from the given file url path."""
    resp = requests.get(file_url_path)
    resp.raise_for_status()
    return resp.content


def _is_valid_model(*, model_data: bytes, expected_hash: str) -> bool:
    """Returns true if the content is valid by checking the hash."""
    if not expected_hash:
        raise ValueError("expected_hash is required")
    return hashlib.sha256(model_data).hexdigest() == expected_hash


def _maybe_remove_file(file_path: str) -> None:
    """Removes the file if exists."""
    if not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError:
        # Don't raise if we cannot remove file.
        pass


def _maybe_load_from_cache(*, file_path: str, expected_hash: str) -> bytes:
    """Loads the content from the cache path."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "rb") as f:
        content = f.read()
    if _is_valid_model(model_data=content, expected_hash=expected_hash):
        return content

    # Cached file corrupted.
    _maybe_remove_file(file_path)


def _maybe_save_to_cache(*, cache_dir: str, cache_path: str, content: bytes) -> None:
    """Saves the content to the cache path."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = cache_dir + "." + str(uuid.uuid4()) + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(content)
        os.rename(tmp_path, cache_path)
    except OSError:
        # Don't raise if we cannot write file.
        pass


def _load_from_url(*, file_url: str, expected_hash: str) -> bytes:
    """Loads model bytes from the given file url."""
    content = _load_file(file_url)
    if not _is_valid_model(model_data=content, expected_hash=expected_hash):
        actual_hash = hashlib.sha256(content).hexdigest()
        raise ValueError(
            f"Downloaded model file is corrupted."
            f" Expected hash {expected_hash}. Got file hash {actual_hash}."
        )
    return content


def _load(*, file_url: str, expected_hash: str) -> bytes:
    """Loads model bytes from the given file url.

    1. If the find local cached file for the given url and the cached file hash
       matches the expected hash, the cached file is returned.
    2. If local cached file is not found or the hash does not match, the file is
       downloaded from the given url. And write to local cache and return the
       file bytes.
    3. If the file downloaded from the given url does not match the expected
       hash, raise ValueError.

    Args:
        file_url: The url of the file to load.
        expected_hash: The expected hash of the file.

    Returns:
        The file bytes.
    """
    model_dir = os.path.join(tempfile.gettempdir(), "vertexai_tokenizer_model")
    filename = hashlib.sha1(file_url.encode()).hexdigest()
    model_path = os.path.join(model_dir, filename)

    model_data = _maybe_load_from_cache(
        file_path=model_path, expected_hash=expected_hash
    )
    if not model_data:
        model_data = _load_from_url(file_url=file_url, expected_hash=expected_hash)

    _maybe_save_to_cache(cache_dir=model_dir, cache_path=model_path, content=model_data)
    return model_data


def _load_model_proto_bytes(tokenizer_name: str) -> bytes:
    """Loads model proto bytes from the given tokenizer name."""
    if tokenizer_name not in _TOKENIZERS:
        raise ValueError(
            f"Tokenizer {tokenizer_name} is not supported."
            f"Supported tokenizers: {list(_TOKENIZERS.keys())}"
        )
    return _load(
        file_url=_TOKENIZERS[tokenizer_name].model_url,
        expected_hash=_TOKENIZERS[tokenizer_name].model_hash,
    )


@functools.lru_cache()
def load_model_proto(tokenizer_name) -> sentencepiece_model_pb2.ModelProto:
    """Loads model proto from the given tokenizer name."""
    model_proto = sentencepiece_model_pb2.ModelProto()
    model_proto.ParseFromString(_load_model_proto_bytes(tokenizer_name))
    return model_proto


def get_tokenizer_name(model_name: str):
    """Gets the tokenizer name for the given model name."""
    if model_name in _GEMINI_MODEL_NAMES:
        return _GEMMA_TOKENIZER
    if model_name in _GEMINI_STABLE_MODEL_NAMES:
        return _GEMMA_TOKENIZER
    raise ValueError(
        f"Model {model_name} is not supported. Supported models: {', '.join(_GEMINI_STABLE_MODEL_NAMES)}.\n"  # pylint: disable=line-too-long
    )


@functools.lru_cache()
def get_sentencepiece(tokenizer_name: str) -> spm.SentencePieceProcessor:
    """Loads sentencepiece tokenizer from the given tokenizer name."""
    processor = spm.SentencePieceProcessor()
    processor.LoadFromSerializedProto(_load_model_proto_bytes(tokenizer_name))
    return processor
