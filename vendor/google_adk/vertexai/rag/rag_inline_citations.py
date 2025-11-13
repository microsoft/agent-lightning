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
"""Helper functions for processing and formatting citations from RAG generation outputs."""


def populate_cited_chunk_references(
    grounding_supports,
    grounding_chunks,
    cited_refs_dict,
) -> None:
    """Populates cited_refs_dict with URI information for all unique chunk indices found in grounding_supports.

    Args:
        grounding_supports: A list of support items, where each item might contain
          grounding chunk indices.
        grounding_chunks: A list of all available chunk items from which to retrieve
          context and URI.
        cited_refs_dict: A dictionary to populate with chunk_idx as key and URI
          as value.

    Raises:
        TypeError: If grounding_chunks is not a list, or cited_refs_dict is not a
          dictionary.
        ValueError: If grounding_chunks or cited_refs_dict is None.
                    If a chunk_item at a valid index is None.
                    If 'retrieved_context' or 'uri' attribute of a chunk is None.
        IndexError: If a chunk_idx is out of bounds for grounding_chunks.
        AttributeError: If 'retrieved_context' or 'uri' attribute is missing from
          a chunk or its context.
    """
    if grounding_chunks is None or not grounding_chunks:
        raise ValueError("grounding_chunks cannot be None or empty.")
    if grounding_supports is None or not grounding_supports:
        raise ValueError("grounding_supports cannot be None or empty.")
    if not isinstance(grounding_chunks, list):
        raise TypeError("grounding_chunks must be a list.")
    if not isinstance(grounding_supports, list):
        raise TypeError("grounding_supports must be a list.")
    if cited_refs_dict is None:
        raise ValueError("cited_refs_dict cannot be None.")
    if not isinstance(cited_refs_dict, dict):
        raise TypeError("cited_refs_dict must be a dictionary.")

    for support in grounding_supports:
        current_support_chunk_indices = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices is not None
        ):
            valid_indices = [
                idx for idx in support.grounding_chunk_indices if isinstance(idx, int)
            ]
            current_support_chunk_indices = sorted(list(set(valid_indices)))

        for chunk_idx in current_support_chunk_indices:
            if chunk_idx not in cited_refs_dict:
                if not (0 <= chunk_idx < len(grounding_chunks)):
                    raise IndexError(
                        f"Chunk index {chunk_idx} is out of bounds for grounding_chunks of size {len(grounding_chunks)}."
                    )
                chunk_item = grounding_chunks[chunk_idx]
                if chunk_item is None:
                    raise ValueError(f"Chunk item at index {chunk_idx} is None.")
                if not hasattr(chunk_item, "retrieved_context"):
                    raise AttributeError(
                        f"Chunk item at index {chunk_idx} is missing 'retrieved_context' attribute."
                    )
                retrieved_context_obj = chunk_item.retrieved_context
                if retrieved_context_obj is None:
                    raise ValueError(
                        f"Attribute 'retrieved_context' for chunk {chunk_idx} is None."
                    )
                if not hasattr(retrieved_context_obj, "uri"):
                    raise AttributeError(
                        f"retrieved_context for chunk {chunk_idx} is missing 'uri' attribute."
                    )
                uri = retrieved_context_obj.uri
                if uri is None:
                    raise ValueError(f"Attribute 'uri' for chunk {chunk_idx} is None.")
                cited_refs_dict[chunk_idx] = uri


def format_bibliography(cited_refs_dict, grounding_chunks) -> str:
    """Formats the bibliography string from the populated cited_refs_dict.

    Omits page information if page numbers are not valid (e.g., not >= 1).

    Args:
        cited_refs_dict: A dictionary with chunk_idx as key and URI as value.
          It's expected that populate_cited_chunk_references has successfully
          populated this dict.
        grounding_chunks: A list of all available chunk items, used to retrieve page
          span information.

    Returns:
        A string representing the formatted bibliography, with each reference
        on a new line.

    Raises:
        TypeError: If cited_refs_dict is not a dictionary or grounding_chunks is not a list.
        ValueError: If cited_refs_dict or grounding_chunks is None.
                    If a chunk_item in grounding_chunks referenced by cited_refs_dict is None.
        IndexError: If a chunk_idx from cited_refs_dict is out of bounds for
          grounding_chunks.
    """
    if cited_refs_dict is None:
        raise ValueError("cited_refs_dict cannot be None.")
    if not isinstance(cited_refs_dict, dict):
        raise TypeError("cited_refs_dict must be a dictionary.")
    if grounding_chunks is None:
        raise ValueError("grounding_chunks cannot be None.")
    if not isinstance(grounding_chunks, list):
        raise TypeError("grounding_chunks must be a list.")

    reference_lines = []
    for chunk_idx_ref in sorted(list(cited_refs_dict.keys())):
        uri = cited_refs_dict[chunk_idx_ref]
        page_info_str = ""
        if not (
            isinstance(chunk_idx_ref, int)
            and 0 <= chunk_idx_ref < len(grounding_chunks)
        ):
            raise IndexError(
                f"Chunk index {chunk_idx_ref} from cited_refs_dict is invalid or out of bounds "
                f"for grounding_chunks of size {len(grounding_chunks)}."
            )
        chunk_item = grounding_chunks[chunk_idx_ref]
        if chunk_item is None:
            raise ValueError(
                f"Chunk item at index {chunk_idx_ref} in grounding_chunks is None, "
                "but was referenced in cited_refs_dict."
            )
        page_span_data = None
        if (
            hasattr(chunk_item, "retrieved_context")
            and chunk_item.retrieved_context
            and hasattr(chunk_item.retrieved_context, "rag_chunk")
            and chunk_item.retrieved_context.rag_chunk
            and hasattr(chunk_item.retrieved_context.rag_chunk, "page_span")
            and chunk_item.retrieved_context.rag_chunk.page_span
        ):
            page_span_data = chunk_item.retrieved_context.rag_chunk.page_span
        if (
            page_span_data
            and hasattr(page_span_data, "first_page")
            and hasattr(page_span_data, "last_page")
        ):
            first_page_val = page_span_data.first_page
            last_page_val = page_span_data.last_page
            is_first_page_valid_num = (
                isinstance(first_page_val, int) and first_page_val >= 1
            )
            is_last_page_valid_num = (
                isinstance(last_page_val, int) and last_page_val >= 1
            )
            if is_first_page_valid_num and is_last_page_valid_num:
                if last_page_val >= first_page_val:
                    page_info_str = (
                        f", p.{first_page_val}-{last_page_val}"
                        if first_page_val != last_page_val
                        else f", p.{first_page_val}"
                    )
        reference_lines.append(f"[{chunk_idx_ref}] {uri}{page_info_str}")
    return "\n".join(reference_lines)
