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

from google.cloud.discoveryengine_v1alpha.types import grounding

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "CheckGroundingSpec",
        "CheckGroundingRequest",
        "CheckGroundingResponse",
    },
)


class CheckGroundingSpec(proto.Message):
    r"""Specification for the grounding check.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        citation_threshold (float):
            The threshold (in [0,1]) used for determining whether a fact
            must be cited for a claim in the answer candidate. Choosing
            a higher threshold will lead to fewer but very strong
            citations, while choosing a lower threshold may lead to more
            but somewhat weaker citations. If unset, the threshold will
            default to 0.6.

            This field is a member of `oneof`_ ``_citation_threshold``.
    """

    citation_threshold: float = proto.Field(
        proto.DOUBLE,
        number=1,
        optional=True,
    )


class CheckGroundingRequest(proto.Message):
    r"""Request message for
    [GroundedGenerationService.CheckGrounding][google.cloud.discoveryengine.v1alpha.GroundedGenerationService.CheckGrounding]
    method.

    Attributes:
        grounding_config (str):
            Required. The resource name of the grounding config, such as
            ``projects/*/locations/global/groundingConfigs/default_grounding_config``.
        answer_candidate (str):
            Answer candidate to check. Can have a maximum
            length of 1024 characters.
        facts (MutableSequence[google.cloud.discoveryengine_v1alpha.types.GroundingFact]):
            List of facts for the grounding check.
            We support up to 200 facts.
        grounding_spec (google.cloud.discoveryengine_v1alpha.types.CheckGroundingSpec):
            Configuration of the grounding check.
        user_labels (MutableMapping[str, str]):
            The user labels applied to a resource must meet the
            following requirements:

            - Each resource can have multiple labels, up to a maximum of
              64.
            - Each label must be a key-value pair.
            - Keys have a minimum length of 1 character and a maximum
              length of 63 characters and cannot be empty. Values can be
              empty and have a maximum length of 63 characters.
            - Keys and values can contain only lowercase letters,
              numeric characters, underscores, and dashes. All
              characters must use UTF-8 encoding, and international
              characters are allowed.
            - The key portion of a label must be unique. However, you
              can use the same key with multiple resources.
            - Keys must start with a lowercase letter or international
              character.

            See `Google Cloud
            Document <https://cloud.google.com/resource-manager/docs/creating-managing-labels#requirements>`__
            for more details.
    """

    grounding_config: str = proto.Field(
        proto.STRING,
        number=1,
    )
    answer_candidate: str = proto.Field(
        proto.STRING,
        number=2,
    )
    facts: MutableSequence[grounding.GroundingFact] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=grounding.GroundingFact,
    )
    grounding_spec: "CheckGroundingSpec" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="CheckGroundingSpec",
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )


class CheckGroundingResponse(proto.Message):
    r"""Response message for the
    [GroundedGenerationService.CheckGrounding][google.cloud.discoveryengine.v1alpha.GroundedGenerationService.CheckGrounding]
    method.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        support_score (float):
            The support score for the input answer
            candidate. Higher the score, higher is the
            fraction of claims that are supported by the
            provided facts. This is always set when a
            response is returned.

            This field is a member of `oneof`_ ``_support_score``.
        cited_chunks (MutableSequence[google.cloud.discoveryengine_v1alpha.types.FactChunk]):
            List of facts cited across all claims in the
            answer candidate. These are derived from the
            facts supplied in the request.
        claims (MutableSequence[google.cloud.discoveryengine_v1alpha.types.CheckGroundingResponse.Claim]):
            Claim texts and citation info across all
            claims in the answer candidate.
    """

    class Claim(proto.Message):
        r"""Text and citation info for a claim in the answer candidate.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            start_pos (int):
                Position indicating the start of the claim in
                the answer candidate, measured in bytes.

                This field is a member of `oneof`_ ``_start_pos``.
            end_pos (int):
                Position indicating the end of the claim in
                the answer candidate, exclusive.

                This field is a member of `oneof`_ ``_end_pos``.
            claim_text (str):
                Text for the claim in the answer candidate.
                Always provided regardless of whether citations
                or anti-citations are found.
            citation_indices (MutableSequence[int]):
                A list of indices (into 'cited_chunks') specifying the
                citations associated with the claim. For instance [1,3,4]
                means that cited_chunks[1], cited_chunks[3], cited_chunks[4]
                are the facts cited supporting for the claim. A citation to
                a fact indicates that the claim is supported by the fact.
            grounding_check_required (bool):
                Indicates that this claim required grounding check. When the
                system decided this claim doesn't require
                attribution/grounding check, this field will be set to
                false. In that case, no grounding check was done for the
                claim and therefore
                [citation_indices][google.cloud.discoveryengine.v1alpha.CheckGroundingResponse.Claim.citation_indices],
                [anti_citation_indices][google.cloud.discoveryengine.v1alpha.CheckGroundingResponse.Claim.anti_citation_indices],
                and
                [score][google.cloud.discoveryengine.v1alpha.CheckGroundingResponse.Claim.score]
                should not be returned.

                This field is a member of `oneof`_ ``_grounding_check_required``.
        """

        start_pos: int = proto.Field(
            proto.INT32,
            number=1,
            optional=True,
        )
        end_pos: int = proto.Field(
            proto.INT32,
            number=2,
            optional=True,
        )
        claim_text: str = proto.Field(
            proto.STRING,
            number=3,
        )
        citation_indices: MutableSequence[int] = proto.RepeatedField(
            proto.INT32,
            number=4,
        )
        grounding_check_required: bool = proto.Field(
            proto.BOOL,
            number=6,
            optional=True,
        )

    support_score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    cited_chunks: MutableSequence[grounding.FactChunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=grounding.FactChunk,
    )
    claims: MutableSequence[Claim] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Claim,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
