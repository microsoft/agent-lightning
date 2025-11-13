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

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.speech.v1p1beta1",
    manifest={
        "CustomClass",
        "PhraseSet",
        "SpeechAdaptation",
        "TranscriptNormalization",
    },
)


class CustomClass(proto.Message):
    r"""A set of words or phrases that represents a common concept
    likely to appear in your audio, for example a list of passenger
    ship names. CustomClass items can be substituted into
    placeholders that you set in PhraseSet phrases.

    Attributes:
        name (str):
            The resource name of the custom class.
        custom_class_id (str):
            If this custom class is a resource, the custom_class_id is
            the resource id of the CustomClass. Case sensitive.
        items (MutableSequence[google.cloud.speech_v1p1beta1.types.CustomClass.ClassItem]):
            A collection of class items.
        kms_key_name (str):
            Output only. The `KMS key
            name <https://cloud.google.com/kms/docs/resource-hierarchy#keys>`__
            with which the content of the ClassItem is encrypted. The
            expected format is
            ``projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}``.
        kms_key_version_name (str):
            Output only. The `KMS key version
            name <https://cloud.google.com/kms/docs/resource-hierarchy#key_versions>`__
            with which content of the ClassItem is encrypted. The
            expected format is
            ``projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/cryptoKeyVersions/{crypto_key_version}``.
        uid (str):
            Output only. System-assigned unique
            identifier for the CustomClass. This field is
            not used.
        display_name (str):
            Output only. User-settable, human-readable
            name for the CustomClass. Must be 63 characters
            or less. This field is not used.
        state (google.cloud.speech_v1p1beta1.types.CustomClass.State):
            Output only. The CustomClass lifecycle state.
            This field is not used.
        delete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time at which this resource
            was requested for deletion. This field is not
            used.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time at which this resource
            will be purged. This field is not used.
        annotations (MutableMapping[str, str]):
            Output only. Allows users to store small
            amounts of arbitrary data. Both the key and the
            value must be 63 characters or less each. At
            most 100 annotations.
            This field is not used.
        etag (str):
            Output only. This checksum is computed by the
            server based on the value of other fields. This
            may be sent on update, undelete, and delete
            requests to ensure the client has an up-to-date
            value before proceeding. This field is not used.
        reconciling (bool):
            Output only. Whether or not this CustomClass
            is in the process of being updated. This field
            is not used.
    """

    class State(proto.Enum):
        r"""Set of states that define the lifecycle of a CustomClass.

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified state.  This is only used/useful
                for distinguishing unset values.
            ACTIVE (2):
                The normal and active state.
            DELETED (4):
                This CustomClass has been deleted.
        """
        STATE_UNSPECIFIED = 0
        ACTIVE = 2
        DELETED = 4

    class ClassItem(proto.Message):
        r"""An item of the class.

        Attributes:
            value (str):
                The class item's value.
        """

        value: str = proto.Field(
            proto.STRING,
            number=1,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    custom_class_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    items: MutableSequence[ClassItem] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=ClassItem,
    )
    kms_key_name: str = proto.Field(
        proto.STRING,
        number=6,
    )
    kms_key_version_name: str = proto.Field(
        proto.STRING,
        number=7,
    )
    uid: str = proto.Field(
        proto.STRING,
        number=8,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=9,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=10,
        enum=State,
    )
    delete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=11,
        message=timestamp_pb2.Timestamp,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
    )
    annotations: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=13,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=14,
    )
    reconciling: bool = proto.Field(
        proto.BOOL,
        number=15,
    )


class PhraseSet(proto.Message):
    r"""Provides "hints" to the speech recognizer to favor specific
    words and phrases in the results.

    Attributes:
        name (str):
            The resource name of the phrase set.
        phrases (MutableSequence[google.cloud.speech_v1p1beta1.types.PhraseSet.Phrase]):
            A list of word and phrases.
        boost (float):
            Hint Boost. Positive value will increase the probability
            that a specific phrase will be recognized over other similar
            sounding phrases. The higher the boost, the higher the
            chance of false positive recognition as well. Negative boost
            values would correspond to anti-biasing. Anti-biasing is not
            enabled, so negative boost will simply be ignored. Though
            ``boost`` can accept a wide range of positive values, most
            use cases are best served with values between 0 (exclusive)
            and 20. We recommend using a binary search approach to
            finding the optimal value for your use case as well as
            adding phrases both with and without boost to your requests.
        kms_key_name (str):
            Output only. The `KMS key
            name <https://cloud.google.com/kms/docs/resource-hierarchy#keys>`__
            with which the content of the PhraseSet is encrypted. The
            expected format is
            ``projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}``.
        kms_key_version_name (str):
            Output only. The `KMS key version
            name <https://cloud.google.com/kms/docs/resource-hierarchy#key_versions>`__
            with which content of the PhraseSet is encrypted. The
            expected format is
            ``projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/cryptoKeyVersions/{crypto_key_version}``.
        uid (str):
            Output only. System-assigned unique
            identifier for the PhraseSet. This field is not
            used.
        display_name (str):
            Output only. User-settable, human-readable
            name for the PhraseSet. Must be 63 characters or
            less. This field is not used.
        state (google.cloud.speech_v1p1beta1.types.PhraseSet.State):
            Output only. The CustomClass lifecycle state.
            This field is not used.
        delete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time at which this resource
            was requested for deletion. This field is not
            used.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time at which this resource
            will be purged. This field is not used.
        annotations (MutableMapping[str, str]):
            Output only. Allows users to store small
            amounts of arbitrary data. Both the key and the
            value must be 63 characters or less each. At
            most 100 annotations.
            This field is not used.
        etag (str):
            Output only. This checksum is computed by the
            server based on the value of other fields. This
            may be sent on update, undelete, and delete
            requests to ensure the client has an up-to-date
            value before proceeding. This field is not used.
        reconciling (bool):
            Output only. Whether or not this PhraseSet is
            in the process of being updated. This field is
            not used.
    """

    class State(proto.Enum):
        r"""Set of states that define the lifecycle of a CustomClass.

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified state.  This is only used/useful
                for distinguishing unset values.
            ACTIVE (2):
                The normal and active state.
            DELETED (4):
                This CustomClass has been deleted.
        """
        STATE_UNSPECIFIED = 0
        ACTIVE = 2
        DELETED = 4

    class Phrase(proto.Message):
        r"""A phrases containing words and phrase "hints" so that the speech
        recognition is more likely to recognize them. This can be used to
        improve the accuracy for specific words and phrases, for example, if
        specific commands are typically spoken by the user. This can also be
        used to add additional words to the vocabulary of the recognizer.
        See `usage
        limits <https://cloud.google.com/speech-to-text/quotas#content>`__.

        List items can also include pre-built or custom classes containing
        groups of words that represent common concepts that occur in natural
        language. For example, rather than providing a phrase hint for every
        month of the year (e.g. "i was born in january", "i was born in
        febuary", ...), use the pre-built ``$MONTH`` class improves the
        likelihood of correctly transcribing audio that includes months
        (e.g. "i was born in $month"). To refer to pre-built classes, use
        the class' symbol prepended with ``$`` e.g. ``$MONTH``. To refer to
        custom classes that were defined inline in the request, set the
        class's ``custom_class_id`` to a string unique to all class
        resources and inline classes. Then use the class' id wrapped in
        $\ ``{...}`` e.g. "${my-months}". To refer to custom classes
        resources, use the class' id wrapped in ``${}`` (e.g.
        ``${my-months}``).

        Speech-to-Text supports three locations: ``global``, ``us`` (US
        North America), and ``eu`` (Europe). If you are calling the
        ``speech.googleapis.com`` endpoint, use the ``global`` location. To
        specify a region, use a `regional
        endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
        with matching ``us`` or ``eu`` location value.

        Attributes:
            value (str):
                The phrase itself.
            boost (float):
                Hint Boost. Overrides the boost set at the phrase set level.
                Positive value will increase the probability that a specific
                phrase will be recognized over other similar sounding
                phrases. The higher the boost, the higher the chance of
                false positive recognition as well. Negative boost will
                simply be ignored. Though ``boost`` can accept a wide range
                of positive values, most use cases are best served with
                values between 0 and 20. We recommend using a binary search
                approach to finding the optimal value for your use case as
                well as adding phrases both with and without boost to your
                requests.
        """

        value: str = proto.Field(
            proto.STRING,
            number=1,
        )
        boost: float = proto.Field(
            proto.FLOAT,
            number=2,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    phrases: MutableSequence[Phrase] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=Phrase,
    )
    boost: float = proto.Field(
        proto.FLOAT,
        number=4,
    )
    kms_key_name: str = proto.Field(
        proto.STRING,
        number=7,
    )
    kms_key_version_name: str = proto.Field(
        proto.STRING,
        number=8,
    )
    uid: str = proto.Field(
        proto.STRING,
        number=9,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=10,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=11,
        enum=State,
    )
    delete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=13,
        message=timestamp_pb2.Timestamp,
    )
    annotations: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=14,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=15,
    )
    reconciling: bool = proto.Field(
        proto.BOOL,
        number=16,
    )


class SpeechAdaptation(proto.Message):
    r"""Speech adaptation configuration.

    Attributes:
        phrase_sets (MutableSequence[google.cloud.speech_v1p1beta1.types.PhraseSet]):
            A collection of phrase sets. To specify the hints inline,
            leave the phrase set's ``name`` blank and fill in the rest
            of its fields. Any phrase set can use any custom class.
        phrase_set_references (MutableSequence[str]):
            A collection of phrase set resource names to
            use.
        custom_classes (MutableSequence[google.cloud.speech_v1p1beta1.types.CustomClass]):
            A collection of custom classes. To specify the classes
            inline, leave the class' ``name`` blank and fill in the rest
            of its fields, giving it a unique ``custom_class_id``. Refer
            to the inline defined class in phrase hints by its
            ``custom_class_id``.
        abnf_grammar (google.cloud.speech_v1p1beta1.types.SpeechAdaptation.ABNFGrammar):
            Augmented Backus-Naur form (ABNF) is a
            standardized grammar notation comprised by a set
            of derivation rules. See specifications:
            https://www.w3.org/TR/speech-grammar
    """

    class ABNFGrammar(proto.Message):
        r"""

        Attributes:
            abnf_strings (MutableSequence[str]):
                All declarations and rules of an ABNF grammar
                broken up into multiple strings that will end up
                concatenated.
        """

        abnf_strings: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    phrase_sets: MutableSequence["PhraseSet"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="PhraseSet",
    )
    phrase_set_references: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    custom_classes: MutableSequence["CustomClass"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="CustomClass",
    )
    abnf_grammar: ABNFGrammar = proto.Field(
        proto.MESSAGE,
        number=4,
        message=ABNFGrammar,
    )


class TranscriptNormalization(proto.Message):
    r"""Transcription normalization configuration. Use transcription
    normalization to automatically replace parts of the transcript
    with phrases of your choosing. For StreamingRecognize, this
    normalization only applies to stable partial transcripts
    (stability > 0.8) and final transcripts.

    Attributes:
        entries (MutableSequence[google.cloud.speech_v1p1beta1.types.TranscriptNormalization.Entry]):
            A list of replacement entries. We will perform replacement
            with one entry at a time. For example, the second entry in
            ["cat" => "dog", "mountain cat" => "mountain dog"] will
            never be applied because we will always process the first
            entry before it. At most 100 entries.
    """

    class Entry(proto.Message):
        r"""A single replacement configuration.

        Attributes:
            search (str):
                What to replace. Max length is 100
                characters.
            replace (str):
                What to replace with. Max length is 100
                characters.
            case_sensitive (bool):
                Whether the search is case sensitive.
        """

        search: str = proto.Field(
            proto.STRING,
            number=1,
        )
        replace: str = proto.Field(
            proto.STRING,
            number=2,
        )
        case_sensitive: bool = proto.Field(
            proto.BOOL,
            number=3,
        )

    entries: MutableSequence[Entry] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=Entry,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
