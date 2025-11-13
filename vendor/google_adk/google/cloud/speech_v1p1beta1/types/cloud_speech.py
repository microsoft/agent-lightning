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

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.speech_v1p1beta1.types import resource

__protobuf__ = proto.module(
    package="google.cloud.speech.v1p1beta1",
    manifest={
        "RecognizeRequest",
        "LongRunningRecognizeRequest",
        "TranscriptOutputConfig",
        "StreamingRecognizeRequest",
        "StreamingRecognitionConfig",
        "RecognitionConfig",
        "SpeakerDiarizationConfig",
        "RecognitionMetadata",
        "SpeechContext",
        "RecognitionAudio",
        "RecognizeResponse",
        "LongRunningRecognizeResponse",
        "LongRunningRecognizeMetadata",
        "StreamingRecognizeResponse",
        "StreamingRecognitionResult",
        "SpeechRecognitionResult",
        "SpeechRecognitionAlternative",
        "WordInfo",
        "SpeechAdaptationInfo",
    },
)


class RecognizeRequest(proto.Message):
    r"""The top-level message sent by the client for the ``Recognize``
    method.

    Attributes:
        config (google.cloud.speech_v1p1beta1.types.RecognitionConfig):
            Required. Provides information to the
            recognizer that specifies how to process the
            request.
        audio (google.cloud.speech_v1p1beta1.types.RecognitionAudio):
            Required. The audio data to be recognized.
    """

    config: "RecognitionConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RecognitionConfig",
    )
    audio: "RecognitionAudio" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="RecognitionAudio",
    )


class LongRunningRecognizeRequest(proto.Message):
    r"""The top-level message sent by the client for the
    ``LongRunningRecognize`` method.

    Attributes:
        config (google.cloud.speech_v1p1beta1.types.RecognitionConfig):
            Required. Provides information to the
            recognizer that specifies how to process the
            request.
        audio (google.cloud.speech_v1p1beta1.types.RecognitionAudio):
            Required. The audio data to be recognized.
        output_config (google.cloud.speech_v1p1beta1.types.TranscriptOutputConfig):
            Optional. Specifies an optional destination
            for the recognition results.
    """

    config: "RecognitionConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RecognitionConfig",
    )
    audio: "RecognitionAudio" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="RecognitionAudio",
    )
    output_config: "TranscriptOutputConfig" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="TranscriptOutputConfig",
    )


class TranscriptOutputConfig(proto.Message):
    r"""Specifies an optional destination for the recognition
    results.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_uri (str):
            Specifies a Cloud Storage URI for the recognition results.
            Must be specified in the format:
            ``gs://bucket_name/object_name``, and the bucket must
            already exist.

            This field is a member of `oneof`_ ``output_type``.
    """

    gcs_uri: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="output_type",
    )


class StreamingRecognizeRequest(proto.Message):
    r"""The top-level message sent by the client for the
    ``StreamingRecognize`` method. Multiple
    ``StreamingRecognizeRequest`` messages are sent. The first message
    must contain a ``streaming_config`` message and must not contain
    ``audio_content``. All subsequent messages must contain
    ``audio_content`` and must not contain a ``streaming_config``
    message.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        streaming_config (google.cloud.speech_v1p1beta1.types.StreamingRecognitionConfig):
            Provides information to the recognizer that specifies how to
            process the request. The first ``StreamingRecognizeRequest``
            message must contain a ``streaming_config`` message.

            This field is a member of `oneof`_ ``streaming_request``.
        audio_content (bytes):
            The audio data to be recognized. Sequential chunks of audio
            data are sent in sequential ``StreamingRecognizeRequest``
            messages. The first ``StreamingRecognizeRequest`` message
            must not contain ``audio_content`` data and all subsequent
            ``StreamingRecognizeRequest`` messages must contain
            ``audio_content`` data. The audio bytes must be encoded as
            specified in ``RecognitionConfig``. Note: as with all bytes
            fields, proto buffers use a pure binary representation (not
            base64). See `content
            limits <https://cloud.google.com/speech-to-text/quotas#content>`__.

            This field is a member of `oneof`_ ``streaming_request``.
    """

    streaming_config: "StreamingRecognitionConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="streaming_request",
        message="StreamingRecognitionConfig",
    )
    audio_content: bytes = proto.Field(
        proto.BYTES,
        number=2,
        oneof="streaming_request",
    )


class StreamingRecognitionConfig(proto.Message):
    r"""Provides information to the recognizer that specifies how to
    process the request.

    Attributes:
        config (google.cloud.speech_v1p1beta1.types.RecognitionConfig):
            Required. Provides information to the
            recognizer that specifies how to process the
            request.
        single_utterance (bool):
            If ``false`` or omitted, the recognizer will perform
            continuous recognition (continuing to wait for and process
            audio even if the user pauses speaking) until the client
            closes the input stream (gRPC API) or until the maximum time
            limit has been reached. May return multiple
            ``StreamingRecognitionResult``\ s with the ``is_final`` flag
            set to ``true``.

            If ``true``, the recognizer will detect a single spoken
            utterance. When it detects that the user has paused or
            stopped speaking, it will return an
            ``END_OF_SINGLE_UTTERANCE`` event and cease recognition. It
            will return no more than one ``StreamingRecognitionResult``
            with the ``is_final`` flag set to ``true``.

            The ``single_utterance`` field can only be used with
            specified models, otherwise an error is thrown. The
            ``model`` field in
            [RecognitionConfig][google.cloud.speech.v1p1beta1.RecognitionConfig]
            must be set to:

            - ``command_and_search``
            - ``phone_call`` AND additional field
              ``useEnhanced``\ =\ ``true``
            - The ``model`` field is left undefined. In this case the
              API auto-selects a model based on any other parameters
              that you set in ``RecognitionConfig``.
        interim_results (bool):
            If ``true``, interim results (tentative hypotheses) may be
            returned as they become available (these interim results are
            indicated with the ``is_final=false`` flag). If ``false`` or
            omitted, only ``is_final=true`` result(s) are returned.
        enable_voice_activity_events (bool):
            If ``true``, responses with voice activity speech events
            will be returned as they are detected.
        voice_activity_timeout (google.cloud.speech_v1p1beta1.types.StreamingRecognitionConfig.VoiceActivityTimeout):
            If set, the server will automatically close the stream after
            the specified duration has elapsed after the last
            VOICE_ACTIVITY speech event has been sent. The field
            ``voice_activity_events`` must also be set to true.
    """

    class VoiceActivityTimeout(proto.Message):
        r"""Events that a timeout can be set on for voice activity.

        Attributes:
            speech_start_timeout (google.protobuf.duration_pb2.Duration):
                Duration to timeout the stream if no speech
                begins.
            speech_end_timeout (google.protobuf.duration_pb2.Duration):
                Duration to timeout the stream after speech
                ends.
        """

        speech_start_timeout: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=1,
            message=duration_pb2.Duration,
        )
        speech_end_timeout: duration_pb2.Duration = proto.Field(
            proto.MESSAGE,
            number=2,
            message=duration_pb2.Duration,
        )

    config: "RecognitionConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RecognitionConfig",
    )
    single_utterance: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    interim_results: bool = proto.Field(
        proto.BOOL,
        number=3,
    )
    enable_voice_activity_events: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    voice_activity_timeout: VoiceActivityTimeout = proto.Field(
        proto.MESSAGE,
        number=6,
        message=VoiceActivityTimeout,
    )


class RecognitionConfig(proto.Message):
    r"""Provides information to the recognizer that specifies how to
    process the request.

    Attributes:
        encoding (google.cloud.speech_v1p1beta1.types.RecognitionConfig.AudioEncoding):
            Encoding of audio data sent in all ``RecognitionAudio``
            messages. This field is optional for ``FLAC`` and ``WAV``
            audio files and required for all other audio formats. For
            details, see
            [AudioEncoding][google.cloud.speech.v1p1beta1.RecognitionConfig.AudioEncoding].
        sample_rate_hertz (int):
            Sample rate in Hertz of the audio data sent in all
            ``RecognitionAudio`` messages. Valid values are: 8000-48000.
            16000 is optimal. For best results, set the sampling rate of
            the audio source to 16000 Hz. If that's not possible, use
            the native sample rate of the audio source (instead of
            re-sampling). This field is optional for FLAC and WAV audio
            files, but is required for all other audio formats. For
            details, see
            [AudioEncoding][google.cloud.speech.v1p1beta1.RecognitionConfig.AudioEncoding].
        audio_channel_count (int):
            The number of channels in the input audio data. ONLY set
            this for MULTI-CHANNEL recognition. Valid values for
            LINEAR16, OGG_OPUS and FLAC are ``1``-``8``. Valid value for
            MULAW, AMR, AMR_WB and SPEEX_WITH_HEADER_BYTE is only ``1``.
            If ``0`` or omitted, defaults to one channel (mono). Note:
            We only recognize the first channel by default. To perform
            independent recognition on each channel set
            ``enable_separate_recognition_per_channel`` to 'true'.
        enable_separate_recognition_per_channel (bool):
            This needs to be set to ``true`` explicitly and
            ``audio_channel_count`` > 1 to get each channel recognized
            separately. The recognition result will contain a
            ``channel_tag`` field to state which channel that result
            belongs to. If this is not true, we will only recognize the
            first channel. The request is billed cumulatively for all
            channels recognized: ``audio_channel_count`` multiplied by
            the length of the audio.
        language_code (str):
            Required. The language of the supplied audio as a
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag. Example: "en-US". See `Language
            Support <https://cloud.google.com/speech-to-text/docs/languages>`__
            for a list of the currently supported language codes.
        alternative_language_codes (MutableSequence[str]):
            A list of up to 3 additional
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tags, listing possible alternative languages of the
            supplied audio. See `Language
            Support <https://cloud.google.com/speech-to-text/docs/languages>`__
            for a list of the currently supported language codes. If
            alternative languages are listed, recognition result will
            contain recognition in the most likely language detected
            including the main language_code. The recognition result
            will include the language tag of the language detected in
            the audio. Note: This feature is only supported for Voice
            Command and Voice Search use cases and performance may vary
            for other use cases (e.g., phone call transcription).
        max_alternatives (int):
            Maximum number of recognition hypotheses to be returned.
            Specifically, the maximum number of
            ``SpeechRecognitionAlternative`` messages within each
            ``SpeechRecognitionResult``. The server may return fewer
            than ``max_alternatives``. Valid values are ``0``-``30``. A
            value of ``0`` or ``1`` will return a maximum of one. If
            omitted, will return a maximum of one.
        profanity_filter (bool):
            If set to ``true``, the server will attempt to filter out
            profanities, replacing all but the initial character in each
            filtered word with asterisks, e.g. "f**\*". If set to
            ``false`` or omitted, profanities won't be filtered out.
        adaptation (google.cloud.speech_v1p1beta1.types.SpeechAdaptation):
            Speech adaptation configuration improves the accuracy of
            speech recognition. For more information, see the `speech
            adaptation <https://cloud.google.com/speech-to-text/docs/adaptation>`__
            documentation. When speech adaptation is set it supersedes
            the ``speech_contexts`` field.
        transcript_normalization (google.cloud.speech_v1p1beta1.types.TranscriptNormalization):
            Optional. Use transcription normalization to
            automatically replace parts of the transcript
            with phrases of your choosing. For
            StreamingRecognize, this normalization only
            applies to stable partial transcripts (stability
            > 0.8) and final transcripts.
        speech_contexts (MutableSequence[google.cloud.speech_v1p1beta1.types.SpeechContext]):
            Array of
            [SpeechContext][google.cloud.speech.v1p1beta1.SpeechContext].
            A means to provide context to assist the speech recognition.
            For more information, see `speech
            adaptation <https://cloud.google.com/speech-to-text/docs/adaptation>`__.
        enable_word_time_offsets (bool):
            If ``true``, the top result includes a list of words and the
            start and end time offsets (timestamps) for those words. If
            ``false``, no word-level time offset information is
            returned. The default is ``false``.
        enable_word_confidence (bool):
            If ``true``, the top result includes a list of words and the
            confidence for those words. If ``false``, no word-level
            confidence information is returned. The default is
            ``false``.
        enable_automatic_punctuation (bool):
            If 'true', adds punctuation to recognition
            result hypotheses. This feature is only
            available in select languages. Setting this for
            requests in other languages has no effect at
            all. The default 'false' value does not add
            punctuation to result hypotheses.
        enable_spoken_punctuation (google.protobuf.wrappers_pb2.BoolValue):
            The spoken punctuation behavior for the call If not set,
            uses default behavior based on model of choice e.g.
            command_and_search will enable spoken punctuation by default
            If 'true', replaces spoken punctuation with the
            corresponding symbols in the request. For example, "how are
            you question mark" becomes "how are you?". See
            https://cloud.google.com/speech-to-text/docs/spoken-punctuation
            for support. If 'false', spoken punctuation is not replaced.
        enable_spoken_emojis (google.protobuf.wrappers_pb2.BoolValue):
            The spoken emoji behavior for the call
            If not set, uses default behavior based on model
            of choice If 'true', adds spoken emoji
            formatting for the request. This will replace
            spoken emojis with the corresponding Unicode
            symbols in the final transcript. If 'false',
            spoken emojis are not replaced.
        enable_speaker_diarization (bool):
            If 'true', enables speaker detection for each recognized
            word in the top alternative of the recognition result using
            a speaker_label provided in the WordInfo. Note: Use
            diarization_config instead.
        diarization_speaker_count (int):
            If set, specifies the estimated number of speakers in the
            conversation. Defaults to '2'. Ignored unless
            enable_speaker_diarization is set to true. Note: Use
            diarization_config instead.
        diarization_config (google.cloud.speech_v1p1beta1.types.SpeakerDiarizationConfig):
            Config to enable speaker diarization and set
            additional parameters to make diarization better
            suited for your application. Note: When this is
            enabled, we send all the words from the
            beginning of the audio for the top alternative
            in every consecutive STREAMING responses. This
            is done in order to improve our speaker tags as
            our models learn to identify the speakers in the
            conversation over time. For non-streaming
            requests, the diarization results will be
            provided only in the top alternative of the
            FINAL SpeechRecognitionResult.
        metadata (google.cloud.speech_v1p1beta1.types.RecognitionMetadata):
            Metadata regarding this request.
        model (str):
            Which model to select for the given request. Select the
            model best suited to your domain to get best results. If a
            model is not explicitly specified, then we auto-select a
            model based on the parameters in the RecognitionConfig.

            .. raw:: html

                <table>
                  <tr>
                    <td><b>Model</b></td>
                    <td><b>Description</b></td>
                  </tr>
                  <tr>
                    <td><code>latest_long</code></td>
                    <td>Best for long form content like media or conversation.</td>
                  </tr>
                  <tr>
                    <td><code>latest_short</code></td>
                    <td>Best for short form content like commands or single shot directed
                    speech.</td>
                  </tr>
                  <tr>
                    <td><code>command_and_search</code></td>
                    <td>Best for short queries such as voice commands or voice search.</td>
                  </tr>
                  <tr>
                    <td><code>phone_call</code></td>
                    <td>Best for audio that originated from a phone call (typically
                    recorded at an 8khz sampling rate).</td>
                  </tr>
                  <tr>
                    <td><code>video</code></td>
                    <td>Best for audio that originated from video or includes multiple
                        speakers. Ideally the audio is recorded at a 16khz or greater
                        sampling rate. This is a premium model that costs more than the
                        standard rate.</td>
                  </tr>
                  <tr>
                    <td><code>default</code></td>
                    <td>Best for audio that is not one of the specific audio models.
                        For example, long-form audio. Ideally the audio is high-fidelity,
                        recorded at a 16khz or greater sampling rate.</td>
                  </tr>
                  <tr>
                    <td><code>medical_conversation</code></td>
                    <td>Best for audio that originated from a conversation between a
                        medical provider and patient.</td>
                  </tr>
                  <tr>
                    <td><code>medical_dictation</code></td>
                    <td>Best for audio that originated from dictation notes by a medical
                        provider.</td>
                  </tr>
                </table>
        use_enhanced (bool):
            Set to true to use an enhanced model for speech recognition.
            If ``use_enhanced`` is set to true and the ``model`` field
            is not set, then an appropriate enhanced model is chosen if
            an enhanced model exists for the audio.

            If ``use_enhanced`` is true and an enhanced version of the
            specified model does not exist, then the speech is
            recognized using the standard version of the specified
            model.
    """

    class AudioEncoding(proto.Enum):
        r"""The encoding of the audio data sent in the request.

        All encodings support only 1 channel (mono) audio, unless the
        ``audio_channel_count`` and
        ``enable_separate_recognition_per_channel`` fields are set.

        For best results, the audio source should be captured and
        transmitted using a lossless encoding (``FLAC`` or ``LINEAR16``).
        The accuracy of the speech recognition can be reduced if lossy
        codecs are used to capture or transmit audio, particularly if
        background noise is present. Lossy codecs include ``MULAW``,
        ``AMR``, ``AMR_WB``, ``OGG_OPUS``, ``SPEEX_WITH_HEADER_BYTE``,
        ``MP3``, and ``WEBM_OPUS``.

        The ``FLAC`` and ``WAV`` audio file formats include a header that
        describes the included audio content. You can request recognition
        for ``WAV`` files that contain either ``LINEAR16`` or ``MULAW``
        encoded audio. If you send ``FLAC`` or ``WAV`` audio file format in
        your request, you do not need to specify an ``AudioEncoding``; the
        audio encoding format is determined from the file header. If you
        specify an ``AudioEncoding`` when you send send ``FLAC`` or ``WAV``
        audio, the encoding configuration must match the encoding described
        in the audio header; otherwise the request returns an
        [google.rpc.Code.INVALID_ARGUMENT][google.rpc.Code.INVALID_ARGUMENT]
        error code.

        Values:
            ENCODING_UNSPECIFIED (0):
                Not specified.
            LINEAR16 (1):
                Uncompressed 16-bit signed little-endian
                samples (Linear PCM).
            FLAC (2):
                ``FLAC`` (Free Lossless Audio Codec) is the recommended
                encoding because it is lossless--therefore recognition is
                not compromised--and requires only about half the bandwidth
                of ``LINEAR16``. ``FLAC`` stream encoding supports 16-bit
                and 24-bit samples, however, not all fields in
                ``STREAMINFO`` are supported.
            MULAW (3):
                8-bit samples that compand 14-bit audio
                samples using G.711 PCMU/mu-law.
            AMR (4):
                Adaptive Multi-Rate Narrowband codec. ``sample_rate_hertz``
                must be 8000.
            AMR_WB (5):
                Adaptive Multi-Rate Wideband codec. ``sample_rate_hertz``
                must be 16000.
            OGG_OPUS (6):
                Opus encoded audio frames in Ogg container
                (`OggOpus <https://wiki.xiph.org/OggOpus>`__).
                ``sample_rate_hertz`` must be one of 8000, 12000, 16000,
                24000, or 48000.
            SPEEX_WITH_HEADER_BYTE (7):
                Although the use of lossy encodings is not recommended, if a
                very low bitrate encoding is required, ``OGG_OPUS`` is
                highly preferred over Speex encoding. The
                `Speex <https://speex.org/>`__ encoding supported by Cloud
                Speech API has a header byte in each block, as in MIME type
                ``audio/x-speex-with-header-byte``. It is a variant of the
                RTP Speex encoding defined in `RFC
                5574 <https://tools.ietf.org/html/rfc5574>`__. The stream is
                a sequence of blocks, one block per RTP packet. Each block
                starts with a byte containing the length of the block, in
                bytes, followed by one or more frames of Speex data, padded
                to an integral number of bytes (octets) as specified in RFC
                5574. In other words, each RTP header is replaced with a
                single byte containing the block length. Only Speex wideband
                is supported. ``sample_rate_hertz`` must be 16000.
            MP3 (8):
                MP3 audio. MP3 encoding is a Beta feature and only available
                in v1p1beta1. Support all standard MP3 bitrates (which range
                from 32-320 kbps). When using this encoding,
                ``sample_rate_hertz`` has to match the sample rate of the
                file being used.
            WEBM_OPUS (9):
                Opus encoded audio frames in WebM container
                (`WebM <https://www.webmproject.org/docs/container/>`__).
                ``sample_rate_hertz`` must be one of 8000, 12000, 16000,
                24000, or 48000.
            ALAW (10):
                8-bit samples that compand 13-bit audio
                samples using G.711 PCMU/a-law.
        """
        ENCODING_UNSPECIFIED = 0
        LINEAR16 = 1
        FLAC = 2
        MULAW = 3
        AMR = 4
        AMR_WB = 5
        OGG_OPUS = 6
        SPEEX_WITH_HEADER_BYTE = 7
        MP3 = 8
        WEBM_OPUS = 9
        ALAW = 10

    encoding: AudioEncoding = proto.Field(
        proto.ENUM,
        number=1,
        enum=AudioEncoding,
    )
    sample_rate_hertz: int = proto.Field(
        proto.INT32,
        number=2,
    )
    audio_channel_count: int = proto.Field(
        proto.INT32,
        number=7,
    )
    enable_separate_recognition_per_channel: bool = proto.Field(
        proto.BOOL,
        number=12,
    )
    language_code: str = proto.Field(
        proto.STRING,
        number=3,
    )
    alternative_language_codes: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=18,
    )
    max_alternatives: int = proto.Field(
        proto.INT32,
        number=4,
    )
    profanity_filter: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    adaptation: resource.SpeechAdaptation = proto.Field(
        proto.MESSAGE,
        number=20,
        message=resource.SpeechAdaptation,
    )
    transcript_normalization: resource.TranscriptNormalization = proto.Field(
        proto.MESSAGE,
        number=24,
        message=resource.TranscriptNormalization,
    )
    speech_contexts: MutableSequence["SpeechContext"] = proto.RepeatedField(
        proto.MESSAGE,
        number=6,
        message="SpeechContext",
    )
    enable_word_time_offsets: bool = proto.Field(
        proto.BOOL,
        number=8,
    )
    enable_word_confidence: bool = proto.Field(
        proto.BOOL,
        number=15,
    )
    enable_automatic_punctuation: bool = proto.Field(
        proto.BOOL,
        number=11,
    )
    enable_spoken_punctuation: wrappers_pb2.BoolValue = proto.Field(
        proto.MESSAGE,
        number=22,
        message=wrappers_pb2.BoolValue,
    )
    enable_spoken_emojis: wrappers_pb2.BoolValue = proto.Field(
        proto.MESSAGE,
        number=23,
        message=wrappers_pb2.BoolValue,
    )
    enable_speaker_diarization: bool = proto.Field(
        proto.BOOL,
        number=16,
    )
    diarization_speaker_count: int = proto.Field(
        proto.INT32,
        number=17,
    )
    diarization_config: "SpeakerDiarizationConfig" = proto.Field(
        proto.MESSAGE,
        number=19,
        message="SpeakerDiarizationConfig",
    )
    metadata: "RecognitionMetadata" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="RecognitionMetadata",
    )
    model: str = proto.Field(
        proto.STRING,
        number=13,
    )
    use_enhanced: bool = proto.Field(
        proto.BOOL,
        number=14,
    )


class SpeakerDiarizationConfig(proto.Message):
    r"""Config to enable speaker diarization.

    Attributes:
        enable_speaker_diarization (bool):
            If 'true', enables speaker detection for each recognized
            word in the top alternative of the recognition result using
            a speaker_label provided in the WordInfo.
        min_speaker_count (int):
            Minimum number of speakers in the
            conversation. This range gives you more
            flexibility by allowing the system to
            automatically determine the correct number of
            speakers. If not set, the default value is 2.
        max_speaker_count (int):
            Maximum number of speakers in the
            conversation. This range gives you more
            flexibility by allowing the system to
            automatically determine the correct number of
            speakers. If not set, the default value is 6.
        speaker_tag (int):
            Output only. Unused.
    """

    enable_speaker_diarization: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    min_speaker_count: int = proto.Field(
        proto.INT32,
        number=2,
    )
    max_speaker_count: int = proto.Field(
        proto.INT32,
        number=3,
    )
    speaker_tag: int = proto.Field(
        proto.INT32,
        number=5,
    )


class RecognitionMetadata(proto.Message):
    r"""Description of audio data to be recognized.

    Attributes:
        interaction_type (google.cloud.speech_v1p1beta1.types.RecognitionMetadata.InteractionType):
            The use case most closely describing the
            audio content to be recognized.
        industry_naics_code_of_audio (int):
            The industry vertical to which this speech
            recognition request most closely applies. This
            is most indicative of the topics contained in
            the audio.  Use the 6-digit NAICS code to
            identify the industry vertical - see
            https://www.naics.com/search/.
        microphone_distance (google.cloud.speech_v1p1beta1.types.RecognitionMetadata.MicrophoneDistance):
            The audio type that most closely describes
            the audio being recognized.
        original_media_type (google.cloud.speech_v1p1beta1.types.RecognitionMetadata.OriginalMediaType):
            The original media the speech was recorded
            on.
        recording_device_type (google.cloud.speech_v1p1beta1.types.RecognitionMetadata.RecordingDeviceType):
            The type of device the speech was recorded
            with.
        recording_device_name (str):
            The device used to make the recording.
            Examples 'Nexus 5X' or 'Polycom SoundStation IP
            6000' or 'POTS' or 'VoIP' or 'Cardioid
            Microphone'.
        original_mime_type (str):
            Mime type of the original audio file. For example
            ``audio/m4a``, ``audio/x-alaw-basic``, ``audio/mp3``,
            ``audio/3gpp``. A list of possible audio mime types is
            maintained at
            http://www.iana.org/assignments/media-types/media-types.xhtml#audio
        obfuscated_id (int):
            Obfuscated (privacy-protected) ID of the
            user, to identify number of unique users using
            the service.
        audio_topic (str):
            Description of the content. Eg. "Recordings
            of federal supreme court hearings from 2012".
    """

    class InteractionType(proto.Enum):
        r"""Use case categories that the audio recognition request can be
        described by.

        Values:
            INTERACTION_TYPE_UNSPECIFIED (0):
                Use case is either unknown or is something
                other than one of the other values below.
            DISCUSSION (1):
                Multiple people in a conversation or discussion. For example
                in a meeting with two or more people actively participating.
                Typically all the primary people speaking would be in the
                same room (if not, see PHONE_CALL)
            PRESENTATION (2):
                One or more persons lecturing or presenting
                to others, mostly uninterrupted.
            PHONE_CALL (3):
                A phone-call or video-conference in which two
                or more people, who are not in the same room,
                are actively participating.
            VOICEMAIL (4):
                A recorded message intended for another
                person to listen to.
            PROFESSIONALLY_PRODUCED (5):
                Professionally produced audio (eg. TV Show,
                Podcast).
            VOICE_SEARCH (6):
                Transcribe spoken questions and queries into
                text.
            VOICE_COMMAND (7):
                Transcribe voice commands, such as for
                controlling a device.
            DICTATION (8):
                Transcribe speech to text to create a written
                document, such as a text-message, email or
                report.
        """
        INTERACTION_TYPE_UNSPECIFIED = 0
        DISCUSSION = 1
        PRESENTATION = 2
        PHONE_CALL = 3
        VOICEMAIL = 4
        PROFESSIONALLY_PRODUCED = 5
        VOICE_SEARCH = 6
        VOICE_COMMAND = 7
        DICTATION = 8

    class MicrophoneDistance(proto.Enum):
        r"""Enumerates the types of capture settings describing an audio
        file.

        Values:
            MICROPHONE_DISTANCE_UNSPECIFIED (0):
                Audio type is not known.
            NEARFIELD (1):
                The audio was captured from a closely placed
                microphone. Eg. phone, dictaphone, or handheld
                microphone. Generally if there speaker is within
                1 meter of the microphone.
            MIDFIELD (2):
                The speaker if within 3 meters of the
                microphone.
            FARFIELD (3):
                The speaker is more than 3 meters away from
                the microphone.
        """
        MICROPHONE_DISTANCE_UNSPECIFIED = 0
        NEARFIELD = 1
        MIDFIELD = 2
        FARFIELD = 3

    class OriginalMediaType(proto.Enum):
        r"""The original media the speech was recorded on.

        Values:
            ORIGINAL_MEDIA_TYPE_UNSPECIFIED (0):
                Unknown original media type.
            AUDIO (1):
                The speech data is an audio recording.
            VIDEO (2):
                The speech data originally recorded on a
                video.
        """
        ORIGINAL_MEDIA_TYPE_UNSPECIFIED = 0
        AUDIO = 1
        VIDEO = 2

    class RecordingDeviceType(proto.Enum):
        r"""The type of device the speech was recorded with.

        Values:
            RECORDING_DEVICE_TYPE_UNSPECIFIED (0):
                The recording device is unknown.
            SMARTPHONE (1):
                Speech was recorded on a smartphone.
            PC (2):
                Speech was recorded using a personal computer
                or tablet.
            PHONE_LINE (3):
                Speech was recorded over a phone line.
            VEHICLE (4):
                Speech was recorded in a vehicle.
            OTHER_OUTDOOR_DEVICE (5):
                Speech was recorded outdoors.
            OTHER_INDOOR_DEVICE (6):
                Speech was recorded indoors.
        """
        RECORDING_DEVICE_TYPE_UNSPECIFIED = 0
        SMARTPHONE = 1
        PC = 2
        PHONE_LINE = 3
        VEHICLE = 4
        OTHER_OUTDOOR_DEVICE = 5
        OTHER_INDOOR_DEVICE = 6

    interaction_type: InteractionType = proto.Field(
        proto.ENUM,
        number=1,
        enum=InteractionType,
    )
    industry_naics_code_of_audio: int = proto.Field(
        proto.UINT32,
        number=3,
    )
    microphone_distance: MicrophoneDistance = proto.Field(
        proto.ENUM,
        number=4,
        enum=MicrophoneDistance,
    )
    original_media_type: OriginalMediaType = proto.Field(
        proto.ENUM,
        number=5,
        enum=OriginalMediaType,
    )
    recording_device_type: RecordingDeviceType = proto.Field(
        proto.ENUM,
        number=6,
        enum=RecordingDeviceType,
    )
    recording_device_name: str = proto.Field(
        proto.STRING,
        number=7,
    )
    original_mime_type: str = proto.Field(
        proto.STRING,
        number=8,
    )
    obfuscated_id: int = proto.Field(
        proto.INT64,
        number=9,
    )
    audio_topic: str = proto.Field(
        proto.STRING,
        number=10,
    )


class SpeechContext(proto.Message):
    r"""Provides "hints" to the speech recognizer to favor specific
    words and phrases in the results.

    Attributes:
        phrases (MutableSequence[str]):
            A list of strings containing words and phrases "hints" so
            that the speech recognition is more likely to recognize
            them. This can be used to improve the accuracy for specific
            words and phrases, for example, if specific commands are
            typically spoken by the user. This can also be used to add
            additional words to the vocabulary of the recognizer. See
            `usage
            limits <https://cloud.google.com/speech-to-text/quotas#content>`__.

            List items can also be set to classes for groups of words
            that represent common concepts that occur in natural
            language. For example, rather than providing phrase hints
            for every month of the year, using the $MONTH class improves
            the likelihood of correctly transcribing audio that includes
            months.
        boost (float):
            Hint Boost. Positive value will increase the probability
            that a specific phrase will be recognized over other similar
            sounding phrases. The higher the boost, the higher the
            chance of false positive recognition as well. Negative boost
            values would correspond to anti-biasing. Anti-biasing is not
            enabled, so negative boost will simply be ignored. Though
            ``boost`` can accept a wide range of positive values, most
            use cases are best served with values between 0 and 20. We
            recommend using a binary search approach to finding the
            optimal value for your use case.
    """

    phrases: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    boost: float = proto.Field(
        proto.FLOAT,
        number=4,
    )


class RecognitionAudio(proto.Message):
    r"""Contains audio data in the encoding specified in the
    ``RecognitionConfig``. Either ``content`` or ``uri`` must be
    supplied. Supplying both or neither returns
    [google.rpc.Code.INVALID_ARGUMENT][google.rpc.Code.INVALID_ARGUMENT].
    See `content
    limits <https://cloud.google.com/speech-to-text/quotas#content>`__.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        content (bytes):
            The audio data bytes encoded as specified in
            ``RecognitionConfig``. Note: as with all bytes fields, proto
            buffers use a pure binary representation, whereas JSON
            representations use base64.

            This field is a member of `oneof`_ ``audio_source``.
        uri (str):
            URI that points to a file that contains audio data bytes as
            specified in ``RecognitionConfig``. The file must not be
            compressed (for example, gzip). Currently, only Google Cloud
            Storage URIs are supported, which must be specified in the
            following format: ``gs://bucket_name/object_name`` (other
            URI formats return
            [google.rpc.Code.INVALID_ARGUMENT][google.rpc.Code.INVALID_ARGUMENT]).
            For more information, see `Request
            URIs <https://cloud.google.com/storage/docs/reference-uris>`__.

            This field is a member of `oneof`_ ``audio_source``.
    """

    content: bytes = proto.Field(
        proto.BYTES,
        number=1,
        oneof="audio_source",
    )
    uri: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="audio_source",
    )


class RecognizeResponse(proto.Message):
    r"""The only message returned to the client by the ``Recognize`` method.
    It contains the result as zero or more sequential
    ``SpeechRecognitionResult`` messages.

    Attributes:
        results (MutableSequence[google.cloud.speech_v1p1beta1.types.SpeechRecognitionResult]):
            Sequential list of transcription results
            corresponding to sequential portions of audio.
        total_billed_time (google.protobuf.duration_pb2.Duration):
            When available, billed audio seconds for the
            corresponding request.
        speech_adaptation_info (google.cloud.speech_v1p1beta1.types.SpeechAdaptationInfo):
            Provides information on adaptation behavior
            in response
        request_id (int):
            The ID associated with the request. This is a
            unique ID specific only to the given request.
        using_legacy_models (bool):
            Whether request used legacy asr models (was
            not automatically migrated to use conformer
            models).
    """

    results: MutableSequence["SpeechRecognitionResult"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="SpeechRecognitionResult",
    )
    total_billed_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=3,
        message=duration_pb2.Duration,
    )
    speech_adaptation_info: "SpeechAdaptationInfo" = proto.Field(
        proto.MESSAGE,
        number=7,
        message="SpeechAdaptationInfo",
    )
    request_id: int = proto.Field(
        proto.INT64,
        number=8,
    )
    using_legacy_models: bool = proto.Field(
        proto.BOOL,
        number=9,
    )


class LongRunningRecognizeResponse(proto.Message):
    r"""The only message returned to the client by the
    ``LongRunningRecognize`` method. It contains the result as zero or
    more sequential ``SpeechRecognitionResult`` messages. It is included
    in the ``result.response`` field of the ``Operation`` returned by
    the ``GetOperation`` call of the ``google::longrunning::Operations``
    service.

    Attributes:
        results (MutableSequence[google.cloud.speech_v1p1beta1.types.SpeechRecognitionResult]):
            Sequential list of transcription results
            corresponding to sequential portions of audio.
        total_billed_time (google.protobuf.duration_pb2.Duration):
            When available, billed audio seconds for the
            corresponding request.
        output_config (google.cloud.speech_v1p1beta1.types.TranscriptOutputConfig):
            Original output config if present in the
            request.
        output_error (google.rpc.status_pb2.Status):
            If the transcript output fails this field
            contains the relevant error.
        speech_adaptation_info (google.cloud.speech_v1p1beta1.types.SpeechAdaptationInfo):
            Provides information on speech adaptation
            behavior in response
        request_id (int):
            The ID associated with the request. This is a
            unique ID specific only to the given request.
    """

    results: MutableSequence["SpeechRecognitionResult"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="SpeechRecognitionResult",
    )
    total_billed_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=3,
        message=duration_pb2.Duration,
    )
    output_config: "TranscriptOutputConfig" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="TranscriptOutputConfig",
    )
    output_error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=7,
        message=status_pb2.Status,
    )
    speech_adaptation_info: "SpeechAdaptationInfo" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="SpeechAdaptationInfo",
    )
    request_id: int = proto.Field(
        proto.INT64,
        number=9,
    )


class LongRunningRecognizeMetadata(proto.Message):
    r"""Describes the progress of a long-running ``LongRunningRecognize``
    call. It is included in the ``metadata`` field of the ``Operation``
    returned by the ``GetOperation`` call of the
    ``google::longrunning::Operations`` service.

    Attributes:
        progress_percent (int):
            Approximate percentage of audio processed
            thus far. Guaranteed to be 100 when the audio is
            fully processed and the results are available.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Time when the request was received.
        last_update_time (google.protobuf.timestamp_pb2.Timestamp):
            Time of the most recent processing update.
        uri (str):
            Output only. The URI of the audio file being
            transcribed. Empty if the audio was sent as byte
            content.
        output_config (google.cloud.speech_v1p1beta1.types.TranscriptOutputConfig):
            Output only. A copy of the
            TranscriptOutputConfig if it was set in the
            request.
    """

    progress_percent: int = proto.Field(
        proto.INT32,
        number=1,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    last_update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    uri: str = proto.Field(
        proto.STRING,
        number=4,
    )
    output_config: "TranscriptOutputConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="TranscriptOutputConfig",
    )


class StreamingRecognizeResponse(proto.Message):
    r"""``StreamingRecognizeResponse`` is the only message returned to the
    client by ``StreamingRecognize``. A series of zero or more
    ``StreamingRecognizeResponse`` messages are streamed back to the
    client. If there is no recognizable audio, and ``single_utterance``
    is set to false, then no messages are streamed back to the client.

    Here's an example of a series of ``StreamingRecognizeResponse``\ s
    that might be returned while processing audio:

    1. results { alternatives { transcript: "tube" } stability: 0.01 }

    2. results { alternatives { transcript: "to be a" } stability: 0.01
       }

    3. results { alternatives { transcript: "to be" } stability: 0.9 }
       results { alternatives { transcript: " or not to be" } stability:
       0.01 }

    4. results { alternatives { transcript: "to be or not to be"
       confidence: 0.92 } alternatives { transcript: "to bee or not to
       bee" } is_final: true }

    5. results { alternatives { transcript: " that's" } stability: 0.01
       }

    6. results { alternatives { transcript: " that is" } stability: 0.9
       } results { alternatives { transcript: " the question" }
       stability: 0.01 }

    7. results { alternatives { transcript: " that is the question"
       confidence: 0.98 } alternatives { transcript: " that was the
       question" } is_final: true }

    Notes:

    - Only two of the above responses #4 and #7 contain final results;
      they are indicated by ``is_final: true``. Concatenating these
      together generates the full transcript: "to be or not to be that
      is the question".

    - The others contain interim ``results``. #3 and #6 contain two
      interim ``results``: the first portion has a high stability and is
      less likely to change; the second portion has a low stability and
      is very likely to change. A UI designer might choose to show only
      high stability ``results``.

    - The specific ``stability`` and ``confidence`` values shown above
      are only for illustrative purposes. Actual values may vary.

    - In each response, only one of these fields will be set: ``error``,
      ``speech_event_type``, or one or more (repeated) ``results``.

    Attributes:
        error (google.rpc.status_pb2.Status):
            If set, returns a [google.rpc.Status][google.rpc.Status]
            message that specifies the error for the operation.
        results (MutableSequence[google.cloud.speech_v1p1beta1.types.StreamingRecognitionResult]):
            This repeated list contains zero or more results that
            correspond to consecutive portions of the audio currently
            being processed. It contains zero or one ``is_final=true``
            result (the newly settled portion), followed by zero or more
            ``is_final=false`` results (the interim results).
        speech_event_type (google.cloud.speech_v1p1beta1.types.StreamingRecognizeResponse.SpeechEventType):
            Indicates the type of speech event.
        speech_event_time (google.protobuf.duration_pb2.Duration):
            Time offset between the beginning of the
            audio and event emission.
        total_billed_time (google.protobuf.duration_pb2.Duration):
            When available, billed audio seconds for the
            stream. Set only if this is the last response in
            the stream.
        speech_adaptation_info (google.cloud.speech_v1p1beta1.types.SpeechAdaptationInfo):
            Provides information on adaptation behavior
            in response
        request_id (int):
            The ID associated with the request. This is a
            unique ID specific only to the given request.
    """

    class SpeechEventType(proto.Enum):
        r"""Indicates the type of speech event.

        Values:
            SPEECH_EVENT_UNSPECIFIED (0):
                No speech event specified.
            END_OF_SINGLE_UTTERANCE (1):
                This event indicates that the server has detected the end of
                the user's speech utterance and expects no additional
                speech. Therefore, the server will not process additional
                audio (although it may subsequently return additional
                results). The client should stop sending additional audio
                data, half-close the gRPC connection, and wait for any
                additional results until the server closes the gRPC
                connection. This event is only sent if ``single_utterance``
                was set to ``true``, and is not used otherwise.
            SPEECH_ACTIVITY_BEGIN (2):
                This event indicates that the server has detected the
                beginning of human voice activity in the stream. This event
                can be returned multiple times if speech starts and stops
                repeatedly throughout the stream. This event is only sent if
                ``voice_activity_events`` is set to true.
            SPEECH_ACTIVITY_END (3):
                This event indicates that the server has detected the end of
                human voice activity in the stream. This event can be
                returned multiple times if speech starts and stops
                repeatedly throughout the stream. This event is only sent if
                ``voice_activity_events`` is set to true.
            SPEECH_ACTIVITY_TIMEOUT (4):
                This event indicates that the user-set
                timeout for speech activity begin or end has
                exceeded. Upon receiving this event, the client
                is expected to send a half close. Further audio
                will not be processed.
        """
        SPEECH_EVENT_UNSPECIFIED = 0
        END_OF_SINGLE_UTTERANCE = 1
        SPEECH_ACTIVITY_BEGIN = 2
        SPEECH_ACTIVITY_END = 3
        SPEECH_ACTIVITY_TIMEOUT = 4

    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    results: MutableSequence["StreamingRecognitionResult"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="StreamingRecognitionResult",
    )
    speech_event_type: SpeechEventType = proto.Field(
        proto.ENUM,
        number=4,
        enum=SpeechEventType,
    )
    speech_event_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=8,
        message=duration_pb2.Duration,
    )
    total_billed_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=5,
        message=duration_pb2.Duration,
    )
    speech_adaptation_info: "SpeechAdaptationInfo" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="SpeechAdaptationInfo",
    )
    request_id: int = proto.Field(
        proto.INT64,
        number=10,
    )


class StreamingRecognitionResult(proto.Message):
    r"""A streaming speech recognition result corresponding to a
    portion of the audio that is currently being processed.

    Attributes:
        alternatives (MutableSequence[google.cloud.speech_v1p1beta1.types.SpeechRecognitionAlternative]):
            May contain one or more recognition hypotheses (up to the
            maximum specified in ``max_alternatives``). These
            alternatives are ordered in terms of accuracy, with the top
            (first) alternative being the most probable, as ranked by
            the recognizer.
        is_final (bool):
            If ``false``, this ``StreamingRecognitionResult`` represents
            an interim result that may change. If ``true``, this is the
            final time the speech service will return this particular
            ``StreamingRecognitionResult``, the recognizer will not
            return any further hypotheses for this portion of the
            transcript and corresponding audio.
        stability (float):
            An estimate of the likelihood that the recognizer will not
            change its guess about this interim result. Values range
            from 0.0 (completely unstable) to 1.0 (completely stable).
            This field is only provided for interim results
            (``is_final=false``). The default of 0.0 is a sentinel value
            indicating ``stability`` was not set.
        result_end_time (google.protobuf.duration_pb2.Duration):
            Time offset of the end of this result
            relative to the beginning of the audio.
        channel_tag (int):
            For multi-channel audio, this is the channel number
            corresponding to the recognized result for the audio from
            that channel. For audio_channel_count = N, its output values
            can range from '1' to 'N'.
        language_code (str):
            Output only. The
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag of the language in this result. This language
            code was detected to have the most likelihood of being
            spoken in the audio.
    """

    alternatives: MutableSequence["SpeechRecognitionAlternative"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="SpeechRecognitionAlternative",
    )
    is_final: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    stability: float = proto.Field(
        proto.FLOAT,
        number=3,
    )
    result_end_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=4,
        message=duration_pb2.Duration,
    )
    channel_tag: int = proto.Field(
        proto.INT32,
        number=5,
    )
    language_code: str = proto.Field(
        proto.STRING,
        number=6,
    )


class SpeechRecognitionResult(proto.Message):
    r"""A speech recognition result corresponding to a portion of the
    audio.

    Attributes:
        alternatives (MutableSequence[google.cloud.speech_v1p1beta1.types.SpeechRecognitionAlternative]):
            May contain one or more recognition hypotheses (up to the
            maximum specified in ``max_alternatives``). These
            alternatives are ordered in terms of accuracy, with the top
            (first) alternative being the most probable, as ranked by
            the recognizer.
        channel_tag (int):
            For multi-channel audio, this is the channel number
            corresponding to the recognized result for the audio from
            that channel. For audio_channel_count = N, its output values
            can range from '1' to 'N'.
        result_end_time (google.protobuf.duration_pb2.Duration):
            Time offset of the end of this result
            relative to the beginning of the audio.
        language_code (str):
            Output only. The
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag of the language in this result. This language
            code was detected to have the most likelihood of being
            spoken in the audio.
    """

    alternatives: MutableSequence["SpeechRecognitionAlternative"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="SpeechRecognitionAlternative",
    )
    channel_tag: int = proto.Field(
        proto.INT32,
        number=2,
    )
    result_end_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=4,
        message=duration_pb2.Duration,
    )
    language_code: str = proto.Field(
        proto.STRING,
        number=5,
    )


class SpeechRecognitionAlternative(proto.Message):
    r"""Alternative hypotheses (a.k.a. n-best list).

    Attributes:
        transcript (str):
            Transcript text representing the words that
            the user spoke. In languages that use spaces to
            separate words, the transcript might have a
            leading space if it isn't the first result. You
            can concatenate each result to obtain the full
            transcript without using a separator.
        confidence (float):
            The confidence estimate between 0.0 and 1.0. A higher number
            indicates an estimated greater likelihood that the
            recognized words are correct. This field is set only for the
            top alternative of a non-streaming result or, of a streaming
            result where ``is_final=true``. This field is not guaranteed
            to be accurate and users should not rely on it to be always
            provided. The default of 0.0 is a sentinel value indicating
            ``confidence`` was not set.
        words (MutableSequence[google.cloud.speech_v1p1beta1.types.WordInfo]):
            A list of word-specific information for each recognized
            word. Note: When ``enable_speaker_diarization`` is true, you
            will see all the words from the beginning of the audio.
    """

    transcript: str = proto.Field(
        proto.STRING,
        number=1,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=2,
    )
    words: MutableSequence["WordInfo"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="WordInfo",
    )


class WordInfo(proto.Message):
    r"""Word-specific information for recognized words.

    Attributes:
        start_time (google.protobuf.duration_pb2.Duration):
            Time offset relative to the beginning of the audio, and
            corresponding to the start of the spoken word. This field is
            only set if ``enable_word_time_offsets=true`` and only in
            the top hypothesis. This is an experimental feature and the
            accuracy of the time offset can vary.
        end_time (google.protobuf.duration_pb2.Duration):
            Time offset relative to the beginning of the audio, and
            corresponding to the end of the spoken word. This field is
            only set if ``enable_word_time_offsets=true`` and only in
            the top hypothesis. This is an experimental feature and the
            accuracy of the time offset can vary.
        word (str):
            The word corresponding to this set of
            information.
        confidence (float):
            The confidence estimate between 0.0 and 1.0. A higher number
            indicates an estimated greater likelihood that the
            recognized words are correct. This field is set only for the
            top alternative of a non-streaming result or, of a streaming
            result where ``is_final=true``. This field is not guaranteed
            to be accurate and users should not rely on it to be always
            provided. The default of 0.0 is a sentinel value indicating
            ``confidence`` was not set.
        speaker_tag (int):
            Output only. A distinct integer value is assigned for every
            speaker within the audio. This field specifies which one of
            those speakers was detected to have spoken this word. Value
            ranges from '1' to diarization_speaker_count. speaker_tag is
            set if enable_speaker_diarization = 'true' and only for the
            top alternative. Note: Use speaker_label instead.
        speaker_label (str):
            Output only. A label value assigned for every unique speaker
            within the audio. This field specifies which speaker was
            detected to have spoken this word. For some models, like
            medical_conversation this can be actual speaker role, for
            example "patient" or "provider", but generally this would be
            a number identifying a speaker. This field is only set if
            enable_speaker_diarization = 'true' and only for the top
            alternative.
    """

    start_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )
    end_time: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=2,
        message=duration_pb2.Duration,
    )
    word: str = proto.Field(
        proto.STRING,
        number=3,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=4,
    )
    speaker_tag: int = proto.Field(
        proto.INT32,
        number=5,
    )
    speaker_label: str = proto.Field(
        proto.STRING,
        number=6,
    )


class SpeechAdaptationInfo(proto.Message):
    r"""Information on speech adaptation use in results

    Attributes:
        adaptation_timeout (bool):
            Whether there was a timeout when applying
            speech adaptation. If true, adaptation had no
            effect in the response transcript.
        timeout_message (str):
            If set, returns a message specifying which
            part of the speech adaptation request timed out.
    """

    adaptation_timeout: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    timeout_message: str = proto.Field(
        proto.STRING,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
