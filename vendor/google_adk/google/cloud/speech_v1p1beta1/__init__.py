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
from google.cloud.speech_v1p1beta1 import gapic_version as package_version

__version__ = package_version.__version__


from .services.adaptation import AdaptationAsyncClient, AdaptationClient
from .services.speech import SpeechAsyncClient, SpeechClient
from .types.cloud_speech import (
    LongRunningRecognizeMetadata,
    LongRunningRecognizeRequest,
    LongRunningRecognizeResponse,
    RecognitionAudio,
    RecognitionConfig,
    RecognitionMetadata,
    RecognizeRequest,
    RecognizeResponse,
    SpeakerDiarizationConfig,
    SpeechAdaptationInfo,
    SpeechContext,
    SpeechRecognitionAlternative,
    SpeechRecognitionResult,
    StreamingRecognitionConfig,
    StreamingRecognitionResult,
    StreamingRecognizeRequest,
    StreamingRecognizeResponse,
    TranscriptOutputConfig,
    WordInfo,
)
from .types.cloud_speech_adaptation import (
    CreateCustomClassRequest,
    CreatePhraseSetRequest,
    DeleteCustomClassRequest,
    DeletePhraseSetRequest,
    GetCustomClassRequest,
    GetPhraseSetRequest,
    ListCustomClassesRequest,
    ListCustomClassesResponse,
    ListPhraseSetRequest,
    ListPhraseSetResponse,
    UpdateCustomClassRequest,
    UpdatePhraseSetRequest,
)
from .types.resource import (
    CustomClass,
    PhraseSet,
    SpeechAdaptation,
    TranscriptNormalization,
)

from google.cloud.speech_v1.helpers import SpeechHelpers


class SpeechClient(SpeechHelpers, SpeechClient):
    __doc__ = SpeechClient.__doc__


__all__ = (
    "AdaptationAsyncClient",
    "SpeechAsyncClient",
    "AdaptationClient",
    "CreateCustomClassRequest",
    "CreatePhraseSetRequest",
    "CustomClass",
    "DeleteCustomClassRequest",
    "DeletePhraseSetRequest",
    "GetCustomClassRequest",
    "GetPhraseSetRequest",
    "ListCustomClassesRequest",
    "ListCustomClassesResponse",
    "ListPhraseSetRequest",
    "ListPhraseSetResponse",
    "LongRunningRecognizeMetadata",
    "LongRunningRecognizeRequest",
    "LongRunningRecognizeResponse",
    "PhraseSet",
    "RecognitionAudio",
    "RecognitionConfig",
    "RecognitionMetadata",
    "RecognizeRequest",
    "RecognizeResponse",
    "SpeakerDiarizationConfig",
    "SpeechAdaptation",
    "SpeechAdaptationInfo",
    "SpeechClient",
    "SpeechContext",
    "SpeechRecognitionAlternative",
    "SpeechRecognitionResult",
    "StreamingRecognitionConfig",
    "StreamingRecognitionResult",
    "StreamingRecognizeRequest",
    "StreamingRecognizeResponse",
    "TranscriptNormalization",
    "TranscriptOutputConfig",
    "UpdateCustomClassRequest",
    "UpdatePhraseSetRequest",
    "WordInfo",
)
