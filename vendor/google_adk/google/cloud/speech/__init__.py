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
from google.cloud.speech import gapic_version as package_version

__version__ = package_version.__version__


from google.cloud.speech_v1.services.adaptation.async_client import (
    AdaptationAsyncClient,
)
from google.cloud.speech_v1.services.adaptation.client import AdaptationClient
from google.cloud.speech_v1.services.speech.async_client import SpeechAsyncClient
from google.cloud.speech_v1 import SpeechClient
from google.cloud.speech_v1.types.cloud_speech import (
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
from google.cloud.speech_v1.types.cloud_speech_adaptation import (
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
from google.cloud.speech_v1.types.resource import (
    CustomClass,
    PhraseSet,
    SpeechAdaptation,
    TranscriptNormalization,
)

__all__ = (
    "AdaptationClient",
    "AdaptationAsyncClient",
    "SpeechClient",
    "SpeechAsyncClient",
    "LongRunningRecognizeMetadata",
    "LongRunningRecognizeRequest",
    "LongRunningRecognizeResponse",
    "RecognitionAudio",
    "RecognitionConfig",
    "RecognitionMetadata",
    "RecognizeRequest",
    "RecognizeResponse",
    "SpeakerDiarizationConfig",
    "SpeechAdaptationInfo",
    "SpeechContext",
    "SpeechRecognitionAlternative",
    "SpeechRecognitionResult",
    "StreamingRecognitionConfig",
    "StreamingRecognitionResult",
    "StreamingRecognizeRequest",
    "StreamingRecognizeResponse",
    "TranscriptOutputConfig",
    "WordInfo",
    "CreateCustomClassRequest",
    "CreatePhraseSetRequest",
    "DeleteCustomClassRequest",
    "DeletePhraseSetRequest",
    "GetCustomClassRequest",
    "GetPhraseSetRequest",
    "ListCustomClassesRequest",
    "ListCustomClassesResponse",
    "ListPhraseSetRequest",
    "ListPhraseSetResponse",
    "UpdateCustomClassRequest",
    "UpdatePhraseSetRequest",
    "CustomClass",
    "PhraseSet",
    "SpeechAdaptation",
    "TranscriptNormalization",
)
