# Copyright 2023 Google LLC
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
"""Classes for working with vision models."""

from vertexai.vision_models._vision_models import (
    ControlImageConfig,
    ControlReferenceImage,
    EntityLabel,
    GeneratedImage,
    GeneratedMask,
    Image,
    ImageCaptioningModel,
    ImageGenerationModel,
    ImageGenerationResponse,
    ImageQnAModel,
    ImageSegmentationModel,
    ImageSegmentationResponse,
    ImageTextModel,
    MaskImageConfig,
    MaskReferenceImage,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
    RawReferenceImage,
    ReferenceImage,
    Scribble,
    StyleImageConfig,
    StyleReferenceImage,
    SubjectImageConfig,
    SubjectReferenceImage,
    Video,
    VideoEmbedding,
    VideoSegmentConfig,
    WatermarkVerificationModel,
    WatermarkVerificationResponse,
)

__all__ = [
    "ControlImageConfig",
    "ControlReferenceImage",
    "EntityLabel",
    "GeneratedMask",
    "Image",
    "ImageGenerationModel",
    "ImageGenerationResponse",
    "ImageCaptioningModel",
    "ImageQnAModel",
    "ImageSegmentationModel",
    "ImageSegmentationResponse",
    "ImageTextModel",
    "MaskImageConfig",
    "MaskReferenceImage",
    "WatermarkVerificationModel",
    "GeneratedImage",
    "MultiModalEmbeddingModel",
    "MultiModalEmbeddingResponse",
    "RawReferenceImage",
    "ReferenceImage",
    "Scribble",
    "StyleImageConfig",
    "StyleReferenceImage",
    "SubjectImageConfig",
    "SubjectReferenceImage",
    "Video",
    "VideoEmbedding",
    "VideoSegmentConfig",
    "WatermarkVerificationResponse",
]
