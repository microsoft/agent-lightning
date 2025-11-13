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
# pylint: disable=bad-continuation, line-too-long, protected-access
"""Classes for working with vision models."""

import base64
import dataclasses
import hashlib
import io
import json
import pathlib
import typing
from typing import Any, Dict, List, Literal, Optional, Union
import urllib

from google.cloud import storage

from google.cloud.aiplatform import initializer as aiplatform_initializer
from vertexai._model_garden import _model_garden_models
from vertexai._utils import warning_logs
from google.cloud.aiplatform.utils.gcs_utils import blob_from_uri

# pylint: disable=g-import-not-at-top
try:
    from IPython import display as IPython_display
except ImportError:
    IPython_display = None

try:
    from PIL import Image as PIL_Image
except ImportError:
    PIL_Image = None


_SUPPORTED_UPSCALING_SIZES = [2048, 4096]


@dataclasses.dataclass
class MaskImageConfig:
    """Mask image config.

    Attributes:
        mask_mode: Mask mode for the image. Can take the following values:
            * MASK_MODE_DEFAULT: Default mask mode
            * MASK_MODE_USER_PROVIDED: User provided mask
            * MASK_MODE_BACKGROUND: Background mask
            * MASK_MODE_FOREGROUND: Foreground mask
            * MASK_MODE_SEMANTIC: Semantic mask
        dilation: Dilation percentage of the mask provided. Float between 0 and 1.
            Defaults to 1.0
        segmentation_classes: List of class IDs for segmentation. Max of 5 IDs
    """

    mask_mode: Literal[
        "MASK_MODE_DEFAULT",
        "MASK_MODE_USER_PROVIDED",
        "MASK_MODE_BACKGROUND",
        "MASK_MODE_FOREGROUND",
        "MASK_MODE_SEMANTIC",
    ]
    segmentation_classes: Optional[List[int]] = None
    dilation: Optional[float] = None


@dataclasses.dataclass
class ControlImageConfig:
    """Control image config.

    Attributes:
        control_type: Control type for the image. Can take the following values:
            * CONTROL_TYPE_DEFAULT: Default control type
            * CONTROL_TYPE_SCRIBBLE: Scribble control type
            * CONTROL_TYPE_FACE_MESH: Face mesh control type
            * CONTROL_TYPE_CANNY: Canny control type
        enable_control_image_computation: When set to True, the control image
            will be computed by the model based on the control type. When set to
            False, the control image will be provided by the user.
    """

    control_type: Literal[
        "CONTROL_TYPE_DEFAULT",
        "CONTROL_TYPE_SCRIBBLE",
        "CONTROL_TYPE_FACE_MESH",
        "CONTROL_TYPE_CANNY",
    ]
    enable_control_image_computation: Optional[bool] = False


@dataclasses.dataclass
class StyleImageConfig:
    """Style image config.

    Attributes:
        style_description: Style description for the image.
    """

    style_description: str


@dataclasses.dataclass
class SubjectImageConfig:
    """Subject image config.

    Attributes:
        subject_description: Subject description for the image.
        subject_type: Subject type for the image. Can take the following values:
            * SUBJECT_TYPE_DEFAULT: Default subject type
            * SUBJECT_TYPE_PERSON: Person subject type
            * SUBJECT_TYPE_ANIMAL: Animal subject type
            * SUBJECT_TYPE_PRODUCT: Product subject type
    """

    subject_description: str
    subject_type: Literal[
        "SUBJECT_TYPE_DEFAULT",
        "SUBJECT_TYPE_PERSON",
        "SUBJECT_TYPE_ANIMAL",
        "SUBJECT_TYPE_PRODUCT",
    ]


class Image:
    """Image."""

    __module__ = "vertexai.vision_models"

    _loaded_bytes: Optional[bytes] = None
    _loaded_image: Optional["PIL_Image.Image"] = None
    _gcs_uri: Optional[str] = None

    def __init__(
        self,
        image_bytes: Optional[bytes] = None,
        gcs_uri: Optional[str] = None,
    ):
        """Creates an `Image` object.

        Args:
            image_bytes: Image file bytes. Image can be in PNG or JPEG format.
            gcs_uri: Image URI in Google Cloud Storage.
        """
        warning_logs.show_deprecation_warning()
        if bool(image_bytes) == bool(gcs_uri):
            raise ValueError("Either image_bytes or gcs_uri must be provided.")

        self._image_bytes = image_bytes
        self._gcs_uri = gcs_uri

    @staticmethod
    def load_from_file(location: str) -> "Image":
        """Loads image from local file or Google Cloud Storage.

        Args:
            location: Local path or Google Cloud Storage uri from where to load
                the image.

        Returns:
            Loaded image as an `Image` object.
        """
        parsed_url = urllib.parse.urlparse(location)
        if (
            parsed_url.scheme == "https"
            and parsed_url.netloc == "storage.googleapis.com"
        ):
            parsed_url = parsed_url._replace(
                scheme="gs", netloc="", path=f"/{urllib.parse.unquote(parsed_url.path)}"
            )
            location = urllib.parse.urlunparse(parsed_url)

        if parsed_url.scheme == "gs":
            return Image(gcs_uri=location)

        # Load image from local path
        image_bytes = pathlib.Path(location).read_bytes()
        image = Image(image_bytes=image_bytes)
        return image

    @property
    def _blob(self) -> storage.Blob:
        if self._gcs_uri is None:
            raise AttributeError("_blob is only supported when gcs_uri is set.")
        storage_client = storage.Client(
            credentials=aiplatform_initializer.global_config.credentials
        )
        blob = blob_from_uri(uri=self._gcs_uri, client=storage_client)
        # Needed to populate `blob.content_type`
        blob.reload()
        return blob

    @property
    def _image_bytes(self) -> bytes:
        if self._loaded_bytes is None:
            self._loaded_bytes = self._blob.download_as_bytes()
        return self._loaded_bytes

    @_image_bytes.setter
    def _image_bytes(self, value: bytes):
        self._loaded_bytes = value

    @property
    def _pil_image(self) -> "PIL_Image.Image":
        if self._loaded_image is None:
            if not PIL_Image:
                raise RuntimeError(
                    "The PIL module is not available. Please install the Pillow package."
                )
            self._loaded_image = PIL_Image.open(io.BytesIO(self._image_bytes))
        return self._loaded_image

    @property
    def _size(self):
        return self._pil_image.size

    @property
    def _mime_type(self) -> str:
        """Returns the MIME type of the image."""
        if self._gcs_uri:
            return self._blob.content_type
        if PIL_Image:
            return PIL_Image.MIME.get(self._pil_image.format, "image/jpeg")
        # Fall back to jpeg
        return "image/jpeg"

    def show(self):
        """Shows the image.

        This method only works when in a notebook environment.
        """
        if PIL_Image and IPython_display:
            IPython_display.display(self._pil_image)

    def save(self, location: str):
        """Saves image to a file.

        Args:
            location: Local path where to save the image.
        """
        pathlib.Path(location).write_bytes(self._image_bytes)

    def _as_base64_string(self) -> str:
        """Encodes image using the base64 encoding.

        Returns:
            Base64 encoding of the image as a string.
        """
        # ! b64encode returns `bytes` object, not `str`.
        # We need to convert `bytes` to `str`, otherwise we get service error:
        # "received initial metadata size exceeds limit"
        return base64.b64encode(self._image_bytes).decode("ascii")


class ReferenceImage:
    """Reference image.

    This is a new base API object for Imagen 3.0 Capabilities.
    """

    __module__ = "vertexai.vision_models"
    reference_image: Optional[Image] = None
    reference_id: int
    config: Optional[
        Union[MaskImageConfig, ControlImageConfig, StyleImageConfig, SubjectImageConfig]
    ] = None
    reference_type: Optional[str] = None

    def __init__(
        self,
        reference_id,
        image: Optional[Union[bytes, Image, str]] = None,
    ):
        """Creates a `ReferenceImage` object.

        Args:
            reference_id: Reference ID for the image.
            image: Either Image object or Image file bytes. Image can be in PNG or
              JPEG format.
        """
        if image is not None:
            if isinstance(image, Image):
                self.reference_image = image
            elif isinstance(image, bytes):
                self.reference_image = Image(image_bytes=image)
            elif isinstance(image, str):
                self.reference_image = Image(gcs_uri=image)
            else:
                raise ValueError("Image must be either Image object, bytes or gcs_uri.")
        self.reference_id = reference_id


class RawReferenceImage(ReferenceImage):
    """Raw reference image.

    This encapsulates the raw reference image type.
    """

    reference_type = "REFERENCE_TYPE_RAW"


class MaskReferenceImage(ReferenceImage):
    """Mask reference image. This encapsulates the mask reference image type."""

    mask_mode_enum_map = {
        "default": "MASK_MODE_DEFAULT",
        "user_provided": "MASK_MODE_USER_PROVIDED",
        "background": "MASK_MODE_BACKGROUND",
        "foreground": "MASK_MODE_FOREGROUND",
        "semantic": "MASK_MODE_SEMANTIC",
    }
    reference_type = "REFERENCE_TYPE_MASK"

    def __init__(
        self,
        reference_id,
        image: Optional[Union[bytes, Image, str]] = None,
        mask_mode: Optional[
            Literal["default", "user_provided", "background", "foreground", "semantic"]
        ] = None,
        dilation: Optional[float] = None,
        segmentation_classes: Optional[List[int]] = None,
    ):
        """Creates a `MaskReferenceImage` object.

        Args:
            reference_id: Reference ID for the image. Required.
            image: Either Image object or Image file bytes. Image can be in PNG or
              JPEG format.
            mask_mode: Mask mode for the image. Can take the following values:
                * default: Default mask mode
                * user_provided: User provided mask
                * background: Background mask
                * foreground: Foreground mask
                * semantic: Semantic mask
            dilation: Dilation percentage of the mask
            segmentation_classes: List of class IDs for segmentation. Max of 5 IDs
        """
        self.config = MaskImageConfig(
            mask_mode=(
                self.mask_mode_enum_map[mask_mode]
                if mask_mode in self.mask_mode_enum_map
                else "MASK_MODE_DEFAULT"
            ),
            dilation=dilation,
            segmentation_classes=segmentation_classes,
        )
        super().__init__(reference_id, image)


class ControlReferenceImage(ReferenceImage):
    """Control reference image.

    This encapsulates the control reference image type.
    """

    control_type_enum_map = {
        "default": "CONTROL_TYPE_DEFAULT",
        "scribble": "CONTROL_TYPE_SCRIBBLE",
        "face_mesh": "CONTROL_TYPE_FACE_MESH",
        "canny": "CONTROL_TYPE_CANNY",
    }
    reference_type = "REFERENCE_TYPE_CONTROL"

    def __init__(
        self,
        reference_id,
        image: Optional[Union[bytes, Image, str]] = None,
        control_type: Optional[
            Literal["default", "scribble", "face_mesh", "canny"]
        ] = None,
        enable_control_image_computation: Optional[bool] = False,
    ):
        """Creates a `ControlReferenceImage` object.

        Args:
            reference_id: Reference ID for the image. Required.
            image: Either Image object or Image file bytes. Image can be in PNG or
              JPEG format.
            control_type: Control type for the image. Can take the following values:
                * default: Default control type
                * scribble: Scribble control type
                * face_mesh: Face mesh control type
                * canny: Canny control type
            enable_control_image_computation: When set to True, the control image
              will be computed by the model based on the control type. When set to
              False, the control image will be provided by the user.
        """
        super().__init__(reference_id, image)
        self.config = ControlImageConfig(
            control_type=(
                self.control_type_enum_map[control_type]
                if control_type in self.control_type_enum_map
                else "CONTROL_TYPE_DEFAULT"
            ),
            enable_control_image_computation=enable_control_image_computation,
        )


class StyleReferenceImage(ReferenceImage):
    """Style reference image. This encapsulates the style reference image type."""

    reference_type = "REFERENCE_TYPE_STYLE"

    def __init__(
        self,
        reference_id,
        image: Optional[Union[bytes, Image, str]] = None,
        style_description: Optional[str] = None,
    ):
        """Creates a `StyleReferenceImage` object.

        Args:
            reference_id: Reference ID for the image. Required.
            image: Either Image object or Image file bytes. Image can be in PNG or
              JPEG format.
            style_description: Style description for the image.
        """
        super().__init__(reference_id, image)
        self.config = StyleImageConfig(style_description=style_description)


class SubjectReferenceImage(ReferenceImage):
    """Subject reference image.

    This encapsulates the subject reference image type.
    """

    subject_type_enum_map = {
        "default": "SUBJECT_TYPE_DEFAULT",
        "person": "SUBJECT_TYPE_PERSON",
        "animal": "SUBJECT_TYPE_ANIMAL",
        "product": "SUBJECT_TYPE_PRODUCT",
    }
    reference_type = "REFERENCE_TYPE_SUBJECT"

    def __init__(
        self,
        reference_id,
        image: Optional[Union[bytes, Image, str]] = None,
        subject_description: Optional[str] = None,
        subject_type: Optional[
            Literal["default", "person", "animal", "product"]
        ] = None,
    ):
        """Creates a `SubjectReferenceImage` object.

        Args:
            reference_id: Reference ID for the image. Required.
            image: Either Image object or Image file bytes. Image can be in PNG or
              JPEG format.
            subject_description: Subject description for the image.
            subject_type: Subject type for the image. Can take the following values:
                * default: Default subject type
                * person: Person subject type
                * animal: Animal subject type
                * product: Product subject type
        """
        super().__init__(reference_id, image)
        self.config = SubjectImageConfig(
            subject_description=subject_description,
            subject_type=(
                self.subject_type_enum_map[subject_type]
                if subject_type in self.subject_type_enum_map
                else "SUBJECT_TYPE_DEFAULT"
            ),
        )


class Video:
    """Video."""

    __module__ = "vertexai.vision_models"

    _loaded_bytes: Optional[bytes] = None
    _gcs_uri: Optional[str] = None

    def __init__(
        self,
        video_bytes: Optional[bytes] = None,
        gcs_uri: Optional[str] = None,
    ):
        """Creates a `Video` object.

        Args:
            video_bytes: Video file bytes. Video can be in AVI, FLV, MKV, MOV,
                MP4, MPEG, MPG, WEBM, and WMV formats.
            gcs_uri: Image URI in Google Cloud Storage.
        """
        warning_logs.show_deprecation_warning()
        if bool(video_bytes) == bool(gcs_uri):
            raise ValueError("Either video_bytes or gcs_uri must be provided.")

        self._video_bytes = video_bytes
        self._gcs_uri = gcs_uri

    @staticmethod
    def load_from_file(location: str) -> "Video":
        """Loads video from local file or Google Cloud Storage.

        Args:
            location: Local path or Google Cloud Storage uri from where to load
                the video.

        Returns:
            Loaded video as an `Video` object.
        """
        parsed_url = urllib.parse.urlparse(location)
        if (
            parsed_url.scheme == "https"
            and parsed_url.netloc == "storage.googleapis.com"
        ):
            parsed_url = parsed_url._replace(
                scheme="gs", netloc="", path=f"/{urllib.parse.unquote(parsed_url.path)}"
            )
            location = urllib.parse.urlunparse(parsed_url)

        if parsed_url.scheme == "gs":
            return Video(gcs_uri=location)

        # Load video from local path
        video_bytes = pathlib.Path(location).read_bytes()
        video = Video(video_bytes=video_bytes)
        return video

    @property
    def _blob(self) -> storage.Blob:
        if self._gcs_uri is None:
            raise AttributeError("_blob is only supported when gcs_uri is set.")
        storage_client = storage.Client(
            credentials=aiplatform_initializer.global_config.credentials
        )
        blob = blob_from_uri(uri=self._gcs_uri, client=storage_client)
        # Needed to populate `blob.content_type`
        blob.reload()
        return blob

    @property
    def _video_bytes(self) -> bytes:
        if self._loaded_bytes is None:
            self._loaded_bytes = self._blob.download_as_bytes()
        return self._loaded_bytes

    @_video_bytes.setter
    def _video_bytes(self, value: bytes):
        self._loaded_bytes = value

    @property
    def _mime_type(self) -> str:
        """Returns the MIME type of the video."""
        if self._gcs_uri:
            return self._blob.content_type
        # Fall back to mp4
        return "video/mp4"

    def save(self, location: str):
        """Saves video to a file.

        Args:
            location: Local path where to save the video.
        """
        pathlib.Path(location).write_bytes(self._video_bytes)

    def _as_base64_string(self) -> str:
        """Encodes video using the base64 encoding.

        Returns:
            Base64 encoding of the video as a string.
        """
        # ! b64encode returns `bytes` object, not `str`.
        # We need to convert `bytes` to `str`, otherwise we get service error:
        # "received initial metadata size exceeds limit"
        return base64.b64encode(self._video_bytes).decode("ascii")


class VideoSegmentConfig:
    """The specific video segments (in seconds) the embeddings are generated for."""

    __module__ = "vertexai.vision_models"

    start_offset_sec: int
    end_offset_sec: int
    interval_sec: int

    def __init__(
        self,
        start_offset_sec: int = 0,
        end_offset_sec: int = 120,
        interval_sec: int = 16,
    ):
        """Creates a `VideoSegmentConfig` object.

        Args:
            start_offset_sec: Start time offset (in seconds) to generate embeddings for.
            end_offset_sec: End time offset (in seconds) to generate embeddings for.
            interval_sec: Interval to divide video for generated embeddings.
        """
        warning_logs.show_deprecation_warning()
        self.start_offset_sec = start_offset_sec
        self.end_offset_sec = end_offset_sec
        self.interval_sec = interval_sec


class VideoEmbedding:
    """Embeddings generated from video with offset times."""

    __module__ = "vertexai.vision_models"

    start_offset_sec: int
    end_offset_sec: int
    embedding: List[float]

    def __init__(
        self, start_offset_sec: int, end_offset_sec: int, embedding: List[float]
    ):
        """Creates a `VideoEmbedding` object.

        Args:
            start_offset_sec: Start time offset (in seconds) of generated embeddings.
            end_offset_sec: End time offset (in seconds) of generated embeddings.
            embedding: Generated embedding for interval.
        """
        warning_logs.show_deprecation_warning()
        self.start_offset_sec = start_offset_sec
        self.end_offset_sec = end_offset_sec
        self.embedding = embedding


class ImageGenerationModel(
    _model_garden_models._ModelGardenModel  # pylint: disable=protected-access
):
    """Generates images from text prompt.

    Examples::

        model = ImageGenerationModel.from_pretrained("imagegeneration@002")
        response = model.generate_images(
            prompt="Astronaut riding a horse",
            # Optional:
            number_of_images=1,
            seed=0,
        )
        response[0].show()
        response[0].save("image1.png")
    """

    __module__ = "vertexai.preview.vision_models"

    _INSTANCE_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/predict/instance/vision_generative_model_1.0.0.yaml"

    def _generate_images(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        number_of_images: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect_ratio: Optional[Literal["1:1", "9:16", "16:9", "4:3", "3:4"]] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        base_image: Optional["Image"] = None,
        mask: Optional["Image"] = None,
        reference_images: Optional[List["ReferenceImage"]] = None,
        edit_mode: Optional[
            Literal[
                "inpainting-insert",
                "inpainting-remove",
                "outpainting",
                "product-image",
                "background-swap",
                "default",
            ]
        ] = None,
        mask_mode: Optional[Literal["background", "foreground", "semantic"]] = None,
        segmentation_classes: Optional[List[str]] = None,
        mask_dilation: Optional[float] = None,
        product_position: Optional[Literal["fixed", "reposition"]] = None,
        output_mime_type: Optional[Literal["image/png", "image/jpeg"]] = None,
        compression_quality: Optional[float] = None,
        language: Optional[str] = None,
        output_gcs_uri: Optional[str] = None,
        add_watermark: Optional[bool] = None,
        safety_filter_level: Optional[
            Literal[
                "block_most",
                "block_some",
                "block_few",
                "block_fewest",
                "block_low_and_above",
                "block_medium_and_above",
                "block_only_high",
                "block_none",
            ]
        ] = None,
        person_generation: Optional[
            Literal["dont_allow", "allow_adult", "allow_all"]
        ] = None,
    ) -> "ImageGenerationResponse":
        """Generates images from text prompt.

        Args:
            prompt: Text prompt for the image.
            negative_prompt: A description of what you want to omit in the generated
              images.
            number_of_images: Number of images to generate. Range: 1..8.
            width: Width of the image. One of the sizes must be 256 or 1024.
            height: Height of the image. One of the sizes must be 256 or 1024.
            aspect_ratio: Aspect ratio for the image. Supported values are:
                * 1:1 - Square image
                * 9:16 - Portait image
                * 16:9 - Landscape image
                * 4:3 - Landscape, desktop ratio.
                * 3:4 - Portrait, desktop ratio
            guidance_scale: Controls the strength of the prompt. Suggested values
              are - * 0-9 (low strength) * 10-20 (medium strength) * 21+ (high
              strength)
            seed: Image generation random seed.
            base_image: Base image to use for the image generation.
            mask: Mask for the base image.
            reference_images: Reference images for Imagen 3 Capabilities calls.
            edit_mode: Describes the editing mode for the request. Supported values
              are -
                * inpainting-insert: fills the mask area based on the text
                  prompt (requires mask and text)
                * inpainting-remove: removes the object(s) in the mask area. (requires mask)
                * outpainting: extend the image based on the mask area. (Requires
                  mask)
                * product-image: Changes the background for the predominant
                  product or subject in the image
                * background-swap: Changes the background for the predominant
                  product or subject in the image
                * default: Default editing mode
            mask_mode: Solicits generation of the mask (v/s providing mask as an
              input). Supported values are:
                * background: Automatically generates a mask for all regions except
                  the primary subject(s) of the image
                * foreground: Automatically generates a mask for the primary
                  subjects(s) of the image.
                * semantic: Segment one or more of the segmentation classes using
                  class ID
            segmentation_classes: List of class IDs for segmentation. Max of 5 IDs
            mask_dilation: Defines the dilation percentage of the mask provided.
              Float between 0 and 1. Defaults to 0.03
            product_position: Defines whether the product should stay fixed or be
              repositioned. Supported Values:
                * fixed: Fixed position
                * reposition: Can be moved (default)
            output_mime_type: Which image format should the output be saved as.
              Supported values:
                * image/png: Save as a PNG image
                * image/jpeg: Save as a JPEG image
            compression_quality: Level of compression if the output mime type is
              selected to be image/jpeg. Float between 0 to 100
            language: Language of the text prompt for the image. Default: None.
              Supported values are `"en"` for English, `"hi"` for Hindi, `"ja"` for
              Japanese, `"ko"` for Korean, and `"auto"` for automatic language
              detection.
            output_gcs_uri: Google Cloud Storage uri to store the generated images.
            add_watermark: Add a watermark to the generated image
            safety_filter_level: Adds a filter level to Safety filtering. Supported
              values are:
                * block_most : Strongest filtering level, most strict
                  blocking
                * block_some : Block some problematic prompts and responses
                * block_few : Block fewer problematic prompts and responses
                * block_fewest : Block very few problematic prompts and
                  responses
              For Imagen 3.0 and Imagen 2.0 Editing (model_name:
              `imagen-3.0-generate-001`, `imagen-3.0-fast-generate-001`,
              `imagen-2.0-edit-preview-0627` and `imagegeneration@006`), the
              following safety filter levels are supported:
                * block_low_and_above : Block low and above safety scores
                * block_medium_and_above : Block medium and above safety scores
                * block_only_high : Block only high safety scores
                * block_none : Block nothing
            person_generation: Allow generation of people by the model Supported
              values are:
                * dont_allow : Block generation of people
                * allow_adult : Generate adults, but not children
                * allow_all : Generate adults and children

        Returns:
            An `ImageGenerationResponse` object.
        """
        # Note: Only a single prompt is supported by the service.
        instance = {"prompt": prompt}
        shared_generation_parameters = {
            "prompt": prompt,
            # b/295946075 The service stopped supporting image sizes.
            # "width": width,
            # "height": height,
            "number_of_images_in_batch": number_of_images,
        }

        if base_image:
            if base_image._gcs_uri:  # pylint: disable=protected-access
                instance["image"] = {
                    "gcsUri": base_image._gcs_uri  # pylint: disable=protected-access
                }
                shared_generation_parameters["base_image_uri"] = (
                    base_image._gcs_uri
                )  # pylint: disable=protected-access
            else:
                instance["image"] = {
                    "bytesBase64Encoded": base_image._as_base64_string()  # pylint: disable=protected-access
                }
                shared_generation_parameters["base_image_hash"] = hashlib.sha1(
                    base_image._image_bytes  # pylint: disable=protected-access
                ).hexdigest()

        if mask:
            if mask._gcs_uri:  # pylint: disable=protected-access
                instance["mask"] = {
                    "image": {
                        "gcsUri": mask._gcs_uri  # pylint: disable=protected-access
                    },
                }
                shared_generation_parameters["mask_uri"] = (
                    mask._gcs_uri
                )  # pylint: disable=protected-access
            else:
                instance["mask"] = {
                    "image": {
                        "bytesBase64Encoded": mask._as_base64_string()  # pylint: disable=protected-access
                    },
                }
                shared_generation_parameters["mask_hash"] = hashlib.sha1(
                    mask._image_bytes  # pylint: disable=protected-access
                ).hexdigest()

        if reference_images:
            instance["referenceImages"] = []
            for reference_image in reference_images:
                reference_image_instance = {}
                if not reference_image.reference_image:
                    if reference_image.reference_type != "REFERENCE_TYPE_MASK":
                        raise ValueError(
                            "Reference image must have an image or a gcs uri."
                        )
                else:
                    reference_image_instance["referenceImage"] = {}
                    if (
                        reference_image.reference_image._gcs_uri
                    ):  # pylint: disable=protected-access
                        reference_image_instance["referenceImage"] = {
                            "gcsUri": reference_image.reference_image._gcs_uri  # pylint: disable=protected-access
                        }
                        shared_generation_parameters[
                            f"reference_image_uri_{reference_image.reference_id}"
                        ] = (
                            reference_image.reference_image._gcs_uri
                        )  # pylint: disable=protected-access
                    elif reference_image.reference_image._image_bytes:
                        reference_image_instance["referenceImage"] = {
                            "bytesBase64Encoded": reference_image.reference_image._as_base64_string()  # pylint: disable=protected-access
                        }
                        shared_generation_parameters[
                            f"reference_image_hash_{reference_image.reference_id}"
                        ] = hashlib.sha1(
                            reference_image.reference_image._image_bytes  # pylint: disable=protected-access
                        ).hexdigest()

                reference_image_instance["referenceId"] = (
                    reference_image.reference_id
                )  # pylint: disable=protected-access
                reference_image_instance["referenceType"] = (
                    reference_image.reference_type
                )  # pylint: disable=protected-access
                shared_generation_parameters[
                    f"reference_type_{reference_image.reference_id}"
                ] = reference_image.reference_type  # pylint: disable=protected-access
                if isinstance(reference_image.config, MaskImageConfig):
                    reference_image_instance["maskImageConfig"] = {
                        "maskMode": reference_image.config.mask_mode,
                    }
                    if reference_image.config.dilation:
                        reference_image_instance["maskImageConfig"][
                            "dilation"
                        ] = reference_image.config.dilation
                    if reference_image.config.segmentation_classes:
                        reference_image_instance["maskImageConfig"][
                            "maskClasses"
                        ] = reference_image.config.segmentation_classes
                    shared_generation_parameters[
                        f"reference_image_mask_config_{reference_image.reference_id}"
                    ] = str(
                        reference_image.config
                    )  # pylint: disable=protected-access
                if isinstance(reference_image.config, ControlImageConfig):
                    reference_image_instance["controlImageConfig"] = {
                        "controlType": reference_image.config.control_type,
                        "enableControlImageComputation": (
                            reference_image.config.enable_control_image_computation
                        ),
                    }
                    shared_generation_parameters[
                        f"reference_image_control_config_{reference_image.reference_id}"
                    ] = str(
                        reference_image.config
                    )  # pylint: disable=protected-access
                if isinstance(reference_image.config, SubjectImageConfig):
                    reference_image_instance["subjectImageConfig"] = {
                        "subjectType": reference_image.config.subject_type,
                        "subjectDescription": reference_image.config.subject_description,
                    }
                    shared_generation_parameters[
                        f"reference_image_subject_config_{reference_image.reference_id}"
                    ] = str(
                        reference_image.config
                    )  # pylint: disable=protected-access
                if isinstance(reference_image.config, StyleImageConfig):
                    reference_image_instance["styleImageConfig"] = {
                        "styleDescription": reference_image.config.style_description,
                    }
                    shared_generation_parameters[
                        f"reference_image_style_config_{reference_image.reference_id}"
                    ] = str(
                        reference_image.config
                    )  # pylint: disable=protected-access
                instance["referenceImages"].append(reference_image_instance)

        edit_config = {}
        output_options = {}
        parameters = {}
        max_size = max(width or 0, height or 0) or None
        if aspect_ratio is not None:
            parameters["aspectRatio"] = aspect_ratio
        elif max_size:
            # Note: The size needs to be a string
            parameters["sampleImageSize"] = str(max_size)
            if height is not None and width is not None and height != width:
                parameters["aspectRatio"] = f"{width}:{height}"

        parameters["sampleCount"] = number_of_images
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
            shared_generation_parameters["negative_prompt"] = negative_prompt

        if seed is not None:
            # Note: String seed and numerical seed give different results
            parameters["seed"] = seed
            shared_generation_parameters["seed"] = seed

        if guidance_scale is not None:
            parameters["guidanceScale"] = guidance_scale
            shared_generation_parameters["guidance_scale"] = guidance_scale

        if language is not None:
            parameters["language"] = language
            shared_generation_parameters["language"] = language

        if output_gcs_uri is not None:
            parameters["storageUri"] = output_gcs_uri
            shared_generation_parameters["storage_uri"] = output_gcs_uri

        if edit_mode is not None:
            if reference_images is not None:
                edit_mode_to_enum_map = {
                    "inpainting-insert": "EDIT_MODE_INPAINT_INSERTION",
                    "inpainting-remove": "EDIT_MODE_INPAINT_REMOVAL",
                    "outpainting": "EDIT_MODE_OUTPAINT",
                    "background-swap": "EDIT_MODE_BGSWAP",
                }
                capability_mode = (
                    edit_mode_to_enum_map[edit_mode]
                    if edit_mode in edit_mode_to_enum_map
                    else "EDIT_MODE_DEFAULT"
                )
                parameters["editMode"] = capability_mode
                shared_generation_parameters["edit_mode"] = capability_mode
            else:
                edit_config["editMode"] = (
                    edit_mode if edit_mode != "background-swap" else "inpainting-insert"
                )
                shared_generation_parameters["edit_mode"] = edit_mode

        if mask is None and edit_mode is not None and edit_mode != "product-image":
            if mask_mode is not None:
                if "maskMode" not in edit_config:
                    edit_config["maskMode"] = {}
                edit_config["maskMode"]["maskType"] = mask_mode
                shared_generation_parameters["mask_mode"] = mask_mode

            if segmentation_classes is not None:
                if "maskMode" not in edit_config:
                    edit_config["maskMode"] = {}
                edit_config["maskMode"]["classes"] = segmentation_classes
                shared_generation_parameters["classes"] = segmentation_classes

        if mask_dilation is not None:
            edit_config["maskDilation"] = mask_dilation
            shared_generation_parameters["mask_dilation"] = mask_dilation

        if product_position is not None:
            edit_config["productPosition"] = product_position
            shared_generation_parameters["product_position"] = product_position

        if output_mime_type is not None:
            output_options["mimeType"] = output_mime_type
            shared_generation_parameters["mime_type"] = output_mime_type

        if compression_quality is not None:
            output_options["compressionQuality"] = compression_quality
            shared_generation_parameters["compression_quality"] = compression_quality

        if add_watermark is not None:
            parameters["addWatermark"] = add_watermark
            shared_generation_parameters["add_watermark"] = add_watermark

        if safety_filter_level is not None:
            parameters["safetySetting"] = safety_filter_level
            shared_generation_parameters["safety_filter_level"] = safety_filter_level

        if person_generation is not None:
            parameters["personGeneration"] = person_generation
            shared_generation_parameters["person_generation"] = person_generation

        if edit_config:
            parameters["editConfig"] = edit_config

        if output_options:
            parameters["outputOptions"] = output_options

        response = self._endpoint.predict(
            instances=[instance],
            parameters=parameters,
        )

        generated_images: List["GeneratedImage"] = []
        for idx, prediction in enumerate(response.predictions):
            generation_parameters = dict(shared_generation_parameters)
            generation_parameters["index_of_image_in_batch"] = idx
            encoded_bytes = prediction.get("bytesBase64Encoded")
            generated_image = GeneratedImage(
                image_bytes=base64.b64decode(encoded_bytes) if encoded_bytes else None,
                generation_parameters=generation_parameters,
                gcs_uri=prediction.get("gcsUri"),
            )
            generated_images.append(generated_image)

        return ImageGenerationResponse(images=generated_images)

    def generate_images(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        number_of_images: int = 1,
        aspect_ratio: Optional[Literal["1:1", "9:16", "16:9", "4:3", "3:4"]] = None,
        guidance_scale: Optional[float] = None,
        language: Optional[str] = None,
        seed: Optional[int] = None,
        output_gcs_uri: Optional[str] = None,
        add_watermark: Optional[bool] = True,
        safety_filter_level: Optional[
            Literal["block_most", "block_some", "block_few", "block_fewest"]
        ] = None,
        person_generation: Optional[
            Literal["dont_allow", "allow_adult", "allow_all"]
        ] = None,
    ) -> "ImageGenerationResponse":
        """Generates images from text prompt.

        Args:
            prompt: Text prompt for the image.
            negative_prompt: A description of what you want to omit in the generated
                images.
            number_of_images: Number of images to generate. Range: 1..8.
            aspect_ratio: Changes the aspect ratio of the generated image Supported
                values are:
                * "1:1" : 1:1 aspect ratio
                * "9:16" : 9:16 aspect ratio
                * "16:9" : 16:9 aspect ratio
                * "4:3" : 4:3 aspect ratio
                * "3:4" : 3:4 aspect_ratio
            guidance_scale: Controls the strength of the prompt. Suggested values are:
                * 0-9 (low strength)
                * 10-20 (medium strength)
                * 21+ (high strength)
            language: Language of the text prompt for the image. Default: None.
                Supported values are `"en"` for English, `"hi"` for Hindi, `"ja"`
                for Japanese, `"ko"` for Korean, and `"auto"` for automatic language
                detection.
            seed: Image generation random seed.
            output_gcs_uri: Google Cloud Storage uri to store the generated images.
            add_watermark: Add a watermark to the generated image
            safety_filter_level: Adds a filter level to Safety filtering. Supported
                values are:
                * "block_most" : Strongest filtering level, most strict
                blocking
                * "block_some" : Block some problematic prompts and responses
                * "block_few" : Block fewer problematic prompts and responses
                * "block_fewest" : Block very few problematic prompts and responses
            person_generation: Allow generation of people by the model Supported
                values are:
                * "dont_allow" : Block generation of people
                * "allow_adult" : Generate adults, but not children
                * "allow_all" : Generate adults and children
        Returns:
            An `ImageGenerationResponse` object.
        """
        return self._generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            guidance_scale=guidance_scale,
            language=language,
            seed=seed,
            output_gcs_uri=output_gcs_uri,
            add_watermark=add_watermark,
            safety_filter_level=safety_filter_level,
            person_generation=person_generation,
        )

    def edit_image(
        self,
        *,
        prompt: str,
        base_image: Optional["Image"] = None,
        mask: Optional["Image"] = None,
        reference_images: Optional[List["ReferenceImage"]] = None,
        negative_prompt: Optional[str] = None,
        number_of_images: int = 1,
        guidance_scale: Optional[float] = None,
        edit_mode: Optional[
            Literal[
                "inpainting-insert",
                "inpainting-remove",
                "outpainting",
                "product-image",
            ]
        ] = None,
        mask_mode: Optional[Literal["background", "foreground", "semantic"]] = None,
        segmentation_classes: Optional[List[str]] = None,
        mask_dilation: Optional[float] = None,
        product_position: Optional[Literal["fixed", "reposition"]] = None,
        output_mime_type: Optional[Literal["image/png", "image/jpeg"]] = None,
        compression_quality: Optional[float] = None,
        language: Optional[str] = None,
        seed: Optional[int] = None,
        output_gcs_uri: Optional[str] = None,
        safety_filter_level: Optional[
            Literal["block_most", "block_some", "block_few", "block_fewest"]
        ] = None,
        person_generation: Optional[
            Literal["dont_allow", "allow_adult", "allow_all"]
        ] = None,
    ) -> "ImageGenerationResponse":
        """Edits an existing image based on text prompt.

        Args:
            prompt: Text prompt for the image.
            base_image: Base image from which to generate the new image.
            mask: Mask for the base image.
            reference_images: List of reference images to use for Imagen 3
              Capabilities. Please refer to the documentation for the ReferenceImage
              class for how to create a ReferenceImage object.
            negative_prompt: A description of what you want to omit in the generated
              images.
            number_of_images: Number of images to generate. Range: 1..8.
            guidance_scale: Controls the strength of the prompt.
                Suggested values are:
                * 0-9 (low strength)
                * 10-20 (medium strength)
                * 21+ (high strength)
            edit_mode: Describes the editing mode for the request. Supported values are:
                * inpainting-insert: fills the mask area based on the text prompt
                (requires mask and text)
                * inpainting-remove: removes the object(s) in the mask area.
                (requires mask)
                * outpainting: extend the image based on the mask area.
                (Requires mask)
                * product-image: Changes the background for the predominant product
                or subject in the image
            mask_mode: Solicits generation of the mask (v/s providing mask as an
                input). Supported values are:
                * background: Automatically generates a mask for all regions except
                the primary subject(s) of the image
                * foreground: Automatically generates a mask for the primary
                subjects(s) of the image.
                * semantic: Segment one or more of the segmentation classes using
                class ID
            segmentation_classes: List of class IDs for segmentation. Max of 5 IDs
            mask_dilation: Defines the dilation percentage of the mask provided.
                Float between 0 and 1. Defaults to 0.03
            product_position: Defines whether the product should stay fixed or be
                repositioned. Supported Values:
                * fixed: Fixed position
                * reposition: Can be moved (default)
            output_mime_type: Which image format should the output be saved as.
                Supported values:
                * image/png: Save as a PNG image
                * image/jpeg: Save as a JPEG image
            compression_quality: Level of compression if the output mime type is
              selected to be image/jpeg. Float between 0 to 100
            language: Language of the text prompt for the image. Default: None.
                Supported values are `"en"` for English, `"hi"` for Hindi,
                `"ja"` for Japanese, `"ko"` for Korean, and `"auto"` for
                automatic language detection.
            seed: Image generation random seed.
            output_gcs_uri: Google Cloud Storage uri to store the edited images.
            safety_filter_level: Adds a filter level to Safety filtering. Supported
                values are:
                * "block_most" : Strongest filtering level, most strict
                blocking
                * "block_some" : Block some problematic prompts and responses
                * "block_few" : Block fewer problematic prompts and responses
                * "block_fewest" : Block very few problematic prompts and responses
            person_generation: Allow generation of people by the model Supported
                values are:
                * "dont_allow" : Block generation of people
                * "allow_adult" : Generate adults, but not children
                * "allow_all" : Generate adults and children

        Returns:
            An `ImageGenerationResponse` object.
        """
        return self._generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            number_of_images=number_of_images,
            guidance_scale=guidance_scale,
            seed=seed,
            base_image=base_image,
            mask=mask,
            reference_images=reference_images,
            edit_mode=edit_mode,
            mask_mode=mask_mode,
            segmentation_classes=segmentation_classes,
            mask_dilation=mask_dilation,
            product_position=product_position,
            output_mime_type=output_mime_type,
            compression_quality=compression_quality,
            language=language,
            output_gcs_uri=output_gcs_uri,
            add_watermark=False,  # Not supported for editing yet
            safety_filter_level=safety_filter_level,
            person_generation=person_generation,
        )

    def upscale_image(
        self,
        image: Union["Image", "GeneratedImage"],
        new_size: Optional[int] = 2048,
        upscale_factor: Optional[Literal["x2", "x4"]] = None,
        output_mime_type: Optional[Literal["image/png", "image/jpeg"]] = "image/png",
        output_compression_quality: Optional[int] = None,
        output_gcs_uri: Optional[str] = None,
    ) -> "Image":
        """Upscales an image.

        This supports upscaling images generated through the `generate_images()`
        method, or upscaling a new image.

        Examples::

            # Upscale a generated image
            model = ImageGenerationModel.from_pretrained("imagegeneration@002")
            response = model.generate_images(
                prompt="Astronaut riding a horse",
            )
            model.upscale_image(image=response[0])

            # Upscale a new 1024x1024 image
            my_image = Image.load_from_file("my-image.png")
            model.upscale_image(image=my_image)

            # Upscale a new arbitrary sized image using a x2 or x4 upscaling factor
            my_image = Image.load_from_file("my-image.png")
            model.upscale_image(image=my_image, upscale_factor="x2")

            # Upscale an image and get the result in JPEG format
            my_image = Image.load_from_file("my-image.png")
            model.upscale_image(image=my_image, output_mime_type="image/jpeg",
            output_compression_quality=90)

        Args:
            image (Union[GeneratedImage, Image]): Required. The generated image
                to upscale.
            new_size (int): The size of the biggest dimension of the upscaled
                image.
                Only 2048 and 4096 are currently supported. Results in a
                2048x2048 or 4096x4096 image. Defaults to 2048 if not provided.
            upscale_factor: The upscaling factor. Supported values are "x2" and
                "x4". Defaults to None.
            output_mime_type: The mime type of the output image. Supported values
                are "image/png" and "image/jpeg". Defaults to "image/png".
            output_compression_quality: The compression quality of the output
                image
                as an int (0-100). Only applicable if the output mime type is
                "image/jpeg". Defaults to None.
            output_gcs_uri: Google Cloud Storage uri to store the upscaled
                images.

        Returns:
            An `Image` object.
        """
        target_image_size = new_size if new_size else None
        longest_dim = max(image._size[0], image._size[1])

        if not new_size and not upscale_factor:
            raise ValueError("Either new_size or upscale_factor must be provided.")

        if not upscale_factor:
            x2_factor = 2.0
            x4_factor = 4.0
            epsilon = 0.1
            is_upscaling_x2_request = abs(new_size / longest_dim - x2_factor) < epsilon
            is_upscaling_x4_request = abs(new_size / longest_dim - x4_factor) < epsilon

            if not is_upscaling_x2_request and not is_upscaling_x4_request:
                raise ValueError(
                    "Only x2 and x4 upscaling are currently supported. Requested"
                    f" upscaling factor: {new_size / longest_dim}"
                )
        else:
            if upscale_factor == "x2":
                target_image_size = longest_dim * 2
            else:
                target_image_size = longest_dim * 4
        if new_size not in _SUPPORTED_UPSCALING_SIZES:
            raise ValueError(
                "Only the folowing square upscaling sizes are currently supported:"
                f" {_SUPPORTED_UPSCALING_SIZES}."
            )

        instance = {"prompt": ""}

        if image._gcs_uri:  # pylint: disable=protected-access
            instance["image"] = {
                "gcsUri": image._gcs_uri  # pylint: disable=protected-access
            }
        else:
            instance["image"] = {
                "bytesBase64Encoded": image._as_base64_string()  # pylint: disable=protected-access
            }

        parameters = {
            "sampleCount": 1,
            "mode": "upscale",
        }

        if upscale_factor:
            parameters["upscaleConfig"] = {"upscaleFactor": upscale_factor}

        else:
            parameters["sampleImageSize"] = str(new_size)

        if output_gcs_uri is not None:
            parameters["storageUri"] = output_gcs_uri

        parameters["outputOptions"] = {"mimeType": output_mime_type}
        if output_mime_type == "image/jpeg" and output_compression_quality is not None:
            parameters["outputOptions"][
                "compressionQuality"
            ] = output_compression_quality

        response = self._endpoint.predict(
            instances=[instance],
            parameters=parameters,
        )

        upscaled_image = response.predictions[0]

        if isinstance(image, GeneratedImage):
            generation_parameters = image.generation_parameters

        else:
            generation_parameters = {}

        generation_parameters["upscaled_image_size"] = target_image_size

        encoded_bytes = upscaled_image.get("bytesBase64Encoded")
        return GeneratedImage(
            image_bytes=base64.b64decode(encoded_bytes) if encoded_bytes else None,
            generation_parameters=generation_parameters,
            gcs_uri=upscaled_image.get("gcsUri"),
        )


@dataclasses.dataclass
class ImageGenerationResponse:
    """Image generation response.

    Attributes:
        images: The list of generated images.
    """

    __module__ = "vertexai.preview.vision_models"

    images: List["GeneratedImage"]

    def __iter__(self) -> typing.Iterator["GeneratedImage"]:
        """Iterates through the generated images."""
        yield from self.images

    def __getitem__(self, idx: int) -> "GeneratedImage":
        """Gets the generated image by index."""
        return self.images[idx]


_EXIF_USER_COMMENT_TAG_IDX = 0x9286
_IMAGE_GENERATION_PARAMETERS_EXIF_KEY = (
    "google.cloud.vertexai.image_generation.image_generation_parameters"
)


class GeneratedImage(Image):
    """Generated image."""

    __module__ = "vertexai.preview.vision_models"

    def __init__(
        self,
        image_bytes: Optional[bytes],
        generation_parameters: Dict[str, Any],
        gcs_uri: Optional[str] = None,
    ):
        """Creates a `GeneratedImage` object.

        Args:
            image_bytes: Image file bytes. Image can be in PNG or JPEG format.
            generation_parameters: Image generation parameter values.
            gcs_uri: Image file Google Cloud Storage uri.
        """
        warning_logs.show_deprecation_warning()
        super().__init__(image_bytes=image_bytes, gcs_uri=gcs_uri)
        self._generation_parameters = generation_parameters

    @property
    def generation_parameters(self):
        """Image generation parameters as a dictionary."""
        return self._generation_parameters

    @staticmethod
    def load_from_file(location: str) -> "GeneratedImage":
        """Loads image from file.

        Args:
            location: Local path from where to load the image.

        Returns:
            Loaded image as a `GeneratedImage` object.
        """
        base_image = Image.load_from_file(location=location)
        exif = base_image._pil_image.getexif()  # pylint: disable=protected-access
        exif_comment_dict = json.loads(exif[_EXIF_USER_COMMENT_TAG_IDX])
        generation_parameters = exif_comment_dict[_IMAGE_GENERATION_PARAMETERS_EXIF_KEY]
        return GeneratedImage(
            image_bytes=base_image._image_bytes,  # pylint: disable=protected-access
            generation_parameters=generation_parameters,
            gcs_uri=base_image._gcs_uri,  # pylint: disable=protected-access
        )

    def save(self, location: str, include_generation_parameters: bool = True):
        """Saves image to a file.

        Args:
            location: Local path where to save the image.
            include_generation_parameters: Whether to include the image
                generation parameters in the image's EXIF metadata.
        """
        if include_generation_parameters:
            if not self._generation_parameters:
                raise ValueError("Image does not have generation parameters.")
            if not PIL_Image:
                raise ValueError(
                    "The PIL module is required for saving generation parameters."
                )

            exif = self._pil_image.getexif()
            exif[_EXIF_USER_COMMENT_TAG_IDX] = json.dumps(
                {_IMAGE_GENERATION_PARAMETERS_EXIF_KEY: self._generation_parameters}
            )
            self._pil_image.save(location, exif=exif)
        else:
            super().save(location=location)


class ImageCaptioningModel(
    _model_garden_models._ModelGardenModel  # pylint: disable=protected-access
):
    """Generates captions from image.

    Examples::

        model = ImageCaptioningModel.from_pretrained("imagetext@001")
        image = Image.load_from_file("image.png")
        captions = model.get_captions(
            image=image,
            # Optional:
            number_of_results=1,
            language="en",
        )
    """

    __module__ = "vertexai.vision_models"

    _INSTANCE_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/predict/instance/vision_reasoning_model_1.0.0.yaml"

    def get_captions(
        self,
        image: Image,
        *,
        number_of_results: int = 1,
        language: str = "en",
        output_gcs_uri: Optional[str] = None,
    ) -> List[str]:
        """Generates captions for a given image.

        Args:
            image: The image to get captions for. Size limit: 10 MB.
            number_of_results: Number of captions to produce. Range: 1-3.
            language: Language to use for captions.
                Supported languages: "en", "fr", "de", "it", "es"
            output_gcs_uri: Google Cloud Storage uri to store the captioned images.

        Returns:
            A list of image caption strings.
        """
        instance = {}

        if image._gcs_uri:  # pylint: disable=protected-access
            instance["image"] = {
                "gcsUri": image._gcs_uri  # pylint: disable=protected-access
            }
        else:
            instance["image"] = {
                "bytesBase64Encoded": image._as_base64_string()  # pylint: disable=protected-access
            }
        parameters = {
            "sampleCount": number_of_results,
            "language": language,
        }
        if output_gcs_uri is not None:
            parameters["storageUri"] = output_gcs_uri

        response = self._endpoint.predict(
            instances=[instance],
            parameters=parameters,
        )
        return response.predictions


class ImageQnAModel(
    _model_garden_models._ModelGardenModel  # pylint: disable=protected-access
):
    """Answers questions about an image.

    Examples::

        model = ImageQnAModel.from_pretrained("imagetext@001")
        image = Image.load_from_file("image.png")
        answers = model.ask_question(
            image=image,
            question="What color is the car in this image?",
            # Optional:
            number_of_results=1,
        )
    """

    __module__ = "vertexai.vision_models"

    _INSTANCE_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/predict/instance/vision_reasoning_model_1.0.0.yaml"

    def ask_question(
        self,
        image: Image,
        question: str,
        *,
        number_of_results: int = 1,
    ) -> List[str]:
        """Answers questions about an image.

        Args:
            image: The image to get captions for. Size limit: 10 MB.
            question: Question to ask about the image.
            number_of_results: Number of captions to produce. Range: 1-3.

        Returns:
            A list of answers.
        """
        instance = {"prompt": question}

        if image._gcs_uri:  # pylint: disable=protected-access
            instance["image"] = {
                "gcsUri": image._gcs_uri  # pylint: disable=protected-access
            }
        else:
            instance["image"] = {
                "bytesBase64Encoded": image._as_base64_string()  # pylint: disable=protected-access
            }
        parameters = {
            "sampleCount": number_of_results,
        }

        response = self._endpoint.predict(
            instances=[instance],
            parameters=parameters,
        )
        return response.predictions


class MultiModalEmbeddingModel(_model_garden_models._ModelGardenModel):
    """Generates embedding vectors from images and videos.

    Examples::

        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        image = Image.load_from_file("image.png")
        video = Video.load_from_file("video.mp4")

        embeddings = model.get_embeddings(
            image=image,
            video=video,
            contextual_text="Hello world",
        )
        image_embedding = embeddings.image_embedding
        video_embeddings = embeddings.video_embeddings
        text_embedding = embeddings.text_embedding
    """

    __module__ = "vertexai.vision_models"

    _INSTANCE_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/predict/instance/vision_embedding_model_1.0.0.yaml"

    def get_embeddings(
        self,
        image: Optional[Image] = None,
        video: Optional[Video] = None,
        contextual_text: Optional[str] = None,
        dimension: Optional[int] = None,
        video_segment_config: Optional[VideoSegmentConfig] = None,
    ) -> "MultiModalEmbeddingResponse":
        """Gets embedding vectors from the provided image.

        Args:
            image (Image): Optional. The image to generate embeddings for. One of
              `image`, `video`, or `contextual_text` is required.
            video (Video): Optional. The video to generate embeddings for. One of
              `image`, `video` or `contextual_text` is required.
            contextual_text (str): Optional. Contextual text for your input image or video.
              If provided, the model will also generate an embedding vector for the
              provided contextual text. The returned image and text embedding
              vectors are in the same semantic space with the same dimensionality,
              and the vectors can be used interchangeably for use cases like
              searching image by text or searching text by image. One of `image`, `video` or
              `contextual_text` is required.
            dimension (int): Optional. The number of embedding dimensions. Lower
              values offer decreased latency when using these embeddings for
              subsequent tasks, while higher values offer better accuracy.
              Available values: `128`, `256`, `512`, and `1408` (default).
            video_segment_config (VideoSegmentConfig): Optional. The specific
              video segments (in seconds) the embeddings are generated for.

        Returns:
            MultiModalEmbeddingResponse:
                The image and text embedding vectors.
        """

        if not image and not video and not contextual_text:
            raise ValueError(
                "One of `image`, `video`, or `contextual_text` is required."
            )

        instance = {}

        if image:
            if image._gcs_uri:  # pylint: disable=protected-access
                instance["image"] = {
                    "gcsUri": image._gcs_uri  # pylint: disable=protected-access
                }
            else:
                instance["image"] = {
                    "bytesBase64Encoded": image._as_base64_string()  # pylint: disable=protected-access
                }

        if video:
            if video._gcs_uri:  # pylint: disable=protected-access
                instance["video"] = {
                    "gcsUri": video._gcs_uri  # pylint: disable=protected-access
                }
            else:
                instance["video"] = {
                    "bytesBase64Encoded": video._as_base64_string()  # pylint: disable=protected-access
                }  # pylint: disable=protected-access

            if video_segment_config:
                instance["video"]["videoSegmentConfig"] = {
                    "startOffsetSec": video_segment_config.start_offset_sec,
                    "endOffsetSec": video_segment_config.end_offset_sec,
                    "intervalSec": video_segment_config.interval_sec,
                }

        if contextual_text:
            instance["text"] = contextual_text

        parameters = {}
        if dimension:
            parameters["dimension"] = dimension

        response = self._endpoint.predict(
            instances=[instance],
            parameters=parameters,
        )
        image_embedding = response.predictions[0].get("imageEmbedding")
        video_embeddings = []
        for video_embedding in response.predictions[0].get("videoEmbeddings", []):
            video_embeddings.append(
                VideoEmbedding(
                    embedding=video_embedding["embedding"],
                    start_offset_sec=video_embedding["startOffsetSec"],
                    end_offset_sec=video_embedding["endOffsetSec"],
                )
            )
        text_embedding = (
            response.predictions[0].get("textEmbedding")
            if "textEmbedding" in response.predictions[0]
            else None
        )
        return MultiModalEmbeddingResponse(
            image_embedding=image_embedding,
            video_embeddings=video_embeddings,
            _prediction_response=response,
            text_embedding=text_embedding,
        )


@dataclasses.dataclass
class MultiModalEmbeddingResponse:
    """The multimodal embedding response.

    Attributes:
        image_embedding (List[float]):
            Optional. The embedding vector generated from your image.
        video_embeddings (List[VideoEmbedding]):
            Optional. The embedding vectors generated from your video.
        text_embedding (List[float]):
            Optional. The embedding vector generated from the contextual text provided for your image or video.
    """

    __module__ = "vertexai.vision_models"

    _prediction_response: Any
    image_embedding: Optional[List[float]] = None
    video_embeddings: Optional[List[VideoEmbedding]] = None
    text_embedding: Optional[List[float]] = None


class ImageTextModel(ImageCaptioningModel, ImageQnAModel):
    """Generates text from images.

    Examples::

        model = ImageTextModel.from_pretrained("imagetext@001")
        image = Image.load_from_file("image.png")

        captions = model.get_captions(
            image=image,
            # Optional:
            number_of_results=1,
            language="en",
        )

        answers = model.ask_question(
            image=image,
            question="What color is the car in this image?",
            # Optional:
            number_of_results=1,
        )
    """

    __module__ = "vertexai.vision_models"

    # NOTE: Using this ImageTextModel class is recommended over using ImageQnAModel or ImageCaptioningModel,
    # since SDK Model Garden classes should follow the design pattern of exactly 1 SDK class to 1 Model Garden schema URI

    _INSTANCE_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/predict/instance/vision_reasoning_model_1.0.0.yaml"


@dataclasses.dataclass
class WatermarkVerificationResponse:

    __module__ = "vertexai.preview.vision_models"

    _prediction_response: Any
    watermark_verification_result: Optional[str] = None


class WatermarkVerificationModel(_model_garden_models._ModelGardenModel):
    """Verifies if an image has a watermark."""

    __module__ = "vertexai.preview.vision_models"

    _INSTANCE_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/predict/instance/watermark_verification_model_1.0.0.yaml"

    def verify_image(self, image: Image) -> WatermarkVerificationResponse:
        """Verifies the watermark of an image.

        Args:
            image: The image to verify.

        Returns:
            A WatermarkVerificationResponse, containing the confidence level of
            the image being watermarked.
        """
        if not image:
            raise ValueError("Image is required.")

        instance = {}

        if image._gcs_uri:
            instance["image"] = {"gcsUri": image._gcs_uri}
        else:
            instance["image"] = {"bytesBase64Encoded": image._as_base64_string()}

        parameters = {}
        response = self._endpoint.predict(
            instances=[instance],
            parameters=parameters,
        )

        verification_likelihood = response.predictions[0].get("decision")
        return WatermarkVerificationResponse(
            _prediction_response=response,
            watermark_verification_result=verification_likelihood,
        )


class Scribble:
    """Input scribble for image segmentation."""

    __module__ = "vertexai.preview.vision_models"

    _image_: Optional[Image] = None

    def __init__(
        self,
        image_bytes: Optional[bytes],
        gcs_uri: Optional[str] = None,
    ):
        """Creates a `Scribble` object.

        Args:
            image_bytes: Mask image file bytes.
            gcs_uri: Mask image file Google Cloud Storage uri.
        """
        if bool(image_bytes) == bool(gcs_uri):
            raise ValueError("Either image_bytes or gcs_uri must be provided.")

        self._image_ = Image(image_bytes, gcs_uri)

    @property
    def image(self) -> Optional[Image]:
        """The scribble image."""
        return self._image_


@dataclasses.dataclass
class EntityLabel:
    """Entity label holding a text label and any associated confidence score."""

    __module__ = "vertexai.preview.vision_models"

    label: Optional[str] = None
    score: Optional[float] = None


class GeneratedMask(Image):
    """Generated image mask."""

    __module__ = "vertexai.preview.vision_models"

    __labels__: Optional[List[EntityLabel]] = None

    def __init__(
        self,
        image_bytes: Optional[bytes],
        gcs_uri: Optional[str] = None,
        labels: Optional[List[EntityLabel]] = None,
    ):
        """Creates a `GeneratedMask` object.

        Args:
            image_bytes: Mask image file bytes.
            gcs_uri: Mask image file Google Cloud Storage uri.
            labels: Generated entity labels. Each text label might be associated
                with a confidence score.
        """

        super().__init__(
            image_bytes=image_bytes,
            gcs_uri=gcs_uri,
        )
        self.__labels__ = labels

    @property
    def labels(self) -> Optional[List[EntityLabel]]:
        """The entity labels of the masked object."""
        return self.__labels__


@dataclasses.dataclass
class ImageSegmentationResponse:
    """Image Segmentation response.

    Attributes:
        masks: The list of generated masks.
    """

    __module__ = "vertexai.preview.vision_models"

    _prediction_response: Any
    masks: List[GeneratedMask]

    def __iter__(self) -> typing.Iterator[GeneratedMask]:
        """Iterates through the generated masks."""
        yield from self.masks

    def __getitem__(self, idx: int) -> GeneratedMask:
        """Gets the generated masks by index."""
        return self.masks[idx]


class ImageSegmentationModel(_model_garden_models._ModelGardenModel):
    """Segments an image."""

    __module__ = "vertexai.preview.vision_models"

    _INSTANCE_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/predict/instance/image_segmentation_model_1.0.0.yaml"

    def segment_image(
        self,
        base_image: Image,
        prompt: Optional[str] = None,
        scribble: Optional[Scribble] = None,
        mode: Literal[
            "foreground", "background", "semantic", "prompt", "interactive"
        ] = "foreground",
        max_predictions: Optional[int] = None,
        confidence_threshold: Optional[float] = 0.1,
        mask_dilation: Optional[float] = None,
        binary_color_threshold: Optional[float] = None,
    ) -> ImageSegmentationResponse:
        """Segments an image.

        Args:
            base_image: The base image to segment.
            prompt: The prompt to guide the segmentation. Valid for the prompt and
                semantic modes.
            scribble: The scribble in the form of an image mask to guide the
                segmentation. Valid for the interactive mode. The scribble image
                should be a black-and-white PNG file equal in size to the base
                image. White pixels represent the scribbled brush stroke which
                select objects in the base image to segment.
            mode: The segmentation mode. Supported values are:
                * foreground: segment the foreground object of an image
                * background: segment the background of an image
                * semantic: specify the objects to segment with a comma delimited
                    list of objects from the class set in the prompt.
                * prompt: use an open-vocabulary text prompt to select objects to
                    segment.
                * interactive: draw scribbles with a brush stroke to guide the
                    segmentation. The default is foreground.
            max_predictions: The maximum number of predictions to make. Valid for
                the prompt mode. Default is unlimited.
            confidence_threshold: A threshold to filter predictions by confidence
                score. The value must be in the range of 0.0 and 1.0. The default is
                0.1.
            mask_dilation: A value to dilate the masks by. The value must be in the
                range of 0.0 (no dilation) and 1.0 (the whole image will be masked).
                The default is 0.0.
            binary_color_threshold: The threshold to convert the grayscale soft
                mask to a binary color black and white mask. The value must be
                in the range of 0 and 255, or -1 to disable the thresholding.
                The default is 96.

        Returns:
            An `ImageSegmentationResponse` object with the generated masks,
            entities, and labels (if any).
        """
        if not base_image:
            raise ValueError("Base image is required.")
        instance = {}

        if base_image._gcs_uri:
            instance["image"] = {"gcsUri": base_image._gcs_uri}
        else:
            instance["image"] = {"bytesBase64Encoded": base_image._as_base64_string()}

        if prompt:
            instance["prompt"] = prompt

        parameters = {}
        if scribble and scribble.image:
            scribble_image = scribble.image
            if scribble_image._gcs_uri:
                instance["scribble"] = {"image": {"gcsUri": scribble_image._gcs_uri}}
            else:
                instance["scribble"] = {
                    "image": {"bytesBase64Encoded": scribble_image._as_base64_string()}
                }
        parameters["mode"] = mode
        if max_predictions:
            parameters["maxPredictions"] = max_predictions
        if confidence_threshold:
            parameters["confidenceThreshold"] = confidence_threshold
        if mask_dilation:
            parameters["maskDilation"] = mask_dilation
        if binary_color_threshold:
            parameters["binaryColorThreshold"] = binary_color_threshold

        response = self._endpoint.predict(
            instances=[instance],
            parameters=parameters,
        )

        masks: List[GeneratedMask] = []
        for prediction in response.predictions:
            encoded_bytes = prediction.get("bytesBase64Encoded")
            labels = []
            if "labels" in prediction:
                for label in prediction["labels"]:
                    labels.append(
                        EntityLabel(
                            label=label.get("label"),
                            score=label.get("score"),
                        )
                    )
            generated_image = GeneratedMask(
                image_bytes=base64.b64decode(encoded_bytes) if encoded_bytes else None,
                gcs_uri=prediction.get("gcsUri"),
                labels=labels,
            )
            masks.append(generated_image)

        return ImageSegmentationResponse(
            _prediction_response=response,
            masks=masks,
        )
