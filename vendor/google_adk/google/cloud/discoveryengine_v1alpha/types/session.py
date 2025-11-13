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
import proto  # type: ignore

from google.cloud.discoveryengine_v1alpha.types import answer as gcd_answer
from google.cloud.discoveryengine_v1alpha.types import common

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "FileSource",
        "Session",
        "Query",
        "ImageCharacteristics",
        "VideoCharacteristics",
        "FileCharacteristics",
        "FileView",
        "FileMetadata",
    },
)


class FileSource(proto.Enum):
    r"""Original source of the file.

    Values:
        FILE_SOURCE_UNSPECIFIED (0):
            Default value. Unknown source.
        FILE_SOURCE_INLINE (1):
            The data of the file was provided inline
            (e.g. pasted from the clipboard).
        FILE_SOURCE_LOCAL (2):
            The file was uploaded from a local file.
        FILE_SOURCE_CLOUD_STORAGE (3):
            The file was uploaded from Cloud Storage.
        FILE_SOURCE_CLOUD_DRIVE (4):
            The file was uploaded from Drive.
        FILE_SOURCE_URL (5):
            The file was retrieved from a URL (e.g.
            public web).
    """
    FILE_SOURCE_UNSPECIFIED = 0
    FILE_SOURCE_INLINE = 1
    FILE_SOURCE_LOCAL = 2
    FILE_SOURCE_CLOUD_STORAGE = 3
    FILE_SOURCE_CLOUD_DRIVE = 4
    FILE_SOURCE_URL = 5


class Session(proto.Message):
    r"""External session proto definition.

    Attributes:
        name (str):
            Immutable. Fully qualified name
            ``projects/{project}/locations/global/collections/{collection}/engines/{engine}/sessions/*``
        display_name (str):
            Optional. The display name of the session.

            This field is used to identify the session in
            the UI. By default, the display name is the
            first turn query text in the session.
        state (google.cloud.discoveryengine_v1alpha.types.Session.State):
            The state of the session.
        user_pseudo_id (str):
            A unique identifier for tracking users.
        turns (MutableSequence[google.cloud.discoveryengine_v1alpha.types.Session.Turn]):
            Turns.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the session started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the session finished.
        is_pinned (bool):
            Optional. Whether the session is pinned,
            pinned session will be displayed on the top of
            the session list.
    """

    class State(proto.Enum):
        r"""Enumeration of the state of the session.

        Values:
            STATE_UNSPECIFIED (0):
                State is unspecified.
            IN_PROGRESS (1):
                The session is currently open.
        """
        STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1

    class Turn(proto.Message):
        r"""Represents a turn, including a query from the user and a
        answer from service.

        Attributes:
            query (google.cloud.discoveryengine_v1alpha.types.Query):
                Optional. The user query. May not be set if
                this turn is merely regenerating an answer to a
                different turn
            answer (str):
                Optional. The resource name of the answer to
                the user query.
                Only set if the answer generation (/answer API
                call) happened in this turn.
            detailed_answer (google.cloud.discoveryengine_v1alpha.types.Answer):
                Output only. In
                [ConversationalSearchService.GetSession][google.cloud.discoveryengine.v1alpha.ConversationalSearchService.GetSession]
                API, if
                [GetSessionRequest.include_answer_details][google.cloud.discoveryengine.v1alpha.GetSessionRequest.include_answer_details]
                is set to true, this field will be populated when getting
                answer query session.
            query_config (MutableMapping[str, str]):
                Optional. Represents metadata related to the
                query config, for example LLM model and version
                used, model parameters (temperature, grounding
                parameters, etc.). The prefix "google." is
                reserved for Google-developed functionality.
        """

        query: "Query" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Query",
        )
        answer: str = proto.Field(
            proto.STRING,
            number=2,
        )
        detailed_answer: gcd_answer.Answer = proto.Field(
            proto.MESSAGE,
            number=7,
            message=gcd_answer.Answer,
        )
        query_config: MutableMapping[str, str] = proto.MapField(
            proto.STRING,
            proto.STRING,
            number=16,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=7,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=2,
        enum=State,
    )
    user_pseudo_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    turns: MutableSequence[Turn] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Turn,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    is_pinned: bool = proto.Field(
        proto.BOOL,
        number=8,
    )


class Query(proto.Message):
    r"""Defines a user inputed query.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        text (str):
            Plain text.

            This field is a member of `oneof`_ ``content``.
        query_id (str):
            Output only. Unique Id for the query.
    """

    text: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="content",
    )
    query_id: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ImageCharacteristics(proto.Message):
    r"""Standard characteristics of an image media view.

    Attributes:
        width (int):
            Output only. Image width in pixels.
        height (int):
            Output only. Image height in pixels.
        color_space (google.cloud.discoveryengine_v1alpha.types.ImageCharacteristics.ColorSpace):
            Output only. Color space of the image (e.g.,
            "RGB", "CMYK", "Grayscale").
        bit_depth (int):
            Output only. Bit depth of the image (e.g.,
            8-bit, 16-bit).
    """

    class ColorSpace(proto.Enum):
        r"""Possible color spaces of an image (e.g., "RGB", "CMYK",
        "Grayscale").

        Values:
            COLOR_SPACE_UNSPECIFIED (0):
                Default value. Unknown color space.
            RGB (1):
                Red, green, blue colorspace.
            CMYK (2):
                Cyan, magenta, yellow, and black colorspace.
            GRAYSCALE (3):
                Grayscale colorspace.
            YUV (4):
                YUV colorspace.
            OTHER_COLOR_SPACE (5):
                Other colorspace.
        """
        COLOR_SPACE_UNSPECIFIED = 0
        RGB = 1
        CMYK = 2
        GRAYSCALE = 3
        YUV = 4
        OTHER_COLOR_SPACE = 5

    width: int = proto.Field(
        proto.INT32,
        number=1,
    )
    height: int = proto.Field(
        proto.INT32,
        number=2,
    )
    color_space: ColorSpace = proto.Field(
        proto.ENUM,
        number=3,
        enum=ColorSpace,
    )
    bit_depth: int = proto.Field(
        proto.INT32,
        number=4,
    )


class VideoCharacteristics(proto.Message):
    r"""Standard characteristics of a video media view.

    Attributes:
        width (int):
            Output only. Video width in pixels.
        height (int):
            Output only. Video height in pixels.
        duration (google.protobuf.duration_pb2.Duration):
            Output only. Video duration.
        frame_rate (float):
            Output only. Frame rate (frames per second).
        audio_codecs (MutableSequence[str]):
            Output only. Audio codecs used in the video.
        video_codecs (MutableSequence[str]):
            Output only. Video codecs used in the video.
        video_bitrate_kbps (int):
            Output only. Bitrate of the video in kbps.
        audio_bitrate_kbps (int):
            Output only. Bitrate of the audio in kbps.
    """

    width: int = proto.Field(
        proto.INT32,
        number=1,
    )
    height: int = proto.Field(
        proto.INT32,
        number=2,
    )
    duration: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=3,
        message=duration_pb2.Duration,
    )
    frame_rate: float = proto.Field(
        proto.DOUBLE,
        number=4,
    )
    audio_codecs: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=5,
    )
    video_codecs: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=6,
    )
    video_bitrate_kbps: int = proto.Field(
        proto.INT32,
        number=7,
    )
    audio_bitrate_kbps: int = proto.Field(
        proto.INT32,
        number=8,
    )


class FileCharacteristics(proto.Message):
    r"""Caracteristics of other file types.

    Attributes:
        characteristics (MutableMapping[str, str]):
            Output only. Generic map of characteristics.
    """

    characteristics: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=6,
    )


class FileView(proto.Message):
    r"""Represents a specific alternate version or "view" of a file
    object, such as a summary, a thumbnail, a translated version,
    etc.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        image_characteristics (google.cloud.discoveryengine_v1alpha.types.ImageCharacteristics):
            Output only. Characteristics of an image
            media view.

            This field is a member of `oneof`_ ``characteristics``.
        video_characteristics (google.cloud.discoveryengine_v1alpha.types.VideoCharacteristics):
            Output only. Characteristics of a video media
            view.

            This field is a member of `oneof`_ ``characteristics``.
        file_characteristics (google.cloud.discoveryengine_v1alpha.types.FileCharacteristics):
            Output only. Characteristics of other file
            types.

            This field is a member of `oneof`_ ``characteristics``.
        view_id (str):
            Output only. Globally Unique id for this
            specific view.
        uri (str):
            Output only. The URI to access this media
            view.
        mime_type (str):
            Output only. MIME type (e.g., "image/jpeg",
            "image/png", "text/plain", "video/mp4")
        byte_size (int):
            Output only. The size of the view in bytes.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the view was created.
    """

    image_characteristics: "ImageCharacteristics" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="characteristics",
        message="ImageCharacteristics",
    )
    video_characteristics: "VideoCharacteristics" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="characteristics",
        message="VideoCharacteristics",
    )
    file_characteristics: "FileCharacteristics" = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="characteristics",
        message="FileCharacteristics",
    )
    view_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    mime_type: str = proto.Field(
        proto.STRING,
        number=3,
    )
    byte_size: int = proto.Field(
        proto.INT64,
        number=4,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )


class FileMetadata(proto.Message):
    r"""Represents a file attached to a session (context file)

    Attributes:
        file_id (str):
            Output only. The ID of the file.
        name (str):
            Output only. The name of the file uploaded.
        mime_type (str):
            The content type of the file, see
            https://www.iana.org/assignments/media-types/media-types.xhtml.
        byte_size (int):
            Output only. The size of the context file in
            bytes.
        original_uri (str):
            Optional. The original location of the file.
            It may be a local file path, or any other URI
            that allows accessing the file in an external
            system. There are two scenarios in which this
            url may be empty:

            1. If the file was sent as inline data (e.g.
                pasted from the clipboard).
            2. If the original location is not available.

            Note that there's no guarantee that the URI will
            be pointing to a valid or actually existing
            file. For example, a file might have been
            uploaded to the session, and then deleted from
            the original source.
        original_source_type (google.cloud.discoveryengine_v1alpha.types.FileSource):
            Optional. The type of the original source of
            the file.
        upload_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the file was uploaded
            (If this is a file generated by an internal
            process and then made available to the session,
            this indicates the moment it happened).
        last_add_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the file was added to
            the session. Note that if a file was added, then
            modified externally, then added again, the add
            time will be updated.
        metadata (MutableMapping[str, str]):
            Optional. Represents metadata related to the
            file that can suit particular use cases. The
            prefix "google." is reserved for the key for use
            by Google, but other prefixes can be freely
            used.
        download_uri (str):
            Output only. The
            [AssistantService.DownloadSessionFile][google.cloud.discoveryengine.v1alpha.AssistantService.DownloadSessionFile]
            URL to download the file. This URL will need the same
            credentials as
            [AssistantService.ListSessionFileMetadata][google.cloud.discoveryengine.v1alpha.AssistantService.ListSessionFileMetadata]
            method and will provide the resource.
        file_origin_type (google.cloud.discoveryengine_v1alpha.types.FileOriginType):
            Optional. The origin of the file.
        views (MutableMapping[str, google.cloud.discoveryengine_v1alpha.types.FileView]):
            Output only. Alternate views of this file object. Each file
            view is attached to a specific role. Possible example keys:

            - "thumbnail"
            - "mobile_thumbnail"
            - "clip"
            - "summary"
            - "translation".
    """

    file_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    mime_type: str = proto.Field(
        proto.STRING,
        number=3,
    )
    byte_size: int = proto.Field(
        proto.INT64,
        number=4,
    )
    original_uri: str = proto.Field(
        proto.STRING,
        number=9,
    )
    original_source_type: "FileSource" = proto.Field(
        proto.ENUM,
        number=10,
        enum="FileSource",
    )
    upload_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
    )
    last_add_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=13,
        message=timestamp_pb2.Timestamp,
    )
    metadata: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=18,
    )
    download_uri: str = proto.Field(
        proto.STRING,
        number=20,
    )
    file_origin_type: common.FileOriginType = proto.Field(
        proto.ENUM,
        number=21,
        enum=common.FileOriginType,
    )
    views: MutableMapping[str, "FileView"] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=22,
        message="FileView",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
