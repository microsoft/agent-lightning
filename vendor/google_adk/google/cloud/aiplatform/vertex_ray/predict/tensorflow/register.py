"""Regsiter Tensorflow for Ray on Vertex AI."""

# -*- coding: utf-8 -*-

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
import os
import logging
import ray
from typing import Callable, Optional, Union, TYPE_CHECKING
import warnings
from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.vertex_ray.predict.util import constants
from google.cloud.aiplatform.vertex_ray.predict.util import (
    predict_utils,
)
from google.cloud.aiplatform.vertex_ray.util._validation_utils import (
    _V2_4_WARNING_MESSAGE,
    _V2_9_WARNING_MESSAGE,
)


try:
    from ray.train import tensorflow as ray_tensorflow

    if TYPE_CHECKING:
        import tensorflow as tf

except ModuleNotFoundError as mnfe:
    raise ModuleNotFoundError("Tensorflow isn't installed.") from mnfe


def register_tensorflow(
    checkpoint: ray_tensorflow.TensorflowCheckpoint,
    artifact_uri: Optional[str] = None,
    _model: Optional[Union["tf.keras.Model", Callable[[], "tf.keras.Model"]]] = None,
    display_name: Optional[str] = None,
    tensorflow_version: Optional[str] = None,
    **kwargs,
) -> aiplatform.Model:
    """Uploads a Ray Tensorflow Checkpoint as Tensorflow Model to Model Registry.

    Example usage:
        from vertex_ray.predict import tensorflow

        def create_model():
            model = tf.keras.Sequential(...)
            ...
            return model

        result = trainer.fit()
        my_model = tensorflow.register_tensorflow(
            checkpoint=result.checkpoint,
            _model=create_model,
            artifact_uri="gs://{gcs-bucket-name}/path/to/store",
            use_gpu=True
        )

        1. `use_gpu` will be passed to aiplatform.Model.upload_tensorflow_saved_model()
        2. The `create_model` provides the model_definition which is required if
        you create the TensorflowCheckpoint using `from_model` method.
        More here, https://docs.ray.io/en/latest/train/api/doc/ray.train.tensorflow.TensorflowCheckpoint.get_model.html#ray.train.tensorflow.TensorflowCheckpoint.get_model

    Args:
        checkpoint: TensorflowCheckpoint instance.
        artifact_uri (str):
            Optional. The path to the directory where Model Artifacts will be saved. If
            not set, will use staging bucket set in aiplatform.init().
        _model: Tensorflow Model Definition. Refer
            https://docs.ray.io/en/latest/train/api/doc/ray.train.tensorflow.TensorflowCheckpoint.get_model.html#ray.train.tensorflow.TensorflowCheckpoint.get_model
        display_name (str):
            Optional. The display name of the Model. The name can be up to 128
            characters long and can be consist of any UTF-8 characters.
        tensorflow_version (str):
                Optional. The version of the Tensorflow serving container.
                Supported versions:
                https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
                If the version is not specified, the latest version is used.
        **kwargs:
            Any kwargs will be passed to aiplatform.Model registration.

    Returns:
        model (aiplatform.Model):
                Instantiated representation of the uploaded model resource.

    Raises:
        ValueError: Invalid Argument.
    """
    if ray.__version__ == "2.9.3":
        warnings.warn(_V2_9_WARNING_MESSAGE, DeprecationWarning, stacklevel=1)
    if tensorflow_version is None:
        tensorflow_version = constants._TENSORFLOW_VERSION
    artifact_uri = artifact_uri or initializer.global_config.staging_bucket
    predict_utils.validate_artifact_uri(artifact_uri)
    prefix = "ray-on-vertex-registered-tensorflow-model"
    display_model_name = (
        (f"{prefix}-{utils.timestamped_unique_name()}")
        if display_name is None
        else display_name
    )
    tf_model = _get_tensorflow_model_from(checkpoint, model=_model)
    model_dir = os.path.join(artifact_uri, prefix)
    try:
        import tensorflow as tf

        tf.saved_model.save(tf_model, model_dir)
    except ImportError:
        logging.warning("TensorFlow must be installed to save the trained model.")
    return aiplatform.Model.upload_tensorflow_saved_model(
        saved_model_dir=model_dir,
        display_name=display_model_name,
        tensorflow_version=tensorflow_version,
        **kwargs,
    )


def _get_tensorflow_model_from(
    checkpoint: ray_tensorflow.TensorflowCheckpoint,
    model: Optional[Union["tf.keras.Model", Callable[[], "tf.keras.Model"]]] = None,
) -> "tf.keras.Model":
    """Converts a TensorflowCheckpoint to Tensorflow Model.

    Args:
        checkpoint: TensorflowCheckpoint instance.
        model: Tensorflow Model Defination.

    Returns:
        A Tensorflow Native Framework Model.

    Raises:
        ValueError: Invalid Argument.
        RuntimeError: Ray version 2.4.0 is not supported.
    """
    ray_version = ray.__version__
    if ray_version == "2.4.0":
        raise RuntimeError(_V2_4_WARNING_MESSAGE)

    try:
        import tensorflow as tf

        try:
            return tf.saved_model.load(checkpoint.path)
        except OSError:
            return tf.saved_model.load("gs://" + checkpoint.path)

    except ImportError:
        logging.warning("TensorFlow must be installed to load the trained model.")
