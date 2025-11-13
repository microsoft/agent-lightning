"""Regsiter Scikit Learn for Ray on Vertex AI."""

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
import pickle
import warnings
import ray
import ray.cloudpickle as cpickle
import tempfile
from typing import Optional, TYPE_CHECKING

from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.utils import gcs_utils
from google.cloud.aiplatform.vertex_ray.predict.util import constants
from google.cloud.aiplatform.vertex_ray.predict.util import (
    predict_utils,
)
from google.cloud.aiplatform.vertex_ray.util._validation_utils import (
    _V2_4_WARNING_MESSAGE,
    _V2_9_WARNING_MESSAGE,
)


try:
    from ray.train import sklearn as ray_sklearn

    if TYPE_CHECKING:
        import sklearn

except ImportError as ie:
    if ray.__version__ < "2.42.0":
        raise ModuleNotFoundError("Sklearn isn't installed.") from ie
    else:
        sklearn = None


def register_sklearn(
    checkpoint: "ray_sklearn.SklearnCheckpoint",
    artifact_uri: Optional[str] = None,
    display_name: Optional[str] = None,
    **kwargs,
) -> aiplatform.Model:
    """Uploads a Ray Sklearn Checkpoint as Sklearn Model to Model Registry.

    Example usage:
        from vertex_ray.predict import sklearn
        from ray.train.sklearn import SklearnCheckpoint

        trainer = SklearnTrainer(estimator=RandomForestClassifier, ...)
        result = trainer.fit()
        sklearn_checkpoint = SklearnCheckpoint.from_checkpoint(result.checkpoint)

        my_model = sklearn.register_sklearn(
            checkpoint=sklearn_checkpoint,
            artifact_uri="gs://{gcs-bucket-name}/path/to/store"
        )


    Args:
        checkpoint: SklearnCheckpoint instance.
        artifact_uri (str):
            Optional. The path to the directory where Model Artifacts will be saved. If
            not set, will use staging bucket set in aiplatform.init().
        display_name (str):
            Optional. The display name of the Model. The name can be up to 128
            characters long and can be consist of any UTF-8 characters.
        **kwargs:
            Any kwargs will be passed to aiplatform.Model registration.

    Returns:
        model (aiplatform.Model):
                Instantiated representation of the uploaded model resource.

    Raises:
        ValueError: Invalid Argument.
        RuntimeError: Only Ray version 2.9.3 is supported.
    """
    ray_version = ray.__version__
    if ray_version != "2.9.3":
        raise RuntimeError(
            f"Ray version {ray_version} is not supported to upload Sklearn"
            " model to Vertex Model Registry yet. Please use Ray 2.9.3."
        )
    if ray_version == "2.9.3":
        warnings.warn(_V2_9_WARNING_MESSAGE, DeprecationWarning, stacklevel=1)

    artifact_uri = artifact_uri or initializer.global_config.staging_bucket
    predict_utils.validate_artifact_uri(artifact_uri)
    display_model_name = (
        (f"ray-on-vertex-registered-sklearn-model-{utils.timestamped_unique_name()}")
        if display_name is None
        else display_name
    )
    estimator = _get_estimator_from(checkpoint)

    model_dir = os.path.join(artifact_uri, display_model_name)
    file_path = os.path.join(model_dir, constants._PICKLE_FILE_NAME)

    with tempfile.NamedTemporaryFile(suffix=constants._PICKLE_EXTENTION) as temp_file:
        pickle.dump(estimator, temp_file)
        gcs_utils.upload_to_gcs(temp_file.name, file_path)
        return aiplatform.Model.upload_scikit_learn_model_file(
            model_file_path=temp_file.name, display_name=display_model_name, **kwargs
        )


def _get_estimator_from(
    checkpoint: "ray_sklearn.SklearnCheckpoint",
) -> "sklearn.base.BaseEstimator":
    """Converts a SklearnCheckpoint to sklearn estimator.

    Args:
        checkpoint: SklearnCheckpoint instance.

    Returns:
        A Sklearn BaseEstimator

    Raises:
        ValueError: Invalid Argument.
        RuntimeError: Model not found.
        RuntimeError: Ray version 2.4 is not supported.
        RuntimeError: Only Ray version 2.9.3 is supported.
    """

    ray_version = ray.__version__
    if ray_version == "2.4.0":
        raise RuntimeError(_V2_4_WARNING_MESSAGE)
    if ray_version != "2.9.3":
        raise RuntimeError(
            f"Ray version {ray_version} is not supported to convert a Sklearn"
            " checkpoint to sklearn estimator on Vertex yet. Please use Ray 2.9.3."
        )

    try:
        return checkpoint.get_model()
    except AttributeError:
        model_file_name = ray.train.sklearn.SklearnCheckpoint.MODEL_FILENAME

    model_path = os.path.join(checkpoint.path, model_file_name)

    if os.path.exists(model_path):
        with open(model_path, mode="rb") as f:
            obj = pickle.load(f)
    else:
        try:
            # Download from GCS to temp and then load_model
            with tempfile.TemporaryDirectory() as temp_dir:
                gcs_utils.download_from_gcs("gs://" + checkpoint.path, temp_dir)
                with open(f"{temp_dir}/{model_file_name}", mode="rb") as f:
                    obj = cpickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"{model_file_name} not found in this checkpoint due to: {e}."
            )
    return obj
