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

"""Utility to interact with Ray-on-Vertex dashboard."""

from ray.dashboard.modules import dashboard_sdk as oss_dashboard_sdk

from .util import _gapic_utils
from .util import _validation_utils


def get_job_submission_client_cluster_info(
    address: str, *args, **kwargs
) -> oss_dashboard_sdk.ClusterInfo:
    """A vertex_ray implementation of get_job_submission_client_cluster_info().

    Implements
    https://github.com/ray-project/ray/blob/ray-2.47.1/python/ray/dashboard/modules/dashboard_sdk.py#L84
    This will be called in from Ray Job API Python client.

    Args:
        address: Address without the module prefix `vertex_ray` but otherwise
            the same format as passed to ray.init(address="vertex_ray://...").
        *args: Remainder of positional args that might be passed down from
            the framework.
        **kwargs: Remainder of keyword args that might be passed down from
            the framework.

    Returns:
        An instance of ClusterInfo that contains address, cookies and
        metadata for SubmissionClient to use.

    Raises:
        RuntimeError if head_address is None.
    """
    if _validation_utils.valid_dashboard_address(address):
        dashboard_address = address
    else:
        address = _validation_utils.maybe_reconstruct_resource_name(address)
        _validation_utils.valid_resource_name(address)

        resource_name = address
        response = _gapic_utils.get_persistent_resource(resource_name)

        dashboard_address = response.resource_runtime.access_uris.get(
            "RAY_DASHBOARD_URI", None
        )

        if dashboard_address is None:
            raise RuntimeError(
                "[Ray on Vertex AI]: Unable to obtain a response from the backend."
            )
    # If passing the dashboard uri, programmatically get headers
    bearer_token = _validation_utils.get_bearer_token()
    if kwargs.get("headers", None) is None:
        kwargs["headers"] = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(bearer_token),
        }
    return oss_dashboard_sdk.get_job_submission_client_cluster_info(
        address=dashboard_address,
        _use_tls=True,
        *args,
        **kwargs,
    )
