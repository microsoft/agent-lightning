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
"""Utility functions for prompt optimizer."""

import json
from . import types


def _get_service_account(
    config: types.PromptOptimizerConfigOrDict,
) -> str:
    """Get the service account from the config for the custom job."""
    if isinstance(config, dict):
        config = types.PromptOptimizerConfig.model_validate(config)

    if config.service_account and config.service_account_project_number:
        raise ValueError(
            "Only one of service_account or "
            "service_account_project_number can be provided."
        )
    elif config.service_account:
        return config.service_account
    elif config.service_account_project_number:
        return f"{config.service_account_project_number}-compute@developer.gserviceaccount.com"
    else:
        raise ValueError(
            "Either service_account or service_account_project_number " "is required."
        )


def _clean_and_parse_optimized_prompt(output_str: str):
    """Cleans a string response returned from the prompt optimizer endpoint.

    Args:
        output_str: The optimized prompt string containing the JSON data,
          potentially with markdown formatting like ```json ... ```.

    Returns:
        The parsed JSON data, or None if parsing fails.
    """
    lines = output_str.strip().split("\n")
    # Remove markdown delimiters
    if lines and lines[0].strip().startswith("```"):
        cleaned_string = "\n".join(lines[1:-1])
    else:
        cleaned_string = output_str

    # remove any 'json' labels if they exist on the first line.
    if cleaned_string.strip().startswith("json"):
        cleaned_string = cleaned_string.strip()[4:].strip()

    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        # TODO(b/437144880): raise errors.ClientError here instead
        raise ValueError(
            f"Failed to parse the response from prompt optimizer endpoint. {e}"
        ) from e


def _parse(output_str: str) -> types.ParsedResponse:
    """Parses the output string from the prompt optimizer endpoint."""
    parsed_out = _clean_and_parse_optimized_prompt(output_str)
    return types.ParsedResponse(**parsed_out)
