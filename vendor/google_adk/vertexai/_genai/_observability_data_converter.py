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
"""Dataset converter for Google Observability GenAI data."""

import json
import logging
from typing import Any, Optional

from google.genai import types as genai_types
from typing_extensions import override

from . import _evals_utils
from . import types


logger = logging.getLogger("vertexai_genai._observability_data_converters")


def _load_jsonl(data: Any, case_id: str) -> list[dict[Any, Any]]:
    """Parses the raw JSONL data into a list of dict possible."""
    if isinstance(data, str):
        json_list = []
        for line in data.splitlines():
            loaded_json = json.loads(line)
            if not isinstance(loaded_json, dict):
                raise TypeError(
                    f"Decoded JSON payload is not a dict for case "
                    f"{case_id}. Type found: {type(loaded_json).__name__}"
                )
            json_list.append(loaded_json)
        return json_list
    else:
        raise TypeError(
            f"Payload is not a JSONL string for case {case_id}. Type "
            f"found: {type(data).__name__}"
        )


class ObservabilityDataConverter(_evals_utils.EvalDataConverter):
    """Converter for dataset in GCP Observability GenAI format."""

    def _message_to_content(self, message: dict[str, Any]) -> genai_types.Content:
        """Converts Observability GenAI Message format to Content."""
        parts = []
        message_parts = message.get("parts", [])
        if isinstance(message_parts, list):
            for message_part in message_parts:
                part = None
                part_type = message_part.get("type", "")
                if part_type == "text":
                    part = genai_types.Part(text=message_part.get("content", ""))
                elif part_type == "blob":
                    part = genai_types.Part(
                        inline_data=genai_types.Blob(
                            data=message_part.get("data", ""),
                            mime_type=message_part.get("mime_type", ""),
                        )
                    )
                elif part_type == "file_data":
                    part = genai_types.Part(
                        file_data=genai_types.FileData(
                            file_uri=message_part.get("file_uri", ""),
                            mime_type=message_part.get("mime_type", ""),
                        )
                    )
                elif part_type == "tool_call":
                    # O11y format requires use of id in place of name
                    part = genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            id=message_part.get("id", ""),
                            name=message_part.get("id", ""),
                            args=message_part.get("arguments", {}),
                        )
                    )
                elif part_type == "tool_call_response":
                    # O11y format requires use of id in place of name
                    part = genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            id=message_part.get("id", ""),
                            name=message_part.get("id", ""),
                            response=message_part.get("result", {}),
                        )
                    )
                else:
                    logger.warning(
                        "Skipping message part due to unrecognized message "
                        "part type of '%s'",
                        part_type,
                    )

                if part is not None:
                    parts.append(part)

        return genai_types.Content(parts=parts, role=message.get("role", ""))

    def _parse_messages(
        self,
        eval_case_id: str,
        request_msgs: list[Any],
        response_msgs: list[Any],
        system_instruction_msg: Optional[dict[str, Any]] = None,
    ) -> types.EvalCase:
        """Parses a set of Observability messages into an EvalCase."""
        # System instruction message
        system_instruction = None
        if system_instruction_msg is not None:
            system_instruction = self._message_to_content(system_instruction_msg)

        # Request messages
        prompt = None
        conversation_history = []
        if request_msgs:
            # Extract latest message as prompt
            prompt = self._message_to_content(request_msgs[-1])

            # All previous messages are conversation history
            if len(request_msgs) > 1:
                for i, msg in enumerate(request_msgs[:-1]):
                    conversation_history.append(
                        types.evals.Message(
                            turn_id=str(i),
                            content=self._message_to_content(msg),
                            author=msg.get("role", ""),
                        )
                    )

        # Output messages
        responses = []
        for msg in response_msgs:
            response = types.ResponseCandidate(response=self._message_to_content(msg))
            responses.append(response)

        return types.EvalCase(
            eval_case_id=eval_case_id,
            prompt=prompt,
            responses=responses,
            system_instruction=system_instruction,
            conversation_history=conversation_history,
            reference=None,
        )

    @override
    def convert(self, raw_data: list[dict[str, Any]]) -> types.EvaluationDataset:
        """Converts a list of GCP Observability GenAI cases into an EvaluationDataset."""
        eval_cases = []

        for i, case in enumerate(raw_data):
            eval_case_id = f"observability_eval_case_{i}"

            if "request" not in case or "response" not in case:
                logger.warning(
                    "Skipping case %s due to missing 'request' or 'response' key.",
                    eval_case_id,
                )
                continue

            request_data = case.get("request", [])
            request_list = _load_jsonl(request_data, eval_case_id)

            response_data = case.get("response", [])
            response_list = _load_jsonl(response_data, eval_case_id)

            system_dict = None
            if "system_instruction" in case:
                system_data = case.get("system_instruction", {})
                system_list = _load_jsonl(system_data, eval_case_id)
                system_dict = system_list[0] if system_list else {}

            eval_case = self._parse_messages(
                eval_case_id, request_list, response_list, system_dict
            )
            eval_cases.append(eval_case)

        return types.EvaluationDataset(eval_cases=eval_cases)
