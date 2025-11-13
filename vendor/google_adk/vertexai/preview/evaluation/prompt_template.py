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
"""Prompt template for creating prompts with variables."""

import re
from typing import Set

_VARIABLE_NAME_REGEX = r"\{([_a-zA-Z][_a-zA-Z0-9]*)\}"


class PromptTemplate:
    """A prompt template for creating prompts with variables.

    The `PromptTemplate` class allows users to define a template string with
    variables represented in curly braces `{variable}`. The variable
    names cannot contain spaces and must start with a letter or underscore,
    followed by letters, digits, or underscore. These variables can be
    replaced with specific values using the `assemble` method, providing
    flexibility in generating dynamic prompts.

    Usage:

        ```
        template_str = "Hello, {name}! Today is {day}. How are you?"
        prompt_template = PromptTemplate(template_str)
        completed_prompt = prompt_template.assemble(name="John", day="Monday")
        print(completed_prompt)
        ```
    """

    def __init__(self, template: str):
        """Initializes the PromptTemplate with a given template.

        Args:
            template: The template string with variables. Variables should be
              represented in curly braces `{variable}`.
        """
        self.template = str(template)
        self.variables = self._get_variables()

    def _get_variables(self) -> Set[str]:
        """Extracts and return a set of variable names from the template."""
        return set(re.findall(_VARIABLE_NAME_REGEX, self.template))

    def assemble(self, **kwargs) -> "PromptTemplate":
        """Replaces only the provided variables in the template with specific values.

        Args:
            **kwargs: Keyword arguments where keys are placeholder names and values
              are the replacements.

        Returns:
            A new PromptTemplate instance with the updated template string.
        """
        assembled_string = self.template
        for variable_name, value in kwargs.items():
            if variable_name not in self.variables:
                raise ValueError(
                    f"Invalid variable name '{variable_name}'. "
                    f"Valid variables are: {self.variables}"
                )
            placeholder = "{" + variable_name + "}"
            assembled_string = assembled_string.replace(placeholder, str(value))
        return PromptTemplate(assembled_string)

    def __str__(self) -> str:
        """Returns the template string."""
        return self.template

    def __repr__(self) -> str:
        """Returns a string representation of the PromptTemplate."""
        return f"PromptTemplate('{self.template}')"
