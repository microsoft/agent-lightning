import re
from typing import Dict
import shlex


class Action:
    """
    Represents an action with:
      - function_name (e.g. 'file_editor')
      - parameters    (a dictionary of parameter_name -> value)
    """

    def __init__(self, function_name: str, parameters: Dict[str, str], function_id: str = None):
        self.function_name = function_name
        self.parameters = parameters

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, object]:
        return {"function": self.function_name, "parameters": self.parameters}
