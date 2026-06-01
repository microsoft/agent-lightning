from typing import Dict
from .action import Action

CONTINUE_MSG = """
You forgot to use a function call in your response. 
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

class Observation:
    def __init__(self, bash_output, error_code, action: Action, num_lines: int = 40, 
                 stdout: str = None, stderr: str = None):
        self.bash_output = bash_output
        self.error_code = error_code
        self.action = action
        self.num_lines = num_lines
        # New fields for separated stdout/stderr
        self.stdout = stdout
        self.stderr = stderr

    def _truncate_output(self, output: str) -> str:
        """Truncate output if too long, keeping first and last num_lines."""
        if not output:
            return ""
        lines = output.splitlines()
        if len(lines) > 2 * self.num_lines:
            top_lines = "\n".join(lines[:self.num_lines])
            bottom_lines = "\n".join(lines[-self.num_lines:])
            divider = "-" * 50
            return (
                f"{top_lines}\n"
                f"{divider}\n"
                f"<Observation truncated in middle for saving context>\n"
                f"{divider}\n"
                f"{bottom_lines}"
            )
        return output

    def __str__(self):
        # Get function name safely
        func_name = getattr(self.action, 'function_name', '') if self.action else ''
        
        if not func_name:
            return CONTINUE_MSG
        elif func_name == "finish" or func_name == "submit":
            return "<<< Finished >>>"
        else:
            if func_name == "execute_bash" or func_name == "bash":
                # Check if we have separated stdout/stderr
                if self.stdout is not None or self.stderr is not None:
                    # Use separated stdout/stderr format
                    stdout_str = self._truncate_output(self.stdout or "")
                    stderr_str = self._truncate_output(self.stderr or "")
                    
                    output_parts = [f"Exit code: {self.error_code}"]
                    # Use escaped brackets to prevent Rich from interpreting them as markup
                    output_parts.append(f"Execution output of \\[{func_name}]:")
                    
                    if stdout_str.strip():
                        output_parts.append(f"\\[STDOUT]\n{stdout_str}")
                    else:
                        output_parts.append("\\[STDOUT]\n")
                    
                    if stderr_str.strip():
                        output_parts.append(f"\\[STDERR]\n{stderr_str}")
                    # Only show STDERR section if there's actual stderr content
                    # This avoids cluttering when there are no warnings/errors
                    
                    output = "\n".join(output_parts)
                else:
                    # Fallback to legacy combined output
                    truncated_output = self._truncate_output(self.bash_output or "")
                    output = (
                        f"Exit code: {self.error_code}\n"
                        f"Execution output of \\[{func_name}]:\n"
                        f"{truncated_output}"
                    )
            else:
                output = f"Execution output of \\[{func_name}]:\n{self.bash_output}"
            return output
