"""
Tool definitions for the Docker Agent
"""

ALLOWED_STR_REPLACE_EDITOR_COMMANDS = ["view", "create", "str_replace", "insert"]

_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

str_replace_editor_tool = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": _STR_REPLACE_EDITOR_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": f"The command to run. Allowed options are: {', '.join(f'`{cmd}`' for cmd in ALLOWED_STR_REPLACE_EDITOR_COMMANDS)}.",
                    "enum": ALLOWED_STR_REPLACE_EDITOR_COMMANDS,
                    "type": "string",
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string",
                },
                "file_text": {
                    "description": "Required for the `create` command, contains the content of the file to be created.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required for the `str_replace` command, specifies the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional for the `str_replace` command to specify the replacement string. Required for the `insert` command to specify the string to insert.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required for the `insert` command. The `new_str` will be inserted AFTER the line specified.",
                    "type": "integer",
                },
                "view_range": {
                    "description": "Optional for the `view` command when `path` points to a file. Specifies the line range to view. E.g., [11, 12] shows lines 11 and 12. Indexing starts at 1. Use [start_line, -1] to show all lines from `start_line` to the end.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["command", "path"],
        },
    },
}


_BASH_DESCRIPTION = """
Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. For example: `python my_script.py`.
"""

execute_bash_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _BASH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command (and optional arguments) to execute. For example: 'python my_script.py'",
                }
            },
            "required": ["command"],
        },
    },
}


_SUBMIT_DESCRIPTION = """
A simple submit tool to finish tasks.

This tool signals completion of a task or submission of results.
No parameters required - simply call to indicate task completion.
"""

submit_tool = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": _SUBMIT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}
