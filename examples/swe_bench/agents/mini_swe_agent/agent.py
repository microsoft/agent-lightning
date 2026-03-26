"""Mini SWE Agent — lightweight coding agent for SWE-bench.

Uses the OpenAI API (via agl-lite gateway) with tool_use for file operations.
Designed for testing and OSS models. Not production-quality — keeps it simple.
"""

import json
import os
import subprocess
import sys

from openai import OpenAI

SYSTEM_PROMPT = """\
You are an expert software engineer. You are given a bug report for a Python repository.
Your task is to fix the bug by modifying the source code.

You have access to the following tools:
- `bash`: Run a bash command in the repository root (/testbed).
- `write_file`: Write content to a file.
- `read_file`: Read content from a file.

Steps:
1. Read the bug report carefully.
2. Explore the repository to understand the codebase (use `bash` with find, grep, etc.).
3. Identify the root cause of the bug.
4. Make minimal changes to fix the bug.
5. Do NOT modify test files.
6. When done, respond with "DONE".
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in /testbed. Returns stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to /testbed"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to /testbed"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def run_bash(command: str) -> str:
    """Execute bash command in /testbed, return output."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd="/testbed",
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        # Truncate long output
        if len(output) > 8000:
            output = output[:4000] + "\n... (truncated) ...\n" + output[-4000:]
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out (120s)"
    except Exception as e:
        return f"ERROR: {e}"


def read_file(path: str) -> str:
    """Read file from /testbed."""
    try:
        full_path = os.path.join("/testbed", path)
        with open(full_path) as f:
            content = f.read()
        if len(content) > 10000:
            content = content[:5000] + "\n... (truncated) ...\n" + content[-5000:]
        return content
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    """Write file to /testbed."""
    try:
        full_path = os.path.join("/testbed", path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"ERROR: {e}"


def handle_tool_call(name: str, arguments: dict) -> str:
    """Dispatch tool call."""
    if name == "bash":
        return run_bash(arguments["command"])
    elif name == "read_file":
        return read_file(arguments["path"])
    elif name == "write_file":
        return write_file(arguments["path"], arguments["content"])
    else:
        return f"Unknown tool: {name}"


def main():
    problem_statement = os.environ.get("AGL_TASK_INPUT", "")
    model = os.environ.get("AGL_MODEL_NAME", "default")
    max_turns = int(os.environ.get("AGL_MAX_TURNS", "30"))

    if not problem_statement:
        print("ERROR: AGL_TASK_INPUT not set")
        sys.exit(1)

    client = OpenAI()  # Uses OPENAI_BASE_URL and OPENAI_API_KEY from env

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Bug report:\n\n{problem_statement}"},
    ]

    print(f"=== Mini SWE Agent ===")
    print(f"Model: {model}")
    print(f"Max turns: {max_turns}")
    print(f"Problem: {problem_statement[:200]}...")
    print()

    for turn in range(max_turns):
        print(f"--- Turn {turn + 1}/{max_turns} ---")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"ERROR: API call failed: {e}")
            break

        choice = response.choices[0]
        message = choice.message

        # Add assistant message to history.
        messages.append(message.model_dump())

        if message.content:
            print(f"Assistant: {message.content[:500]}")
            if "DONE" in message.content.upper():
                print("Agent signaled DONE.")
                break

        if not message.tool_calls:
            print("No tool calls, agent finished.")
            break

        # Process tool calls.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"  Tool: {name}({json.dumps(args)[:200]})")
            result = handle_tool_call(name, args)
            print(f"  Result: {result[:300]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    print("=== Agent complete ===")


if __name__ == "__main__":
    main()
