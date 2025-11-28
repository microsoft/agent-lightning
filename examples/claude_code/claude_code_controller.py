# Copyright (c) Microsoft. All rights reserved.

"""Controller module for managing Claude Code executions in containerized environments.

This module provides the ClaudeController class that manages the execution of Claude Code
within Docker containers. It handles container initialization, command execution, and
patch application for SWE-bench evaluation tasks.

Key features:
- Container lifecycle management
- Claude Code execution via CLI or Python SDK
- Patch application and result retrieval
- Integration with SWE-bench evaluation framework
"""

from functools import partial
from typing import Any, Dict, Literal

import dotenv
from swebench_utils.docker_runtime import Runtime
from swebench_utils.logger import logger


class ClaudeController:
    system_prompt = """
 You are an expert software engineer solving swebench bug fixing tasks.
"""
    user_prompt = """
You are given a code repository in the current directory (/testbed).
The bug description is:
{description}
=================================================
You task is to fix the bug with the following steps:
(1) write test cases to reproduce the bug.
(2) explore the source codes to locate the bug.
(3) edit the source codes to fix the bug.
(4) rerun your written test cases to validate that the bug is fixed. If not, go back to explore the source codes and fix the codes again.
(5) remember to delete the test cases you write at last.
Please do not commit your edits. We will do it later.
"""

    def __init__(self, image: str, instance: Dict[str, Any], run_id: str, endpoint: str, api_key: str) -> None:
        self.image = image
        self.instance = instance
        self.run_id = run_id
        self.endpoint = endpoint
        self.api_key = api_key
        self.container: Runtime = self.init_container(self.image, self.instance)
        return

    def init_container(self, image: str, instance: Dict[str, Any]) -> Runtime:
        container = Runtime.start_session(
            image,
            instance,
            log_function=partial(logger, run_id=self.run_id, instance_id=instance["instance_id"]),
        )
        container.send_command("curl -fsSL https://claude.ai/install.sh | bash")
        container.send_command('alias claude="$HOME/.local/bin/claude"')
        dotenv.load_dotenv()
        # anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        # container.send_command(f"export ANTHROPIC_API_KEY={anthropic_api_key}")
        # if (not os.getenv("ANTHROPIC_BASE_URL")) or (not os.getenv("ANTHROPIC_AUTH_TOKEN")):
        #     raise RuntimeError("ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN not found!")
        container.send_command(f"export ANTHROPIC_BASE_URL={self.endpoint}")
        container.send_command(f"export ANTHROPIC_AUTH_TOKEN={self.api_key}")
        container.send_command("export IS_SANDBOX=1")
        return container

    def _run_cli(self, instance: Dict[str, Any], max_step: int, timelimit: int) -> None:
        # prepare prompt safely: write it to a file inside the container using a single-quoted heredoc
        # directly applying prompt for heredoc may raise error for windows line ending \r\n
        prompt_text = self.user_prompt.format(description=instance["problem_statement"].replace('"""', "'''"))
        # choose a simple filename and a heredoc delimiter unlikely to collide
        heredoc_cmd = "cat > /tmp/cc_prompt.txt <<'CC_PROMPT'\n" + prompt_text + "\nCC_PROMPT\n"
        self.container.send_command(heredoc_cmd)

        # self.container.send_command("mkdir -p /testbed/.claude")
        # with open("utils/settings.template.json") as f:
        #     setting = f.read()
        # setting_cmd = "cat > /testbed/.claude/settings.json <<'CC_SETTING'\n" + setting + "\nCC_SETTING\n"
        # self.container.send_command(setting_cmd)

        # with open("utils/handle_hook.template.sh") as f:
        #     handler = f.read()
        # handler_cmd = "cat > /tmp/handle_hook.sh <<'CC_HOOK'\n" + handler + "\nCC_HOOK\n"
        # self.container.send_command(handler_cmd)
        # self.container.send_command("chmod +x /tmp/handle_hook.sh")

        # run claude reading the prompt from the file to avoid shell interpolation issues
        claude_cmd = f'claude -p "$(cat /tmp/cc_prompt.txt)" --append-system-prompt "{self.system_prompt}" --max-turns {max_step} --dangerously-skip-permissions --output-format json --verbose'
        self.container.send_command(claude_cmd, timelimit * 60)
        # self.container.send_command("cat /tmp/hook.out")
        return

    def _run_python_sdk(self, instance: Dict[str, Any], max_step: int, timelimit: int):
        self.container.send_command(
            f"""
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed. Installing Python 3.12..."
    sudo apt-get update && sudo apt-get install -y python3.12
else
    echo "Python is already installed."
fi
"""
        )
        self.container.send_command("python3 -m pip install claude-code-sdk")
        with open("src/agent/cc/claude_code_main.py.template") as f:
            entrance_template = f.read()
        entrance_template.replace("SYS_PROMPT", self.system_prompt).replace(
            "PROMPT", self.user_prompt.format(description=instance["problem_statement"].replace('"""', "'''"))
        ).replace("MAX_STEP", str(max_step))
        self.container.send_command(f"cat > /tmp/claude_code_main.py <<'CC_MAIN'\n{entrance_template}\nCC_MAIN\n")
        self.container.send_command("python3 /tmp/claude_code_main.py", timelimit * 60)
        return

    def run_instance(
        self,
        instance: Dict[str, Any],
        max_step: int = 40,
        timelimit: int = 30,
        run_method: Literal["python", "cli"] = "python",
    ) -> Dict[str, Any]:
        """
        timelimit: in minute
        """
        if run_method == "python":
            self._run_python_sdk(instance, max_step, timelimit)
        elif run_method == "cli":
            self._run_cli(instance, max_step, timelimit)
        else:
            raise ValueError(f"wrong run_method {run_method}, run_method should be in [python, cli]")
        result = self.container.send_command("git --no-pager diff HEAD")
        git_diff = result.output.replace("git --no-pager diff HEAD\n", "")
        return {
            "instance_id": instance["instance_id"],
            "model_patch": git_diff,
            "model_name_or_path": "cc",
        }

    def __del__(self):
        if hasattr(self, "container"):
            self.container.cleanup()
