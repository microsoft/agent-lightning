"""
Agent for Docker-based execution
"""
import os
import re
import copy
import json
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from rich.markup import escape

from .action import Action
from .trajectory import TrajectoryStep, Trajectory
from .tools import str_replace_editor_tool, execute_bash_tool, submit_tool
from .openai_client import create_chat_completion, openai_timeout


def get_logger(name: str) -> logging.Logger:
    """Get a logger with RichHandler for colorful output."""
    logger = logging.getLogger(name)
    
    # Remove existing handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    
    # Remove root logger handlers
    root_logger = logging.getLogger()
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    
    logger.setLevel(logging.INFO)
    
    # Create RichHandler
    rich_handler = RichHandler(
        rich_tracebacks=True, 
        show_path=False,
        show_time=True,
        show_level=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)
    
    return logger


logger = get_logger(__name__)

def stream_and_collect(response):
    full_text = ""

    for chunk in response:
        delta = chunk["choices"][0]["delta"]

        if "content" in delta:
            text = delta["content"]
            print(text, end="", flush=True)
            full_text += text

    print()
    return full_text


@dataclass
class AgentArgs:
    """Agent configuration arguments."""
    system_prompt: str
    instance_prompt: str
    llm_name: str
    llm_base_url: Optional[str] = None
    max_retries: int = 5
    save_openai_response: bool = False
    output_dir: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None
    quiet: bool = False  # Disable console output (for benchmark mode)


class Agent:
    """
    Agent handles the behavior of the model and how it interacts with the environment.
    """

    def __init__(self, args: AgentArgs, logger=None):
        self.args = args
        self.llm_name = args.llm_name
        self.quiet = args.quiet  # Disable console output (for benchmark mode)
        
        # Console for output (quiet mode captures to StringIO for later retrieval)
        if self.quiet:
            import io
            self._console_buffer = io.StringIO()
            self.console = Console(file=self._console_buffer, record=True, force_terminal=True, width=100)
        else:
            self._console_buffer = None
            self.console = Console()
        
        # Set up logger (disable in quiet mode)
        if logger is None:
            self.logger = get_logger("Agent")
        else:
            self.logger = logger
        
        if self.quiet:
            self.logger.setLevel(logging.WARNING)  # Only show warnings and errors
        
        # Configure LLM base URL
        self.llm_base_url = args.llm_base_url
        if self.llm_base_url is None:
            if ("openai/" in self.llm_name) or ("hosted_vllm" in self.llm_name):
                self.llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.max_retries = args.max_retries
        
        # Extra body params (e.g., {"chat_template_kwargs": {"thinking": True}))
        self.extra_body = args.extra_body
        
        self.logger.info(f"Initialized Agent with LLM: {self.llm_name}")
        self.logger.info(f"🔗 LLM Base URL: {self.llm_base_url}")
        self.logger.info(f"📦 Extra body: {self.extra_body}")
        
        # Save OpenAI response settings
        self.save_openai_response = args.save_openai_response
        self.output_dir = args.output_dir
        self.llm_call_count = 0
        if self.save_openai_response:
            self.logger.info(f"📝 Save OpenAI response enabled, output_dir: {self.output_dir}")
        
        # Initialize trajectory
        self.trajectory_steps: List[TrajectoryStep] = []
        self.history: List[Dict[str, str]] = []

    def get_console_output(self) -> str:
        """Get captured console output (only available in quiet mode)."""
        if self.quiet and hasattr(self.console, 'export_text'):
            return self.console.export_text()
        return ""

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in messages."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    def model_query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens_per_call: int = 65536,
        max_token_limit: int = 65536,
    ) -> Tuple[Any, float]:
        """Query the LLM with the messages."""
        tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
        
        retries = 0
        messages_ = copy.deepcopy(messages)
        
        # Check token count
        total_tokens = self._count_tokens(messages_)
        self.logger.info(f"Total tokens in conversation: {total_tokens}")
        
        if total_tokens > max_token_limit:
            logger.warning(f"Total tokens: {total_tokens} > {max_token_limit}")
            raise ValueError(f"Total tokens: {total_tokens} > {max_token_limit}")
        
        # For locally hosted models with custom base URL, keep the API key from env
        # (some servers require a dummy key even if not used for auth)
        
        start_time = time.time()
        response = None
        
        while retries < self.max_retries:
            try:
                timeout = openai_timeout(default=1200)
                response = create_chat_completion(
                    model=self.llm_name,
                    tools=tools,
                    messages=messages_,
                    temperature=temperature,
                    timeout=timeout,
                    base_url=self.llm_base_url,
                    max_tokens=max_tokens_per_call,
                    extra_body=self.extra_body,
                )

                self.logger.info(f"LLM query complete")
                
                if self.save_openai_response and self.output_dir:
                    self._save_openai_response(messages_, response, temperature, max_tokens_per_call)
                
                break
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                
                # Check if it's a token limit error
                if "token" in error_msg.lower() and ("exceed" in error_msg.lower() or "limit" in error_msg.lower() or "maximum" in error_msg.lower()):
                    self.logger.warning(f"⚠️ Token limit error detected, attempting to handle...")
                    
                    # Parse error message:
                    # "maximum context length of 163840 tokens. You requested a total of 179069 tokens: 113533 tokens from the input messages and 65536 tokens for the completion"
                    context_match = re.search(r'(?:maximum|context)[^\d]*(\d+)\s*tokens', error_msg, re.IGNORECASE)
                    input_match = re.search(r'(\d+)\s*tokens?\s*(?:from\s*(?:the\s*)?(?:input|messages?)|in\s*(?:the\s*)?(?:input|prompt|messages?))', error_msg, re.IGNORECASE)
                    
                    max_context = int(context_match.group(1)) if context_match else None
                    current_input_tokens = int(input_match.group(1)) if input_match else None
                    
                    # Minimum completion tokens to preserve
                    min_completion_tokens = 8192
                    
                    self.logger.warning(f"📊 Parsed error: max_context={max_context}, input={current_input_tokens}, max_tokens_per_call={max_tokens_per_call}")
                    
                    if max_context and current_input_tokens:
                        # Strategy 1: Reduce max_tokens (completion tokens)
                        available_for_completion = max_context - current_input_tokens - 100  # 100 buffer
                        
                        if available_for_completion >= min_completion_tokens:
                            new_max_tokens = min(available_for_completion, max_tokens_per_call)
                            if new_max_tokens < max_tokens_per_call:
                                self.logger.warning(f"📉 Token limit exceeded, reducing max_tokens: {max_tokens_per_call} -> {new_max_tokens} (input {current_input_tokens}, context {max_context})")
                                max_tokens_per_call = new_max_tokens
                                continue
                        
                        # Not enough space even with min_completion_tokens
                        self.logger.warning(f"⚠️ Input tokens ({current_input_tokens}) too large, cannot fit in context ({max_context}) with min completion space")
                        raise
                    else:
                        # Cannot parse complete info, try reducing max_tokens
                        if max_tokens_per_call > min_completion_tokens:
                            new_max_tokens = max(int(max_tokens_per_call * 0.5), min_completion_tokens)
                            self.logger.warning(f"📉 Token limit exceeded (incomplete parse), reducing max_tokens: {max_tokens_per_call} -> {new_max_tokens}")
                            max_tokens_per_call = new_max_tokens
                            continue
                        else:
                            self.logger.warning(f"⚠️ max_tokens already at minimum ({min_completion_tokens}), cannot reduce further")
                            raise
                
                retries += 1
                
                if "RateLimitError" in str(e):
                    time.sleep(60)
                
                if retries >= self.max_retries:
                    raise e
        
        exec_time = time.time() - start_time
        return response, exec_time

    def _save_openai_response(self, messages: List[Dict], response, temperature: float, max_tokens: int):
        """Save OpenAI request and response to output_dir for debugging."""
        try:
            self.llm_call_count += 1
            save_path = self.output_dir
            os.makedirs(save_path, exist_ok=True)

            request_data = {
                "call_id": self.llm_call_count,
                "model": self.llm_name,
                "base_url": self.llm_base_url,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": self.extra_body,
            }
            request_file = os.path.join(save_path, f"request_{self.llm_call_count:03d}.json")
            with open(request_file, "w", encoding="utf-8") as f:
                json.dump(request_data, f, indent=2, ensure_ascii=False)

            response_data = {
                "call_id": self.llm_call_count,
                "response": response.model_dump() if hasattr(response, "model_dump") else response.to_dict() if hasattr(response, "to_dict") else str(response),
            }
            response_file = os.path.join(save_path, f"response_{self.llm_call_count:03d}.json")
            with open(response_file, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved OpenAI request/response #{self.llm_call_count} to {save_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save OpenAI response: {e}")

    def parse_response(self, response) -> Tuple[str, Action, Optional[str]]:
        """Parse the response from the LLM.
        
        Returns:
            Tuple of (thought, action, tool_call_id)
        """
        thought = response.choices[0].message.content
        if not thought:
            thought = ""
        
        tool_call_id = None
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id  # Use the actual tool_call_id from response
            action = Action(function_name=function_name, parameters=parameters)
        except Exception:
            action = Action(function_name="", parameters={})
        
        return thought, action, tool_call_id

    def run(
        self,
        runtime,  # DockerRuntime
        problem_statement: str,
        max_steps: int = 30,
        max_token_limit: int = 65536,
        max_tokens_per_call: int = 65536,
        temperature: float = 1.0,
    ) -> Trajectory:
        """
        Run the agent on the task.
        
        Args:
            runtime: DockerRuntime instance
            problem_statement: The task description
            max_steps: Maximum number of steps
            max_token_limit: Maximum tokens for the conversation
            max_tokens_per_call: Maximum tokens per LLM call
            temperature: Temperature for LLM
        
        Returns:
            Trajectory object with all steps
        """
        self.reset()
        self.logger.info(f"Starting agent run:")
        self.logger.info(f"max_steps={max_steps}")
        self.logger.info(f"max_token_limit={max_token_limit}")
        self.logger.info(f"max_tokens_per_call={max_tokens_per_call}")
        self.logger.info(f"temperature={temperature}")
        
        # Prepare system prompt
        system_prompt = self.system_prompt_template.format(
            command_docs="",
            demo="",
        )
        
        # Prepare user prompt using instance_prompt template if available
        if self.instance_prompt_template:
            user_prompt = self.instance_prompt_template.format(
                problem_statement=problem_statement,
            )
        else:
            user_prompt = problem_statement
        
        # Initialize conversation
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Print prompts
        self.console.print(Panel(
            escape(system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt),
            title="[bold cyan]SYSTEM PROMPT[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        ))
        self.console.print(Panel(
            escape(user_prompt[:2000] + "..." if len(user_prompt) > 2000 else user_prompt),
            title="[bold cyan]USER PROMPT[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        ))
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Pretty step header
            self.console.print()
            self.console.rule(f"[bold blue]Step {step_count + 1}/{max_steps}[/bold blue]", style="blue")
            
            # Add step count message to history (R2E style: remaining = max - current)
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                step_msg = f"Steps Remaining: {steps_remaining}"
            else:
                step_msg = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1]["content"] += f"\n{step_msg}"
            self.logger.info(step_msg)
            
            # Create messages for this step
            messages = self.history.copy()
            
            # Query LLM
            try:
                response, exec_time = self.model_query(
                    messages=messages,
                    temperature=temperature,
                    max_token_limit=max_token_limit,
                    max_tokens_per_call=max_tokens_per_call,
                )
            except Exception as e:
                self.logger.error(f"LLM query failed: {e}")
                break
            
            # Extract reasoning_content from response
            reasoning_content = ""
            try:
                message = response.choices[0].message
                if hasattr(message, 'provider_specific_fields') and message.provider_specific_fields:
                    reasoning_content = message.provider_specific_fields.get('thinking') or message.provider_specific_fields.get('reasoning_content') or ""
            except Exception as e:
                self.logger.warning(f"fail to extract reasoning_content: {e}")

            # Pretty print reasoning_content if present
            if reasoning_content:
                thought_display = reasoning_content[:2000] + "..." if len(reasoning_content) > 2000 else reasoning_content
                self.console.print(Panel(
                    escape(thought_display),
                    title="[bold magenta]🧠 REASONING CONTENT[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                ))

            # Parse response
            thought, action, tool_call_id = self.parse_response(response)
            
            # Pretty print thought
            if thought:
                thought_display = thought[:1000] + "..." if len(thought) > 1000 else thought
                self.console.print(Panel(
                    escape(thought_display),
                    title="[bold magenta]💭 THOUGHT[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                ))
            
            # Pretty print action
            action_text = f"[bold]{action.function_name}[/bold]"
            if action.parameters:
                params_str = json.dumps(action.parameters, indent=2, ensure_ascii=False)
                if len(params_str) > 300:
                    params_str = params_str[:300] + "..."
                action_text += f"\n{escape(params_str)}"
            self.console.print(Panel(
                action_text,
                title="[bold yellow]⚡ ACTION[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            ))
            
            # Execute action
            observation = self._execute_action(action, runtime)
            
            # Pretty print observation
            obs_display = observation[:800] + "..." if len(observation) > 800 else observation
            self.console.print(Panel(
                escape(obs_display),
                title="[bold green]👁 OBSERVATION[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))
            
            # Record step
            step_record = TrajectoryStep(
                thought=thought,
                reasoning_content=reasoning_content,
                action=action.to_dict(),
                observation=observation,
                metadata={"step": step_count + 1, "exec_time": exec_time},
            )
            self.trajectory_steps.append(step_record)
            
            # Update history with original assistant message (preserves reasoning_content etc.)
            assistant_msg = response.choices[0].message
            if hasattr(assistant_msg, 'model_dump'):
                assistant_msg_dict = assistant_msg.model_dump(exclude_none=True)
            elif hasattr(assistant_msg, 'to_dict'):
                assistant_msg_dict = assistant_msg.to_dict()
            else:
                assistant_msg_dict = dict(assistant_msg)
            # print(assistant_msg_dict)
            self.history.append(assistant_msg_dict)
            
            # Add tool result or CONTINUE_MSG
            if action.function_name and tool_call_id:
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": observation,
                })
            elif not action.function_name:
                # Model didn't use tool_calls, send CONTINUE_MSG as user message
                self.history.append({
                    "role": "user",
                    "content": observation,  # This is CONTINUE_MSG
                })
            
            # Check if done
            if action.function_name == "submit":
                done = True
                self.logger.info("Agent submitted answer")
            
            # Increment step count at end of loop (R2E style)
            step_count += 1
        
        # Create trajectory
        trajectory = Trajectory(
            problem_statement=problem_statement,
            steps=self.trajectory_steps,
            metadata={"total_steps": step_count},
        )
        
        return trajectory

    def _execute_action(self, action: Action, runtime) -> str:
        """Execute an action in the runtime."""
        from .observation import Observation
        
        if not action.function_name or not isinstance(action.parameters, dict):
            # Use Observation class to return CONTINUE_MSG
            # This handles: empty function_name, or malformed parameters (e.g., str instead of dict)
            obs = Observation(bash_output="", error_code=0, action=action)
            return str(obs)
        
        if action.function_name == "execute_bash":
            command = action.parameters.get("command", "")
            if not command:
                return "Error: No command provided."
            # Use demux_run to get separate stdout and stderr
            stdout, stderr, exit_code = runtime.demux_run(command)
            
            obs = Observation(
                bash_output=stdout + stderr,  # Combined for backward compatibility
                error_code=exit_code,
                action=action,
                stdout=stdout,
                stderr=stderr,
            )
            return str(obs)
        
        elif action.function_name == "str_replace_editor":
            result = self._execute_str_replace_editor(action, runtime)
            obs = Observation(bash_output=result, error_code=0, action=action)
            return str(obs)
        
        elif action.function_name == "submit":
            obs = Observation(bash_output="", error_code=0, action=action)
            return str(obs)
        
        else:
            return f"Unknown action: {action.function_name}"

    def _execute_str_replace_editor(self, action: Action, runtime) -> str:
        """Execute str_replace_editor action."""
        SNIPPET_LINES = 4
        MAX_RESPONSE_LEN = 10000
        TRUNCATED_MESSAGE = (
            "<response clipped><NOTE>To save on context only part of this file has been "
            "shown to you. You should retry this tool after you have searched inside the file "
            "with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
        )
        
        def maybe_truncate(content: str, truncate_after: int = MAX_RESPONSE_LEN) -> str:
            if not truncate_after or len(content) <= truncate_after:
                return content
            return content[:truncate_after] + TRUNCATED_MESSAGE
        
        def make_output(file_content: str, file_descriptor: str, init_line: int = 1) -> str:
            """Format file content with line numbers like cat -n."""
            file_content = maybe_truncate(file_content)
            file_content = file_content.expandtabs()
            lines = file_content.split("\n")
            numbered = "\n".join(f"{i + init_line:6}\t{line}" for i, line in enumerate(lines))
            return f"Here's the result of running `cat -n` on {file_descriptor}:\n{numbered}\n"
        
        params = action.parameters
        command = params.get("command", "")
        path = params.get("path", "")
        
        if not command or not path:
            return "Error: command and path are required."
        
        if command == "view":
            # First check if path is a directory or file
            check_cmd = f"test -d {path} && echo 'dir' || (test -f {path} && echo 'file' || echo 'notfound')"
            path_type, _ = runtime.run(check_cmd)
            path_type = path_type.strip()
            
            if path_type == "dir":
                # List directory contents up to 2 levels deep (only .py and .rst files, like R2E-Gym)
                cmd = f"find {path} -maxdepth 2 -not -path '*/.*' \\( -type d -o -name '*.py' -o -name '*.rst' \\) | head -100"
                output, _ = runtime.run(cmd)
                if output:
                    return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden:\n{output}"
                return f"(empty directory: {path})"
            elif path_type == "file":
                # Only allow viewing .py and .rst files (like R2E-Gym)
                if not (path.endswith('.py') or path.endswith('.rst')):
                    return f"ERROR: Viewing non-Python files is disallowed for saving context. File '{path}' is not a .py or .rst file."
                
                view_range = params.get("view_range")
                # Read full file first
                file_content, _ = runtime.run(f"cat {path}")
                if not file_content:
                    return f"(empty file: {path})"
                
                file_content = file_content.expandtabs()
                lines = file_content.split('\n')
                total_lines = len(lines)
                
                if view_range and len(view_range) == 2:
                    start, end = view_range
                    # Validate range
                    if not (1 <= start <= total_lines):
                        return f"Error: Invalid view_range {view_range}: start line must be in [1, {total_lines}]"
                    if end != -1 and (end < start or end > total_lines):
                        return f"Error: Invalid view_range {view_range}: end must be >= start and <= {total_lines}, or -1 to view until end."
                    
                    # Slice lines (1-based indexing)
                    if end == -1:
                        sliced_lines = lines[start-1:]
                    else:
                        sliced_lines = lines[start-1:end]
                    
                    numbered = "\n".join(f"{i + start:6}\t{line}" for i, line in enumerate(sliced_lines))
                else:
                    # Show full file
                    start = 1
                    numbered = "\n".join(f"{i + 1:6}\t{line}" for i, line in enumerate(lines))
                
                result = f"Here's the result of running `cat -n` on {path}:\n{numbered}"
                return maybe_truncate(result)
            else:
                return f"Error: Path not found: {path}"
        
        elif command == "create":
            file_text = params.get("file_text", "")
            # Convert to string in case model returns non-string (e.g., int)
            file_text = str(file_text)
            # Escape the text for shell
            escaped_text = file_text.replace("'", "'\"'\"'")
            cmd = f"mkdir -p $(dirname {path}) && echo '{escaped_text}' > {path}"
            output, exit_code = runtime.run(cmd)
            if "Error" in str(exit_code):
                return f"Error creating file: {output}"
            
            # Return file content for review (like R2E-Gym)
            success_msg = f"File created at {path}. "
            success_msg += make_output(file_text, str(path))
            success_msg += "Review the file and make sure that it is as expected. Edit the file if necessary."
            return success_msg
        
        elif command == "str_replace":
            old_str = params.get("old_str", "")
            new_str = params.get("new_str", "")
            
            if not old_str:
                return "Error: old_str is required for str_replace."
            
            # Read the file
            file_content, _ = runtime.run(f"cat {path}")
            if not file_content:
                return f"Error: Could not read file {path}"
            
            # Expand tabs (like R2E-Gym)
            file_content = file_content.expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs() if new_str else ""
            
            # Check occurrences
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                return f"Error: No occurrences of old_str found in {path} for replacement."
            if occurrences > 1:
                return f"Error: Multiple occurrences ({occurrences}) of old_str found in {path}. Please ensure it is unique before using str_replace."
            
            # Find replacement line for snippet
            replacement_line = file_content.split(old_str)[0].count("\n")
            
            # Replace
            new_content = file_content.replace(old_str, new_str if new_str else "", 1)
            
            # Write back
            escaped_content = new_content.replace("'", "'\"'\"'")
            cmd = f"echo '{escaped_content}' > {path}"
            _, exit_code = runtime.run(cmd)
            
            if "Error" in str(exit_code):
                return f"Error writing file: {path}"
            
            # Return snippet around the change for review
            new_lines = new_content.split("\n")
            start_line = max(0, replacement_line - SNIPPET_LINES)
            end_line = replacement_line + SNIPPET_LINES + (new_str or "").count("\n")
            snippet = "\n".join(new_lines[start_line:end_line + 1])
            
            success_msg = f"The file {path} has been edited. "
            success_msg += make_output(snippet, f"a snippet of {path}", start_line + 1)
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
            return success_msg
        
        elif command == "insert":
            insert_line = params.get("insert_line", 0)
            new_str = params.get("new_str", "")
            
            if not new_str:
                return "Error: new_str is required for insert."
            
            # Read the file
            file_content, _ = runtime.run(f"cat {path}")
            # Expand tabs (like R2E-Gym)
            file_content = file_content.expandtabs() if file_content else ""
            new_str = new_str.expandtabs()
            lines = file_content.split('\n') if file_content else []
            
            # Validate insert_line
            if insert_line < 0 or insert_line > len(lines):
                return f"Error: Invalid insert_line {insert_line}. Must be in [0, {len(lines)}]."
            
            # Insert
            new_str_lines = new_str.split("\n")
            new_lines = lines[:insert_line] + new_str_lines + lines[insert_line:]
            new_content = '\n'.join(new_lines)
            
            # Write back
            escaped_content = new_content.replace("'", "'\"'\"'")
            cmd = f"echo '{escaped_content}' > {path}"
            _, exit_code = runtime.run(cmd)
            
            if "Error" in str(exit_code):
                return f"Error writing file: {path}"
            
            # Return snippet around the inserted content for review
            snippet_lines = (
                lines[max(0, insert_line - SNIPPET_LINES):insert_line]
                + new_str_lines
                + lines[insert_line:insert_line + SNIPPET_LINES]
            )
            snippet = "\n".join(snippet_lines)
            
            success_msg = f"The file {path} has been edited. "
            success_msg += make_output(snippet, "a snippet of the edited file", max(1, insert_line - SNIPPET_LINES + 1))
            success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
            return success_msg
        
        else:
            return f"Unknown command: {command}"
