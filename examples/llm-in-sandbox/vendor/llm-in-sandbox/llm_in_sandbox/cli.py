#!/usr/bin/env python
# Copyright (c) Microsoft. All rights reserved.

"""
LLM-in-Sandbox CLI - Run LLM agents in local Docker containers
"""
import os
import sys
import json
import yaml
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Optional
from importlib import resources
import shutil
import traceback

import docker
import fire
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

from .docker_runtime import DockerRuntime, LocalRuntime
from .agent import Agent, AgentArgs, get_logger
from .trajectory import Trajectory



# Rich console
console = Console()

# Default Docker image name and version
DEFAULT_DOCKER_IMAGE = "cdx123/llm-in-sandbox:v0.1"
SETTINGS_ENV_VAR = "LLM_IN_SANDBOX_CONFIG"
DEFAULT_SETTINGS_LOCATIONS = [
    Path.cwd() / "llm-in-sandbox.yaml",
    Path.cwd() / "llm_in_sandbox.yaml",
    Path.home() / ".llm-in-sandbox" / "config.yaml",
    Path.home() / ".llm-in-sandbox.yaml",
]


def get_default_config_path() -> Path:
    """Get the default prompt config file path."""
    # Try to get from package resources
    try:
        with resources.files("llm_in_sandbox.config") as config_dir:
            return Path(config_dir) / "general.yaml"
    except (TypeError, FileNotFoundError):
        # Fallback to relative path
        return Path(__file__).parent / "config" / "general.yaml"


def load_runtime_settings(explicit_path: Optional[str] = None):
    """Load CLI defaults (llm_name/llm_base_url) from YAML config files."""
    candidates = []
    seen = set()

    def _add_candidate(candidate):
        if not candidate:
            return
        path = Path(candidate).expanduser()
        if path in seen:
            return
        seen.add(path)
        candidates.append(path)

    _add_candidate(explicit_path)
    _add_candidate(os.environ.get(SETTINGS_ENV_VAR))
    for default_path in DEFAULT_SETTINGS_LOCATIONS:
        _add_candidate(default_path)

    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "r") as f:
                data = yaml.safe_load(f) or {}
            return data, candidate

    return {}, None


def find_dockerfile() -> Optional[Path]:
    """Find the Dockerfile for building the default image."""
    # Try 1: Development mode - docker/ is sibling to llm_in_sandbox/
    script_dir = Path(__file__).parent
    dev_docker_dir = script_dir.parent / "docker"
    if (dev_docker_dir / "Dockerfile").exists():
        return dev_docker_dir / "Dockerfile"
    
    # Try 2: Installed mode - check sys.prefix for shared data
    installed_docker_dir = Path(sys.prefix) / "share" / "llm-in-sandbox" / "docker"
    if (installed_docker_dir / "Dockerfile").exists():
        return installed_docker_dir / "Dockerfile"
    
    return None


def ensure_docker_image(image_name: str, logger) -> bool:
    """Check if Docker image exists. If not, try to pull it from Docker Hub."""
    client = docker.from_env()
    
    try:
        client.images.get(image_name)
        return True  # Image exists
    except docker.errors.ImageNotFound:
        # Try to pull from Docker Hub
        console.print(Panel.fit(
            f"[yellow]🐳 Docker image '{image_name}' not found locally.[/yellow]\n"
            f"[dim]Attempting to pull from Docker Hub...[/dim]",
            border_style="yellow",
        ))
        try:
            logger.info(f"Pulling Docker image '{image_name}' from Docker Hub...")
            client.images.pull(image_name)
            console.print(Panel.fit(
                f"[green]✅ Successfully pulled Docker image '{image_name}'![/green]",
                border_style="green",
            ))
            return True
        except docker.errors.APIError as e:
            logger.warning(f"Failed to pull image: {e}")
            return False


def build_docker_image(
    image_name: str = DEFAULT_DOCKER_IMAGE,
    force: bool = False,
):
    """
    Build the Docker image for LLM-in-Sandbox.
    
    This command builds the default Docker image used by the agent.
    You only need to run this once before using the 'run' command.
    
    Args:
        image_name: Docker image name to build (default: llm-in-sandbox:v0.1)
        force: Force rebuild even if image already exists
    
    Example:
        llm-in-sandbox build
        llm-in-sandbox build --force  # Force rebuild
        llm-in-sandbox build --image_name my-custom-image:v1
    """
    logger = get_logger("llm-in-sandbox")
    client = docker.from_env()
    
    # Check if image already exists
    if not force:
        try:
            client.images.get(image_name)
            console.print(Panel.fit(
                f"[green]✅ Docker image '{image_name}' already exists![/green]\n"
                f"[dim]Use --force to rebuild[/dim]",
                border_style="green",
            ))
            return
        except docker.errors.ImageNotFound:
            pass
    
    # Find Dockerfile
    dockerfile = find_dockerfile()
    if dockerfile is None:
        console.print(Panel.fit(
            f"[red]❌ Cannot find Dockerfile to build '{image_name}'[/red]\n"
            f"[dim]Please build manually: docker build -t {image_name} <path-to-dockerfile>[/dim]",
            border_style="red",
        ))
        sys.exit(1)
    
    # Build image
    console.print()
    console.print(Panel.fit(
        f"[yellow]🐳 Building Docker image '{image_name}'...[/yellow]\n"
        f"[dim]Dockerfile: {dockerfile}[/dim]",
        border_style="yellow",
    ))
    console.print()
    
    docker_dir = dockerfile.parent
    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", str(dockerfile), str(docker_dir)],
            check=True,
        )
        console.print()
        console.print(Panel.fit(
            f"[green]✅ Docker image '{image_name}' built successfully![/green]\n"
            f"[dim]You can now run: llm-in-sandbox run --query \"Your task\"[/dim]",
            border_style="green",
        ))
    except subprocess.CalledProcessError as e:
        console.print(Panel.fit(
            f"[red]❌ Failed to build Docker image (exit code {e.returncode})[/red]",
            border_style="red",
        ))
        sys.exit(1)
    except FileNotFoundError:
        console.print(Panel.fit(
            f"[red]❌ Docker not found. Please install Docker first.[/red]",
            border_style="red",
        ))
        sys.exit(1)


def load_prompt_config(config_path: str) -> dict:
    """Load prompt configuration from a yaml file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_agent_query(
    query: str,
    llm_name: Optional[str] = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    runtime_type: str = 'docker', # docker or local
    max_steps: int = 100,
    temperature: float = 1.0,
    max_token_limit: int = 65536,
    max_tokens_per_call: int = 65536,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    prompt_config: Optional[str] = None,
    save_openai_response: bool = False,
    extra_body: Optional[str] = None,
    settings: Optional[str] = None,
):
    """
    Run an LLM agent in a Docker container to complete a task.
    
    Args:
        query: The task description / problem statement
        llm_name: LLM model name 
        docker_image: Docker image to use (default: cdx123/llm-in-sandbox:v0.1)
        max_steps: Maximum number of steps (default: 100)
        temperature: Temperature for LLM (default: 1.0)
        max_token_limit: Maximum token limit for the whole trajectory
        max_tokens_per_call: Maximum tokens per LLM API call
        input_dir: Local directory to copy into container at container's {input_dir}
        output_dir: Local directory to save container's {output_dir} contents
        llm_base_url: LLM API base URL (default: from LLM_BASE_URL env var)
        api_key: API key for the LLM service (default: from OPENAI_API_KEY env var)
        prompt_config: Path to yaml file with system_prompt and instance_prompt (default: ./config/general.yaml)
        save_openai_response: Whether to save full OpenAI responses
        extra_body: Extra JSON body to include in LLM API calls, e.g., '{"chat_template_kwargs": {"thinking": True}}'
        settings: Optional path to a YAML file that provides defaults such as llm_name and llm_base_url
    
    Returns:
        Trajectory object with all steps and results
    """
    logger = get_logger("llm-in-sandbox")

    runtime_settings, runtime_settings_path = load_runtime_settings(settings)
    if runtime_settings_path:
        logger.info(f"Loaded runtime settings from: {runtime_settings_path}")

    def _with_setting(value, key):
        if value in (None, ""):
            return runtime_settings.get(key)
        return value

    llm_name = _with_setting(llm_name, "llm_name")
    llm_base_url = _with_setting(llm_base_url, "llm_base_url")
    api_key = _with_setting(api_key, "api_key")
    prompt_config = _with_setting(prompt_config, "prompt_config")

    if not llm_name:
        raise ValueError(
            "llm_name is required. Provide --llm_name or set it in a settings YAML file."
        )
    
    # Set API key based on model type
    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)
        os.environ["ANTHROPIC_API_KEY"] = str(api_key)
        os.environ["AZURE_OPENAI_API_KEY"] = str(api_key)
    else:
        # Set dummy key if not provided (some servers don't need auth)
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "dummy"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = "dummy"
    
    # Load prompt config from yaml (use default if not provided)
    config_path = prompt_config if prompt_config else get_default_config_path()
    if Path(config_path).exists():
        logger.info(f"Loading prompt config from: {config_path}")
        config = load_prompt_config(config_path)
        system_prompt = config.get("system_prompt", "")
        instance_prompt = config.get("instance_prompt", "")
        # Get container paths from config (defaults: /testbed, /testbed/input, /testbed/output)
        working_dir = config.get("working_dir", "/testbed")
        container_input_dir = config.get("input_dir", "/testbed/input")
        container_output_dir = config.get("output_dir", "/testbed/output")
        # Replace placeholders in prompts
        system_prompt = system_prompt.replace("{input_dir}", container_input_dir).replace("{output_dir}", container_output_dir).replace("{working_dir}", working_dir)
        instance_prompt = instance_prompt.replace("{input_dir}", container_input_dir).replace("{output_dir}", container_output_dir).replace("{working_dir}", working_dir)
    else:
        raise FileNotFoundError(f"Prompt config not found: {config_path}")
    
    # Auto-add openai/ prefix for custom LLM endpoints
    if not llm_name.startswith(("openai/", "anthropic/", "azure/", "hosted_vllm/")):
        llm_name = f"openai/{llm_name}"
        logger.info(f"Auto-added 'openai/' prefix to model: {llm_name}")
    
    # Set up output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path.cwd() / "output" / timestamp
    else:
        output_dir = Path(output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set LLM base URL
    if llm_base_url:
        os.environ["LLM_BASE_URL"] = llm_base_url
    
    # Ensure Docker image exists (auto-build if default image)
    if not ensure_docker_image(docker_image, logger):
        console.print(Panel.fit(
            f"[red]❌ Docker image '{docker_image}' not found![/red]\n"
            f"[dim]Please build it first: llm-in-sandbox build[/dim]",
            border_style="red",
        ))
        sys.exit(1)
    

    if runtime_type == "docker":
        # Initialize Docker runtime
        logger.info(f"Starting Docker container...")
        runtime = DockerRuntime(
            docker_image=docker_image,
            repo_path=working_dir,
            logger=logger,
        )
    elif runtime_type == "local":
        runtime = LocalRuntime(
            repo_path=working_dir,
            logger=logger,
        )
    else:
        raise NotImplementedError
        
    # Copy input files to container if provided
    if input_dir and os.path.isdir(input_dir):
        logger.info(f"Copying input files from {input_dir} to container's {container_input_dir}")
        runtime.copy_dir_to_container(input_dir, container_input_dir)
    
    def _fix_string_bools(obj):
        """Recursively convert string 'true'/'false' to bool True/False."""
        if isinstance(obj, dict):
            return {k: _fix_string_bools(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_fix_string_bools(item) for item in obj]
        elif isinstance(obj, str):
            if obj.lower() == 'true':
                return True
            elif obj.lower() == 'false':
                return False
        return obj
    
    try:
        # Handle extra_body: could be dict (from fire) or JSON string
        extra_body_dict = None
        if extra_body:
            if isinstance(extra_body, dict):
                extra_body_dict = extra_body
            elif isinstance(extra_body, str):
                try:
                    extra_body_dict = json.loads(extra_body)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extra_body JSON: {e}")
                    raise ValueError(f"Invalid extra_body JSON: {extra_body}")
            # Fix string bools like 'true' -> True
            extra_body_dict = _fix_string_bools(extra_body_dict)
            logger.info(f"Using extra_body: {extra_body_dict}")
        
        # Initialize agent
        agent_args = AgentArgs(
            system_prompt=system_prompt,
            instance_prompt=instance_prompt,
            llm_name=llm_name,
            llm_base_url=llm_base_url or os.environ.get("LLM_BASE_URL"),
            save_openai_response=save_openai_response,
            output_dir=str(output_dir),
            extra_body=extra_body_dict,
        )
        agent = Agent(args=agent_args, logger=logger)
        
        # Run agent
        logger.info(f"Starting agent...")
        trajectory = agent.run(
            runtime=runtime,
            problem_statement=query,
            max_steps=max_steps,
            temperature=temperature,
            max_token_limit=max_token_limit,
            max_tokens_per_call=max_tokens_per_call,
        )
        
        # Copy output files from container to files/ subdirectory
        files_dir = output_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Copying output files from container's {container_output_dir} to {files_dir}")
        try:
            runtime.copy_from_container(container_output_dir, str(files_dir))
        except Exception as e:
            logger.warning(f"Could not copy output from container: {e}")
        
        # Save trajectory
        trajectory_file = output_dir / "trajectory.json"
        
        with open(trajectory_file, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Pretty completion banner
        console.print()
        console.print(Panel.fit(
            f"[bold green]✅ Agent completed in {len(trajectory.steps)} steps[/bold green]",
            border_style="green",
        ))
        
        # Print output paths
        console.print()
        console.print("[bold]📦 Output saved to:[/bold]")
        paths_table = Table(show_header=False, box=None, padding=(0, 2))
        paths_table.add_column("Label", style="bold blue")
        paths_table.add_column("Path", style="white")
        paths_table.add_row("Agent output files", str(files_dir))
        paths_table.add_row("Execution trajectory", str(trajectory_file))
        console.print(paths_table)
        
        # Print answer.txt if exists
        answer_file = files_dir / "answer.txt"
        if answer_file.exists():
            answer_content = answer_file.read_text().strip()
            if answer_content:
                console.print()
                console.print(Panel(
                    f"{escape(answer_content)}\n\n[dim]📁 {answer_file}[/dim]",
                    title="[bold cyan]📄 Answer[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                ))
        
    finally:
        # Clean up
        logger.info(f"Cleaning up Docker container...")
        runtime.close()


def run_benchmark(
    task: str,
    llm_name: str = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    max_steps: int = 100,
    temperature: float = None,
    max_token_limit: int = 65536,
    max_tokens_per_call: int = 65536,
    max_response_len: int = 65536,
    output_dir: str = None,
    llm_base_url: str = None,
    api_key: str = None,
    extra_body: str = None,
    settings: str = None,
    num_workers: int = None,
    start_id: int = None,
    end_id: int = None,
    mode: str = "llm-in-sandbox",
    save_openai_response: bool = False,
):
    """
    Run benchmark on a specific task.
    
    Args:
        mode: "llm-in-sandbox" (default) or "llm" (vanilla LLM without sandbox)
    
    Example:
        llm-in-sandbox benchmark --task math --llm_name openai/gpt-5 --num_workers 4
        llm-in-sandbox benchmark --task math --llm_name openai/gpt-5 --mode llm
    """
    from llm_in_sandbox.benchmark.runner import run_benchmark as _run_benchmark
    
    logger = get_logger("llm-in-sandbox")
    runtime_settings, _ = load_runtime_settings(settings)
    
    # Resolve parameters: CLI args > env vars > settings file > defaults
    llm_name = llm_name or os.environ.get("LLM_NAME") or runtime_settings.get("llm_name")
    llm_base_url = llm_base_url or os.environ.get("LLM_BASE_URL") or runtime_settings.get("llm_base_url")
    api_key = api_key or os.environ.get("LLM_API_KEY") or runtime_settings.get("api_key")
    if temperature is None:
        temperature = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    if num_workers is None:
        num_workers = int(os.environ.get("LLM_NUM_WORKERS", "1"))
    
    if not llm_name:
        raise ValueError("llm_name is required")
    
    # Set API keys (use placeholder for local vLLM)
    api_key = api_key or "sk-placeholder"
    os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_API_KEY"] = str(api_key)
    if llm_base_url:
        os.environ["LLM_BASE_URL"] = llm_base_url
    if not llm_name.startswith(("openai/", "anthropic/", "azure/", "hosted_vllm/")):
        llm_name = f"openai/{llm_name}"
    
    # Setup output directory: output/{timestamp}_{task}_{llm_name}_{mode}/
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "vanillaLLM" if mode == "llm" else "LLMinSandbox"
    llm_name_safe = llm_name.replace("/", "_")  # openai/qwen3_coder -> openai_qwen3_coder
    output_dir = Path(output_dir or Path.cwd() / "output") / f"{timestamp}_{task}_{llm_name_safe}_{mode_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate mode
    if mode not in ("llm", "llm-in-sandbox"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'llm' or 'llm-in-sandbox'")
    
    # Only check docker image for llm-in-sandbox mode
    if mode == "llm-in-sandbox":
        if not ensure_docker_image(docker_image, logger):
            console.print(f"[red]Docker image '{docker_image}' not found![/red]")
            sys.exit(1)
    
    def _fix_string_bools(obj):
        """Recursively convert string 'true'/'false' to bool True/False."""
        if isinstance(obj, dict):
            return {k: _fix_string_bools(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_fix_string_bools(item) for item in obj]
        elif isinstance(obj, str):
            if obj.lower() == 'true':
                return True
            elif obj.lower() == 'false':
                return False
        return obj
    
    # Handle extra_body: could be dict (from fire) or JSON string
    extra_body_dict = None
    if extra_body:
        if isinstance(extra_body, dict):
            extra_body_dict = extra_body
        elif isinstance(extra_body, str):
            try:
                extra_body_dict = json.loads(extra_body)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extra_body JSON: {e}")
                raise ValueError(f"Invalid extra_body JSON: {extra_body}")
        # Fix string bools like 'true' -> True
        extra_body_dict = _fix_string_bools(extra_body_dict)
        logger.info(f"Using extra_body: {extra_body_dict}")
    
    # Agent configuration (passed to subprocesses)
    agent_config = {
        "docker_image": docker_image,
        "llm_name": llm_name,
        "llm_base_url": llm_base_url or os.environ.get("LLM_BASE_URL"),
        "max_steps": max_steps,
        "temperature": temperature,
        "max_token_limit": max_token_limit,
        "max_tokens_per_call": max_tokens_per_call,
        "max_response_len": max_response_len,
        "extra_body": extra_body_dict,
        "save_openai_response": save_openai_response,
    }
    
    # Run benchmark
    results = _run_benchmark(
        task_name=task,
        agent_config=agent_config,
        output_dir=str(output_dir),
        num_workers=num_workers,
        start_id=start_id,
        end_id=end_id,
        mode=mode,
    )
    
    return results

def setup(data_folder_name, data_filename, data_index):
    '''
    Setup the {data_index}-th sample in /data/{data_folder_name}/{data_filename}.
    Put the sample json in /data/sample.json (remove answer).
    Setup input_files in /testbed/documents .
    '''
    try:
        with open(f"/data/{data_folder_name}/{data_filename}") as f:
            sample_list = json.load(f)
            sample = sample_list[data_index]

        extra_info = sample['extra_info']
        input_files = extra_info.get("input_files", "{}")
        input_files = json.loads(input_files)

        target_folder = "/testbed/documents"
        for filename, content in input_files.items():
            if content is None:
                continue
            if os.path.exists(target_folder) is False:
                os.makedirs(target_folder)
            target_path = os.path.join(target_folder, filename)
            # sometimes, filename can be like "folder1/folder2/file.txt", we need to create the intermediate folders
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

        # remove this folder to avoid data leak
        shutil.rmtree("/data")
        os.makedirs("/data")
        # remove ground-truth
        del sample['reward_model']
        if "ground_truth" in sample['extra_info']:
            del sample['extra_info']['ground_truth']
        with open("/data/sample.json", "w") as f:
            json.dump(sample, f)
    except Exception as e:
        print("Error during setup:", str(e))

def run_in_container():
    from llm_in_sandbox.benchmark.runner import load_task_config
    import yaml

    data_folder_name = os.environ['DATA_FOLDER_NAME']
    data_filename = os.environ['DATA_FILENAME']
    data_index = int(os.environ['DATA_INDEX'])
    setup(data_folder_name, data_filename, data_index)

    logger = get_logger("llm-in-sandbox")

    llm_name = os.environ["LLM_NAME"]
    llm_base_url = os.environ["LLM_BASE_URL"]
    api_key = os.environ["LLM_API_KEY"]
    temperature = float(os.environ["LLM_TEMPERATURE"])
    os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"] = str(api_key)
    max_steps = 30
    max_token_limit = 60000
    if "MAX_TOKENS_PER_CALL" in os.environ:
        max_tokens_per_call = int(os.environ["MAX_TOKENS_PER_CALL"])
    else:
        max_tokens_per_call = 20000

    with open("/data/sample.json") as f:
        sample = json.load(f)

    domain = sample['extra_info']['domain']
    domain = domain.replace("_mini", "")
    task_config = load_task_config(domain)

    system_prompt = task_config["system_prompt"]
    instance_prompt = task_config.get("instance_prompt", "")

    working_dir = '/testbed'
    output_dir = '/testbed'
    input_dir = "/testbed/documents"
    system_prompt = system_prompt.replace("{working_dir}", working_dir).replace("{input_dir}", input_dir).replace("{output_dir}", output_dir)
    instance_prompt = instance_prompt.replace("{working_dir}", working_dir).replace("{input_dir}", input_dir).replace("{output_dir}", output_dir)
            
    # Initialize agent
    agent_args = AgentArgs(
        system_prompt=system_prompt,
        instance_prompt=instance_prompt,
        llm_name=llm_name,
        llm_base_url=llm_base_url,
        save_openai_response=False,
        output_dir=output_dir,
        extra_body={},
    )
    agent = Agent(args=agent_args, logger=logger)
    
    try:
        # Run agent
        logger.info(f"Starting agent...")
        trajectory = agent.run(
            runtime=LocalRuntime(), # run in local
            problem_statement=sample['extra_info']['problem_statement'],
            max_steps=max_steps,
            temperature=temperature,
            max_token_limit=max_token_limit,
            max_tokens_per_call=max_tokens_per_call,
        )
    except Exception as e:
        print("Error while running agent: ", traceback.format_exc())

    # print answer to std
    ans_path = "/testbed/answer.txt"
    if os.path.exists(ans_path):
        with open(ans_path) as f:
            answer = f.read().strip()
        answer = answer[:2000]  # just in case it is too long
    else:
        answer = ""  # do not use N/A here, it will match "A"
    print("##########")  # note: #### is not enough, which cause performance degradation
    print(answer)
    print("##########")


def main():
    """Main entry point for CLI."""
    fire.Fire({
        "run": run_agent_query,
        "build": build_docker_image,
        "benchmark": run_benchmark,
        "run_in_container": run_in_container,
    })


if __name__ == "__main__":
    main()
