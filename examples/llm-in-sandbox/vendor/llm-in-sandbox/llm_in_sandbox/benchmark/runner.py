"""
Benchmark runner with parallel execution support.
"""

import atexit
import importlib.util
import json
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import yaml


@dataclass
class BenchmarkResult:
    """Result of a single benchmark problem."""
    problem_id: str
    score: float
    agent_answer: str
    ground_truth: str
    problem_statement: str = ""
    trajectory: List = field(default_factory=list)
    error: Optional[str] = None


def _cleanup_docker_containers(docker_image: str):
    """Clean up all Docker containers for the specified image."""
    try:
        import subprocess
        # First stop, then remove (force remove may not work on running containers)
        subprocess.run(
            f"docker ps -aq --filter 'ancestor={docker_image}' | xargs -r docker stop 2>/dev/null",
            shell=True,
        )
        subprocess.run(
            f"docker ps -aq --filter 'ancestor={docker_image}' | xargs -r docker rm -f 2>/dev/null",
            shell=True,
        )
        print(f"Cleaned up Docker containers for {docker_image}")
    except Exception as e:
        print(f"Error cleaning up containers: {e}")


def load_reward_function(task_name: str) -> Callable:
    """Load reward function from benchmark/{task_name}/reward.py"""
    benchmark_dir = Path(__file__).parent
    reward_path = benchmark_dir / task_name / "reward.py"
    
    if not reward_path.exists():
        raise FileNotFoundError(f"Reward function not found: {reward_path}")
    
    spec = importlib.util.spec_from_file_location("reward", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "compute_score"):
        raise AttributeError(f"reward.py must define compute_score function")
    
    return module.compute_score


def load_prompt_function(task_name: str) -> Callable:
    """Load prompt creation function from benchmark/{task_name}/vanilla_llm_prompt.py"""
    benchmark_dir = Path(__file__).parent
    prompt_path = benchmark_dir / task_name / "vanilla_llm_prompt.py"
    
    if not prompt_path.exists():
        # Default: just return problem_statement
        return lambda problem_data: problem_data['problem_statement']
    
    spec = importlib.util.spec_from_file_location("vanilla_llm_prompt", prompt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "create_prompt"):
        # Default: just return problem_statement
        return lambda problem_data: problem_data['problem_statement']
    
    return module.create_prompt


def load_task_config(task_name: str) -> dict:
    """Load task config from benchmark/{task_name}/config.yaml"""
    benchmark_dir = Path(__file__).parent
    config_path = benchmark_dir / task_name / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_from_config(task_config: dict):
    """Load dataset from HuggingFace or local JSON file."""
    from datasets import load_dataset
    
    dataset_name = task_config["dataset"]
    split = task_config.get("split", "test")
    
    if dataset_name == "json":
        # Load from local JSON file, resolve relative paths against benchmark dir
        data_files = task_config.get("data_files")
        if data_files and not Path(data_files).is_absolute():
            data_files = str(Path(__file__).parent / data_files)
        ds = load_dataset("json", data_files=data_files, split="train")
    else:
        # Load from HuggingFace
        config = task_config.get("config")
        ds = load_dataset(dataset_name, config, split=split)
    
    return ds


def create_agent_runner(
    docker_image: str,
    llm_name: str,
    llm_base_url: str,
    max_steps: int,
    temperature: float,
    max_token_limit: int,
    max_tokens_per_call: int,
    extra_body: dict = None,
    task_system_prompt: str = None,
    task_instance_prompt: str = None,
    save_openai_response: bool = False,
    working_dir: str = None,
    input_dir: str = None,
    output_dir: str = None,
    **kwargs,  # Ignore extra params like max_response_len (for vanilla LLM only)
) -> Callable:
    """
    Factory function to create an agent runner.
    Returns a function: (query, input_files, local_output_dir) -> (answer, trajectory, console_output)
    """
    # Import here to avoid circular imports
    import logging
    import tempfile
    import shutil
    from llm_in_sandbox.docker_runtime import DockerRuntime
    from llm_in_sandbox.agent import Agent, AgentArgs
    from llm_in_sandbox.cli import get_default_config_path, load_prompt_config
    
    # Construct answer_path from output_dir
    answer_path = f"{output_dir}/answer.txt"
    
    # Quiet logger - errors will be caught as exceptions
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.WARNING)
    
    def agent_runner(query: str, input_files: dict, local_output_dir: str) -> str:
        """Run agent on a single problem."""
        runtime = DockerRuntime(
            docker_image=docker_image,
            repo_path=working_dir,
            logger=logger,
        )
        
        # Register atexit cleanup in case process is killed
        atexit.register(runtime.close)
        
        # Copy input files using temp directory
        temp_dir = None
        if input_files:
            temp_dir = tempfile.mkdtemp()
            for filename, content in input_files.items():
                if content is None:
                    continue
                temp_path = Path(temp_dir) / filename
                temp_path.write_text(content)
            runtime.copy_dir_to_container(temp_dir, input_dir)
        
        try:
            # Use task-specific prompts if provided, otherwise use general.yaml defaults
            if task_system_prompt:
                system_prompt = task_system_prompt
                instance_prompt = task_instance_prompt or ""
            else:
                config_path = get_default_config_path()
                config = load_prompt_config(config_path)
                system_prompt = config.get("system_prompt", "")
                instance_prompt = config.get("instance_prompt", "")
            
            agent_args = AgentArgs(
                system_prompt=system_prompt,
                instance_prompt=instance_prompt,
                llm_name=llm_name,
                llm_base_url=llm_base_url,
                output_dir=local_output_dir,
                extra_body=extra_body,
                quiet=True,  # Capture console output in benchmark mode
                save_openai_response=save_openai_response,
            )
            agent = Agent(args=agent_args, logger=logger)
            
            trajectory = agent.run(
                runtime=runtime,
                problem_statement=query,
                max_steps=max_steps,
                temperature=temperature,
                max_token_limit=max_token_limit,
                max_tokens_per_call=max_tokens_per_call,
            )
            
            # Get captured console output
            console_output = agent.get_console_output()
            
            # Read answer directly from container
            answer = ""
            try:
                output, _ = runtime.run(f"cat {answer_path} 2>/dev/null || echo ''")
                answer = output.strip()
            except Exception:
                pass
            
            # Return answer, trajectory, and console output
            return answer, trajectory, console_output
            
        finally:
            # Unregister atexit since we're cleaning up normally
            atexit.unregister(runtime.close)
            runtime.close()
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    return agent_runner


def run_vanilla_llm(query: str, agent_config: dict) -> tuple:
    """
    Run vanilla LLM without sandbox.
    Returns: (answer, console_output)
    """
    import re
    import time
    from llm_in_sandbox.openai_client import create_chat_completion
    
    llm_name = agent_config["llm_name"]
    llm_base_url = agent_config.get("llm_base_url")
    temperature = agent_config.get("temperature", 1.0)
    # Use max_response_len for vanilla LLM
    max_tokens = agent_config.get("max_response_len", 65536)
    extra_body = agent_config.get("extra_body")
    
    # Token limit handling config
    max_retries = 5
    min_completion_tokens = 8192
    current_max_tokens = max_tokens
    
    for attempt in range(max_retries):
        try:
            response = create_chat_completion(
                model=llm_name,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                max_tokens=current_max_tokens,
                timeout=1800,
                base_url=llm_base_url,
                extra_body=extra_body,
            )
            answer = response.choices[0].message.content or ""
            console_output = f"Query:\n{query}\n\nResponse:\n{answer}\n"
            return answer, console_output
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ [DEBUG] attempt: {attempt+1}/{max_retries} OpenAI call failed:")
            print(f"   - Error type: {type(e).__name__}")
            print(f"   - Error message: {error_msg}")
            
            # Check if it's a token limit error
            if "token" in error_msg.lower() and ("exceed" in error_msg.lower() or "limit" in error_msg.lower() or "maximum" in error_msg.lower()):
                # Parse error message:
                # "maximum context length of 163840 tokens. You requested a total of 179069 tokens: 113533 tokens from the input messages and 65536 tokens for the completion"
                context_match = re.search(r'(?:maximum|context)[^\d]*(\d+)\s*tokens', error_msg, re.IGNORECASE)
                input_match = re.search(r'(\d+)\s*tokens?\s*(?:from\s*(?:the\s*)?(?:input|messages?)|in\s*(?:the\s*)?(?:input|prompt|messages?))', error_msg, re.IGNORECASE)
                
                max_context = int(context_match.group(1)) if context_match else None
                current_input_tokens = int(input_match.group(1)) if input_match else None
                
                print(f"📊 Parsed error: max_context={max_context}, input={current_input_tokens}, current_max_tokens={current_max_tokens}")
                
                if max_context and current_input_tokens:
                    # Strategy 1: Reduce max_tokens (completion tokens)
                    available_for_completion = max_context - current_input_tokens - 100  # 100 buffer
                    
                    if available_for_completion >= min_completion_tokens:
                        new_max_tokens = min(available_for_completion, current_max_tokens)
                        if new_max_tokens < current_max_tokens:
                            print(f"📉 Token limit exceeded, reducing max_tokens: {current_max_tokens} -> {new_max_tokens} (input {current_input_tokens}, context {max_context})")
                            current_max_tokens = new_max_tokens
                            continue
                    
                    # Not enough space
                    print(f"⚠️  Input tokens ({current_input_tokens}) too large, cannot fit in context ({max_context}) with min completion space")
                    raise
                else:
                    # Cannot parse complete info, try reducing max_tokens
                    if current_max_tokens > min_completion_tokens:
                        new_max_tokens = max(int(current_max_tokens * 0.5), min_completion_tokens)
                        print(f"📉 Token limit exceeded (incomplete parse), reducing max_tokens: {current_max_tokens} -> {new_max_tokens}")
                        current_max_tokens = new_max_tokens
                        continue
                    else:
                        print(f"⚠️  max_tokens already at minimum ({min_completion_tokens}), cannot reduce further")
                        raise
            else:
                # Non-token-limit error
                if "RateLimitError" in str(e):
                    print(f"Rate limit hit, sleeping 60s...")
                    time.sleep(60)
                    continue
                raise
    
    # Retries exhausted
    raise RuntimeError(f"Failed after {max_retries} retries")


def run_single_problem(args: dict) -> BenchmarkResult:
    """Run agent on a single problem and compute score. Works with ProcessPoolExecutor."""
    problem = args["problem"]
    agent_config = args["agent_config"]
    task_name = args["task_name"]
    prompt_config = args["prompt_config"]
    output_dir = args["output_dir"]
    logs_dir = args["logs_dir"]
    mode = args.get("mode", "llm-in-sandbox")
    
    problem_id = problem["id"]
    ground_truth = problem["ground_truth"]
    problem_statement = problem["problem_statement"]
    
    # Create per-problem output directory for OpenAI logs.
    save_openai_response = agent_config.get("save_openai_response", False)
    problem_output_dir = os.path.join(output_dir, "openai_logs", problem_id) if save_openai_response else None
    if problem_output_dir:
        os.makedirs(problem_output_dir, exist_ok=True)
    
    compute_score_func = load_reward_function(task_name)
    
    try:
        if mode == "llm":
            # Vanilla LLM mode - direct API call without sandbox
            # Use task-specific prompt function
            create_prompt_func = load_prompt_function(task_name)
            query = create_prompt_func(problem)
            agent_answer, console_output = run_vanilla_llm(query, agent_config)
            trajectory = []
        else:
            # LLM-in-Sandbox mode - use task-specific system prompt
            # Get container paths from config (raises KeyError if missing)
            working_dir = prompt_config.get("working_dir", "/testbed")
            input_dir = prompt_config.get("input_dir", "/testbed/documents")
            output_dir_config = prompt_config.get("output_dir", "/testbed")
            
            # Replace directory placeholders in prompts (use str.replace to preserve other placeholders like {problem_statement})
            system_prompt = prompt_config["system_prompt"]
            instance_prompt = prompt_config["instance_prompt"]
            system_prompt = system_prompt.replace("{working_dir}", working_dir).replace("{input_dir}", input_dir).replace("{output_dir}", output_dir_config)
            instance_prompt = instance_prompt.replace("{working_dir}", working_dir).replace("{input_dir}", input_dir).replace("{output_dir}", output_dir_config)
            
            run_agent_func = create_agent_runner(
                **agent_config,
                task_system_prompt=system_prompt,
                task_instance_prompt=instance_prompt,
                working_dir=working_dir,
                input_dir=input_dir,
                output_dir=output_dir_config,
            )
            input_files_raw = problem.get("input_files") or {}
            # Parse JSON string if needed (HuggingFace stores as string)
            if isinstance(input_files_raw, str):
                import json
                input_files = json.loads(input_files_raw) if input_files_raw else {}
            else:
                input_files = input_files_raw
            # Pass only problem_statement as query (system prompt is in agent config now)
            agent_answer, trajectory, console_output = run_agent_func(
                query=problem_statement,
                input_files=input_files,
                local_output_dir=problem_output_dir,
            )
        
        # Convert trajectory to list of dicts if needed
        traj_list = []
        if hasattr(trajectory, 'steps'):
            traj_list = [s.to_dict() if hasattr(s, 'to_dict') else s for s in trajectory.steps]
        elif isinstance(trajectory, list):
            traj_list = trajectory
        
        # Compute score (pass problem fields as kwargs for task-specific scoring)
        # Remove 'ground_truth' from kwargs to avoid duplicate argument error
        problem_kwargs = {k: v for k, v in problem.items() if k != 'ground_truth'}
        score = compute_score_func(agent_answer, ground_truth, **problem_kwargs)
        
        # Save log as .txt (captured console output + result summary)
        log_text = console_output
        log_text += f"\n{'=' * 80}\n"
        log_text += f"### Result ###\n"
        log_text += f"Problem ID: {problem_id}\n"
        log_text += f"Agent Answer: {agent_answer}\n"
        log_text += f"Ground Truth: {ground_truth}\n"
        log_text += f"Score: {score:.4f}\n"
        log_text += f"{'=' * 80}\n"
        
        log_path = os.path.join(logs_dir, f"{problem_id}.txt")
        with open(log_path, "w") as f:
            f.write(log_text)
        
        return BenchmarkResult(
            problem_id=problem_id,
            score=score,
            agent_answer=agent_answer,
            ground_truth=ground_truth,
            problem_statement=problem_statement,
            trajectory=traj_list,
        )
        
    except (ImportError, ModuleNotFoundError) as e:
        # Configuration errors should stop the entire benchmark
        raise
    except Exception as e:
        import traceback
        error_str = str(e)
        # Authentication errors should stop the entire benchmark
        if "AuthenticationError" in error_str or "api_key" in error_str.lower():
            raise
        
        # Docker/containerd errors should stop the entire benchmark
        if "containerd.sock" in error_str or "connection refused" in error_str.lower():
            raise RuntimeError(
                f"Docker daemon is not running or crashed. Original error: {error_str}\n"
                f"Please restart Docker by running the following commands in your terminal:\n"
                f"  pkill -9 dockerd 2>/dev/null; pkill -9 containerd 2>/dev/null\n"
                f"  sleep 1 && containerd &\n"
                f"  sleep 3 && rm -f /var/run/docker.pid && dockerd &\n"
            )
        
        # Print error to console with full traceback
        print(f"[{problem_id}] Error: {e}")
        traceback.print_exc()
        
        # Save error log as .txt
        log_text = f"Problem ID: {problem_id}\n"
        log_text += f"ERROR: {error_str}\n"
        log_text += f"Ground Truth: {ground_truth}\n"
        
        log_path = os.path.join(logs_dir, f"{problem_id}.txt")
        with open(log_path, "w") as f:
            f.write(log_text)
        
        return BenchmarkResult(
            problem_id=problem_id,
            score=0.0,
            agent_answer="",
            ground_truth=ground_truth,
            problem_statement=problem_statement,
            trajectory=[],
            error=error_str,
        )


def run_benchmark(
    task_name: str,
    agent_config: dict,
    output_dir: str,
    num_workers: int = 1,
    start_id: int = None,
    end_id: int = None,
    mode: str = "llm-in-sandbox",
) -> dict:
    """
    Run benchmark on a task with parallel execution.
    
    Args:
        task_name: Name of the benchmark task (math, chem, physics, etc.)
        agent_config: Dict with agent configuration (docker_image, llm_name, etc.)
        output_dir: Directory to save outputs (already named as {timestamp}_{task})
        num_workers: Number of parallel workers (ProcessPoolExecutor)
        start_id: Start index (0-based, inclusive)
        end_id: End index (0-based, exclusive)
        mode: "llm-in-sandbox" (default) or "llm" (vanilla LLM)
    
    Returns:
        Dictionary with results and statistics
    
    Output structure:
        {output_dir}/
            logs/           # Per-problem logs (human-readable .txt)
            trajectory.json # All trajectories
            results.json    # Final results
    """
    from rich.console import Console
    
    console = Console()
    docker_image = agent_config.get("docker_image")
    
    # Load task config
    task_config = load_task_config(task_name)
    compute_score = load_reward_function(task_name)
    
    # Test reward function early to catch missing dependencies
    try:
        compute_score("test", "test")
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        raise
    except Exception:
        pass  # Other errors are OK, we just want to check imports
    
    # Load prompt config (supports both merged and separate configs)
    if "system_prompt" in task_config:
        # New format: prompt config merged into config.yaml
        prompt_config = {
            "system_prompt": task_config["system_prompt"],
            "instance_prompt": task_config.get("instance_prompt", ""),
        }
    elif "prompt_config" in task_config:
        # Legacy format: separate prompt_config.yaml file
        prompt_config_path = task_config["prompt_config"]
        with open(prompt_config_path, "r") as f:
            prompt_config = yaml.safe_load(f)
    else:
        raise ValueError(f"Task config must have either 'system_prompt' or 'prompt_config'")
    
    # Load dataset
    dataset = load_dataset_from_config(task_config)
    problems = list(dataset)
    
    # Filter by start_id and end_id
    if start_id is not None or end_id is not None:
        start_id = start_id or 0
        end_id = end_id or len(problems)
        problems = problems[start_id:end_id]
    
    total_problems = len(problems)
    
    mode_display = "[green]LLM-in-Sandbox[/green]" if mode == "llm-in-sandbox" else "[yellow]Vanilla LLM[/yellow]"
    console.print(f"[bold cyan]🚀 Benchmark: {task_name}[/bold cyan] ({mode_display})")
    if start_id is not None or end_id is not None:
        console.print(f"   Range: [{start_id}, {end_id})")
    console.print(f"   Problems: {total_problems}, Workers: {num_workers}")
    console.print(f"   Output: {output_dir}")
    console.print(f"   Model: {agent_config.get('llm_name', 'N/A')}")
    console.print(f"   Temperature: {agent_config.get('temperature', 'N/A')}")
    if mode == "llm-in-sandbox":
        console.print(f"   Max Steps: {agent_config.get('max_steps', 'N/A')}")
        console.print(f"   Max Token Limit: {agent_config.get('max_token_limit', 'N/A')}")
        console.print(f"   Max Tokens Per Call: {agent_config.get('max_tokens_per_call', 'N/A')}")
    elif mode == "llm":
        console.print(f"   Max Response Length: {agent_config.get('max_response_len', 'N/A')}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    results = []
    completed_count = 0
    running_sum = 0.0
    error_count = 0
    
    from tqdm import tqdm
    
    console.print(f"[cyan]Starting {total_problems} problems with {num_workers} workers...[/cyan]")
    
    import time
    start_time = time.time()
    
    pbar = tqdm(total=total_problems, desc="Running", dynamic_ncols=True)
    
    def update_pbar(result):
        nonlocal completed_count, running_sum, error_count
        completed_count += 1
        running_sum += result.score
        if result.error:
            error_count += 1
        mean_score = running_sum / completed_count
        pbar.set_postfix({"score": f"{mean_score:.3f}", "errors": error_count})
        pbar.update(1)
    
    if num_workers == 1:
        # Sequential execution
        for problem in problems:
            args = {
                "problem": problem,
                "agent_config": agent_config,
                "task_name": task_name,
                "prompt_config": prompt_config,
                "output_dir": output_dir,
                "logs_dir": logs_dir,
                "mode": mode,
            }
            result = run_single_problem(args)
            results.append(result)
            update_pbar(result)
    else:
        # Parallel execution with ProcessPoolExecutor
        worker_args_list = [
            {
                "problem": problem,
                "agent_config": agent_config,
                "task_name": task_name,
                "prompt_config": prompt_config,
                "output_dir": output_dir,
                "logs_dir": logs_dir,
                "mode": mode,
            }
            for problem in problems
        ]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Register cleanup on SIGINT/SIGTERM
            def cleanup_handler(signum, frame):
                pbar.close()
                console.print("\n[yellow]Interrupted! Cleaning up...[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)
                if docker_image:
                    _cleanup_docker_containers(docker_image)
                sys.exit(1)
            
            old_sigint = signal.signal(signal.SIGINT, cleanup_handler)
            old_sigterm = signal.signal(signal.SIGTERM, cleanup_handler)
            
            try:
                futures = {
                    executor.submit(run_single_problem, args): args["problem"]["id"]
                    for args in worker_args_list
                }
                
                # Timeout per problem: 30 min max
                problem_timeout = 1800
                for future in as_completed(futures, timeout=problem_timeout * len(futures)):
                    try:
                        result = future.result(timeout=problem_timeout)
                    except TimeoutError:
                        problem_id = futures[future]
                        console.print(f"[red]Timeout: {problem_id}[/red]")
                        result = BenchmarkResult(
                            problem_id=problem_id,
                            score=0.0,
                            agent_answer="",
                            ground_truth="",
                            problem_statement="",
                            trajectory=[],
                            error="Timeout",
                        )
                    results.append(result)
                    update_pbar(result)
            finally:
                # Restore original signal handlers
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
    
    pbar.close()
    
    # Compute final statistics
    elapsed_time = time.time() - start_time
    scores = [r.score for r in results]
    errors = [r for r in results if r.error]
    
    stats = {
        "task": task_name,
        "total": len(results),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "num_errors": len(errors),
        "elapsed_time_seconds": elapsed_time,
    }
    
    # Save trajectory.json (each instance has trajectory + reward + answer + ground_truth + problem_statement)
    trajectory_data = {
        r.problem_id: {
            "problem_statement": r.problem_statement,
            "trajectory": r.trajectory,
            "reward": r.score,
            "agent_answer": r.agent_answer,
            "ground_truth": r.ground_truth,
        } for r in results
    }
    trajectory_path = os.path.join(output_dir, "trajectory.json")
    with open(trajectory_path, "w") as f:
        json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
    
    # Save results.json (config + stats summary)
    results_data = {
        "config": {
            "mode": mode,
            **agent_config,
        },
        **stats,
    }
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    elapsed_str = f"{int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s"
    
    console.print()
    console.print(f"[bold]📊 Results: {task_name}[/bold]")
    console.print(f"   Mean Score: [green]{stats['mean_score']:.4f}[/green]")
    console.print(f"   Errors: [red]{stats['num_errors']}[/red]" if stats['num_errors'] > 0 else "   Errors: 0")
    console.print(f"   Total Time: {elapsed_str}")
    console.print(f"   Output: {output_dir}")
    
    return results_data
