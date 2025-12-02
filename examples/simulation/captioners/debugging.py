import os
import json
from rich import print
from rich.panel import Panel
from rich.text import Text

def print_step(reasoning, executed_action, env_obs, reward, terminated):
    print("[bold white]===== STEP RESULT =====[/bold white]")
    print(f"[bold green]Reasoning:[/bold green] {reasoning}")
    print(f"[bold blue]Executed Action:[/bold blue] {executed_action}")
    print(f"[bold yellow]Observation:[/bold yellow] {env_obs['text']}")
    print(f"[bold magenta]Reward:[/bold magenta] {reward}")
    print(f"[bold red]Done:[/bold red] {terminated}")
    print("[bold white]======================[/bold white]")

def print_llm_chat_input(obs):
    print("[bold white]===== LLM Input =====[/bold white]")
    for item in obs:
        role = item.type
        content = item.content

        # Choose color by role
        if "System" in role:
            role_color = "cyan"
        elif "User" in role:
            role_color = "green"
        else:
            role_color = "white"

        # Wrap each block in a panel
        print(Panel.fit(
            Text(content, style=role_color),
            title=f"[bold {role_color}]{role.upper()}[/bold {role_color}]",
            border_style=role_color
        ))

    print("[bold white]======================[/bold white]")

def print_llm_single_input(obs):
    print("[bold white]===== LLM Input =====[/bold white]")
    role_color = "cyan"

    print(Panel.fit(
        Text(obs[0].content, style=role_color),
        border_style=role_color
    ))   

    print("[bold white]======================[/bold white]")


def save_chat_rollout(obs, filename):
    data = []

    for item in obs:
        entry = {
            "role": item.type,
            "content": item.content
        }
        data.append(entry)

    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✔ JSON saved to {filename}")