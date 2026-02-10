"""
This sample code demonstrates how to use the MultiProviderClient with the APO algorithm 
to tune mathematical reasoning prompts using a hybrid model setup.
"""

import logging
import re
import asyncio
import multiprocessing
from typing import Tuple, cast, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()
import agentlightning as agl
from agentlightning import Trainer, setup_logging, PromptTemplate
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset
from litellm import completion
from agentlightning.utils.multi_provider_client import MultiProviderClient


# --- 1. Dataset Logic ---
def load_math_tasks() -> List[Dict[str, str]]:
    """Small mock GSM8k-style dataset."""
    return [
        {"question": "If I have 3 apples and buy 2 more, how many do I have?", "expected": "5"},
        {"question": "A train travels 60 miles in 1 hour. How far in 3 hours?", "expected": "180"},
        {"question": "What is the square root of 144?", "expected": "12"},
        {"question": "If a shirt costs $20 and is 10% off, what is the price?", "expected": "18"},
    ]

def load_train_val_dataset() -> Tuple[Dataset[Dict[str, str]], Dataset[Dict[str, str]]]:
    dataset_full = load_math_tasks()
    train_split = len(dataset_full) // 2
    # Use list() and cast to satisfy Pylance's SupportsIndex/slice checks
    dataset_train = cast(Dataset[Dict[str, str]], list(dataset_full[:train_split]))
    dataset_val = cast(Dataset[Dict[str, str]], list(dataset_full[train_split:]))
    return dataset_train, dataset_val

# --- 2. Agent Logic ---
class MathAgent(agl.LitAgent):
    def __init__(self):
        super().__init__()

    def rollout(self, task: Any, resources: Dict[str, Any], rollout: Any) -> float:
        # Pylance fix: Explicitly cast task to Dict
        t = cast(Dict[str, str], task)
        prompt_template: PromptTemplate = resources.get("prompt_template") # type: ignore
        
        # Ensure template access is type-safe
        template_str = getattr(prompt_template, "template", str(prompt_template))
        prompt = template_str.format(question=t["question"])
        
        # Direct LiteLLM call
        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = str(response.choices[0].message.content)
        
        # Reward: Numerical exact match check
        pred_nums = re.findall(r"[-+]?\d*\.\d+|\d+", answer.split("Answer:")[-1])
        reward = 1.0 if pred_nums and pred_nums[-1] == t["expected"] else 0.0
        
        agl.emit_reward(reward)
        return reward

# --- 3. Logging & Main ---
def setup_apo_logger(file_path: str = "apo_math.log") -> None:
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)

def main() -> None:
    setup_logging()
    setup_apo_logger()

    multi_client = MultiProviderClient()

    initial_prompt_str = "Solve: {question}"

    algo = APO[Dict[str, str]](
        multi_client,
        gradient_model="gemini/gemini-2.0-flash",
        apply_edit_model="groq/llama-3.3-70b-versatile",
        val_batch_size=2,
        gradient_batch_size=2,
        beam_width=1,
        branch_factor=1,
        beam_rounds=1,
    )

    trainer = Trainer(
        algorithm=algo,
        n_runners=2,
        initial_resources={
            "prompt_template": PromptTemplate(template=initial_prompt_str, engine="f-string")
        },
        adapter=TraceToMessages(),
    )

    dataset_train, dataset_val = load_train_val_dataset()
    agent = MathAgent()
    
    print("\n" + "="*60)
    print("🚀 HYBRID APO OPTIMIZATION STARTING")
    print("-" * 60)

    trainer.fit(agent=agent, train_dataset=dataset_train, val_dataset=dataset_val)

    # Print Final Prompt from the store
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE")
    print("-" * 60)
    print(f"INITIAL PROMPT:\n{initial_prompt_str}")

    
    # Accessing the latest optimized prompt from the trainer store
    try:
        latest_resources = asyncio.run(trainer.store.query_resources())
        if latest_resources:
            final_res = latest_resources[-1].resources.get("prompt_template")
            final_prompt = getattr(final_res, "template", str(final_res))
            print(f"FINAL OPTIMIZED PROMPT:\n{final_prompt}")
    except Exception as e:
        print(f"Optimization finished. Check apo_math.log for detailed iteration results. Error: {e}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()