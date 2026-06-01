"""
Instruction following reward function (IFBench).
Copied from R2E-Gym utils.py

Requires IFBench: See README.md for installation instructions.
"""
# Filter out annoying pkg_resources deprecation warnings from syllapy
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import json
import os


# ====== CONFIGURE IFBENCH PATH ======
# Set IFBENCH_PATH environment variable before running:
# export IFBENCH_PATH=/path/to/your/IFBench
IFBENCH_PATH = os.path.expanduser(os.environ.get("IFBENCH_PATH", "~/IFBench"))
# ====================================


def compute_score(agent_answer: str, ground_truth: str, **kwargs) -> float:
    """
    Compute score for instruction following problems using IFBench.
    
    Uses loose evaluation mode to check if all instructions are followed.
    ground_truth is a JSON string with instruction_id_list and kwargs.
    """
    try:
        gt_data = json.loads(ground_truth)
        instruction_id_list = gt_data.get("instruction_id_list", [])
        kwargs_list = gt_data.get("kwargs", [])
        
        # Try to load IFBench
        try:
            import sys
            if IFBENCH_PATH not in sys.path:
                sys.path.insert(0, IFBENCH_PATH)
            
            import instructions_registry
            instruction_dict = instructions_registry.INSTRUCTION_DICT
        except ImportError:
            raise ImportError(
                f"IFBench not found at {IFBENCH_PATH}. "
                "Please set IFBENCH_PATH environment variable and follow the installation instructions in README.md"
            )
        
        response = agent_answer
        
        # Loose evaluation: create response variants
        r = response.split("\n")
        response_variants = [
            response,
            response.replace("*", ""),
            "\n".join(r[1:]).strip(),
            "\n".join(r[:-1]).strip(),
            "\n".join(r[1:-1]).strip(),
        ]
        
        for idx, instruction_id in enumerate(instruction_id_list):
            if instruction_id not in instruction_dict:
                print(f"Warning: Unknown instruction: {instruction_id}")
                return 0.0
            
            instruction_cls = instruction_dict[instruction_id]
            instruction = instruction_cls(instruction_id)
            
            # Filter None kwargs
            inst_kwargs = {k: v for k, v in kwargs_list[idx].items() if v is not None}
            instruction.build_description(**inst_kwargs)
            
            # Check if any variant follows the instruction
            is_following = False
            for variant in response_variants:
                if variant.strip() and instruction.check_following(variant):
                    is_following = True
                    break
            
            if not is_following:
                return 0.0
        
        return 1.0
        
    except json.JSONDecodeError:
        # Silently return 0 for invalid JSON (e.g., during startup test)
        return 0.0
    except Exception as e:
        print(f"Error in compute_score: {e}")
        return 0.0
