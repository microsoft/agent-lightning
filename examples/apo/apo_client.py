import dotenv
import os
import random
import re
from agentlightning import configure_logger
from agentlightning.litagent import LitAgent
from agentlightning.trainer import Trainer
from agentlightning.types import Rollout
from typing import List, Dict, Tuple, Optional

def call_api(user_prompt: str = "") -> str:
    return user_prompt + " (simulated API response)"


class GSM8KAgent(LitAgent):
    
    def _is_number(self, s: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a string is a number and return its numeric value.
        """
        try:
            return True, str(float(s))
        except ValueError:
            try:
                import unicodedata
                return True, str(unicodedata.numeric(s))
            except (TypeError, ValueError):
                return False, None
    
    def extract_answer(self, completion: str) -> Optional[str]:
        """
        Extract the answer from the model completion.
        """
        if not completion:
            return None

        preds = completion.split("The answer is")
        pred = preds[1] if len(preds) > 1 else preds[-1]

        pred = pred.replace(",", "")
        numbers = re.findall(r"-?\d+\.?\d*", pred)

        if not numbers:
            return None

        pred = numbers[-1].rstrip(".")
        is_number, pred = self._is_number(pred)

        return pred if is_number else None

    def check_answer(self, pred: str, answer: str, tolerance: float = 1e-6) -> bool:
        """
        Check if the predicted answer matches the ground truth.
        """
        pred_answer = self.extract_answer(pred)
        gt_label = self.extract_answer(answer)

        if pred_answer is None or gt_label is None:
            return False

        try:
            return abs(float(pred_answer) - float(gt_label)) <= tolerance
        except ValueError:
            return False

    def training_rollout(self, task, rollout_id, resources):
        print("Resources:", resources)
        print("Task:", task)
        user_prompt = resources["prompt"].template

        # user_prompt = user_prompt.render_query(question=task['question'])
        user_prompt = user_prompt.replace("{question}", task['question'])
        user_prompt = re.sub(r'\n{3,}', '\n\n', user_prompt)

        query_output = call_api(user_prompt)
        question = task['question']
        ground_truth = task['answer']
        score = int(self.check_answer(query_output, task['answer']))
        return Rollout(
            rollout_id=rollout_id,
            final_reward=score,
            metadata={
                "query": question,
                "query_output": query_output,
                "ground_truth": ground_truth
            }
        )


if __name__ == "__main__":
    configure_logger()
    dotenv.load_dotenv()
    agent = GSM8KAgent()
    trainer = Trainer(n_workers=2)
    trainer.fit(agent, backend="http://127.0.0.1:9997")
