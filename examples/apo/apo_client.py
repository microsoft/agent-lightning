import dotenv
import os
import random

from openai import OpenAI

from agentlightning import configure_logger
from agentlightning.litagent import LitAgent
from agentlightning.trainer import Trainer

from tqdm import tqdm

import dotenv
import os
import random
import re
from agentlightning import configure_logger
from agentlightning.litagent import LitAgent
from agentlightning.trainer import Trainer
from agentlightning.types import Rollout
from typing import List, Dict, Tuple, Optional

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
    
    def eval_one_case(self, user_prompt, question, answer):
        user_prompt = user_prompt.replace("{question}", question)
        openai = OpenAI(
            api_key="111",
            base_url="http://localhost:8000/v1",
        )

        result = openai.chat.completions.create(
            model="/home/jiahangxu/working/models/Meta-Llama-3-8B-Instruct",
            messages=[
                # {"role": "system", "content": resources["system_prompt"].template},
                {"role": "user", "content": user_prompt},
            ],
        ).choices[0].message.content

        question = question
        ground_truth = answer
        score = int(self.check_answer(result, answer))
        # print(f"Question: {question}")
        # print(f"Query Output: {result}")
        # print(f"Ground Truth: {ground_truth}")
        # print(f"Score: {score}")
        return question, result, ground_truth, score
            
    def training_rollout(self, task, rollout_id, resources):
        user_prompt = resources["prompt"].template

        if isinstance(task, dict) and 'question' in task:
            question, result, ground_truth, score = self.eval_one_case(user_prompt, task['question'], task['answer'])
        
        elif isinstance(task, list):
            question, result, ground_truth, score = [], [], [], []
            for t in tqdm(task):
                q, r, gt, s = self.eval_one_case(user_prompt, t['question'], t['answer'])
                question.append(q)
                result.append(r)
                ground_truth.append(gt)
                score.append(s)

        return Rollout(
            rollout_id=rollout_id,
            final_reward=sum(score) / len(score) if isinstance(score, list) else score,
            metadata={
                "query": question,
                "query_output": result,
                "ground_truth": ground_truth
            })

if __name__ == "__main__":
    configure_logger()
    dotenv.load_dotenv()
    agent = GSM8KAgent()
    trainer = Trainer(n_workers=2)
    trainer.fit(agent, backend="http://127.0.0.1:9997")
