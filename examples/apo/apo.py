import asyncio

from agentlightning.server import AgentLightningServer
from agentlightning.types import NamedResources, PromptTemplate

import asyncio
import random
from agentlightning.server import AgentLightningServer
from agentlightning.types import NamedResources, PromptTemplate
# from cfpo import POAgents
from typing import List, Dict, Tuple, Optional
from po_agent.agent import POAgent
import os

def call_api(user_prompt: str = "") -> str:
    import anthropic
    print("Calling API with user prompt:", user_prompt)

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=20000,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text


import datasets
import random
import re
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import logging


class GSM8KDataLoader:
    def __init__(
        self,
        train_size: int = 100,
        minibatch_size: int = 5,
        valid_size: int = 200,
        test_size: int = -1,
        answer_marker: str = "The answer is",
    ):
        """
        Initialize the GSM8K task.
        """
        data_dir = "openai/gsm8k"
        self.train_size = train_size
        self.minibatch_size = minibatch_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.answer_marker = answer_marker
        self.dataset = self.load_task_dataset(data_dir)
        self.train_set, self.valid_set, self.test_set = self.dataset


    def load_task_dataset(self, data_dir) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load and preprocess the GSM8K dataset.
        """
        dataset = datasets.load_dataset(path=data_dir, name='main')
        train_examples = self._pre_process(dataset["train"])
        test_examples = self._pre_process(dataset['test'])

        # Split dataset into train, validation, and test sets
        test_set = test_examples if self.test_size == -1 else test_examples[:self.test_size]

        train_size = len(train_examples)
        if self.valid_size > train_size:
            raise ValueError("valid_size is greater than the number of train examples.")

        valid_indices = random.sample(range(train_size), self.valid_size)
        valid_set = [train_examples[i] for i in valid_indices]

        remaining_train_indices = [i for i in range(train_size) if i not in valid_indices]
        if self.train_size == -1:
            train_set = [train_examples[i] for i in remaining_train_indices]
        else:
            if self.train_size > len(remaining_train_indices):
                raise ValueError("train_size is greater than the remaining number of train examples after validation set selection.")
            train_set = [train_examples[i] for i in random.sample(remaining_train_indices, self.train_size)]

        return train_set, valid_set, test_set

    def _pre_process(self, dataset) -> List[Dict]:
        """
        Preprocess the dataset.
        """
        out_doc = []
        for doc in dataset:
            label = doc['answer'].split('####')[-1].strip()
            text = doc['answer'].split('####')[0].strip()

            lines = text.split('\n')
            processed_lines = [f"{line.strip()}." if not line.strip().endswith('.') else line.strip() for line in lines]
            processed_text = ' '.join(processed_lines).strip()

            answer = f"{processed_text} {self.answer_marker} {label}."
            question = re.sub(r'\s+', ' ', doc['question'])
            answer = re.sub(r'\s+', ' ', answer)

            out_doc.append({"question": question, "answer": answer})
        return out_doc
    
    def sample_minibatch(self) -> List[Dict]:
        """
        Sample a minibatch from the training set.
        """
        minibatch = random.sample(self.train_set, k=min(self.minibatch_size, len(self.train_set)))
        return minibatch


async def example_apo():
    """
    An example of how a prompt optimization works.
    """
    gsm8k_dataloader = GSM8KDataLoader()
    prompt_optimizer = POAgent(task_intention="solve a reasoning task and answer the following mathematical problem",
    optimizer_api=call_api)

    server = AgentLightningServer(host="127.0.0.1", port=9997)
    await server.start()

    prompt_candidates_and_reward = [("Please solve the following question: ", None)]

    # for prompt, _ in prompt_candidates_and_reward:
    #     task_id_list = []
    #     # 1. The optimization algorithm updates the prompt template
    #     print(f"\n[Algo] Updating prompt template to: '{prompt}'")
    #     prompt = prompt + "\n{question}" if "\n{question}" not in prompt else prompt
    #     resources: NamedResources = {"prompt": PromptTemplate(template=prompt, engine="f-string")}
    #     # How the resource is used fully depends on the client implementation.
    #     await server.update_resources(resources)

    #     minibatch = gsm8k_dataloader.sample_minibatch()
    #     print(f"[Algo] Sampled {len(minibatch)} tasks from the GSM8K dataset for this round.")

    #     # 2. Get the results of prompt in this minibatch
    #     querys, ground_truths, query_outputs, scores = [], [], [], []
    #     for data in minibatch:
    #         print(data)
    #         print("[Algo] Queuing task for clients...")
    #         task_id = await server.queue_task(sample=data, mode='train')
    #         print(f"[Algo] Task '{task_id}' is now available for clients.")
    #         task_id_list.append(task_id)
        
    #     for task_id in task_id_list:
    #         rollout = await server.poll_completed_rollout(task_id, timeout=30)
    #         assert rollout, "Expected a completed rollout from the client."
    #         print(f"[Algo] Received Result: {rollout}")
    #         querys.append(rollout.metadata["query"])
    #         ground_truths.append(rollout.metadata["ground_truth"])
    #         query_outputs.append(rollout.metadata["query_output"])
    #         scores.append(rollout.final_reward)
        
    #     prompt_candidates = prompt_optimizer.diagnosing(prompt, querys, query_outputs, ground_truths, scores)[0]
    #     print(f"[Algo] Found {len(prompt_candidates)} prompt candidates for the next round.")

        
    #     # avg_reward = sum(scores) / len(scores) if scores else 0
    #     # print(f"[Algo] Average reward for prompt '{prompt}': {avg_reward}")
    #     for candidate in prompt_candidates:
    #         print(f"[Algo] Candidate prompt: '{candidate[0]}'")
    #         prompt_candidates_and_reward.append((candidate[0], None))

    for i, (prompt, eval_score) in enumerate(prompt_candidates_and_reward):
        if eval_score is not None:
            continue
        task_id_list = []
        # 1. The optimization algorithm updates the prompt template
        print(f"\n[Algo] Updating prompt template to: '{prompt}'")
        prompt = prompt + "\n{question}" if "\n{question}" not in prompt else prompt
        resources: NamedResources = {"prompt": PromptTemplate(template=prompt, engine="f-string")}
        # How the resource is used fully depends on the client implementation.
        await server.update_resources(resources)

        valid_set = gsm8k_dataloader.valid_set
        print(f"[Algo] Sampled {len(valid_set)} tasks from the GSM8K dataset for this round.")

        # 2. Get the results of prompt in this minibatch
        querys, ground_truths, query_outputs, scores = [], [], [], []
        for data in valid_set:
            print(data)
            print("[Algo] Queuing task for clients...")
            task_id = await server.queue_task(sample=data, mode='train')
            print(f"[Algo] Task '{task_id}' is now available for clients.")
            task_id_list.append(task_id)
        
        for task_id in task_id_list:
            rollout = await server.poll_completed_rollout(task_id, timeout=30)
            assert rollout, "Expected a completed rollout from the client."
            print(f"[Algo] Received Result: {rollout}")
            querys.append(rollout.metadata["query"])
            ground_truths.append(rollout.metadata["ground_truth"])
            query_outputs.append(rollout.metadata["query_output"])
            scores.append(rollout.final_reward)
        
        prompt_candidates_and_reward[i] = (prompt, sum(scores) / len(scores) if scores else 0)
        print(f"[Algo] Average reward for prompt '{prompt}': {prompt_candidates_and_reward[i][1]}")
        


    await server.stop()


if __name__ == "__main__":
    asyncio.run(example_apo())
