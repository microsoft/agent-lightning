# Copyright (c) Microsoft. All rights reserved.

import os
import random
from typing import Any

import dotenv
from openai import OpenAI

from agentlightning import configure_logger
from agentlightning.litagent import LitAgent
from agentlightning.trainer import Trainer


class SimpleAgent(LitAgent[Any]):

    def training_rollout(self, task, rollout_id, resources):  # type: ignore
        print("Resources:", resources)  # type: ignore

        openai = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"],
        )

        result = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": resources["system_prompt"].template},  # type: ignore
                {"role": "user", "content": task["prompt"]},
            ],
        )
        print("Result:", result)

        return random.uniform(0, 1)


if __name__ == "__main__":
    configure_logger()
    dotenv.load_dotenv()
    agent = SimpleAgent()
    trainer = Trainer(n_workers=2)
    trainer.fit(agent, "http://127.0.0.1:9997")
