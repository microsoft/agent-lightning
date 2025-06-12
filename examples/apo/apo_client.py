import dotenv
import os
import random

from openai import OpenAI

from agentlightning import LitAgent, configure_logger
from agentlightning.prompt.resource import NamedResources
from agentlightning.prompt.client import TrainerV1


class SimpleAgent(LitAgent):

    def training_rollout(self, sample, *, sampling_parameters=None, rollout_id=None):
        resources = NamedResources.load(sampling_parameters)
        print("Resources:", resources)

        openai = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"],
        )

        result = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": resources["system_prompt"].template},
                {"role": "user", "content": sample["prompt"]},
            ],
        )
        print("Result:", result)

        return random.uniform(0, 1)


if __name__ == "__main__":
    configure_logger()
    dotenv.load_dotenv()
    agent = SimpleAgent()
    trainer = TrainerV1()
    trainer.fit(agent, endpoint="http://127.0.0.1:9997")
