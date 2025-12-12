from collections import deque
from typing import Any, List, Optional

from autogen_core.models import AssistantMessage, SystemMessage, UserMessage


class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning.

    Maintains a configurable history of text, images, and chain-of-thought reasoning to
    construct prompt messages for conversational agents.
    """

    def __init__(
        self,
        max_text_history: int = 16,
        instruction: Optional[str] = None,
        single_obs_template: Optional[str] = None,
    ):
        self.max_history = max_text_history
        self.instruction = instruction
        self._events = []
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.admissible_actions = None
        self.single_obs_template = single_obs_template

        self.step_count = 1

    def get_pure_env_obs(self, env_obs):
        text = env_obs["text"]
        return text

    def update_step_count(self):
        self.step_count += 1

    def update_instruction_prompt(self, instruction: str):
        self.instruction = instruction

    def update_single_obs_template(self, single_obs_template_wo_his: str, single_obs_template: str):
        self.single_obs_template_wo_his = single_obs_template_wo_his
        self.single_obs_template = single_obs_template

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        text = obs["text"]
        image = obs.get("image", None)

        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "image": image,
            }
        )

    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def update_admissible_actions(self, admissible_actions):
        self.admissible_actions = admissible_actions

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_chat_prompt(self):
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: A list of messages constructed from the event history.
        """
        if self.max_history != -1:
            events = self._events[-(self.max_history * 2 + 1) :]
        else:
            events = self._events

        messages = []

        # Add system prompt if present
        # if self.system_prompt:
        #     messages.append(SystemMessage(content=self.system_prompt))

        for idx, event in enumerate(events):
            event_type = event.get("type")
            message = None

            if event_type == "observation":
                # Determine prefix for observation
                content = event.get("text", "")
                if idx == 0:
                    content += "\n" + self.instruction
                message = UserMessage(source="user", content=content)

            elif event_type == "action":
                # Use reasoning if available, otherwise use action text
                reasoning = event.get("reasoning")
                content = f"Previous plan:\n{reasoning}" if reasoning else event.get("action", "")
                message = AssistantMessage(source="assistant", content=content)

            if message:
                messages.append(message)

        return messages

    def get_single_prompt(self):
        if self.max_history != -1:
            events = self._events[-(self.max_history * 2 + 1) :]
        else:
            events = self._events

        current_obs = events[-1]["text"]

        if len(events) == 1:
            template = self.single_obs_template_wo_his
            kwargs = {"current_observation": current_obs}
            if "{admissible_actions}" in template:
                kwargs["admissible_actions"] = self.admissible_actions
            single_prompt = template.format(**kwargs)
        else:
            template = self.single_obs_template
            history = ""
            obs_count = 0
            for idx, event in enumerate(events):
                if events[idx]["type"] == "observation" and idx != len(events) - 1:
                    next_event = events[idx + 1]
                    history += f"[Observation {max(self.step_count-self.max_history+obs_count, 1)}: '{event['text']}', "
                    history += (
                        f"Action {max(self.step_count-self.max_history+obs_count, 1)}: '{next_event['action']}']\n "
                    )
                    obs_count += 1

            kwargs = {
                "step_count": self.step_count - 1,
                "history_length": min(self.step_count - 1, self.max_history),
                "history": history,
                "current_step": self.step_count,
                "current_observation": current_obs,
            }
            if "{admissible_actions}" in template:
                kwargs["admissible_actions"] = self.admissible_actions
            single_prompt = template.format(**kwargs)

        return [UserMessage(source="user", content=single_prompt)]
