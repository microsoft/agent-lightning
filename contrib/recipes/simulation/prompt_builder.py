from typing import Optional

from autogen_core.models import AssistantMessage, UserMessage

class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions.

    Maintains a configurable history of text to
    construct prompt messages for conversational agents.
    """

    def __init__(
        self,
        max_history: int = -1,
        prompt_type: str = "chat"
    ):
        self.max_history = max_history
        self.prompt_type = prompt_type

        self._events = []
        self.admissible_actions = None

        self.step_count = 1

    def update_step_count(self):
        self.step_count += 1

    def update_instruction_prompt(self, instruction: str):
        self.instruction = instruction

    def update_single_obs_template(self, single_obs_template_wo_his: str, single_obs_template: str):
        self.single_obs_template_wo_his = single_obs_template_wo_his
        self.single_obs_template = single_obs_template

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history."""
        self._events.append(
            {
                "type": "observation",
                "text": obs,
            }
        )

    def update_action(self, action: str):
        """Add an action to the prompt history."""
        self._events.append(
            {
                "type": "action",
                "action": action,
            }
        )

    def update_admissible_actions(self, admissible_actions):
        self.admissible_actions = admissible_actions

    def init(self, env):
        """Clear the event history."""
        self._events.clear()

        if self.prompt_type == "chat":
            inst_prompt = env.get_instruction_prompt(info)
            self.update_instruction_prompt(inst_prompt)
        elif self.prompt_type == "single":
            template_wo_his, template = env.get_single_obs_template()
            self.update_single_obs_template(template_wo_his, template)
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

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

        for idx, event in enumerate(events):
            event_type = event.get("type")
            message = None

            if event_type == "observation":
                content = event.get("text", "")
                if idx == 0 and self.instruction:
                    content += "\n" + self.instruction
                message = UserMessage(source="user", content=content)

            elif event_type == "action":
                content = event.get("action", "")
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

    def get_prompt(self):
        if self.prompt_type == "chat":
            prompt = self.get_chat_prompt()
        elif self.prompt_type == "single":
            prompt = self.get_single_prompt()
        
        return prompt