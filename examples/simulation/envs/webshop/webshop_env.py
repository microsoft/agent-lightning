from typing import Optional
from webshop.web_agent_site.envs import WebAgentTextEnv

def make_webshop_env(env_name, task, config, render_mode: Optional[str] = None):
    task_id=idx
    session_id=session_id
    env = WebAgentTextEnv(observation_mode="text", human_goals=True)
    
    return env