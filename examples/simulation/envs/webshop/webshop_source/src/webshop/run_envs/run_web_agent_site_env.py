"""
Test the site gym environment.

TODO: move to testing dir for more rigorous tests
"""
import gym
from rich import print
from rich.markup import escape

from ..web_agent_site.envs import WebAgentSiteEnv
from ..web_agent_site.models import (
    HumanPolicy,
    RandomPolicy,
)
from ..web_agent_site.utils import DEBUG_PROD_SIZE


if __name__ == '__main__':
    #env = gym.make('WebAgentSite-v0')
    #env = WebAgentSiteEnv(render=True, pause=2.0)
    #env = WebAgentSiteEnv(observation_mode='html', render=False)
    env = WebAgentSiteEnv(observation_mode='text', render=False, num_products=DEBUG_PROD_SIZE)
    global_step = 0
    
    try:
        #policy = HumanPolicy()
        policy = RandomPolicy()
    
        observation = env.observation
        while True:
            print(observation)
            available_actions = env.get_available_actions()
            print('Available actions:', available_actions)
            action = policy.forward(observation, available_actions)
            observation, reward, done, info = env.step(action)
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            if done:
                break
            global_step += 1
    finally:
        env.close()