from typing import Optional

from envs.scienceworld.base import SciWorldEnv
from scienceworld import ScienceWorldEnv


def build_simplification_str(args):
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")
    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")
    if args["open_containers"]:
        simplifications.append("openContainers")
    if args["open_doors"]:
        simplifications.append("openDoors")
    if args["no_electrical"]:
        simplifications.append("noElectricalAction")
    return args["simplifications_preset"] or ",".join(simplifications)


def parse_args():
    from types import SimpleNamespace

    args = SimpleNamespace(
        jar_path=None,
        task_num=0,  # 7
        var_num=0,
        env_step_limit=100,
        num_episodes=5,
        seed=None,
        output_path_prefix="save-histories",
        max_episode_per_file=1000,
        simplifications_preset="easy",
        teleport=False,
        self_watering_plants=False,
        open_containers=True,
        open_doors=True,
        no_electrical=False,
    )
    params = vars(args)
    return params


def make_scienceworld_env(env_name, task, config, render_mode: Optional[str] = None):
    env_args = parse_args()
    sciworld_env = ScienceWorldEnv("", serverPath=env_args["jar_path"], envStepLimit=env_args["env_step_limit"])
    env = SciWorldEnv(sciworld_env, **config)

    sub_task_name = task["sub_task_name"]
    variation_idx = task["variation_idx"]

    env.env.load(
        sub_task_name, variation_idx, simplificationStr=build_simplification_str(env_args), generateGoldPath=True
    )

    if "max_steps" in task:
        env.max_steps = int(task["max_steps"])
    else:
        env.set_max_steps(sub_task_name)

    return env
