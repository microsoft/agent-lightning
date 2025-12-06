# BabaIsAI

### Installing BabaIsAI

The easiest way to install TextWorld is via [`pip`](https://pypi.org/):

    pip install git+https://github.com/nacloos/baba-is-ai

Or, after cloning the repo, go inside the root folder of the project (i.e. alongside `setup.py`) and run

    pip install .

### Usage

To play an environment, run this command:
```bash
python baba/play.py --env two_room-break_stop-make_win
```
The `--env` argument specifies the ID of the environment. Once the game opens, use the arrow keys to move the agent.

You can also create a Gym environment object:
```python
import baba

env_id = "two_room-break_stop-make_win"
env = baba.make(f"env/{env_id}")
```

To list all available environment IDs, run this code:
```python
import baba

print(baba.make("env/*").keys())
```