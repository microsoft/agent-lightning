import os

def mkdirs(dirpath: str) -> str:
    """ Create a directory and all its parents.

    If the folder already exists, its path is returned without raising any exceptions.

    Arguments:
        dirpath: Path where a folder need to be created.

    Returns:
        Path to the (created) folder.
    """
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    return dirpath

_default_alfworld_cache = os.path.expanduser("examples/simulation/envs/alfworld/alfworld_source")
ALFWORLD_DATA = mkdirs(os.getenv("ALFWORLD_DATA", _default_alfworld_cache))
os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA

print(ALFWORLD_DATA)

import subprocess

subprocess.run([
    "alfworld-download",
    "--data-dir", ALFWORLD_DATA,
    "--extra"
])