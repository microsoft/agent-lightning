# TextWorld

## Download games

We use pregenerated games from https://github.com/conglu1997/intelligent-go-explore/tree/main/textworld/tw_games

Download and unzip them

    curl -L -o tw-games.zip 'https://drive.google.com/uc?export=download&id=1aeT-45-OBxiHzD9Xn99E5OvC86XmqhzA'
    unzip tw-games.zip


## Installation

TextWorld supports __Python 3.9/3.10/3.11/3.12__ for __Linux__ and __macOS__ systems only at the moment. For __Windows__ users, docker can be used as a workaround (see Docker section below).

### Requirements

TextWorld requires some system libraries for its native components.
On a Debian/Ubuntu-based system, these can be installed with

    sudo apt update && sudo apt install build-essential libffi-dev python3-dev curl git

And on macOS, with

    brew install libffi curl git

> **Note:** We advise our users to use virtual environments to avoid Python packages from different projects to interfere with each other. Popular choices are [Conda Environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and [Virtualenv](https://virtualenv.pypa.io/en/stable/)

### Installing TextWorld

The easiest way to install TextWorld is via [`pip`](https://pypi.org/):

    pip install textworld

Or, after cloning the repo, go inside the root folder of the project (i.e. alongside `setup.py`) and run

    pip install .

#### Visualization

TextWorld comes with some tools to visualize game states. Make sure all dependencies are installed by running

    pip install textworld[vis]

Then, you will need to install either the [Chrome](https://sites.google.com/a/chromium.org/chromedriver/) or [Firefox](https://github.com/mozilla/geckodriver) webdriver (depending on which browser you have currently installed).
If you have Chrome already installed you can use the following command to install chromedriver

    pip install chromedriver_installer

Current visualization tools include: `take_screenshot`, `visualize` and `show_graph` from [`textworld.render`](https://textworld.readthedocs.io/en/latest/textworld.render.html).
