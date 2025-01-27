from copy import deepcopy
import warnings
import os, sys
import gymnasium as gym

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiGym_Models") + len("PhysiGym_Models")
]
sys.path.append(absolute_path)

from rl.tdmpc2.tdmpc2_jax.envs.wrappers.pixels import PixelWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from rl.tdmpc2.tdmpc2_jax.envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)
