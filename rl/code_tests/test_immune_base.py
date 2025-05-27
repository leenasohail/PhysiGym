import gymnasium as gym
import physigym
import random
import numpy as np
import os, sys
import time


# Setup environment
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
]
sys.path.append(absolute_path)

from rl.utils.wrappers.wrapper_physicell_tumor_immune_base import PhysiCellModelWrapper

env = gym.make("physigym/ModelPhysiCellEnv", observation_type="simple")
env = PhysiCellModelWrapper(env, list_variable_name=["drug_1"])
# Fill both buffers
done = True
episode = -1

while episode < 3:
    if done:
        episode += 1
        o, info = env.reset()
        df_cell_obs = info["df_cell"]
        actions = np.array([0.5], dtype=np.float16)

    o, r, t, ter, info = env.step(actions)
    done = t or ter
    next_df_cell_obs = info["df_cell"]
    print(o)

    df_cell_obs = next_df_cell_obs
