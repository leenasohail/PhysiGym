import gymnasium as gym
import physigym
import random
import numpy as np
from rl.utils.wrappers.wrapper_physicell_complex_tme import PhysiCellModelWrapper
import matplotlib.pyplot as plt
import os

env = gym.make("physigym/ModelPhysiCellEnv", observation_type="image_gray")
env = PhysiCellModelWrapper(env, list_variable_name=["anti_M2", "anti_pd1"])
# env = gym.wrappers.RecordEpisodeStatistics(env)
obs , info = env.reset(seed=1)
episode = 1
color_mapping = env.unwrapped.color_mapping
df_cell_obs = info["df_cell"]

while True:
    actions = np.array(env.action_space.sample())   
    next_obs, rewards, terminations, truncations, info = env.step(actions)
    next_df_cell_obs = info["df_cell"]
    obs = next_obs
    df_cell_obs = next_df_cell_obs
        
    # print(o)
    if terminations or truncations:
        # del obs
        # del df_cell_obs
        obs, info = env.reset()
        df_cell_obs = info["df_cell"]
