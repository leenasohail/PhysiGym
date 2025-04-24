import gymnasium as gym
import physigym
import random
import numpy as np
from rl.utils.wrappers.wrapper_physicell_complex_tme import PhysiCellModelWrapper
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

env = gym.make("physigym/ModelPhysiCellEnv", observation_type="simple")
env = PhysiCellModelWrapper(env, list_variable_name=["anti_M2", "anti_pd1"])
env = gym.wrappers.RecordEpisodeStatistics(env)
_, info = env.reset(seed=1)
df = info["df_cell"]  # To delete
episode = 1
color_mapping = env.unwrapped.color_mapping
cumulative_reward = 0
step = 1
liste = []
while episode < 50:
    begin_time = time.time()
    actions = np.array(env.action_space.sample())
    o, r, t, ter, info = env.step(actions)
    nb_cancer_cells = info["number_cancer_cells"]
    nb_number_m2 = info["number_m2"]
    nb_cd8 = info["number_cd8"]
    nb_cd8exhausted = info["number_cd8exhausted"]
    end_time = time.time()
    time_step = end_time - begin_time
    # end time
    liste.append(
        [
            episode,
            step,
            actions[0],
            actions[1],
            nb_cancer_cells,
            nb_number_m2,
            nb_cd8,
            nb_cd8exhausted,
            r[0],
            time_step,
        ]
    )
    step += 1

    # print(o)
    if t or ter:
        episode += 1
        liste.append(cumulative_reward)
        step = 0
        o, info = env.reset()

# Create the DataFrame with appropriate column names
df = pd.DataFrame(
    liste,
    columns=[
        "episode",
        "step",
        "anti_M2",
        "anti_pd1",
        "number_cancer_cells",
        "number_m2",
        "number_cd8",
        "number_cd8exhausted",
        "reward",
        "cumulative_reward",
        "time_step_seconds",
    ],
)
df_sorted = df.sort_values(by=["episode", "step"])
df_sorted.to_csv("stochastic_results.csv", index=False)


print(df.head())
