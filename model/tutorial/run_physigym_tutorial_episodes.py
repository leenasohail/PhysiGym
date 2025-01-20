#####
# title: run_physigym_tutorial_episodes.py
#
# language: python3
# library: gymnasium, numpy, physicell embedding, physigym
#
# date: 2024-spring
# license: <has to be comatiple with bsb-3-clause>
# author: <your name goes here>
# input: https://gymnasium.farama.org/main/
# original source code: https://github.com/Dante-Berth/PhysiGym
# modified source code: <https://>
#
# run:
#   1. copy this file into the PhysiCell root folder
#   2. python3 run_physigym_tutorial_episodes.py
#
# description:
#   python script to run multiple episodes from the physigym tutorial model.
#####


# library
from embedding import physicell
import gymnasium
import numpy as np
import physigym

# load PhysiCell Gymnasium environment
# %matplotlib
# env = gymnasium.make('physigym/ModelPhysiCellEnv-v0', settingxml='config/PhysiCell_settings.xml', figsize=(8,6), render_mode='human', render_fps=10)
env = gymnasium.make('physigym/ModelPhysiCellEnv-v0')

# episode loop
for i_episode in range(3):

    # set episode output folder
    env.get_wrapper_attr('x_root').xpath("//save/folder")[0].text = f'output/episode{str(i_episode).zfill(8)}'

    # reset the environment
    r_reward = 0.0
    o_observation, d_info = env.reset()

    # time step loop
    b_episode_over = False
    while not b_episode_over:

        # policy according to o_observation
        i_observation = o_observation[0]
        if (i_observation >= physicell.get_parameter('cell_count_target')):
            d_action = {'drug_dose': np.array([1.0 - r_reward])}
        else:
            d_action = {'drug_dose': np.array([0.0])}

        # action
        o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
        b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
