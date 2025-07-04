#####
# title: run_physigym_template.py
#
# language: python3
# library: gymnasium,
#   and the extending and physigym custom_modules
#
# date:
# license: <compatible with bsb-3-clause>
# author: <your name goes here>
# input: https://gymnasium.farama.org/main/
# original source code: https://github.com/Dante-Berth/PhysiGym
# modified source code: <https://>
#
# run:
#   1. copy this file into the PhysiCell root folder
#   2. python3 run_physigym_template.py
#
# description:
#   python script to run a single episode from the physigym template model.
#####


# library
from extending import physicell
import gymnasium
import physigym

# load PhysiCell Gymnasium environment
# %matplotlib
# env = gymnasium.make('physigym/ModelPhysiCellEnv-v0', settingxml='config/PhysiCell_settings.xml', figsize=(8,6), render_mode='human', render_fps=10)
env = gymnasium.make('physigym/ModelPhysiCellEnv-v0')

# reset the environment
r_reward = 0.0
o_observation, d_info = env.reset()

# time step loop
b_episode_over = False
while not b_episode_over:

    # policy according to o_observation
    b_observation = o_observation
    if (b_observation):
        d_action = {}
    else:
        d_action = {}

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
