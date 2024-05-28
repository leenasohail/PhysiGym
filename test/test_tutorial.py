####
# title: test/test_episode.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/test_episode.py
#
# description:
#     unit test code for the physigym project
#     note: pytest and physigym enviroment are incompatible.
#####


# modules
import gymnasium
import matplotlib.pyplot as plt
import os
import pcdl
import physigym  # import the Gymnasium PhysiCell bridge module
import random
import shutil


#############
# run tests #
#############

print('\nUNITTEST run test ...')
os.chdir('../PhysiCell')
os.system('rm timeseries_*_episode*.csv')
# set variables
i_cell_target = 128

# load PhysiCell Gymnasium environment
env = gymnasium.make(
    'physigym/ModelPhysiCellEnv-v0',
    #settingxml='config/PhysiCell_settings.xml',
    #render_mode='rgb_array',
    #render_fps=10
)

# episode loop
ddf_cell = {}
ddf_conc = {}
for i_episode in range(4):
    # reset output folder
    shutil.rmtree('output/')
    os.mkdir('output/')

    # reset variable
    random.seed(0)

    # reset the environment
    i_observation, d_info = env.reset(seed=0)
    r_reward = 0

    # episode time step loop
    b_episode_over = False
    while not b_episode_over:
        # policy according to i_observation
        print(f'r_reward: {r_reward}')
        if (i_observation > i_cell_target):
            d_action = {
                'drug_dose': 1  #- r_reward
            }
        else:
            d_action = {
                'drug_dose': 0
            }

        #d_action = {'drug_dose': random.random() * 0.1}

        # action
        o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)

        # check if episode finsih
        b_episode_over = b_terminated or b_truncated
        print(f'dt_gym env step: {env.unwrapped.step_env}\tepisode: {env.unwrapped.episode}\tepisode step: {env.unwrapped.step_episode}\tover: {b_episode_over}\tb_terminated: {b_terminated}\tb_truncated: {b_truncated}')

    # get timeseries data
    mcdsts = pcdl.TimeSeries('output/')
    ddf_cell.update({i_episode: mcdsts.get_cell_df().drop({'runtime'}, axis=1)})
    ddf_conc.update({i_episode: mcdsts.get_conc_df().drop({'runtime'}, axis=1)})
    ddf_cell[i_episode].to_csv(f'timeseries_cell_episode{str(i_episode).zfill(3)}.csv')
    ddf_conc[i_episode].to_csv(f'timeseries_conc_episode{str(i_episode).zfill(3)}.csv')

    # plot timeseries
    fig, axs = plt.subplots(nrows=3, ncols=1 ,figsize=(8,12))
    mcdsts.plot_timeseries('cell_type', ax=axs[0])
    mcdsts.plot_timeseries(None, 'drug', ylim=(0.0,1.0), ax=axs[1])
    mcdsts.plot_timeseries('cell_type', 'death_rates_0', ax=axs[2])
    fig.suptitle(f'timeseries episode {str(i_episode).zfill(3)}')
    plt.tight_layout()
    fig.savefig(f'timeseries_plot_tutorial_episode{str(i_episode).zfill(3)}.png')

# free PhysiCell Gymnasium environment
env.close()

# finish
os.chdir('../PhysiGym')
print('UNITTEST: ok!')

