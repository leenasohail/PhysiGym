####
# title: test/test_epoch.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/test_epoch.py
#
# description:
#     unit test code for the physigym project
#     note: pytest and physigym enviroment are incompatible.
#####


# modules
import gymnasium
import os
import pcdl
import physigym  # import the Gymnasium PhysiCell bridge module
import shutil


#############
# run tests #
#############

print('\nUNITTEST run test ...')
os.chdir('../PhysiCell')

# set variables
i_cell_target = 128

# load PhysiCell Gymnasium environment
env = gymnasium.make(
    'physigym/ModelPhysiCellEnv-v0',
    #settingxml='config/PhysiCell_settings.xml',
    #render_mode='rgb_array',
    #render_fps=10
)


# epoch loop
ddf_cell = {}
ddf_conc = {}
for i_epoch in range(3):
    # reset output folder
    shutil.rmtree('output/')
    os.mkdir('output/')

    # reset the environment
    i_observation, d_info = env.reset()

    # episode time step loop
    b_episode_over = False
    i_step = 0
    while not b_episode_over:
        i_step += 1
        # policy according to i_observation
        if (i_observation > i_cell_target):
            d_action = {'subs_conc': 1 - r_reward}
        else:
            d_action = {'subs_conc': 0}

        # action
        o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
        b_episode_over = b_terminated or b_truncated
        b_episode_over = True
        print(f'epoch: {i_epoch}\tstep: {i_step}\tover: {b_episode_over}')

    # episode finishing
    env.close()

    # get timeseries data
    mcdsts = pcdl.TimeSeries('output/')
    ddf_cell.update({i_epoch: mcdsts.get_cell_df().drop({'runtime'}, axis=1)})
    ddf_conc.update({i_epoch: mcdsts.get_conc_df().drop({'runtime'}, axis=1)})

    # check results
    for i_epoch in ddf_cell.keys():
        ddf_cell[i_epoch].to_csv(f'timeseries_cell_epoch{str(i_epoch).zfill(3)}.csv')
        ddf_conc[i_epoch].to_csv(f'timeseries_conc_epoch{str(i_epoch).zfill(3)}.csv')

os.chdir('../PhysiGym')
print('UNITTEST: ok!')

