####
# title: test/physigym_drift_ice.py
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
#     note: pytest and physigym environment are incompatible.
#####


# modules
import glob
import os
import pandas as pd


#############
# run tests #
#############

print('\nUNITTEST check for drift ...')
os.chdir('../PhysiCell')

##################
# load conc data #
##################
ddf_conc = {}
for s_file in sorted(glob.glob('timeseries_conc_episode*csv')):
    i_episode = int(s_file.replace('timeseries_conc_episode','').replace('.csv',''))
    df_conc = pd.read_csv(s_file, index_col=0)
    ddf_conc.update({i_episode: df_conc})

# check results
for i_episode in ddf_conc.keys():
    print(f'processing: conc episode {i_episode}')
    for s_column in  ddf_conc[0].columns:
        if any(ddf_conc[0][s_column] != ddf_conc[i_episode][s_column]):
            print(f'\tepisode conc: {i_episode}\tcolumn: {s_column}')
print()

##################
# load cell data #
##################
ddf_cell = {}
for s_file in sorted(glob.glob('timeseries_cell_episode*csv')):
    i_episode = int(s_file.replace('timeseries_cell_episode','').replace('.csv',''))
    df_cell = pd.read_csv(s_file, index_col=0)
    ddf_cell.update({i_episode: df_cell})

# check results
for i_episode in ddf_cell.keys():
    print(f'processing: cell episode {i_episode}')
    try:
        for s_column in  ddf_cell[0].columns:
            if any(ddf_cell[0][s_column] != ddf_cell[i_episode][s_column]):
                print(f'\tcell episode: {i_episode}\tcolumn: {s_column}')
    except ValueError:
        print(f'\tcell episode: {i_episode}\t{ddf_cell[0].shape} {ddf_cell[i_episode].shape}\terror: series not identically labeled')

# finish
os.chdir('../PhysiGym')
print('UNITTEST: ok!')

