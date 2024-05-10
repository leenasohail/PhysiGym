####
# title: test_py3pc_embed.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# description:
#   pytest unit test library for the physicellembedding py3pc_embed project
#   + https://docs.pytest.org/
#
#   note:
#   assert actual == expected, message
#   == value equality
#   is reference equality
#   pytest.approx for real values
#####


# modules
import os
import pcdl
import shutil


###############################
# install install py3pc_embed #
###############################

print('\nUNITTEST: install physigym ...')
os.chdir('../PhysiCell')
os.system('rm -fr user_projects/utest')
os.system('cp -r ../PhysiGym/physigym user_projects/utest')
os.system('make save PROJ=backup')
os.system('make data-cleanup clean reset')
os.system("sed -i 's/cp .\/user_projects\/$(PROJ)\/custom_modules\//cp -r .\/user_projects\/$(PROJ)\/custom_modules\//' ./Makefile")
os.system('make load PROJ=utest')
os.chdir('../PhysiGym')
print('UNITTEST: ok!')


##########################
# copy utest model files #
##########################

print('\nUNITTEST: copy utest model files ...')
shutil.copyfile(
    'test/config/PhysiCell_settings.xml',
    '../PhysiCell/config/PhysiCell_settings.xml',
)
shutil.copyfile(
    'test/config/cell_rules.csv',
    '../PhysiCell/config/cell_rules.csv',
)
print('UNITTEST: ok!')


#######################
# manipulate custom.h #
#######################

print('\nUNITTEST manipulate custom.h ...')
f = open('../PhysiCell/custom_modules/custom.h', 'a')
f.writelines([
'\n',
'int set_microenv(std::string s_substrate, double r_conc);\n',
])
f.close()
print('UNITTEST: ok!')


########################################
# manipulate custom_modules/custom.cpp #
########################################

print('\nUNITTEST manipulate custom.cpp ...')
f = open('../PhysiCell/custom_modules/custom.cpp', 'a')
f.writelines([
'\n',
'int set_microenv(std::string s_substrate, double r_conc) {\n',
'    // update substrate concentration\n',
'    int k = microenvironment.find_density_index(s_substrate);\n',
'    for (unsigned int n=0; n < microenvironment.number_of_voxels(); n++) {\n',
'        microenvironment(n)[k] += r_conc;\n',
'    }\n',
'    return 0;\n',
'}\n',
])
f.close()
print('UNITTEST: ok!')


###########################################################
# manipulate custom_modules/embedding/physicellmodule.cpp #
###########################################################

print('\nUNITTEST manipulate embedding physicellmodule.cpp ...')
fr = open('physigym/custom_modules/embedding/physicellmodule.cpp', 'r')
fw = open('../PhysiCell/custom_modules/embedding/physicellmodule.cpp', 'w')

b_jump = False
for s_line in fr:
    # declare action
    if (s_line.find('// Put physigym related parameter, variable, and vector action mapping here!') > -1):
        b_jump = True
        fw.writelines([
        '    // update drug concentration\n',
        '    set_microenv("substrate_a", parameters.doubles("subs_conc"));\n',
        '    set_microenv("substrate_b", parameters.doubles("subs_conc"));\n',
        '    set_microenv("substrate_c", parameters.doubles("subs_conc"));\n',
        '}\n',
        '\n',
        ])

    elif (s_line.find('// do observation') > -1):
        b_jump = False
        fw.write(s_line)


    # declare observation
    elif (s_line.find('// Put physigym related parameter, variable, and vector observation mapping here!') > -1):
        b_jump = True
        fw.writelines([
        '    // receive cell count\n',
        '    parameters.ints("cell_count") = (*all_cells).size();\n',
        '\n',
        '    // receive apoptosis rate\n',
        '    for (Cell* pCell : (*all_cells)) {\n',
        '        pCell->custom_data["apoptosis_rate"] = get_single_behavior(pCell, "apoptosis");\n',
        '    }\n',
        '}\n',
        '\n',
        ])

    elif (s_line.find('// on phenotype time step') > -1):
        b_jump = False
        fw.write(s_line)

    # write line
    elif not b_jump:
        fw.write(s_line)

    # else
    else:
        print(f'skip: {s_line.strip()}')

fw.close()
fr.close()
print('UNITTEST: ok!')


##############################################################
# manipulate custom_modules/physigym/envs/physicell_model.py #
##############################################################

print('\nUNITTEST manipulate physigym physicell_model.py ...')
fr = open('physigym/custom_modules/physigym/physigym/envs/physicell_model.py', 'r')
fw = open('../PhysiCell/custom_modules/physigym/physigym/envs/physicell_model.py', 'w')

b_jump = False
for s_line in fr:

    # declare action space
    if (s_line.find("action_space = spaces.Dict({") > -1):
        b_jump = True
        fw.writelines([
        "        action_space = spaces.Dict({\n",
        "            'drug_conc': spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float64)\n",
        "        })\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return action_space") > -1):
        b_jump = False
        fw.write(s_line)

    # declare observation space
    elif (s_line.find("observation_space = spaces.Discrete(2)") > -1):
        b_jump = True
        fw.writelines([
        "        observation_space = spaces.Box(low=0, high=(2**16 - 1), shape=(), dtype=np.uint16)\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return observation_space") > -1):
        b_jump = False
        fw.write(s_line)

    # declare get_img function
    elif (s_line.find("# substrate data #") > -1):
        b_jump = True
        fw.writelines([
        "        # substrate data #\n",
        "        ##################\n",
        "\n",
        "        df_conc = pd.DataFrame(physicell.get_microenv('drug'), columns=['x','y','z','drug'])\n",
        "        df_conc = df_conc.loc[df_conc.z == 0.0, :]\n",
        "        df_mesh = df_conc.pivot(index='y', columns='x', values='drug')\n",
        "        ax.contourf(\n",
        "            df_mesh.columns, df_mesh.index, df_mesh.values,\n",
        "            vmin=0.0, vmax=0.2, cmap='Reds',\n",
        "            #alpha=0.5,\n",
        "        )\n",
        "\n",
        "        #######################\n",
        "        # substrate color bar #\n",
        "        #######################\n",
        "\n",
        "        self.fig.colorbar(\n",
        "            mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=0.2), cmap='Reds'),\n",
        "            label='drug_conc',\n",
        "            ax=ax,\n",
        "        )\n",
        "\n",
        "        #############\n",
        "        # cell data #\n",
        "        #############\n",
        "\n",
        "        df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID', 'x','y', 'z'])\n",
        "        df_variable = pd.DataFrame(physicell.get_variable('apoptosis_rate'), columns=['apoptosis_rate'])\n",
        "        df_cell = pd.merge(df_cell, df_variable, left_index=True, right_index=True, how='left')\n",
        "        df_cell = df_cell.loc[df_cell.z == 0.0, :]\n",
        "        df_cell.plot(\n",
        "            kind='scatter', x='x', y='y', c='apoptosis_rate',\n",
        "            xlim=[\n",
        "                int(self.x_root.xpath('//domain/x_min')[0].text),\n",
        "                int(self.x_root.xpath('//domain/x_max')[0].text),\n",
        "            ],\n",
        "            ylim=[\n",
        "                int(self.x_root.xpath('//domain/y_min')[0].text),\n",
        "                int(self.x_root.xpath('//domain/y_max')[0].text),\n",
        "            ],\n",
        "            vmin=0.0, vmax=0.1, cmap='viridis',\n",
        "            grid=True,\n",
        "            title=f'dt_gym step {str(self.iteration).zfill(3)}: {df_cell.shape[0]} / 128 [cell]',\n",
        "            ax=ax,\n",
        "        )\n",
        "\n",
        "        ###############",
        ])

    elif (s_line.find("# save to file #") > -1):
        b_jump = False
        fw.write(s_line)

    # declare get_observation function
    elif (s_line.find(" o_observation = {'discrete': True}") > -1):
        b_jump = True
        fw.writelines([
        "        o_observation = np.array(physicell.get_parameter('cell_count'), dtype=np.uint16)\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return o_observation") > -1):
        b_jump = False
        fw.write(s_line)

    # declare get_info function
    # nop

    # declare get_terminated function
    elif (s_line.find("b_terminated = False") > -1):
        b_jump = True
        fw.writelines([
        "        b_terminated = physicell.get_parameter('cell_count') <= 0\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return b_terminated") > -1):
        b_jump = False
        fw.write(s_line)

    # declare get_reward function
    elif (s_line.find("r_reward = 0.0") > -1):
        b_jump = True
        fw.writelines([
        "        i_cellcount = np.clip(physicell.get_parameter('cell_count'), a_min=0, a_max=256)\n",
        "        if (i_cellcount == 128):\n",
        "            r_reward == 1\n",
        "        elif (i_cellcount < 128):\n",
        "            r_reward = i_cellcount / 128\n",
        "        elif (i_cellcount > 128):\n",
        "            r_reward = (i_cellcount - 128) / 128\n",
        "        else:\n",
        "            sys.exit('Error @ CorePhysiCellEnv.get_reward : strange clipped cell_count detected {i_cellcount}.')\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return r_reward") > -1):
        b_jump = False
        fw.write(s_line)

    # write line
    elif not b_jump:
        fw.write(s_line)

    # else
    else:
        print(f'skip: {s_line.strip()}')

fw.close()
fr.close()
print('UNITTEST: ok!')


########
# make #
########

print('\nUNITTEST make ...')
os.chdir('../PhysiCell')
os.system('make')
#from embedding import physicell
import gymnasium                                                                
import physigym  # import the Gymnasium PhysiCell bridge module                 
os.chdir('../PhysiGym')
print('UNITTEST: ok!')


#############
# run tests #
#############

print('\nUNITTEST run test ...')
class TestPhysiGym(object):
    ''' test for physigym. '''

    def test_epoch(self):
        os.chdir('../PhysiCell')
        # set variables
        i_cell_target = 128

        # load PhysiCell Gymnasium environment                                          
        env = gymnasium.make('physigym/ModelPhysiCellEnv-v0', render_mode='human', render_fps=10)

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
            while not b_episode_over:                                                       
                # policy according to i_observation                                         
                if (i_observation > 128):                                                   
                    d_action = {'drug_conc': 1 - r_reward}                                  
                else:                                                                       
                    d_action = {'drug_conc': 0}                                             
                                                                                            
                # action                                                                    
                o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
                b_episode_over = b_terminated or b_truncated                                
                                                                                            
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

        assert False
        os.chdir('../PhysiGym')

print('UNITTEST: ok!')



###########################
# restore backuped model #
###########################

#print('\nUNITTEST restore backuped model ...')
#os.chdir('../PhysiCell')
#os.system('make load=backup')
#os.chdir('../PhysiGym')
#print('UNITTEST: ok!')

