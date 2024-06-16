####
# title: test/install_unittest.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/install_unittest.py
#
# description:
#     install unit test code for the physigym project
#     note: pytest and physigym enviroment are incompatible.
#####


# modules
import os
import shutil


###############################
# install install py3pc_embed #
###############################

print('\nUNITTEST: install physigym ...')
os.chdir('../PhysiCell')
os.system('rm -fr user_projects/unittest')
os.system('cp -r ../PhysiGym/physigym user_projects/unittest')
os.system('make save PROJ=backup')
os.system('make data-cleanup clean reset')
os.system("sed -i 's/cp .\/user_projects\/$(PROJ)\/custom_modules\//cp -r .\/user_projects\/$(PROJ)\/custom_modules\//' ./Makefile")
os.system('make load PROJ=unittest')
os.chdir('../PhysiGym')
print('UNITTEST: ok!')


#############################
# copy unittest model files #
#############################

print('\nUNITTEST: copy unittest config files ...')
shutil.copyfile(
    'test/config_unittest/PhysiCell_settings.xml',
    '../PhysiCell/config/PhysiCell_settings.xml',
)
shutil.copyfile(
    'test/config_unittest/cell_rules.csv',
    '../PhysiCell/config/cell_rules.csv',
)
shutil.copyfile(
    'test/config_unittest/cells.csv',
    '../PhysiCell/config/cells.csv',
)
print('UNITTEST: ok!')


#######################
# manipulate custom.h #
#######################

print('\nUNITTEST manipulate custom.h ...')
f = open('../PhysiCell/custom_modules/custom.h', 'a')
f.writelines([
'\n',
'int set_microenv(std::string s_substrate, double r_dose);\n',
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
'int set_microenv(std::string s_substrate, double r_dose) {\n',
'    // update substrate concentration\n',
'    int k = microenvironment.find_density_index(s_substrate);\n',
'    for (unsigned int n=0; n < microenvironment.number_of_voxels(); n++) {\n',
'        microenvironment(n)[k] += r_dose;\n',
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
        '                // Put physigym related parameter, variable, and vector action mapping here!\n',
        '\n',
        '                // parameter\n',
        '                //my_function( parameters.bools("my_bool")) );\n',
        '                //my_function( parameters.ints("my_int")) );\n',
        '                //my_function( parameters.doubles("my_float") );\n',
        '                //my_function( parameters.strings("my_str") );\n',
        '\n',
        '                // custom variable\n',
        '                //std::string my_variable = "my_variable";\n',
        '                //for (Cell* pCell : (*all_cells)) {\n',
        '                //    my_function( pCell->custom_data[my_variable] );\n',
        '                //}\n',
        '\n',
        '                // custom vector\n',
        '                //std::string my_vector = "my_vector";\n',
        '                //for (Cell* pCell : (*all_cells)) {\n',
        '                //    int vectindex = pCell->custom_data.find_vector_variable_index(my_vector);\n',
        '                //    if (vectindex > -1) {\n',
        '                //        my_function( pCell->custom_data.vector_variables[vectindex].value );\n',
        '                //    } else {\n',
        '                //        char error[64];\n',
        '                //        sprintf(error, "Error: unknown custom_data vector! %s", my_vector);\n',
        '                //        PyErr_SetString(PyExc_ValueError, error);\n',
        '                //        return NULL;\n',
        '                //    }\n',
        '                //}\n',
        '\n',
        '                // update substrate concentration\n',
        '                set_microenv("substrate_a", parameters.doubles("substrate_dose_a"));\n',
        '                set_microenv("substrate_b", parameters.doubles("substrate_dose_b"));\n',
        '                set_microenv("substrate_c", parameters.doubles("substrate_dose_c"));\n',
        '            }\n',
        '\n',
        ])

    elif (s_line.find('// do observation') > -1):
        b_jump = False
        fw.write(s_line)


    # declare observation
    elif (s_line.find('// Put physigym related parameter, variable, and vector observation mapping here!') > -1):
        b_jump = True
        fw.writelines([
        '                // Put physigym related parameter, variable, and vector observation mapping here!\n',
        '\n',
        '                // parameter\n',
        '                //parameters.bools("my_bool") = value;\n',
        '                //parameters.ints("my_int") = value;\n',
        '                //parameters.doubles("my_float") = value;\n',
        '                //parameters.strings("my_str") = value;\n',
        '\n',
        '                // custom variable\n',
        '                //std::string my_variable = "my_variable";\n',
        '                //for (Cell* pCell : (*all_cells)) {\n',
        '                //    pCell->custom_data[my_variable] = value;\n',
        '                //}\n',
        '\n',
        '                // custom vector\n',
        '                //std::string my_vector = "my_vector";\n',
        '                //for (Cell* pCell : (*all_cells)) {\n',
        '                //    int vectindex = pCell->custom_data.find_vector_variable_index(my_vector);\n',
        '                //    if (vectindex > -1) {\n',
        '                //        pCell->custom_data.vector_variables[vectindex].value = value;\n',
        '                //    } else {\n',
        '                //        char error[64];\n',
        '                //        sprintf(error, "Error: unknown custom_data vector! %s", my_vector);\n',
        '                //        PyErr_SetString(PyExc_ValueError, error);\n',
        '                //        return NULL;\n',
        '                //    }\n',
        '                //}\n',
        '\n',
        '                // receive cell count\n',
        '                parameters.ints("cell_count") = (*all_cells).size();\n',
        '\n',
        '                // receive apoptosis rate\n',
        '                for (Cell* pCell : (*all_cells)) {\n',
        '                    pCell->custom_data["apoptosis_rate"] = get_single_behavior(pCell, "apoptosis");\n',
        '                }\n',
        '            }\n',
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
	#print(f'skip: {s_line.strip()}')
        pass

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
    if (s_line.find("# model dependent action_space processing logic goes here!") > -1):
        b_jump = True
        fw.writelines([
        "        # model dependent action_space processing logic goes here!\n",
        "        d_action_space = spaces.Dict({\n",
        "            'substrate_dose_a': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),\n",
        "            'substrate_dose_b': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),\n",
        "            'substrate_dose_c': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),\n",
        "        })\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return d_action_space") > -1):
        b_jump = False
        fw.write(s_line)

    # declare observation space
    elif (s_line.find("# model dependent observation_space processing logic goes here!") > -1):
        b_jump = True
        fw.writelines([
        "        # model dependent observation_space processing logic goes here!\n",
        "        o_observation_space = spaces.Box(low=0, high=(2**16 - 1), shape=(1,), dtype=np.uint16)\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return o_observation_space") > -1):
        b_jump = False
        fw.write(s_line)

    # declare get_observation function
    elif (s_line.find("# model dependent observation processing logic goes here!") > -1):
        b_jump = True
        fw.writelines([
        "        # model dependent observation processing logic goes here!\n",
        "        o_observation = np.array([physicell.get_parameter('cell_count')], dtype=np.uint16)\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return o_observation") > -1):
        b_jump = False
        fw.write(s_line)

    # declare get_info function
    # nop

    # declare get_terminated function
    elif (s_line.find("# model dependent terminated processing logic goes here!") > -1):
        b_jump = True
        fw.writelines([
        "        # model dependent terminated processing logic goes here!\n",
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
        "            r_reward = 1\n",
        "        elif (i_cellcount < 128):\n",
        "            r_reward = i_cellcount / 128\n",
        "        elif (i_cellcount > 128):\n",
        "            r_reward = 1 - (i_cellcount - 128) / 128\n",
        "        else:\n",
        "            sys.exit('Error @ CorePhysiCellEnv.get_reward : strange clipped cell_count detected {i_cellcount}.')\n",
        "\n",
        "        # output\n",
        ])

    elif (s_line.find("return r_reward") > -1):
        b_jump = False
        fw.write(s_line)

    # declare get_img function
    elif (s_line.find("# substrate data #") > -1):
        b_jump = True
        fw.writelines([
        "        # substrate data #\n",
        "        ##################\n",
        "\n",
        "        # substrate a\n",
        "        df_substrate_a = pd.DataFrame(physicell.get_microenv('substrate_a'), columns=['x','y','z','substarte_a'])\n",
	"        df_substrate_a = df_substrate_a.loc[df_substrate_a.z == 0.0, :]\n",
        "        df_mesh_a = df_substrate_a.pivot(index='y', columns='x', values='substarte_a')\n",
	"        ax.contourf(\n",
        "            df_mesh_a.columns, df_mesh_a.index, df_mesh_a.values,\n",
        "            vmin=0.0, vmax=1.0, cmap='Reds',\n",
        "            alpha=0.333,\n",
	"        )\n",
        "\n",
        "        # substrate b\n",
	"        df_substrate_b = pd.DataFrame(physicell.get_microenv('substrate_b'), columns=['x','y','z','substarte_b'])\n",
        "        df_substrate_b = df_substrate_b.loc[df_substrate_b.z == 0.0, :]\n",
        "        df_mesh_b = df_substrate_b.pivot(index='y', columns='x', values='substarte_b')\n",
	"        ax.contourf(\n",
        "            df_mesh_b.columns, df_mesh_b.index, df_mesh_b.values,\n",
        "            vmin=0.0, vmax=1.0, cmap='Greens',\n",
        "            alpha=0.333,\n",
	"        )\n",
        "\n",
        "        # substrate c\n",
	"        df_substrate_c = pd.DataFrame(physicell.get_microenv('substrate_c'), columns=['x','y','z','substarte_c'])\n",
        "        df_substrate_c = df_substrate_c.loc[df_substrate_c.z == 0.0, :]\n",
        "        df_mesh_c = df_substrate_c.pivot(index='y', columns='x', values='substarte_c')\n",
	"        ax.contourf(\n",
        "            df_mesh_c.columns, df_mesh_c.index, df_mesh_c.values,\n",
        "            vmin=0.0, vmax=1.0, cmap='Blues',\n",
        "            alpha=0.333,\n",
	"        )\n",
	"\n",
        "        #######################\n",
        "        # substrate color bar #\n",
        "        #######################\n",
        "\n",
        "        # substrate a\n",
        "        self.fig.colorbar(\n",
        "            mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='Reds'),\n",
        "            label='substrate_a',\n",
        "            ax=ax,\n",
        "        )\n",
        "\n",
        "        # substrate b\n",
        "        self.fig.colorbar(\n",
        "            mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='Greens'),\n",
        "            label='substrate_b',\n",
        "            ax=ax,\n",
        "        )\n",
        "\n",
        "        # substrate c\n",
        "        self.fig.colorbar(\n",
        "            mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='Blues'),\n",
        "            label='substrate_c',\n",
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
        "            vmin=0.0, vmax=1.0, cmap='turbo',\n",
        "            grid=True,\n",
        "            title=f'dt_gym env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} / 128 [cell]',\n",
        "            ax=ax,\n",
        "        )\n",
        "\n",
        "        ###############\n",
        ])

    elif (s_line.find("# save to file #") > -1):
        b_jump = False
        fw.write(s_line)


    # write line
    elif not b_jump:
        fw.write(s_line)

    # else
    else:
	#print(f'skip: {s_line.strip()}')
        pass

fw.close()
fr.close()
print('UNITTEST: ok!')


########
# make #
########

print('\nUNITTEST make ...')
os.chdir('../PhysiCell')
os.system('make')
os.chdir('../PhysiGym')
print('UNITTEST: ok!')


###########################
# update reference manual #
###########################

print('\nREFERENCE manual update ...')
os.system('python3 man/scarab.py')
print('REFERENCE manual: ok!')

