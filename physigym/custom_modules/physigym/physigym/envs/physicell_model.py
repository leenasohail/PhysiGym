#####
# title: pysigym/envs/physicell_model.py
#
# language: python3
# library: gymnasium v1.0.0a1
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     gymnasium enviroemnt for physicell embedding
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/implementing_custom_wrappers/
#####


# library
from embedding import physicell
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import pandas as pd
from physigym.envs.physicell_core import CorePhysiCellEnv


# function
class ModelPhysiCellEnv(CorePhysiCellEnv):
    """
    input:
        gymnasium.Env

    offspring:
        physigym.ModelPhysiCellEnv

    description:
    """
    def __ini__():
        super(CorePhysiCellEnv, self).__init__()

    def _get_action_space(self):
        """
        description:
            + https://gymnasium.farama.org/main/api/spaces/
        """
        # model dependent processing logic goes here!
        action_space = spaces.Dict({
            'discrete': spaces.Discrete(2)  # boolean, string
            #'discrete': spaces.MultyBinary()  # boolean
            #'discrete': spaces.MultyDiscrete()  # string
            #'discrete': spaces.Text() # e.g. DNA letter
            #'numeric': spaces.Box()   # int, float
        })

        # output
        return action_space


    def _get_observation_space(self):
        """
        description:
            + https://gymnasium.farama.org/main/api/spaces/
        """
        # model dependent processing logic goes here!
        observation_space = spaces.Dict({
            'discrete': spaces.Discrete(2)  # boolean, string
            #'discrete': spaces.MultyBinary()  # boolean
            #'discrete': spaces.MultyDiscrete()  # string
            #'discrete': spaces.Text() # e.g. DNA letter
            #'numeric': spaces.Box()   # int, float
        })

        # output
        return observation_space


    def _get_img(self):
        """
        description:
            templare code to generate a matplotlib figure from the data.
            physicell.get_microenv()
            pjysicell.get_cell()
        """
        # physigym necessary begin: generate invisible figure
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.close()
        # physigym necessary end


        # model dependent ploting logic goes here!
        ax.axis('off')
        ax.axis('equal')

        ##################
        # substrate data #
        ##################

        #df_conc = pd.DataFrame(physicell.get_microenv('my_substrate'), columns=['x','y','z','my_substrate'])
        #df_conc = df_conc.loc[df_conc.z == 0.0, :]
        #df_mesh = df_conc.pivot(index='y', columns='x', values='my_substrate')
        #ax.contourf(
        #    df_mesh.columns, df_mesh.index, df_mesh.values,
        #    vmin=0.0, vmax=0.2, cmap='Reds',
        #    #alpha=0.5,
        #)

        ######################
        # substrate colorbar #
        ######################

        #fig.colorbar(
        #    mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='Reds'),
        #    label='my_substrate',
        #    ax=ax,
        #)

        #############
        # cell data #
        #############

        #df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID', 'x','y', 'z'])
        #df_variable = pd.DataFrame(physicell.get_variable("my_variable"), columns=['my_variable'])
        #df_cell = pd.merge(df_cell, df_variable, left_index=True, right_index=True, how='left')
        #df_cell = df_cell.loc[df_cell.z == 0.0, :]
        #df_cell.plot(
        #    kind='scatter', x='x', y='y', c='my_variable',
        #    xlim=[
        #        int(self.x_root.xpath('//domain/x_min')[0].text,
        #        int(self.x_root.xpath('//domain/x_max')[0].text,
        #    ],
        #    ylim=[
        #        int(self.x_root.xpath('//domain/y_min')[0].text,
        #        int(self.x_root.xpath('//domain/y_max')[0].text,
        #    ],
        #    vmin=0.0, vmax=1.0, cmap='viridis',
        #    grid=True,
        #    title=None,
        #    ax=ax,
        #)

        ################
        # save to file #
        ################

        #plt.tight_layout()
        #s_path = self.x_root.xpath('//save/folder')[0].text
        #fig.savefig(f'{s_path}/timeseries_step{str(self.iteration).zfill(3)}.jpeg', facecolor='white')


        # physigym necessary begin: output
        fig.canvas.draw()
        a_img = np.array(fig.canvas.buffer_rgba(), dtype=np.uint8)
        # physigym necessary end
        return a_img


    def _get_observation(self):
        """
        output:
            compatible with defined onbservation space!
        physicell.get_parameter()
        physicell.get_variable()
        physicell.get_vector()
        """
        # model dependent processing logic goes here!
        o_observation = {'discrete': True}

        # output
        return o_observation


    def _get_info(self):
        """
        physicell.get_parameter()
        physicell.get_variable()
        physicell.get_vector()
        """
        # model dependent processing logic goes here!
        d_info = {}

        # output
        return d_info


    def _get_terminated(self):
        """
        # e.g. if exactely 128 cells
        """
        # model dependent processing logic goes here!
        b_terminated = False

        # output
        return b_terminated


    def _get_reward(self):
        """
        # e.g. how far I am away from 128
        """
        # model dependent processing logic goes here!
        r_reward = 0.0

        # output
        return r_reward
