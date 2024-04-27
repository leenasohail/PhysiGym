#####
# title: pysigym/envs/physicell_model.py
#
# language: python3
# library: gymnasium v1.0.0a1
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     gymnasium enviroemnt for physicell embedding
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
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
        physigym.CorePhysiCellEnv

    output:
        physigym.ModelPhysiCellEnv

    description:
        this is the model physigym enviroment class, built on top of the
        physigym.CorePhysiCellEnv class, which is built on top of the
        gymnasium.Env class.

        fresh from the PhysiGym repo this is only a template class!
        you will have to edit this class, to specify the model specific
        reniforcement learning enviroment.
    """
    def __ini__():
        super(CorePhysiCellEnv, self).__init__()


    def get_action_space(self):
        """
        input:

        output:
            action_space: dictionary
                the dictionary keys have to match the parameter,
                custom variable, or custom vector label.
                the value have to be defind as gymnasium space object.
                + https://gymnasium.farama.org/main/api/spaces/

        description:
            dictionary structur built out of gymnasium.spaces elements.
            this struct has to specify type and range for each
            action parameter, custom variable, and custom vector.
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


    def get_observation_space(self):
        """
        input:

        output:
            observation_space structur.
                the struct have to be built out of gymnasium.spaces elements.
                there are no other limits.
                + https://gymnasium.farama.org/main/api/spaces/

        description:
            data structur built out of gymnasium.spaces elements.
            this struct has to specify type and range
            for each observed variable.
        """
        # model dependent processing logic goes here!
        #observation_space =
        #compositione: spaces.Dict({})
        #compositione: spaces.Tuple(())
        #discrete: spaces.Discrete()  # boolean, string
        #discrete: spaces.MultyBinary()  # boolean
        #discrete: spaces.MultyDiscrete()  # string
        #discrete: spaces.Text()  # char e.g. DNA letters
        #numeric: spaces.Box()  # int, float
        #niche: spaces.Graph(())
        #niche: spaces.Sequence(())  # set of spaces

        observation_space = spaces.Discrete(2)

        # output
        return observation_space


    def get_img(self):
        """
        input:

        output:
            self.fig.savefig
                instance attached matplotlib figure.

        description:
            templare code to generate a matplotlib figure from the data.
            for example from:
            + physicell.get_microenv('my_substrate')
            + physicell.get_cell()
            + physicell.get_variable('my_variable')
            however, there are no limts.
        """
        # model dependent ploting logic goes here!
        self.fig.axes[0].remove()
        ax = self.fig.add_subplot(1,1,1)
        ax.axis('equal')
        #ax.axis('off')

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

        #self.fig.colorbar(
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
        #        int(self.x_root.xpath('//domain/x_min')[0].text),
        #        int(self.x_root.xpath('//domain/x_max')[0].text),
        #    ],
        #    ylim=[
        #        int(self.x_root.xpath('//domain/y_min')[0].text),
        #        int(self.x_root.xpath('//domain/y_max')[0].text),
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
        #self.fig.savefig(f'{s_path}/timeseries_step{str(self.iteration).zfill(3)}.jpeg', facecolor='white')


    def get_observation(self):
        """
        input:

        output:
            o_observation: object compatible with the defined
                observation space struct.

        description:
            data for the observation object for example be retrieved by:
            + physicell.get_parameter('my_parameter')
            + physicell.get_variable('my_variable')
            + physicell.get_vector('my_vector')
            however, there are no limts.
        """
        # model dependent processing logic goes here!
        o_observation = {'discrete': True}

        # output
        return o_observation


    def get_info(self):
        """
        input:

        output: 
            d_info: dictionary

        description:
            function to provide additional information important for
            controlling the action of the policy. for example,
            if we do reinforcement learning on a jump and run game,
            the number of hearts (lives left) from our character.
        """
        # model dependent processing logic goes here!
        d_info = {}

        # output
        return d_info


    def get_terminated(self):
        """
        input:

        output:
            b_terminated: bool

        description:
            function to determ, if the epoch is terminated.
            for example, if we do reinforcement learning on a
            jump and run game, if ouer character died.
            please notice, that this ending is different form
            truncated (the epoch reached the max time limit).
        """
        # model dependent processing logic goes here!
        b_terminated = False

        # output
        return b_terminated


    def get_reward(self):
        """
        input:

        output:
            r_reward: float between or equal to 0.0 and 1.0.
                there are no other limits to the algorithm implementation enforced.
                however, the algorithm is usually based on data as well retrived
                by the get_observation function (o_observation, d_info),
                and possibly by the render function (a_img).

        description:
            cost function.
        """
        # model dependent processing logic goes here!
        r_reward = 0.0

        # output
        return r_reward

