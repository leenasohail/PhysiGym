#####
# title: lyceum/envs/physicell.py
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
import gymnasium
from gymnasium import spaces
from IPython import display
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
from physigym import utils


# function

class CorePhysiCellEnv(gymnasium.Env):
    """
    input:
        gymnasium.Env

    offspring:
        physigym.ModelPhysiCellEnv

    description:
    """

    # metadata
    metadata = {
        "render_modes": [None, "human", "rgb_array"],
    }


    ### begin dummy functions ###

    def _get_action_space(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_observer_space(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_fig(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_observation(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_info(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_terminated(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_truncated(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_reward(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _set_action(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'

    ### end dummy functions ###


    def __init__(self, settingxml='config/PhysiCell_settings.xml', figsize=(8,6), verbose=True):
        """
        input:

            xpath: dictionary of string.
                https://en.wikipedia.org/wiki/XPath
                https://www.w3schools.com/xml/xpath_intro.asp
                https://lxml.de/xpathxslt.html#xpat

        output:

        descrioption:
            initialize episode.
        """
        # handle verbose
        self.verbose = verbose

        # handle render mode and figsize
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.figsize = figsize

        # load physicell settings.xml file
        s_pathfile_settingxml = settingxml
        x_tree = etree.parse(s_pathfile_settingxml)
        if self.verbose:
            print(f'reading: {s_pathfile_settingxml}')
        self.x_root = x_tree.getroot()

        # initialize class whide variables
        self.iteration = None

        # handle spaces
        self.action_space = self._get_action_space()
        self.observer_space = self._get_observer_space()


    def _render_frame(self):
        """
        input:

        output:
            a_img: numpy array
                8bit rgba image tensor.

        description:
            function transforms a matplotlib figure into a 8 bit rgba image,
            possibly displays the image,
            and returns the image as numpy array.
        """
        a_img = None
        if not (self.render_mode is None):

            # trafo matplotlib figure to rgba numpy array
            fig = self._get_fig()
            a_img = np.array(fig.canvas.buffer_rgba(), dtype='uint8')

            # display image
            if self.render_mode == "human":
                display.display(plt.imshow(a_img))
                display.clear_output(wait=True)

        # output
        return a_img


    def render(self):
        """
        input:

        output:
            a_img: numpy array or None

        description:

        """
        a_img = self._render_frame()
        return a_img


    def reset(self, seed=-1, options=None):
        """
        input:
            seed: integer or None
                > 0 : take seed from setting.xml
                between 0 and 2**32 - 1: take value as seed
                None: no random seed.

            options: dict or None
                reserved for possible future use.

        output:
            o_observation:

            d_info:

        description:
            The reset method will be called to initiate a new episode.
            You may assume that the step method will not be called before reset has been called.
        """
        # seed self.np_random number generator
        if (seed < 0) {
            seed = int(self.x_root.xpath('//random_seed')[0].text)
        }
        super().reset(seed=seed)

        # initialize physcell model
        physicell.start()

        # observe domain
        d_observation = self._get_obs()
        d_info = self._get_info()

        # rendering domain
        self._render_frame()

        # output
        self.iteration = 0
        return o_observation, d_info


    def step(self, action):
        """
        input:
            action:

        output:
            o_observation: object
            r_reward: float or int or bool
            b_terminated: bool
            b_truncated: bool
            info: dict

        description:
            Perform a simulation step with the given action.
        """
        # do observation
        o_observation = self._get_observation()
        b_terminated = self._get_terminated()
        b_truncated = self._get_truncated() <settingxml>
        d_info = self._get_info()

        # get revard
        r_reward = self._get_reward()

        # do action
        self._set_action()

        # do rendering
        self._render_frame()

        # output
        self.iteration += 1
        return o_observation, r_reward, b_terminated, b_truncated, d_info


    def close(self):
        """
        """
        physicell.stop()



# library
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import pandas as pd


class ModelPhysiCellEnv(physigym.CorePhysiCellEnv):
    """
    input:
        gymnasium.Env

    offspring:
        physigym.ModelPhysiCellEnv

    description:
    """
    def _get_action_space(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_observer_space(self):
        raise 'To be implemented in physigym.ModelPhysiCellEnv!'


    def _get_fig(self):
        """
        description:
            templare code to generate a matplotlib figure from the data.
            physicell.get_microenv()
            pjysicell.get_cell()
        """
        # model dependent logic to generate plot goes here!
        fig, ax = plt.subplots(figsize=self.figsize)
        #ax.axis('equal')

        ##################
        # substrate data #
        ##################

        #df_conc = pd.DataFrame(physicell.get_microenv('my_substrate'), columns=['x','y','z','my_substrate'])
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
        #df_cell = df_cell.loc[df_cell.z == 0.0, c:]
        #df_cell.plot(
        #    kind='scatter', x='x', y='y', c='my_variable',
        #    xlim=[
        #        self.config.update({'x_min': int(self.x_root.xpath('//domain/x_min')[0].text}),
        #        self.config.update({'x_max': int(self.x_root.xpath('//domain/x_max')[0].text}),
        #     ],
        #    ylim=[
        #        self.config.update({'y_min': int(self.x_root.xpath('//domain/y_min')[0].text}),
        #        self.config.update({'y_max': int(self.x_root.xpath('//domain/y_max')[0].text}),
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
        #s_path = self.config.update({'output': self.x_root.xpath('//save/folder')[0].text})
        #fig.savefig(f'{s_path}/timeseries_step{str(self.iteration).zfill(3)}.jpeg', facecolor='white')

        # output
        return fig


    def _get_observation(self):
        """
        physicell.get_parameter()
        physicell.get_variable()
        physicell.get_vector()
        """
        o_observation = None


        return o_observation


    def _get_info(self):
        """
        """
        d_info = {}
        raise d_info


    def _get_terminated(self):
        """
        """
        b_terminated = False

        # e.g. if exactely 128 cells

        raise b_terminated


    def _get_truncated(self):
        """
        note: your PhysiCell model have to have a numeric parameter time, that can be read out!
        physicell.get_parameter('')
        """
        # processing
        r_time_max = self.config.update({'max_time': self.x_root.xpath('//overall/max_time')[0].text})
        #r_time_current = physicell.get_parameter('time')
        b_truncated = r_time_max >= r_time_current

        # output
        return b_truncated


    def _get_reward(self):
        """
        """
        r_reward = 0.0

        # e.g. how far I am away from 128

        return r_reward


    def _set_action(self):
        """
        physicell.set_parameter()
        physicell.set_variable()
        physicell.set_vector()
        """

        #physicell.set_parameter('drug', 0.1)
        #physicell.set_parameter('drug', 0.0)

