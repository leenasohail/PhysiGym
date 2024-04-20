#####
# title: pysigym/envs/physicell_core.py
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
from IPython import display
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


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
        "render_fps": None,
    }


    ### begin dummy functions ###

    def _get_action_space(self):
        sys.exit('_get_action_space function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_observation_space(self):
        sys.exit('_get_observation_space function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_fig(self):
        sys.exit('_get_fig function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_observation(self):
        sys.exit('_get_observation function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_info(self):
        sys.exit('_get_info function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_terminated(self):
        sys.exit('_get_terminated function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_truncated(self):
        sys.exit('_get_terminated function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_reward(self):
        sys.exit('_get_terminated function to be implemented in physigym.ModelPhysiCellEnv!')


    ### end dummy functions ###


    def __init__(self, settingxml='config/PhysiCell_settings.xml', figsize=(8,6), render_mode=None, verbose=True):
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
        if self.verbose:
            print(f'physigym: initialize enviroment ...')

        # load physicell settings.xml file
        s_pathfile_settingxml = settingxml
        x_tree = etree.parse(s_pathfile_settingxml)
        if self.verbose:
            print(f'physigym: reading {s_pathfile_settingxml}')
        self.x_root = x_tree.getroot()

        # initialize class whide variables
        if self.verbose:
            print(f'physigym: declare class instance wide variables.')
        self.iteration = None

        # handle render mode and figsize
        if self.verbose:
            print(f'physigym: declare render settings.')
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.figsize = figsize

        # handle spaces
        if self.verbose:
            print(f'physigym: declare action and observer space.')
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # output
        if self.verbose:
            print(f'physigym: ok!')


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
        if self.verbose:
            print(f'physigym: render frame.')
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
        if self.verbose :
            print(f'physigym: render time snap shot ...')

        # processing
        a_img = self._render_frame()

        # output
        if self.verbose :
            print(f'ok!')
        return a_img


    def reset(self, seed=-1, options={}):
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
        if self.verbose :
            print(f'physigym: reset epoche ...')

        # seed self.np_random number generator
        if self.verbose:
            print(f'physigym: seed random number generator.')
        if (seed is None) or (seed >= 0):
            i_seed = seed
        else:
            i_seed = int(self.x_root.xpath('//random_seed')[0].text)
        super().reset(seed=i_seed)

        # initialize physcell model
        if self.verbose:
            print(f'physigym: declare PhysiCell model instance.')
        os.makedirs(self.x_root.xpath('//save/folder')[0].text, exist_ok=True)
        physicell.start()

        # observe domain
        if self.verbose:
            print(f'physigym: domain observation.')
        o_observation = self._get_observation()
        d_info = self._get_info()

        # render domain
        self._render_frame()

        # output
        self.iteration = 0
        if self.verbose:
            print(f'physigym: ok!')
        return o_observation, d_info


    def step(self, action):
        """
        input:
            action: dict

        output:
            o_observation: object
            r_reward: float or int or bool
            b_terminated: bool
            b_truncated: bool
            info: dict

        description:
            Perform a simulation step with the given action.
        """
        if self.verbose :
            print(f'physigym: do a dt_gym time step ...')

        # get observation
        if self.verbose:
            print(f'physigym: domain observation.')
        o_observation = self._get_observation()
        b_terminated = self._get_terminated()
        b_truncated = self._get_truncated()
        d_info = self._get_info()

        # get revard
        r_reward = self._get_reward()

        # do rendering
        self._render_frame()

        # do action
        if self.verbose:
            print(f'physigym: action.')
        for s_action, o_value in action.items():

            # parameter action
            if type(o_value) in {bool, int, float, str}:
                physicell.set_parameter(s_action, o_value)

            elif type(o_value) in {list, tuple, set, np.array}:

                # vector action
                if type(o_value[0]) in {list, tuple, np.array}:
                    physicell.set_vector(s_action, o_value)

                # variable action
                else:
                    physicell.set_variable(s_action, o_value)

            # error
            else:
                sys.exit(f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : {s_action} {type(o_value)} unknowen variable type detected!.")

        # do dt_gym time step
        if self.verbose:
            print(f'physigym: dt_gym PhysiCell model time step.')
        physicell.step()

        # output
        self.iteration += 1
        if self.verbose:
            print(f'physigym: ok!')
        return o_observation, r_reward, b_terminated, b_truncated, d_info


    def close(self):
        """
        description:
        """
        if self.verbose :
            print(f'physigym: epoche closure ...')

        # processing
        if self.verbose:
            print(f'physigym: shut down PhysiCell model run.')
        physicell.stop()

        # output
        if self.verbose:
            print(f'physigym: ok!')


    def verbose_true(self):
        """
        """
        self.verbose = True


    def verbose_false(self):
        """
        """
        self.verbose = False

