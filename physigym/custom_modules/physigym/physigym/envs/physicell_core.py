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
#####


# library
from embedding import physicell
import gymnasium
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

    ### begin dummy functions ###

    def _get_action_space(self):
        sys.exit('_get_action_space function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_observation_space(self):
        sys.exit('_get_observation_space function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_img(self):
        sys.exit('_get_img function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_observation(self):
        sys.exit('_get_observation function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_info(self):
        sys.exit('_get_info function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_terminated(self):
        sys.exit('_get_terminated function to be implemented in physigym.ModelPhysiCellEnv!')


    def _get_reward(self):
        sys.exit('_get_terminated function to be implemented in physigym.ModelPhysiCellEnv!')


    ### end dummy functions ###


    # metadata
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }


    def __init__(self, settingxml='config/PhysiCell_settings.xml', figsize=(8,6), render_mode=None, render_fps=None, verbose=True):
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
        assert render_mode is None or render_mode in self.metadata['render_modes'], f"'{render_mode}' is an unknowen render_mode. known are {sorted(self.metadata['render_modes'])}, and None."
        self.render_mode = render_mode
        self.metadata.update({'render_fps': render_fps})
        self.figsize = figsize

        # handle spaces
        if self.verbose:
            print(f'physigym: declare action and observer space.')
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # output
        if self.verbose:
            print(f'physigym: ok!')


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
        if (seed is None) or (seed >= 0):
            i_seed = seed
            if self.verbose:
                print(f'physigym: seed random number generator with {i_seed}.')
        else:
            i_seed = int(self.x_root.xpath('//random_seed')[0].text)
            if self.verbose:
                print(f'physigym: seed random number generator with {i_seed}, the value the from setting.xml file.')
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
        if self.verbose:
            print(f'physigym: render {self.render_mode} frame.')
        if not (self.render_mode is None):
            a_img = self._get_img()
            if (self.render_mode == 'human'):
                plt.cla()
                plt.imshow(a_img)
                plt.pause(self.metadata['render_fps'])

        # output
        self.iteration = 0
        if self.verbose:
            print(f'physigym: ok!')
        return o_observation, d_info


    def render(self):
        """
        input:

        output:
            a_img: numpy array or None

        description:

        """
        if self.verbose :
            print(f'physigym: render {self.render_mode} frame ...')

        # processing
        a_img = None
        if not (self.render_mode is None):
            a_img = self._get_img()
            if (self.render_mode == 'human'):
                plt.cla()
                plt.imshow(a_img)
                plt.pause(self.metadata['render_fps'])
                a_img = None

        # output
        if self.verbose :
            print(f'ok!')
        return a_img


    def _get_truncated(self):
        """
        note: your PhysiCell model have to have a numeric parameter time, that can be read out!
        physicell.get_parameter('')
        """
        # processing
        b_truncated = False
        r_time_max = float(self.x_root.xpath('//overall/max_time')[0].text)
        r_time_current = physicell.get_parameter('time')  # achtung: time has to be declared as parameter of type float in the settings.xml file!
        b_truncated = r_time_current >= r_time_max

        # output
        return b_truncated


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
            print(f'physigym: taking a dt_gym time step ...')

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
        if self.verbose:
            print(f'physigym: render {self.render_mode} frame.')
        if (self.render_mode == 'human'):
            a_img = self._get_img()
            plt.cla()
            plt.imshow(a_img)
            plt.pause(self.metadata['render_fps'])

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
            print(f'physigym: PhysiCell model step.')
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
        run:
            env.unwrapped.verbose_true()
        """
        print(f'physigym: set env.verbose = True.')
        self.verbose = True


    def verbose_false(self):
        """
        run:
            env.unwrapped.verbose_false()
        """
        print(f'physigym: set env.verbose = False.')
        self.verbose = False

