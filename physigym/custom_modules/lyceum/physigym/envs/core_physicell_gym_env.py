#####
# title: lyceum/envs/core_physicell_gym_env.py
#
# language: python3
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     gymnasium enviroemnt for physicell embedding
#     + https://gymnasium.farama.org/tutorials/gymnasium_basics/
#####


# library
from embedding import physicell                                                 
import gymnasium as gym                                                         
from gymnasium import spaces                                                    
from physigym import utils                                                          


#import shutil
#import os
#import sys
#import pickle
#from typing import Union
#from distutils.dir_util import copy_tree
#from IPython.display import display, HTML

#abs_path_pc = os.path.abspath(__file__)[
#    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
#]
#sys.path.append(abs_path_pc)
#sys.path.append(abs_path_pc + "/custom_modules/build")


class CorePhysiCellEnv(gym.Env):

    # bue 20240417: tutorial
    # metadata 
    #metadata = {"key": value, "key": value}
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # spaces


    def __init__(self, config: dict, config_xml: dict):

        # bue 20240417: tutorial
        # self.observation_space
        # self.action_space
        # self._action_to_

        # check if render_mode is specified and set it
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.render_mode = render_mode

        # variabel declaration
        #self.window = None
        #self.clock = None

        # modify physicell settings according to user settings
        self.config_xml = config_xml
        self.xml_file_helper = utils.xml.xml_helper_first_model(config_xml)
        self.xml_file_helper.modify_physicell_settings(self.config_xml)

        # should save PHysicellSettings.xml, cells_rules.csv, custom.cpp and custom.h, custom_modules.cpp
        signature_c = (
            self.xml_file_helper.get_hash_physicell_settings_core()
        )

        signature_d = self.xml_file_helper.get_hash_physicell_settings_specific()

        assert signature_c != signature_d

        self.config = config
        self.new_folder_path = None


    # bue 20240417: tutorial
    #def _get_obs(self) -> dict:
    #    pass

    # bue 20240417: tutorial
    #def _get_info(self) -> dict:
        # bue internal function 
        """function which outputs a dictionnary with valuable information in order to debug

        Returns:
            dict: dictionnary _description_
        """
    #    raise "To be implemented"

    def reset(self):
        """
        The reset method will be called to initiate a new episode.
        You may assume that the step method will not be called before reset has been called.
        """
        super().reset(seed=self.config["seed"])
        physicell.start()  # starting the env call
        self._initial_action = 0
        self.iterator = 0

    #def _reward_function(self, *args, **kwargs):
        """function computing the reward
        Returns:
            ?: reward value
        """
    #    raise "To be implemented"


    # bue 20240417: tutorial
    #def step(self, action):
    #    pass
    #    return observation, reward, terminated, False, info


    #def render(self):


    #def svg_displayer(self, path: Union[str, None] = None) -> None:
        """_summary_

        Args:
            path (Optional[str]): _description_

        Returns:
            _type_: _description_
        """
    #    path = (
    #        self.config["output_folder_path"]
    #        + f"/snapshot{str(self.iterator).zfill(8)}.svg"
    #        if path is None
    #        else path
    #    )
    #    with open(path, "r") as svg_file:
    #        svg_content = svg_file.read()
    #        display(HTML(f"<div>{svg_content}</div>"))
    #    return None

    #def close(self):

    def close(self, exp_folder_path: str) -> None:
        """function ending the simulation and transfers all data to the specified folder.


        Args:
            exp_folder_path (str): The folder where you want to store the simulation data.

        """
        physicell.stop()
        self.move_exp(exp_folder_path)

    #def move_exp(self, exp_folder_path: str):
        """moves the experience from the output folder to experience folder path
        find the maximum iteration among your different episodes and save in a next folder all your outputs
        Args:
            exp_folder_path (str): The folder where you want to store the simulation data.
        """

        # Construct the path for the core model experiment folder using the experiment folder path and the hash of core PhysiCell settings
    #    path_exp_model_core = (
    #        exp_folder_path
    #        + str(self.xml_file_helper.get_hash_physicell_settings_core())
    #        + str("_hash_core")
    #    )

        # Construct the path for the specific model experiment folder using the path of the core model experiment folder and the hash of specific PhysiCell settings
    #    path_exp_model_specific = (
    #        path_exp_model_core
    #        + "/"
    #        + str(self.xml_file_helper.get_hash_physicell_settings_specific())
    #    )

        # Create the core model experiment folder if it doesn't exist
    #    os.makedirs(path_exp_model_core, exist_ok=True)

        # Create the specific model experiment folder if it doesn't exist
    #    os.makedirs(path_exp_model_specific, exist_ok=True)

        # Copy the PhysiCell XML configuration file to the core model experiment folder

    #    physicell_settings_xml = self.config_xml["xml_file_path"].rfind("/")
    #    physicell_path_config = self.config_xml["xml_file_path"][
    #        :physicell_settings_xml
    #    ]

    #    copy_tree(src=physicell_path_config, dst=path_exp_model_core)

        # Create the core model experiment folder
    #    os.makedirs(path_exp_model_core, exist_ok=True)

        # Save the configuration dictionary as a JSON file in the specific model experiment folder (file name: "config.json")
    #    utils.json.save_dict_to_json(
    #        self.config, path_exp_model_specific, "config"
    #    )  # saving config as json
        # Save the PhysiCell XML configuration dictionary as a JSON file in the specific model experiment folder (file name: "config_xml_file.json")
    #    utils.json.save_dict_to_json(
    #        self.config_xml, path_exp_model_specific, "config_xml_file"
    #    )  # saving config_xml_file as json

        #pickle_file_path = os.path.join(path_exp_model_specific,"return_data.pickle")
        #with open(pickle_file_path,"wb") as file:
            #pickle.dump(self.data_save,file)
        # Create a list of episode folders in the specific model experiment directory and extract their last 8 digits
    #    list_nb_episode_folder = [
    #        int(folder[-8:])
    #        for folder in os.listdir(path_exp_model_specific)
    #        if folder.startswith("episode") and folder[-8:].isdigit()
    #    ]

        # Generate a new episode folder name by finding the maximum numeric suffix in the existing episode folders and incrementing it
   #     new_folder_name = (
   #         f"episode_{str(max(list_nb_episode_folder)+1).zfill(8)}"
   #         if len(list_nb_episode_folder) > 0
   #         else f"episode_{str(0).zfill(8)}"
   #     )

        # Construct the path for the new episode folder
   #     new_folder_path = path_exp_model_specific + "/" + new_folder_name

   #     os.makedirs(new_folder_path, exist_ok=True)
        # Move the contents of the output folder to the new episode folder
   #     copy_tree(self.config["output_folder_path"], new_folder_path)
   #     shutil.rmtree(self.config["output_folder_path"])
        # Recreate the output folder after moving its contents
   #     os.makedirs(self.config["output_folder_path"], exist_ok=True)
   #     self.new_folder_path = new_folder_path


    def step(self, action):
        """Perform a simulation step with the given action.

        Args:
            action (?): _description_

        Returns:
            _type_: _description_
        """
        raise "To be implemented"


#if __name__ == "__main__":
#    __file__ = os.getcwd()  # not useful
#    pc_path = os.path.abspath(__file__)[
#        : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
#    ]
#    output_folder_path = abs_path_pc + "/output"
#    xml_file_path = abs_path_pc + "/config/PhysiCell_settings.xml"
#    config_xml_file = {
#        "xml_file_path": xml_file_path,
#        "random_seed": 5,
#        "max_time": 2880,
#        "cell_count": 100,
#    }
#    config_pc_env = {
#        "custom_variable": "drug",
#        "render": None,
#        "seed": 5,
#        "output_folder_path": output_folder_path,
#        "display": False,
#    }
#    env = CorePhysiCellEnv(config_pc_env, config_xml_file)

