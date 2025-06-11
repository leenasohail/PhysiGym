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
#     model specific implementation of the custom_modules/extend module
#     comaptible Gymnasium environment.
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
#####


# library
from extending import physicell
from gymnasium import spaces
import gymnasium as gym
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

    run:
        import gymnasium
        import physigym

        env = gymnasium.make('physigym/ModelPhysiCellEnv')

        o_observation, info = env.reset()
        o_observation, r_reward, b_terminated, b_truncated, info = env.step(action={})
        env.close()

    description:
        this is the model physigym environment class, built on top of the
        physigym.CorePhysiCellEnv class, which is built on top of the
        gymnasium.Env class.

        fresh from the PhysiGym repo this is only a template class!
        you will have to edit this class, to specify the model specific
        reinforcement learning environment.
    """

    def __init__(
        self,
        settingxml="config/PhysiCell_settings.xml",
        figsize=(8, 6),
        render_mode=None,
        render_fps=10,
        verbose=False,
        observation_type="simple",
        reward_type="normalize",
    ):
        self.observation_type = "simple" if None else observation_type
        if self.observation_type not in ["simple", "image_gray", "transformer"]:
            raise ValueError(
                f"Error: unknown observation type: {self.observation_type}"
            )

        # Corrected usage of super()
        super().__init__(
            settingxml=settingxml,
            figsize=figsize,
            render_mode=render_mode,
            render_fps=render_fps,
            verbose=verbose,
        )
        # self.init_cancer_cells = int(
        #     self.x_root.xpath("//user_parameters/number_of_tumor")[0].text
        # )
        self.init_cancer_cells = 512
        self.nb_cell_types = len(self.unique_cell_types)
        self.np_ratio_nb_cancer_cells = None
        self.np_ratio_old_nb_cancer_cells = None
        self.reward_type = reward_type
        self.type_map = {t: i for i, t in enumerate(self.unique_cell_types)}

    def get_action_space(self):
        """
        input:

        output:
            d_action_space: dictionary composition space
                the dictionary keys have to match the parameter,
                custom variable, or custom vector label.
                the value has to be defined as gymnasium space object.
                + https://gymnasium.farama.org/main/api/spaces/
        run:
            internal function, user defined.

        description:
            dictionary structure built out of gymnasium.spaces elements.
            this struct has to specify type and range for each
            action parameter, action custom variable, and action custom vector.
        """

        # model dependent action_space processing logic goes here!
        d_action_space = spaces.Dict(
            {
                "drug_1": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float16),
            }
        )

        # output
        return d_action_space

    def get_observation_space(self):
        """
        input:

        output:
            o_observation_space structure.
                the struct have to be built out of gymnasium.spaces elements.
                there are no other limits.
                + https://gymnasium.farama.org/main/api/spaces/

        run:
            internal function, user defined.

        description:
            data structure built out of gymnasium.spaces elements.
            this struct has to specify type and range
            for each observed variable.
        """
        # template
        # observation_space =
        # compositione: spaces.Dict({})
        # compositione: spaces.Tuple(())
        # discrete: spaces.Discrete()  # boolean, integer
        # discrete: spaces.Text()  # string
        # discrete: spaces.MultiBinary()  # boolean numpy array
        # discrete: spaces.MultiDiscrete() # boolean, integer numpy array
        # numeric: spaces.Box()  # boolean, integer, float numpy array
        # niche: spaces.Graph(())
        # niche: spaces.Sequence(())  # set of spaces

        # model dependent observation_space processing logic goes here!
        if self.observation_type == "simple":
            o_observation_space = spaces.Box(
                low=0,
                high=2**16,
                shape=(len(self.unique_cell_types),),
                dtype=np.float32,
            )
        elif self.observation_type == "image_gray":
            # Define the Box space for the image
            o_observation_space = spaces.Box(
                low=0, high=255, shape=(1, self.height, self.width), dtype=np.uint8
            )
        elif self.observation_type == "transformer":
            o_observation_space = spaces.Dict(
                {
                    "type": spaces.Sequence(
                        spaces.Discrete(len(self.unique_cell_types))
                    ),
                    "dead": spaces.Sequence(spaces.Discrete(2)),
                    "pos": spaces.Sequence(
                        spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
                    ),
                }
            )

        else:
            raise f"Error unknown observation type: {o_observation_space}"

        # output
        return o_observation_space

    def get_observation(self):
        """
        input:

        output:
            o_observation: object compatible with the defined
                observation space struct.

        run:
            internal function, user defined.

        description:
            data for the observation object for example be retrieved by:
            + physicell.get_parameter('my_parameter')
            + physicell.get_variable('my_variable')
            + physicell.get_vector('my_vector')
            however, there are no limits.
        """
        self.df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "type"]
        )

        self.nb_cancer_cells = len(
            self.df_cell.loc[
                (self.df_cell.dead == 0.0) & (self.df_cell.type == "tumor"), :
            ]
        )

        self.nb_cell_1 = len(
            self.df_cell.loc[
                (self.df_cell.dead == 0.0) & (self.df_cell.type == "cell_1"), :
            ]
        )
        self.nb_cell_2 = len(
            self.df_cell.loc[
                (self.df_cell.dead == 0.0) & (self.df_cell.type == "cell_2"), :
            ]
        )

        self.np_ratio_old_nb_cancer_cells = (
            self.np_ratio_nb_cancer_cells
            if self.np_ratio_nb_cancer_cells is not None
            else None
        )
        self.np_ratio_nb_cancer_cells = np.array(
            [self.nb_cancer_cells / self.init_cancer_cells], dtype=np.float64
        )

        self.np_ratio_old_nb_cancer_cells = (
            self.np_ratio_nb_cancer_cells
            if self.np_ratio_old_nb_cancer_cells is None
            else self.np_ratio_old_nb_cancer_cells
        )
        # model dependent observation processing logic goes here!

        if self.observation_type == "simple":
            normalized_concentration_cells = np.zeros((self.nb_cell_types,))
            for i in range(self.nb_cell_types):
                normalized_concentration_cells[i] = len(
                    self.df_cell.loc[
                        (self.df_cell.dead == 0.0)
                        & (self.df_cell.type == self.unique_cell_types[i]),
                        :,
                    ]
                )
            o_observation = normalized_concentration_cells / self.init_cancer_cells - 1
            o_observation = np.array(o_observation, dtype=float)

        elif self.observation_type == "image_gray":
            x = self.df_cell["x"].to_numpy()
            y = self.df_cell["y"].to_numpy()
            cell_id = self.df_cell["ID"].to_numpy()
            # Normalizing the coordinates to fit into the image grid
            x_normalized = (x - self.x_min).astype(int)
            y_normalized = (y - self.y_min).astype(int)
            # Extracting the x, y coordinates and cell id into a numpy array
            self.df_cell["color"] = self.df_cell["type"].map(
                lambda t: self.color_mapping.get(t, (0, 0, 0))
            )  # Default to black if type not found
            self.df_cell["color"] = self.df_cell.apply(
                lambda row: [0, 0, 0] if row["dead"] != 0.0 else row["color"], axis=1
            )

            o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)
            # Assign colors to the image grid
            for i in range(len(cell_id)):
                o_observation[:, x_normalized[i], y_normalized[i]] = self.df_cell[
                    "color"
                ].iloc[i]

            grayscale_image = np.dot(
                o_observation.transpose(1, 2, 0),
                [0.2989, 0.5870, 0.1140],
            ).astype(np.uint8)
            o_observation = grayscale_image[np.newaxis, :, :]
            o_observation = np.array(o_observation, dtype=float)

        elif self.observation_type == "transformer":
            x = self.df_cell["x"].to_numpy()
            y = self.df_cell["y"].to_numpy()
            cell_id = self.df_cell["ID"].to_numpy()
            # Normalizing the coordinates to fit into the image grid
            x_normalized = (x - self.x_min) / (self.x_max - self.x_min)
            y_normalized = (y - self.y_min) / (self.y_max - self.y_min)
            types = (
                self.df_cell["type"]
                .map(self.type_map)
                .fillna(0)
                .to_numpy(dtype=np.int64)
                .reshape(-1, 1)
            )
            dead = self.df_cell["dead"].to_numpy(dtype=np.int64).reshape(-1, 1)
            o_observation = {
                "type": types,  # shape (num_cells,)
                "dead": dead,  # shape (num_cells,)
                "pos": np.stack(
                    [x_normalized, y_normalized], axis=1
                ),  # shape (num_cells, 2)
            }
        else:
            raise f"Observation type: {self.observation_type} does not exist"
        return o_observation

    def get_info(self):
        """
        input:

        output:
            info: dictionary

        run:
            internal function, user defined.

        description:
            function to provide additional information important for
            controlling the action of the policy. for example,
            if we do reinforcement learning on a jump and run game,
            the number of hearts (lives left) from our character.
        """
        # model dependent info processing logic goes here!
        info = {
            "number_cancer_cells": self.nb_cancer_cells,
            "df_cell": self.df_cell,
            "number_cell_1": self.nb_cell_1,
            "number_cell_2": self.nb_cell_1,
        }

        # output
        return info

    def get_terminated(self):
        """
        input:

        output:
            b_terminated: bool

        run:
            internal function, user defined.

        description:
            function to determine if the episode is terminated.
            for example, if we do reinforcement learning on a
            jump and run game, if our character died.
            please notice, that this ending is different form
            truncated (the episode reached the max time limit).
        """
        # model dependent terminated processing logic goes here!
        return True if self.nb_cancer_cells == 0 else False
        # return False

    def get_reward(self):
        """
        input:

        output:
            r_reward: float between or equal to 0.0 and 1.0.
                there are no other limits to the algorithm implementation enforced.
                however, the algorithm is usually based on data retrieved
                by the get_observation function (o_observation, info),
                and possibly by the render function (a_img).

        run:
            internal function, user defined.

        description:
            cost function.
        """
        C_t = self.np_ratio_nb_cancer_cells * self.init_cancer_cells
        C_prev = C_prev = self.np_ratio_old_nb_cancer_cells * self.init_cancer_cells
        return self.normalize(C_t=C_t, C_prev=C_prev)

    def normalize(self, C_t, C_prev):
        if self.reward_type == "sparse":
            if C_prev * C_t == 0:
                return 100
            elif C_prev > C_t:
                return 1
            else:
                return -1
        elif self.reward_type == "linear":
            return (C_prev - C_t) / np.log(C_prev + 1)
        elif self.reward_type == "simple":
            return 1 if C_prev > C_t else 0
        else:
            raise f"The reward type is not implemented{self.reward_type}"

    def get_reset_values(self):
        self.np_ratio_old_nb_cancer_cells = None
        self.np_ratio_nb_cancer_cells = None
        return None
