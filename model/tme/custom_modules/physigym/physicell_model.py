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
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
from matplotlib import cm, colors
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
        nb_steps_max=100,
        observation_type="simple",
    ):
        self.observation_type = "simple" if None else observation_type
        if self.observation_type not in ["simple", "image"]:
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

        self.cell_count_cancer_cell_target = int(
            self.x_root.xpath("//cell_count_cancer_cell_target")[0].text
        )


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
                "drug_apoptosis": spaces.Box(
                    low=0.0, high=30, shape=(1,), dtype=np.float16
                ),
                "drug_reducing_antiapoptosis": spaces.Box(
                    low=0.0, high=30, shape=(1,), dtype=np.float16
                ),
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
                low=0, high=2**8, shape=(1,), dtype=np.float32
            )
        elif self.observation_type == "image":
            # Define the Box space for the image
            o_observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
        elif self.observation_type == "image_rgb_first":
            # Define the Box space for the image
            o_observation_space = spaces.Box(
                low=0, high=255, shape=(3, self.height, self.width), dtype=np.uint8
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
        # model dependent observation processing logic goes here!
        if self.observation_type == "simple":
            o_observation = np.array([physicell.get_parameter("count_cancer_cell")])
            o_observation = (
                o_observation.astype(float)[0] / self.cell_count_cancer_cell_target
            )
            o_observation = np.array([o_observation], dtype=np.float64)
            o_observation = np.clip(
                o_observation, self.observation_space.low, self.observation_space.high
            )

        elif self.observation_type == "image" or self.observation_type == "image_rgb_first":
            x = self.df_cell["x"].to_numpy()
            y = self.df_cell["y"].to_numpy()
            cell_id = self.df_cell["ID"].to_numpy()
            # Normalizing the coordinates to fit into the image grid
            x_normalized = (x - self.x_min).astype(int)
            y_normalized = (y - self.y_min).astype(int)
            # Extracting the x, y coordinates and cell id into a numpy array
            self.df_cell["color"] = self.df_cell["type"].map(lambda t: self.color_mapping.get(t, (0,0,0)))  # Default to black if type not found
            self.df_cell["color"] = self.df_cell.apply(lambda row: [0,0,0] if row["dead"] != 0.0 else row["color"], axis=1)

            if self.observation_type == "image":
                o_observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # Assign colors to the image grid
                for i in range(len(cell_id)):
                    o_observation[x_normalized[i], y_normalized[i], :] = self.df_cell["color"].iloc[i]

            elif self.observation_type == "image_rgb_first":
                o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)
                # Assign colors to the image grid
                for i in range(len(cell_id)):
                    o_observation[:, x_normalized[i], y_normalized[i]] = self.df_cell["color"].iloc[i]
            else:
                raise f"Observation type: {self.observation_type} does not exist"
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
        info = {"number_cancer_cells": physicell.get_parameter("count_cancer_cell"),
                "df_cell":self.df_cell}

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
        cancer_cell = physicell.get_parameter("count_cancer_cell")
        b_terminated = (
            cancer_cell <= self.cell_count_cancer_cell_target // 4
            or cancer_cell > 1.5 * self.cell_count_cancer_cell_target
        )
        return b_terminated

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
        return -np.abs(physicell.get_parameter("count_cancer_cell") - self.cell_count_cancer_cell_target)# **(1+(physicell.get_parameter('time') /self.r_time_max))

    def get_img(self):
        """
        input:

        output:
            self.fig.savefig
                instance attached matplotlib figure.

        run:
            internal function, user defined.

        description:
            template code to generate a matplotlib figure from the data.
            for example from:
            + physicell.get_microenv('my_substrate')
            + physicell.get_cell()
            + physicell.get_variable('my_variable')
            however, there are no limits.
        """
        # model dependent img processing logic goes here!

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.axis("equal")
        # ax.axis('off')

        ##################
        # substrate data #
        ##################

        df_conc = pd.DataFrame(
            physicell.get_microenv("drug_apoptosis"),
            columns=["x", "y", "z", "drug_apoptosis"],
        )
        df_conc = df_conc.loc[df_conc.z == 0.0, :]
        df_mesh = df_conc.pivot(index="y", columns="x", values="drug_apoptosis")
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Reds",
            # alpha=0.5,
        )

        #######################
        # substrate color bar #
        #######################

        self.fig.colorbar(
            mappable=cm.ScalarMappable(
                norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="Reds"
            ),
            label="drug_apoptosis",
            ax=ax,
        )

        #############
        # cell data #
        #############

        df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "type"]
        )

        df_cell = df_cell.loc[df_cell.z == 0.0, :]
        df_cell.plot(
            kind="scatter",
            x="x",
            y="y",
            xlim=[
                int(self.x_root.xpath("//domain/x_min")[0].text),
                int(self.x_root.xpath("//domain/x_max")[0].text),
            ],
            ylim=[
                int(self.x_root.xpath("//domain/y_min")[0].text),
                int(self.x_root.xpath("//domain/y_max")[0].text),
            ],
            grid=True,
            title=f"dt_gym env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} / {physicell.get_parameter('count_nurse_cell') + physicell.get_parameter('count_cancer_cell')} [cell]",
        )
        ###############
        # save to file #
        ################

        # plt.tight_layout()
        # s_path = self.x_root.xpath('//save/folder')[0].text
        # self.fig.savefig(f'{s_path}/timeseries_step{str(self.step_env).zfill(3)}.jpeg', facecolor='white')
    @staticmethod
    def show(o_observation):
        """
        TO DO
        """
        import mpld3
        # Plot the image using imshow
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(o_observation)

        # Set titles and labels
        ax.set_title("Cell Position")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")

        # Use mpld3 to render the plot and open in the default browser
        mpld3.show()
