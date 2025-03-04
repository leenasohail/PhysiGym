import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from embedding import physicell
import matplotlib.pyplot as plt
import pandas as pd
import os
class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_apoptosis",
            "drug_reducing_antiapoptosis",
        ]
    ):
        """
        Args:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
        """
        super().__init__(env)

        # Check that all variable names are strings
        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name
        self.unwrapped_env = env.unwrapped
        self.x_min = self.unwrapped_env.x_min
        self.x_max = self.unwrapped_env.x_max
        self.y_min = self.unwrapped_env.y_min
        self.y_max = self.unwrapped_env.y_max
        # Flatten the action space to match the expected two values
        low = np.array(
            [
                env.action_space[variable_name].low[0]
                for variable_name in list_variable_name
            ]
        )
        high = np.array(
            [
                env.action_space[variable_name].high[0]
                for variable_name in list_variable_name
            ]
        )
        self._action_space = Box(low=low, high=high, dtype=np.float64)

    @property
    def action_space(self):
        """Returns the flattened action space for the wrapper."""
        return self._action_space

    def reset(self, seed=None, options={}):
        """
        Resets the environment and preprocesses the observation.
        """
        o_observation, info = self.env.reset(seed=seed, options=options)

        # Preprocess observation (if needed)
        o_observation = np.array(o_observation, dtype=float)

        return o_observation, info

    def step(self, action: np.ndarray):
        """
        Steps through the environment using the flattened action.

        Args:
            action (np.ndarray): The flattened action array.

        Returns:
            Tuple: Observation, reward, terminated, truncated, info.
        """
        # Convert the flat action array to the dictionary expected by the env
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }
        # Take a step in the environment
        o_observation, r_reward, b_terminated, b_truncated, info = self.env.step(
            d_action
        )
        # Preprocess observation (if needed)
        o_observation = np.array(o_observation, dtype=float)
        info["action"]  = d_action
        self.info = info
        return o_observation, r_reward, b_terminated, b_truncated, info
    
    def render(self, path="./output/image", saving_title:str="output_simulation_image_episode"):
        os.makedirs(path,exist_ok=True)
        df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID','x','y','z','dead','cell_type'])
        fig, ax = plt.subplots(1, 3, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 0.2, 0.2]})
        count_cancer_cell = len(df_cell.loc[(df_cell.z == 0.0) & (df_cell.cell_type == 'cancer_cell'), :])

        for s_celltype, s_color in sorted({'cancer_cell': 'gray', 'nurse_cell': 'red'}.items()):
            df_celltype = df_cell.loc[(df_cell.z == 0.0) & (df_cell.cell_type == s_celltype), :]
            df_celltype.plot(
                kind='scatter', x='x', y='y', c=s_color,
                xlim=[
                   self.x_min,
                    self.x_max,
                ],
                ylim=[
                    self.y_min,
                    self.y_max,
                ],
                grid=True,
                label = s_celltype,
                s=100,
                title=f"episode step {str(self.unwrapped_env.step_episode).zfill(3)}, cancer cell: {count_cancer_cell}",
                ax=ax[0],
            ).legend(loc='lower left')


        # Create a colormap for the color bars (from -1 to 1)
        list_colors = ["royalblue","darkorange"]

        # Function to create fluid-like color bars
        def create_fluid_bar(ax_bar, drug_amount, title, max_amount=30, color="cyan"):
            ax_bar.set_xlim(0, 1)
            ax_bar.set_ylim(0, 1) 
            ax_bar.set_title(title, fontsize=10)
            ax_bar.set_xticks([])
            ax_bar.set_yticks(np.linspace(0, 1, 5))  # 0% to 100% scale

            # Normalize drug amount (convert to percentage of max)
            fill_level = drug_amount / max_amount 

            # Fill up to the corresponding level
            ax_bar.fill_betweenx(np.linspace(0, fill_level, 100), 0, 1, color=color)

            # Draw container border
            ax_bar.spines['left'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['top'].set_visible(True)
            ax_bar.spines['bottom'].set_visible(True)


        action = self.info["action"]
        for i, (key, value) in enumerate(action.items(), start=1):  # Start index from 1
            create_fluid_bar(ax[i], value[0], f"drug_{i}", color=list_colors[i-1])

        plt.savefig(path+f"/{saving_title} step {str(self.unwrapped_env.step_episode).zfill(3)}")
        plt.close(fig)



def wrap_env_with_rescale_stats(env: gym.Env, min_action:float=-1, max_action:float=1):
    """
    Applies RescaleAction to normalize actions between -1 and 1,
    Records Episode Statistics for the environment.
    """
    env = gym.wrappers.RescaleAction(env, min_action=min_action, max_action=max_action)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def wrap_gray_env_image(env,resize_shape=(None,None), stack_size=1, gray=True):
    if resize_shape != (None, None):
        env = gym.wrappers.ResizeObservation(env,resize_shape)
    if gray:
        env = gym.wrappers.GrayscaleObservation(env)
    class UInt8Wrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return obs.astype(np.uint8)
    
    env = UInt8Wrapper(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    if not gray:
        C,H,W,S = env.observation_space.shape
        env = gym.wrappers.ReshapeObservation(env, shape=(C*S, H, W))
    return env
