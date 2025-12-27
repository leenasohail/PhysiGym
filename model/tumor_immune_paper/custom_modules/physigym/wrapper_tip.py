import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from network_field import create_csv
import os
import pandas as pd
import shutil


# ============================================================
# Wrapper: PhysiCellModelWrapper
# ============================================================
class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = ["drug_1"],
        weight: float = 0.8,
        frequency_save_data=256,
    ):
        """
        Wraps a PhysiCell environment to use a flat continuous Box action space.
        Reward = weighted sum between drug penalty and cancer cell signal.
        """
        super().__init__(env)

        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be str, got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name
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
        dtype = env.action_space[list_variable_name[0]].dtype

        self._action_space = Box(low=low, high=high, dtype=dtype)
        self.weight = weight
        self.cell_positions_folder = (
            self.env.get_wrapper_attr("x_root")
            .xpath("//initial_conditions/cell_positions/folder")[0]
            .text
        )
        self.cell_name_file = (
            self.env.get_wrapper_attr("x_root")
            .xpath("//initial_conditions/cell_positions/filename")[0]
            .text
        )
        self.csv_path_init = os.path.join(
            self.cell_positions_folder, self.cell_name_file
        )
        self.generation_cfg = None
        self.output_dir = (
            self.env.get_wrapper_attr("x_root").xpath("//save/folder")[0].text
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.frequency_save_data = frequency_save_data
        self.list_data = []
        self.seed = int(
            self.env.get_wrapper_attr("x_root").xpath("//random_seed")[0].text
        )

    @property
    def action_space(self):
        return self._action_space

    def save_data(self):
        if self.frequency_save_data is not None:
            self.output_dir_episode = os.path.join(
                self.output_dir, f"episode{str(self.env.unwrapped.episode).zfill(8)}"
            )
            os.makedirs(self.output_dir_episode, exist_ok=True)
            episode = self.env.unwrapped.episode
            df = pd.DataFrame(self.list_data)
            df.to_csv(os.path.join(self.output_dir_episode, "data.csv"), index=False)
            dst_path = os.path.join(
                self.output_dir_episode,
                os.path.basename(self.generation_cfg["csv_path"]),
            )
            shutil.copy(self.generation_cfg["csv_path"], dst_path)
            self.list_data = []
            self.output_dir_episode = os.path.join(
                self.output_dir, f"episode{str(episode).zfill(8)}"
            )
            if episode % self.frequency_save_data == 0:
                os.makedirs(self.output_dir_episode, exist_ok=True)
                # manipulate setting xml before reset
                self.env.get_wrapper_attr("x_root").xpath("//save/folder")[
                    0
                ].text = self.output_dir_episode
                self.env.get_wrapper_attr("x_root").xpath("//save/full_data/enable")[
                    0
                ].text = "true"
                self.env.get_wrapper_attr("x_root").xpath("//save/SVG/enable")[
                    0
                ].text = "true"
            else:
                self.env.get_wrapper_attr("x_root").xpath("//save/folder")[
                    0
                ].text = os.path.join(self.output_dir, "devnull")
                self.env.get_wrapper_attr("x_root").xpath("//save/full_data/enable")[
                    0
                ].text = "false"
                self.env.get_wrapper_attr("x_root").xpath("//save/SVG/enable")[
                    0
                ].text = "false"
        else:
            None

    def step(self, action: np.ndarray):
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)

        r_drugs = np.mean(action)
        info["action"] = d_action
        info["step_episode"] = self.env.unwrapped.step_episode

        reward = -(1 - self.weight) * r_drugs + self.weight * r_cancer_cells

        # reward = np.clip(reward, -1,1)
        if self.frequency_save_data is not None:
            data = {
                "step": self.env.unwrapped.step_episode,
                "reward": reward,
                "drug_1": action,
                "r_drugs": r_drugs,
                "r_cancer_cells": r_cancer_cells,
                "number_tumor": info["number_tumor"],
                "number_cell_1": info["number_cell_1"],
                "number_cell_2": info["number_cell_2"],
            }

            self.list_data.append(data)

        return obs, reward, terminated, truncated, info

    def process_update_generation_cfg(self, generation_cfg=None):
        if self.generation_cfg is not None:
            self.generation_cfg["seed"] += self.env.unwrapped.episode
            self.generation_cfg["csv_path"] = self.csv_path_init
            create_csv(**self.generation_cfg)
        else:
            if generation_cfg is not None and self.generation_cfg is None:
                self.generation_cfg = generation_cfg.copy()

                # complete spatial bounds from env
                self.generation_cfg["x_min"] = self.env.unwrapped.x_min
                self.generation_cfg["y_min"] = self.env.unwrapped.y_min
                self.generation_cfg["x_max"] = self.env.unwrapped.x_max
                self.generation_cfg["y_max"] = self.env.unwrapped.y_max

                # ensure seed exists
                self.generation_cfg.setdefault("seed", self.seed)

    def reset(self, seed=None, options=None, generation_cfg=None, **kwargs):
        if options is None:
            options = {}

        if seed is not None:
            self.seed = seed

        self.process_update_generation_cfg(generation_cfg)

        self.save_data()

        # ---- IMPORTANT: forward seed, do not invent one ----
        return self.env.reset(seed=seed, options=options)
