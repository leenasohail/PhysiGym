import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_1",
        ],
        weight: float = 0.8,
        discrete: bool = False,
    ):
        """
        Args:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
            weight (float): Weight corresponding how much weight is added to the reward term related to cancer cells.
        """
        super().__init__(env)

        # Check that all variable names are strings
        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name
        # Flatten the action space to match the expected two values
        self.discrete = discrete
        self.dose_to_class = []
        if self.discrete:
            anti_M2_dose_map = [0.0, 0.5, 1.0]
            anti_pd1_dose_map = [0.0, 0.5, 1.0]

            for m2 in anti_M2_dose_map:
                for pd1 in anti_pd1_dose_map:
                    self.dose_to_class.append([m2, pd1])
            self._action_space = Discrete(n=len(self.dose_to_class))
        else:
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

        self.weight = weight

    @property
    def action_space(self):
        """Returns the flattened action space for the wrapper."""
        return self._action_space

    def step(self, action: np.ndarray):
        """
        Steps through the environment using the flattened action.

        Args:
            action (np.ndarray): The flattened action array.

        Returns:
            Tuple: Observation, reward, terminated, truncated, info.
        """
        if self.discrete:
            action = np.array(self.dose_to_class[action])

        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }
        # Take a step in the environment
        o_observation, r_cancer_cells, b_terminated, b_truncated, info = self.env.step(
            d_action
        )

        r_drugs = 1 - np.mean(action)

        info["action"] = d_action
        info["reward_drugs"] = r_drugs
        info["reward_cancer_cells"] = r_cancer_cells

        r_reward = (1 - self.weight) * r_drugs + self.weight * r_cancer_cells

        return o_observation, r_reward, b_terminated, b_truncated, info
