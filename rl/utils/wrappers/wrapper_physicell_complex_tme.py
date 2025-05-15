import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "anti_M2",
            "anti_pd1",
        ],
        weight: float = 0.8,
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
        self.max_steps = env.unwrapped.max_steps

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
        info["action"] = d_action
        r_reward = (1 - self.weight) * (1 - np.mean(action)) + r_reward * self.weight

        # corporate-manu-sureli/SAC_IMAGE_COMPLEX_TME/run-cqtzu9b8-history:v1
        # r_reward = -1
        # r_reward += 1000  if info["number_cancer_cells"] == 0 else 0
        return o_observation, r_reward, b_terminated, b_truncated, info
