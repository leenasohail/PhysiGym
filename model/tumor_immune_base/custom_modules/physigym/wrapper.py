import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


# ============================================================
# Wrapper: PhysiCellModelWrapper
# ============================================================
class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = ["drug_1"],
        weight: float = 0.8,
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
        self._action_space = Box(low=low, high=high, dtype=np.float64)
        self.weight = weight

    @property
    def action_space(self):
        return self._action_space

    def step(self, action: np.ndarray):
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)

        r_drugs = np.mean(action)
        info["action"] = d_action
        info["reward_drugs"] = r_drugs
        info["reward_cancer_cells"] = r_cancer_cells

        reward = -(1 - self.weight) * r_drugs + self.weight * r_cancer_cells

        return obs, reward, terminated, truncated, info
