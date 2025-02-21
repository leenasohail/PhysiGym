
import gymnasium as gym
import numpy as np
import os
import wandb
import physigym
import stable_baselines3
import sb3_contrib
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import tyro
import time
from gymnasium.spaces import Box
from dataclasses import dataclass

@dataclass
class Args:
    algo_name: str = "TQC"
    """the name of the algo"""
    wandb_project_name: str = "IMAGE_TME_PHYSIGYM"
    """the wandb's project name"""
    wandb_entity: str = "corporate-manu-sureli"
    # Algorithm specific arguments
    env_id: str = "physigym/ModelPhysiCellEnv-v0"
    """the id of the environment"""
    observation_type: str = "image"
# ----------------------
# 🏆 Initialize WandB
# ----------------------
args = tyro.cli(Args)
config = vars(args)

# ----------------------
# 🔍 Choose Algorithm (SB3 or SB3-Contrib)
# ----------------------
algo_name = args.algo_name
if algo_name  in sb3_contrib.__all__:
    algorithm = getattr(sb3_contrib,algo_name)
elif algo_name in stable_baselines3.__all__:
    algorithm = getattr(stable_baselines3,algo_name)
else:
    raise f"Algorith name does not exist: {algo_name}"



run_name = f"{args.env_id}__{args.algo_name}_{args.wandb_entity}_{int(time.time())}"

wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    name=f"{args.algo_name}: {args.observation_type}",
    sync_tensorboard=True,  # Sync TensorBoard logs
    config=config,
    monitor_gym=True,  # Monitor Gym environment
    save_code=True,  # Save the training script
)

# Define WandB logging directory
run_dir = f"runs/{run_name}"
os.makedirs(run_dir, exist_ok=True)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values to TensorBoard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.global_step = 0  # Track global training steps

    def _on_step(self) -> bool:
        # Get information from the environment
        if "reward" in self.locals:
            self.logger.record("env/reward_value", self.locals["rewards"][0] )

        if "number_cancer_cells" in self.locals["infos"][0]:
            self.logger.record("env/cancer_cell_count", self.locals["infos"][0]["number_cancer_cells"])

        if "actions" in self.locals:
            actions = self.locals["actions"][0]
            self.logger.record("env/drug_apoptosis", actions[0])
            self.logger.record("env/drug_reducing_antiapoptosis", actions[1])

        self.global_step += 1  # Increment step counter
        self.logger.dump(step=self.global_step)
        return True

class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_apoptosis",
            "drug_reducing_antiapoptosis",
        ],
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

        return o_observation, r_reward, b_terminated, b_truncated, info

# ----------------------
# 🏗️  Environment Setup
# ----------------------
env = gym.make(args.env_id,observation_type=args.observation_type)
env = PhysiCellModelWrapper(env)
env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, stack_size=1)
obs, info = env.reset()

# ----------------------
# 📂 Logging Setup
# ----------------------
log_dir = f"./tensorboard_logs/{algo_name}" 
os.makedirs(log_dir, exist_ok=True)

# ----------------------
# 🏃 Train the Model (with WandB Callback)
# ----------------------
model = TQC("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
new_logger = configure(log_dir, ["tensorboard"])
model.set_logger(new_logger)
model.learn(total_timesteps=int(2e6), log_interval=1, progress_bar=False, callback=TensorboardCallback())

# ----------------------
# 🎮 Run the Trained Agent
# ----------------------
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
wandb.finish()  # ✅ Finish WandB run