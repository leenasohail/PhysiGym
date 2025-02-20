import optuna
import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.callbacks import BaseCallback
import os
import wandb
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from stable_baselines3.common.logger import configure
import stable_baselines3
import sb3_contrib
import time


class TrackingCallback(BaseCallback):
    def __init__(self, trial, start_tracking_step=50_000, verbose=0, mean_elements=100, eval_frequency=10000):
        """
        Callback to track episode rewards and store them when an episode ends.

        :param start_tracking_step: Step after which we start storing rewards.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.trial = trial
        self.start_tracking_step = start_tracking_step
        self.ep_rewards = []  # Stores rewards at the end of episodes
        self.last_eval_step = 0
        self.global_step = 0  # Track global training steps
        self.mean_elements = mean_elements
        self.eval_frequency = eval_frequency
        self.counter = 0
        self.mean_ep_reward = float("-inf") 

    def _on_step(self) -> bool:

        if self.global_step>=self.start_tracking_step:
            for info in self.locals["infos"]:
                if "episode" in info:  # Only store reward at end of episode
                    self.ep_rewards.append(info["episode"]["r"])
                
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

        if len(self.ep_rewards) >= self.mean_elements and self.global_step%self.eval_frequency==0:
            self.mean_ep_reward = np.mean(self.ep_rewards[-self.mean_elements:])
            self.trial.report(self.mean_ep_reward, self.counter)
            self.counter +=1
            if self.trial.should_prune():  # Check if Optuna wants to prune
                print("⚠️ Trial pruned by Optuna!")
                raise optuna.TrialPruned()

        return True  




class RLHyperparamTuner:
    def __init__(self, algo, env_id, n_trials=100, total_timesteps=int(1e6), pruner_type="median", 
                 start_tracking_step=50000, mean_elements=int(1e3), policy="CnnPolicy", 
                 wandb_project="RL_Optimization", wandb_entity=None, eval_frequency=int(1e4)):
        """
        Class to tune hyperparameters for RL algorithms using Optuna.

        :param algo: Algorithm name (e.g., "ppo", "sac", "tqc")
        :param env_id: Gymnasium environment ID
        :param n_trials: Number of Optuna trials
        :param total_timesteps: Total training timesteps per trial
        :param pruner_type: Type of Optuna pruner ("median", "halving", "hyperband")
        :param start_tracking_step: Number of warmup steps
        :param mean_elements: Number of episodes for averaging reward
        :param wandb_project: WandB project name
        :param wandb_entity: WandB entity (team/user)
        """
        self.algo = algo.lower()
        self.env_id = env_id
        self.n_trials = n_trials
        self.total_timesteps = total_timesteps
        self.start_tracking_step = start_tracking_step
        self.mean_elements = mean_elements
        self.policy = policy
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.eval_frequency = eval_frequency

        # Validate algorithm
        if self.algo not in HYPERPARAMS_SAMPLER:
            raise ValueError(f"Algorithm {self.algo} not supported. Choose from {list(HYPERPARAMS_SAMPLER.keys())}.")

        # Select Optuna pruning strategy
        if pruner_type == "median":
            self.pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
        elif pruner_type == "halving":
            self.pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif pruner_type == "hyperband":
            self.pruner = optuna.pruners.HyperbandPruner()
        else:
            raise ValueError("Invalid pruner_type. Choose from 'median', 'halving', or 'hyperband'.")

    def create_env(self):
        """Create and wrap the environment."""
        env = gym.make(self.env_id)
        env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=1)
        return env

    def objective(self, trial: optuna.Trial):

        """Objective function for Optuna hyperparameter optimization."""
        env = self.create_env()
        hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial, n_actions=env.action_space.shape[0], n_envs=1, additional_args={})

        if self.algo  in sb3_contrib.__all__:
            algorithm = getattr(sb3_contrib,self.algo.upper())
        elif self.algo in stable_baselines3.__all__:
            algorithm = getattr(stable_baselines3,self.algo.upper())
        else:
            raise f"Algorith name does not exist: {self.algo.upper()}"
        # ----------------------
        # 📂 Logging Setup
        # ----------------------
        # WandB run setup
        run_name = f"{self.env_id}__{self.algo}_{int(time.time())}"
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=hyperparams,
            monitor_gym=True,
            save_code=True,
        )

        # Logging directory for TensorBoard
        log_dir = f"/tensorboard_logs/{self.algo}/runs/{run_name}"
        os.makedirs(log_dir, exist_ok=True)

        # Create model
        model = algorithm(self.policy, env, verbose=0, tensorboard_log=log_dir, **hyperparams)
        new_logger = configure(log_dir, ["tensorboard"])
        model.set_logger(new_logger)
        # Create pruning callback
        pruning_callback = TrackingCallback(trial=trial, start_tracking_step=self.start_tracking_step, mean_elements=self.mean_elements, eval_frequency=self.eval_frequency)


        model.learn(total_timesteps=self.total_timesteps, callback=pruning_callback)
        wandb.finish()
        try:
            return pruning_callback.mean_ep_reward  # ✅ Get mean reward from the callback
        except: 
            return None


    def run_optimization(self):
        """Run Optuna optimization."""
        study = optuna.create_study(direction="maximize", pruner=self.pruner)
        study.optimize(self.objective, n_trials=self.n_trials)
        print("✅ Best hyperparameters:", study.best_params)
