#####
# title: model/tumor_immune_base/custom_modules/physigym/sac_tib.py
#
# language: python3
# main libraries: gymnasium, physigym, torch
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     sac implementation for tumor immune base model
#####


#### IMPORT LIBRARIES ####
# Standard Python Libraries
import os
import random
import time
from dataclasses import dataclass

# Gymnasium PhysiCell bridge module
import physigym

# Gymnasium
import gymnasium
from gymnasium import spaces  
import numpy as np

# Torch ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict

import wandb
#import tyro

import numba
from numba import njit, prange


#### Arguments ####
#
# description:
#   The class's arguments you may change such as:
#   cuda, wandb_track, wandb_entity.
####


@dataclass
class Args():
    name: str = "sac"   # the name of this experiment
    weight: float = 0.5   # weight for the reduction of tumor
    reward_type: str = "dummy_linear"   # type of the reward
    seed: int = 1   # seed of the experiment
    torch_deterministic: bool = True   # if toggled, `torch.backends.cudnn.deterministic=False`
    cuda: bool = True   # if toggled, cuda will be enabled by default

    # Weights and Biases specific arguments
    wandb_track: bool = True   # track with wandb
    wandb_entity: str = "corporate-manu-sureli"   # the wandb's entity name
    wandb_project_name: str = "SAC_IMAGE_TIB"   # the wandb's project name

    # Algorithm specific arguments
    env_id: str = "physigym/ModelPhysiCellEnv-v0"   # the id of the environment
    observation_type: str = "image_cell_types"   # the type of observation
    total_timesteps: int = int(1e6)    # the learning rate of the optimizer
    buffer_size: int = int(1e6)    # the replay memory buffer size
    gamma: float = 0.99    # the discount factor gamma
    tau: float = 0.005    # target smoothing coefficient (default: 0.005)
    batch_size: int = 256   # the batch size of sample from the reply memory
    learning_starts: float = 10e3   # timestep to start learning
    policy_lr: float = 3e-4    # the learning rate of the policy network optimizer
    q_lr: float = 3e-4    # the learning rate of the Q network network optimizer
    policy_frequency: int = 2    # the frequency of training policy (delayed)
    target_network_frequency: int = 1   # the frequency of updates for the target nerworks (Denis Yarats' implementation delays this by 2.)
    alpha: float = 0.2   # Entropy regularization coefficient.
    autotune: bool = True   # automatic tuning of the entropy coefficient


#### Wrapper ####
#
# description:
#   PhysiCell Gymnasium environment wrapper.
#####


class PhysiCellModelWrapper(gymnasium.Wrapper):
    def __init__(
            self,
            env: gymnasium.Env,
            ls_var: list[str] = ["drug_1"],
            weight: float = 0.8,
        ):
        """
        input:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
            weight (float): Weight corresponding how much weight is added to the reward term related to cancer cells.

        output:

        description:
        """
        super().__init__(env)

        # Check that all variable names are strings
        if any([not isinstance(s_var, str) for s_var in ls_var]):
            raise ValueError(f"Expected variable_name to be of type str, but got {type(variable_name).__name__}")

        self.ls_var = ls_var

        # bue 20250601: why do whe have to transfrom this? 
        a_low = np.array([env.action_space[s_var].low[0] for s_var in ls_var])
        a_high = np.array([env.action_space[s_var].high[0] for s_var in ls_var])
        self._action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float64)

        self.weight = weight
        self.reward_type = env.unwrapped.reward_type

    @property
    def action_space(self):
        """Returns the flattened action space for the wrapper."""
        return self._action_space

    def step(self, ar_action: np.ndarray):
        """
        Steps through the environment using the flattened action.

        Args:
            action (np.ndarray): The flattened action array.

        Returns:
            Tuple: Observation, reward, terminated, truncated, info.
        """
        # bue 20250701: this is action, not action space 
        d_action = {}
        for s_var, r_value in zip(self.ls_var, ar_action):
            d_action.update({s_var: np.array([r_value])})

        # Take a step in the environment
        o_observation, r_cancer_cells, b_terminated, b_truncated, info = self.env.step(d_action)

        # bue 20250701: what would this mean if we have more than one action space?
        r_drugs = np.mean(ar_action)  

        info["action"] = d_action
        info["reward_drugs"] = r_drugs
        info["reward_cancer_cells"] = r_cancer_cells

        # If you reward function is different from a sum you can add a new condition
        if self.reward_type == "log_exp":
            r_reward = - r_cancer_cells * np.exp(r_drugs - 1)
        else:
            r_reward = - (1 - self.weight) * r_drugs + self.weight * r_cancer_cells

        return o_observation, r_reward, b_terminated, b_truncated, info


#### Neural Networks ####
#
# description:
#   A list of torch objects mainly Neural Networks (Actor/Critic).
####


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class FeatureExtractor(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg

        obs_shape = env.observation_space.shape
        self.is_image = len(obs_shape) == 3  # Check if input is an image (C, H, W)

        if self.is_image:
            # CNN feature extractor
            num_channels = 8
            layers = [
                PixelPreprocess(),
                nn.Conv2d(obs_shape[0], num_channels, 7, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 5, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=1),
                nn.Flatten(),
            ]
            self.feature_extractor = nn.Sequential(*layers)
            self.feature_size = self._get_feature_size(obs_shape)
        else:
            # Directly flatten vector input
            self.feature_extractor = nn.Identity()
            self.feature_size = np.prod(obs_shape)

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        if self.is_image:
            x = self.feature_extractor(x)  # Apply CNN
            x = x.view(x.size(0), -1)  # Flatten
        return x


class QNetwork(nn.Module):
    """Critic network (Q-function)"""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc3 = nn.LazyLinear(1)
        self.mish = nn.Mish()

    def forward(self, x, a):
        x = self.feature_extractor(x)  # Extract features
        x = torch.cat([x, a], dim=1)  # Concatenate state and action

        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """Policy network (Actor)"""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])
        action_dim = np.prod(env.action_space.shape)

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc_mean = nn.LazyLinear(action_dim)
        self.fc_logstd = nn.LazyLinear(action_dim)
        self.relu = nn.ReLU()
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # Stable variance scaling

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


#### Replay Buffers ####
#
####

class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.
    Stores states, actions, rewards, next states, and done flags.
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            device,
            buffer_size,
            batch_size,
            state_type=np.float32,
        ):
        """
        Initializes the replay buffer.

        Parameters:
        - state_dim tuple(int): Dimensionality of the state space.
        - action_dim tuple(int): Dimensionality of the action space.
        - device (torch.device): Device where tensors should be stored.
        - buffer_size (int): Maximum size of the replay buffer.
        - batch_size (int): Number of samples per batch.
        - state_type (numpy dtype, optional): Data type of the state representation (default: np.float32).
        """
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = np.empty((self.buffer_size, *state_dim), dtype=state_type)
        self.next_state = np.empty((self.buffer_size, *state_dim), dtype=state_type)
        self.action = np.empty((self.buffer_size, *action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the current number of stored experiences in the buffer.
        """
        return self.buffer_size if self.full else self.buffer_index

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        - state (np.ndarray): Current state.
        - action (np.ndarray): Action taken.
        - reward (float): Reward received.
        - next_state (np.ndarray): Next state after taking the action.
        - done (bool): Whether the episode has ended.
        """
        self.state[self.buffer_index] = state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_state[self.buffer_index] = next_state
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
        - TensorDict containing sampled states, actions, rewards, next states, and done flags.
        """
        batch_size = self.batch_size

        # Ensure there are enough samples in the buffer
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        # Generate random indices for sampling
        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        # Convert indices to tensors and gather the sampled experiences
        state = torch.as_tensor(self.state[sample_index]).float()
        next_state = torch.as_tensor(self.next_state[sample_index]).float()
        action = torch.as_tensor(self.action[sample_index])
        reward = torch.as_tensor(self.reward[sample_index])
        done = torch.as_tensor(self.done[sample_index])

        # Create a dictionary of the sampled experiences
        sample = TensorDict(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample


#### Algorithm Logic ####
#
# description:
#   The code is mainly inspired from:
#   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
####


def main():
    d_arg = vars(Args())

    # INITIALISATION/ CREATE FOLDERS
    custom_run_name = f"{d_arg['name']}: seed_{d_arg['seed']}_observationtype_{d_arg['observation_type']}_weight_{d_arg['weight']}_rewardtype_{d_arg['reward_type']}_time_{int(time.time())}"
    if d_arg['wandb_track']:
        print("Wandb selected")
        run = wandb.init(
            project = d_arg['wandb_project_name'],
            entity = d_arg['wandb_entity'],
            name=custom_run_name,
            sync_tensorboard=True,
            config=d_arg,
            monitor_gym=True,
            save_code=True,
        )
        s_dir = os.path.join(run.dir, custom_run_name)  # run.dir wandb/run-20250612_123456-abcdef
    else:
        print("Tensorboard selected")
        s_dir = os.path.join("tensorboard", custom_run_name)
    os.makedirs(s_dir, exist_ok=True)

    # organize output folders using run_dir
    writer = SummaryWriter(s_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{s_key}|{s_value}|" for s_key, s_value in sorted(d_arg.items())])),
    )

    # SEEDING
    random.seed(d_arg['seed'])
    np.random.seed(d_arg['seed'])
    torch.manual_seed(d_arg['seed'])
    torch.backends.cudnn.deterministic = d_arg['torch_deterministic']
    device = torch.device("cuda" if torch.cuda.is_available() and d_arg['cuda'] else "cpu") # cpu or cuda

    # initialisation of the environment
    env = gymnasium.make(
        d_arg['env_id'],
        observation_type=d_arg['observation_type'],
        reward_type=d_arg['reward_type'],
    )

    # Wrapper
    env = PhysiCellModelWrapper(env=env)
    cfg = {"cfg_FeatureExtractor": {}}

    # Neural Networks/ Optimisers init.
    actor = Actor(env, cfg).to(device)
    qf1 = QNetwork(env, cfg).to(device)
    qf2 = QNetwork(env, cfg).to(device)
    qf1_target = QNetwork(env, cfg).to(device)
    qf2_target = QNetwork(env, cfg).to(device)
    target_actor = Actor(env, cfg).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg['q_lr']
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=d_arg['policy_lr'])

    # Automatic entropy tuning
    if d_arg['autotune']:
        target_entropy = - torch.prod(
            torch.Tensor(env.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=d_arg['q_lr'])
    else:
        alpha = d_arg['alpha']

    rb = ReplayBuffer(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.shape,
        device=device,
        buffer_size=d_arg['buffer_size'],
        batch_size=d_arg['batch_size'],
        state_type=env.observation_space.dtype,
    )


    # Start the env and init, tracking values
    obs, info = env.reset(seed=d_arg['seed'])
    cumulative_return = 0
    episode = 1
    step_episode = 0
    discounted_cumulative_return = 0
    for global_step in range(d_arg['total_timesteps']):
        # ALGO LOGIC: put action logic here
        if global_step <= d_arg['learning_starts']:
            actions = np.array(env.action_space.sample(), dtype=np.float16)
        else:
            x = obs
            x = torch.Tensor(x).to(device).unsqueeze(0)
            actions, _, _ = actor.get_action(x)
            actions = actions.detach().squeeze(0).cpu().numpy()
        # execute a step forward and log data.
        next_obs, rewards, terminations, truncations, info = env.step(actions)
        step_episode += 1
        done = terminations or truncations
        cumulative_return += rewards
        discounted_cumulative_return += rewards * d_arg['gamma'] ** (step_episode)
        rb.add(obs, actions, rewards, next_obs, done)
        obs = next_obs.copy()
        if global_step == d_arg['batch_size']:
            data = rb.sample()
            data_next_state = data["next_state"]
            data_state = data["state"]
            with torch.no_grad():
                next_state_actions, _, _ = actor.get_action(data_next_state)
                _, _ = (
                    qf1(data_next_state, next_state_actions),
                    qf2(data_next_state, next_state_actions),
                )
                _, _ = (
                    qf1_target(data_next_state, next_state_actions),
                    qf2_target(data_next_state, next_state_actions),
                )
            del next_state_actions, data_next_state, data_state, data

        # ALGO LOGIC: training.
        if global_step > d_arg['learning_starts']:
            data = rb.sample()
            data_next_state = data["next_state"]
            data_state = data["state"]

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data_next_state
                )
                qf1_next_target = qf1_target(data_next_state, next_state_actions)
                qf2_next_target = qf2_target(data_next_state, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data["reward"].flatten() + (
                    1 - data["done"].flatten()
                ) * d_arg['gamma'] * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data_state, data["action"]).view(-1)
            qf2_a_values = qf2(data_state, data["action"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % d_arg['policy_frequency'] == 0:  # TD 3 Delayed update support
                for _ in range(
                    d_arg['policy_frequency']
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data_state)

                    qf1_pi = qf1(data_state, pi)
                    qf2_pi = qf2(data_state, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if d_arg['autotune']:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data_state)

                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()

                        alpha = log_alpha.exp().item()
                    entropy = -log_pi.mean().item()

                losses = {
                    "losses/min_qf_next_target": min_qf_next_target.mean().item(),
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/entropy": entropy,
                }

                for tag, value in losses.items():
                    writer.add_scalar(tag, value, global_step)

            # Update the target networks
            if global_step % d_arg['target_network_frequency'] == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        d_arg['tau'] * param.data + (1 - d_arg['tau']) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        d_arg['tau'] * param.data + (1 - d_arg['tau']) * target_param.data
                    )
        scalars = {
            "env/drug_1": actions[0],
            "env/reward_value": rewards,
            "env/number_cancer_cells": info["number_cancer_cells"],
            "env/number_cell_1": info["number_cell_1"],
            "env/number_cell_2": info["number_cell_2"],
            "env/reward_cancer_cells": info["reward_cancer_cells"],
            "env/reward_drugs": info["reward_drugs"],
        }
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, episode)
        if done:
            norm_coeff = (1 - d_arg['gamma'] ** (step_episode + 1)) / (1 - d_arg['gamma'])
            scalars = {
                "charts/episodic_return": cumulative_return / step_episode,
                "charts/cumulative_return": cumulative_return,
                "charts/episodic_length": step_episode,
                "charts/discounted_cumulative_return": discounted_cumulative_return,
                "charts/normalized_discounted_episodic_return": discounted_cumulative_return
                / norm_coeff,
            }
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, episode)
            episode += 1
            step_episode = 0
            discounted_cumulative_return = 0
            cumulative_return = 0
            obs, info = env.reset()
            done = False
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
