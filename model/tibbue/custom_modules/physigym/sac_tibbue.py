
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
# + https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
#####

# core python
from dataclasses import dataclass, field
import numpy as np
import os
import random
import time

# gymnasium and physigym
import gymnasium
from gymnasium import spaces
import physigym

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard

# wandb
import wandb

# replybuffer
from alexlines1 import ReplayBuffer
#from stable_baselines3.common.buffers import ReplayBuffer


###########
# Wrapper #
###########

class PhysiCellModelWrapper(gymnasium.Wrapper):
    def __init__(
            self,
            env: gymnasium.Env,
            r_weight: float,
            ls_var: list[str],
        ):
        """
        input:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
            r_weight (float): Weight corresponding how much weight is added to the reward term related to cancer cells.

        output:

        description:
        """
        super().__init__(env)

        # Check that all variable names are strings
        if any([not isinstance(s_var, str) for s_var in ls_var]):
            raise ValueError(f"Expected variable names in ls_var to be of type str. {ls_var}")

        self.ls_var = ls_var

        # bue 20250601: why do whe have to transfrom this?
        a_low = np.array([env.action_space[s_var].low[0] for s_var in ls_var])
        a_high = np.array([env.action_space[s_var].high[0] for s_var in ls_var])
        self._action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float64)

        self.r_weight = r_weight
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
        o_observation, r_cancer_cells, b_terminated, b_truncated, d_info = self.env.step(d_action)

        # bue 20250701: what would this mean if we have more than one action space?
        r_drugs = np.mean(ar_action)

        d_info["action"] = d_action
        d_info["reward_drugs"] = r_drugs
        d_info["reward_cancer_cells"] = r_cancer_cells

        # If you reward function is different from a sum you can add a new condition
        if self.reward_type == "log_exp":
            r_reward = - r_cancer_cells * np.exp(r_drugs - 1)
        else:
            r_reward = - (1 - self.r_weight) * r_drugs + self.r_weight * r_cancer_cells

        return o_observation, r_reward, b_terminated, b_truncated, d_info


###################
# Neural Networks #
###################

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


#############
# Arguments #
#############

@dataclass
class Args():
    # basics
    name: str = "sac"   # the name of this experiment

    # tracking
    wandb_track: bool = False   # track with wandb
    wandb_entity: str = "corporate-manu-sureli"   # the wandb's entity name
    wandb_project_name: str = "SAC_IMAGE_TIB"   # the wandb's project name

    # hardware
    cuda: bool = True   # should torch check for gpu (nvidia, amd mroc) accelerator?

    # random seed
    seed: int = 1   # seed of the experiment
    torch_deterministic: bool = True   # torch.backends.cudnn.deterministic

    # physigym
    env_id: str = "physigym/ModelPhysiCellEnv-v0"   # the id of the gymnasium environment
    weight: float = 0.5   # weight for the reduction of tumor
    ls_var: list = field(default_factory=lambda: ["drug_1"])  # list of action varaible names
    observation_type: str = "image_cell_types"   # the type of observation
    reward_type: str = "dummy_linear"   # type of the reward
    total_timesteps: int = int(1e6)    # the learning rate of the optimizer

    # neural network
    alpha: float = 0.2   # set manuall entropy regularization coefficient.
    autotune: bool = True   # automatic tuning the the entropy coefficient.

    # algorithm I
    buffer_size: int = int(1e6)    # the replay memory buffer size
    batch_size: int = 256   # the batch size of sample from the reply memory
    learning_starts: float = 10e3   # timestep to start learning
    policy_frequency: int = 2    # the frequency of training policy (delayed)
    target_network_frequency: int = 1   # the frequency of updates for the target nerworks (Denis Yarats' implementation delays this by 2.)

    # algorithm II
    gamma: float = 0.99    # the discount factor gamma
    tau: float = 0.005    # target smoothing coefficient (default: 0.005)
    q_lr: float = 3e-4    # the learning rate of the Q network network optimizer
    policy_lr: float = 3e-4    # the learning rate of the policy network optimizer


#############
# main loop #
##############
####


def main():
    d_arg = vars(Args())

    # tracking
    s_run = f"{d_arg['name']}_seed_{d_arg['seed']}_observationtype_{d_arg['observation_type']}_weight_{d_arg['weight']}_rewardtype_{d_arg['reward_type']}_time_{int(time.time())}"
    if d_arg['wandb_track']:
        print("tracking: wandb ...")
        run = wandb.init(
            project = d_arg['wandb_project_name'],
            entity = d_arg['wandb_entity'],
            name=s_run,
            sync_tensorboard=True,
            config=d_arg,
            monitor_gym=True,
            save_code=True,
        )
        s_dir = os.path.join(run.dir, s_run)  # run.dir wandb/run-20250612_123456-abcdef
    else:
        print("tracking tensorboard ...")
        s_dir = os.path.join("tensorboard", s_run)
    os.makedirs(s_dir, exist_ok=True)

    # initialize tensorbord writer
    writer = tensorboard.SummaryWriter(s_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{s_key}|{s_value}|" for s_key, s_value in sorted(d_arg.items())])),
    )

    # set random seed
    random.seed(d_arg['seed'])
    np.random.seed(d_arg['seed'])
    torch.manual_seed(d_arg['seed'])
    torch.backends.cudnn.deterministic = d_arg['torch_deterministic']


    # initialize physigym environment
    env = gymnasium.make(
        d_arg['env_id'],
        observation_type=d_arg['observation_type'],
        reward_type=d_arg['reward_type'],
    )
    env = PhysiCellModelWrapper(
        env=env,
        ls_var=d_arg['ls_var'],
        r_weight=d_arg['weight'],

    )

    # initialize neural networks and optimiser.
    device = torch.device("cuda" if torch.cuda.is_available() and d_arg['cuda'] else "cpu") # cpu or gpu
    cfg = {"cfg_FeatureExtractor": {}}
    actor = Actor(env, cfg).to(device)
    qf1 = QNetwork(env, cfg).to(device)
    qf2 = QNetwork(env, cfg).to(device)
    qf1_target = QNetwork(env, cfg).to(device)
    qf2_target = QNetwork(env, cfg).to(device)
    target_actor = Actor(env, cfg).to(device)  # bue: where is target_actor used?
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg['q_lr'])
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=d_arg['policy_lr'])

    # automatic entropy tuning or manual alpha
    if d_arg['autotune']:
        target_entropy = - torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=d_arg['q_lr'])
    else:
        alpha = d_arg['alpha']

    # initilize the reply buffer
    rb = ReplayBuffer(
        o_observation_dim=env.observation_space.shape,
        a_action_dim=env.action_space.shape,
        device=device,
        buffer_size=d_arg['buffer_size'],
        batch_size=d_arg['batch_size'],
        o_observation_type=env.observation_space.dtype,
    )


    # reset gymnasium env
    o_observation, d_info = env.reset(seed=d_arg['seed'])
    r_cumulative_return = 0
    r_discounted_cumulative_return = 0

    for global_step in range(d_arg['total_timesteps']):

        # sample the action space or learn
        if global_step <= d_arg['learning_starts']:
            a_action = np.array(env.action_space.sample(), dtype=np.float16)
        else:
            x = torch.Tensor(o_observation).to(device).unsqueeze(0)
            actions, _, _ = actor.get_action(x)
            a_action = actions.detach().squeeze(0).cpu().numpy()

        # physigym step
        o_observation_next, r_reward, b_terminated, b_truncated, d_info = env.step(a_action)
        b_episode_over = b_terminated or b_truncated
        r_cumulative_return += r_reward
        r_discounted_cumulative_return += r_reward * d_arg['gamma'] ** (env.unwrapped.step_episode)

        # record to reply buffer
        rb.add(
            o_observation=o_observation,
            a_action=a_action,
            o_observation_next=o_observation_next,
            r_reward=r_reward,
            b_episode_over=b_episode_over,
        )

        # process observation
        o_observation = o_observation_next.copy()

        # at the end of the first batch
        if env.unwrapped.step_env == d_arg['batch_size']:
            data = rb.sample()
            with torch.no_grad():
                next_state_actions, _, _ = actor.get_action(data["next_state"])
                qf1(data["next_state"], next_state_actions)
                qf2(data["next_state"], next_state_actions)
                qf1_target(data["next_state"], next_state_actions)
                qf2_target(data["next_state"], next_state_actions)
            del data, next_state_actions

        # learning
        if env.unwrapped.step_env > d_arg['learning_starts']:
            data = rb.sample()
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_state"])
                qf1_next_target = qf1_target(data["next_state"], next_state_actions)
                qf2_next_target = qf2_target(data["next_state"], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data["r_reward"].flatten() + (1 - data["b_episode_over"].flatten()) * d_arg['gamma'] * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data["state"], data["action"]).view(-1)
            qf2_a_values = qf2(data["state"], data["action"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # every policy frequency
            if env.unwrapped.step_env % d_arg['policy_frequency'] == 0:  # TD 3 Delayed update support

                # compensate for the delay by doing 'actor_update_interval' instead of 1
                for _ in range(d_arg['policy_frequency']):
                    pi, log_pi, _ = actor.get_action(data["state"])

                    qf1_pi = qf1(data["state"], pi)
                    qf2_pi = qf2(data["state"], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # entropy autotune
                    if d_arg['autotune']:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data["state"])

                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()

                        alpha = log_alpha.exp().item()

                # write to tensoboard
                writer.add_scalar("losses/min_qf_next_target", min_qf_next_target.mean().item(), env.unwrapped.step_env)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), env.unwrapped.step_env)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), env.unwrapped.step_env)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), env.unwrapped.step_env)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), env.unwrapped.step_env)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, env.unwrapped.step_env)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), env.unwrapped.step_env)
                writer.add_scalar("losses/entropy", - log_pi.mean().item(), env.unwrapped.step_env)  # entropy

            # update the target networks
            if global_step % d_arg['target_network_frequency'] == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(d_arg['tau'] * param.data + (1 - d_arg['tau']) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(d_arg['tau'] * param.data + (1 - d_arg['tau']) * target_param.data)


        # write to tensoboard
        writer.add_scalar("env/drug_1", a_action[0], env.unwrapped.episode)
        writer.add_scalar("env/reward_value", r_reward, env.unwrapped.episode)
        writer.add_scalar("env/number_cancer_cells", d_info["number_cancer_cells"], env.unwrapped.episode)
        writer.add_scalar("env/number_cell_1", d_info["number_cell_1"], env.unwrapped.episode)
        writer.add_scalar("env/number_cell_2", d_info["number_cell_2"], env.unwrapped.episode)
        writer.add_scalar("env/reward_cancer_cells", d_info["reward_cancer_cells"], env.unwrapped.episode)
        writer.add_scalar("env/reward_drugs", d_info["reward_drugs"], env.unwrapped.episode)

        #
        if b_episode_over:
            norm_coeff = (1 - d_arg['gamma'] ** (env.unwrapped.step_episode + 1)) / (1 - d_arg['gamma'])
            writer.add_scalar("charts/episodic_return", r_cumulative_return / env.unwrapped.step_episode, env.unwrapped.episode)
            writer.add_scalar("charts/cumulative_return", r_cumulative_return, env.unwrapped.episode)
            writer.add_scalar("charts/episodic_length", env.unwrapped.step_episode, env.unwrapped.episode)
            writer.add_scalar("charts/discounted_cumulative_return", r_discounted_cumulative_return, env.unwrapped.episode)
            writer.add_scalar("charts/normalized_discounted_episodic_return", r_discounted_cumulative_return / norm_coeff, env.unwrapped.episode)

            o_observation, d_info = env.reset()
            r_cumulative_return = 0
            r_discounted_cumulative_return = 0
            b_episode_over = False

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
