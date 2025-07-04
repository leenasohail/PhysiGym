#####
# title: model/tumor_immune_base/custom_modules/physigym/sac_tib.py
#
# language: python3
# main libraries: gymnasium, physigym, numpy, torch, wandb
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#   sac implementation for tumor immune base model.
#   implementation is based on the following source code:
# + https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
#####

# basic python
import numpy as np
import os
import pandas as pd
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


################################
# Class PhysiCellModel Wrapper #
################################

class PhysiCellModelWrapper(gymnasium.Wrapper):
    def __init__(
            self,
            env: gymnasium.Env,
            ls_action: list[str],
            r_weight: float,
        ):
        """
        input:
            env (gym.Env): the environment to be wrapped.
            ls_action (list[str]): list of variable names corresponding to actions in the original env.
            r_weight (float): weight corresponding how much weight is added to the reward term related to cancer cells.

        output:

        description:
            handle flattened numpy action space.
        """
        super().__init__(env)

        # handle possible keyword arguments input
        self.ls_action = ls_action
        self.r_weight = r_weight

        # numpy flattend action space
        a_low = np.array([env.action_space[s_action].low[0] for s_action in self.ls_action])
        a_high = np.array([env.action_space[s_action].high[0] for s_action in self.ls_action])
        self._action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float64)


    @property
    def action_space(self):
        """
        inoput:

        output:
             self._action_space numpy flattened action space.

        description:
             return action space in a form compatible with the wrapper.
        """
        return self._action_space


    def step(self, ar_action: np.ndarray):
        """
        input:
            ar_action (np.ndarray): The flattened action array.

        output:
           tuple of observation, reward, terminated, truncated, and info.

        description:
            steps through the environment using the flattened action.
        """
        # action, not action space
        d_action = {}
        for s_action, r_value in zip(self.ls_action, ar_action):
            d_action.update({s_action: np.array([r_value])})

        # take a step in the environment
        o_observation, r_tumor, b_terminated, b_truncated, d_info = self.env.step(d_action)

        # the mean of all the actions (e.g. one or multiple drugs).
        r_drugs = np.mean(ar_action)
        r_reward = - (1 - self.r_weight) * r_drugs + self.r_weight * r_tumor

        # update the info dictionary
        d_info["action"] = d_action
        d_info["reward"] = r_reward
        d_info["reward_drugs"] = r_drugs
        d_info["reward_tumor"] = r_tumor

        # going home
        return o_observation, r_reward, b_terminated, b_truncated, d_info


#########################
# Class Neural Networks #
#########################

class PixelPreprocess(nn.Module):
    """Normalizes pixel observations to [-0.5, 0.5]."""

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
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # Stable variance scaling

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

d_arg = {
    # basics
    "name" : "sac",   # str: the name of this experiment

    # hardware
    "cuda" : True,   # bool: should torch check for gpu (nvidia, amd mroc) accelerator?

    # tracking
    "wandb_track" : False,   # bool: track with wandb
    "wandb_entity" : "corporate-manu-sureli",   # str: the wandb s entity name
    "wandb_project_name" : "SAC_IMAGE_TIB",   # str: the wandb s project name

    # random seed
    "seed" : 1,   # int: seed of the experiment
    "torch_deterministic" : True,   # bool: torch.backends.cudnn.deterministic

    # physigym
    "env_id" : "physigym/ModelPhysiCellEnv-v0",   # str: the id of the gymnasium environment
    "cell_type_cmap" : {"tumor" : "yellow", "cell_1" : "blue", "cell_2" : "green"},
    #"cell_type_cmap" : "viridis",
    "render_mode" : "rgb_array",
    #"observation_type" : "scalars",   # str: the type of observation scalars, img_rgba, img_multichannel
    #"observation_type" : "img_multichannel",   # str: the type of observation scalars, img_rgba, img_multichannel
    "observation_type" : "img_rgba",   # str: the type of observation scalars, img_rgba, img_multichannel
    "grid_size_x" : 64,
    "grid_size_y" : 64,
    "normalization_factor" : 512,
    "ls_action" : ["drug_1"],  # list of str: of action varaible names
    "r_weight" : 0.5,   # float: weight for the reduction of tumor
    "total_timesteps" : int(1e6),    # int: the learning rate of the optimizer

    # neural network
    "alpha" : 0.2,   # float: set manuall entropy regularization coefficient.
    "autotune" : True,   # bool: automatic tuning the the entropy coefficient.

    # algorithm I
    "buffer_size" : int(1e6),    # int: the replay memory buffer size
    "batch_size" : 256,   # int: the batch size of sample from the reply memory
    "learning_starts" : 10e3,   # float: timestep to start learning
    "policy_frequency" : 2,    # int: the frequency of training policy (delayed)
    "target_network_frequency" : 1,   # int: the frequency of updates for the target nerworks (Denis Yarats" implementation delays this by 2.)

    # algorithm II
    "gamma" : 0.99,    # float: the discount factor gamma
    "tau" : 0.005,    # float: target smoothing coefficient (default" : 0.005)
    "q_lr" : 3e-4,    # float: the learning rate of the Q network network optimizer
    "policy_lr" : 3e-4,    # float: the learning rate of the policy network optimizer
}


#############
# main loop #
#############

# initialize tracking
s_run = f'{d_arg["name"]}_seed_{d_arg["seed"]}_observationtype_{d_arg["observation_type"]}_weight_{d_arg["r_weight"]}_time_{int(time.time())}'
if d_arg["wandb_track"]:
    print("tracking: wandb ...")
    run = wandb.init(
        project = d_arg["wandb_project_name"],
        entity = d_arg["wandb_entity"],
        name=s_run,
        sync_tensorboard=True,
        config=d_arg,
        monitor_gym=True,
        save_code=True,
    )
    s_dir_run = os.path.join(run.dir, s_run)  # run.dir wandb/run-20250612_123456-abcdef
else:
    print("tracking tensorboard ...")
    s_dir_run = os.path.join("tensorboard", s_run)
s_dir_data = os.path.join(s_dir_run, "data")

# initialize tensorbord writer
writer = tensorboard.SummaryWriter(s_dir_run)
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{s_key}|{s_value}|" for s_key, s_value in sorted(d_arg.items())])),
)

# set random seed
random.seed(d_arg["seed"])
np.random.seed(d_arg["seed"])
torch.manual_seed(d_arg["seed"])
torch.backends.cudnn.deterministic = d_arg["torch_deterministic"]

# initialize physigym environment
env = gymnasium.make(
    d_arg["env_id"],
    cell_type_cmap=d_arg["cell_type_cmap"],
    render_mode=d_arg["render_mode"],
    observation_type=d_arg["observation_type"],
    grid_size_x=d_arg["grid_size_x"],
    grid_size_y=d_arg["grid_size_y"],
    normalization_factor=d_arg["normalization_factor"],
)
env = PhysiCellModelWrapper(
    env=env,
    ls_action=d_arg["ls_action"],
    r_weight=d_arg["r_weight"],

)

# initialize neural networks
o_device = torch.device("cuda" if torch.cuda.is_available() and d_arg["cuda"] else "cpu") # cpu or gpu
cfg = {"cfg_FeatureExtractor": {}}
actor = Actor(env, cfg).to(o_device)
qf1 = QNetwork(env, cfg).to(o_device)
qf2 = QNetwork(env, cfg).to(o_device)
qf1_target = QNetwork(env, cfg).to(o_device)
qf2_target = QNetwork(env, cfg).to(o_device)
target_actor = Actor(env, cfg).to(o_device)  # bue: where is target_actor used?
target_actor.load_state_dict(actor.state_dict())
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["q_lr"])
actor_optimizer = optim.Adam(list(actor.parameters()), lr=d_arg["policy_lr"])
# neural network automatic entropy tuning or manual alpha
if d_arg["autotune"]:
    target_entropy = - torch.prod(torch.Tensor(env.action_space.shape).to(o_device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=o_device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=d_arg["q_lr"])
else:
    alpha = d_arg["alpha"]

# initilize the reply buffer
rb = ReplayBuffer(
    li_observation_dim=env.observation_space.shape,
    li_action_dim=env.action_space.shape,
    o_device=o_device,
    i_buffer_size=d_arg["buffer_size"],
    i_batch_size=d_arg["batch_size"],
    o_observation_type=env.observation_space.dtype,
)

# reset gymnasium env
o_observation, d_info = env.reset(seed=d_arg["seed"])
r_cumulative_return = 0
r_discounted_cumulative_return = 0

# do reinforcement
ld_data = []
while env.unwrapped.step_env < d_arg["total_timesteps"]:

    # sample the action space or learn
    if env.unwrapped.step_env <= d_arg["learning_starts"]:
        a_action = np.array(env.action_space.sample(), dtype=np.float16)
    else:
        x = torch.Tensor(o_observation).to(o_device).unsqueeze(0)
        actions, _, _ = actor.get_action(x)
        a_action = actions.detach().squeeze(0).cpu().numpy()

    # physigym step
    o_observation_next, r_reward, b_terminated, b_truncated, d_info = env.step(a_action)
    b_episode_over = b_terminated or b_truncated
    r_cumulative_return += r_reward
    r_discounted_cumulative_return += r_reward * d_arg["gamma"] ** (env.unwrapped.step_episode)

    # record to reply buffer
    rb.add(
        o_observation=o_observation,
        a_action=a_action,
        o_observation_next=o_observation_next,
        r_reward=r_reward,
        b_episode_over=b_episode_over,
    )

    # handle observation
    o_observation = o_observation_next


    # upadte data output
    d_data = {
        "step": env.unwrapped.step_episode,
        "reward": r_reward,
        "cumulative_return": r_cumulative_return,
        "discounted_cumulative_return": r_discounted_cumulative_return,
        "drug_1": a_action[0],
        "number_tumor": d_info["number_tumor"],
        "number_cell_1": d_info["number_cell_1"],
        "number_cell_2": d_info["number_cell_2"],
    }
    ld_data.append(d_data)


    # for debugung the reply buffer
    #if env.unwrapped.step_env == d_arg["batch_size"]:
    #    data = rb.sample()
    #    with torch.no_grad():
    #        next_state_actions, _, _ = actor.get_action(data["observation_next"])
    #        qf1(data["observation_next"], next_state_actions)
    #        qf2(data["observation_next"], next_state_actions)
    #        qf1_target(data["observation_next"], next_state_actions)
    #        qf2_target(data["observation_next"], next_state_actions)
    #    del data, next_state_actions

    # learning
    if env.unwrapped.step_env > d_arg["learning_starts"]:
        data = rb.sample()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["observation_next"])
            qf1_next_target = qf1_target(data["observation_next"], next_state_actions)
            qf2_next_target = qf2_target(data["observation_next"], next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = data["reward"].flatten() + (1 - data["episode_over"].flatten()) * d_arg["gamma"] * (min_qf_next_target).view(-1)

        qf1_a_values = qf1(data["observation"], data["action"]).view(-1)
        qf2_a_values = qf2(data["observation"], data["action"]).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # every policy frequency
        if env.unwrapped.step_env % d_arg["policy_frequency"] == 0:  # TD 3 Delayed update support

            # compensate for the delay by doing "actor_update_interval" instead of 1
            for _ in range(d_arg["policy_frequency"]):
                pi, log_pi, _ = actor.get_action(data["observation"])

                qf1_pi = qf1(data["observation"], pi)
                qf2_pi = qf2(data["observation"], pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # entropy autotune
                if d_arg["autotune"]:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data["observation"])

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
            entropy = - log_pi.mean().item()
            writer.add_scalar("losses/entropy", entropy, env.unwrapped.step_env)

        # update the target networks
        if env.unwrapped.step_env % d_arg["target_network_frequency"] == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(d_arg["tau"] * param.data + (1 - d_arg["tau"]) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(d_arg["tau"] * param.data + (1 - d_arg["tau"]) * target_param.data)


    # write to tensorboard
    writer.add_scalar("env/drug_1", a_action[0], env.unwrapped.episode)
    writer.add_scalar("env/number_tumor", d_info["number_tumor"], env.unwrapped.episode)
    writer.add_scalar("env/number_cell_1", d_info["number_cell_1"], env.unwrapped.episode)
    writer.add_scalar("env/number_cell_2", d_info["number_cell_2"], env.unwrapped.episode)
    writer.add_scalar("env/reward", r_reward, env.unwrapped.episode)
    writer.add_scalar("env/reward_tumor", d_info["reward_tumor"], env.unwrapped.episode)
    writer.add_scalar("env/reward_drugs", d_info["reward_drugs"], env.unwrapped.episode)

    # if episode is over
    if b_episode_over:
        # write to tensorbord
        #norm_coeff = (1 - d_arg["gamma"] ** (env.unwrapped.step_episode + 1)) / (1 - d_arg["gamma"])
        #writer.add_scalar("charts/episodic_return", r_cumulative_return / env.unwrapped.step_episode, env.unwrapped.episode)
        writer.add_scalar("charts/cumulative_return", r_cumulative_return, env.unwrapped.episode)
        writer.add_scalar("charts/episodic_length", env.unwrapped.step_episode, env.unwrapped.episode)
        writer.add_scalar("charts/discounted_cumulative_return", r_discounted_cumulative_return, env.unwrapped.episode)
        #writer.add_scalar("charts/normalized_discounted_episodic_return", r_discounted_cumulative_return / norm_coeff, env.unwrapped.episode)

        # write data
        df = pd.DataFrame(ld_data)
        s_dir_data_episode = os.path.join(s_dir_data, str(env.unwrapped.episode))
        os.makedirs(s_dir_data_episode, exist_ok=True)
        df.to_csv(os.path.join(s_dir_data_episode, "data.csv"), index=False)

        # reset gymnasium environment and global variables
        o_observation, d_info = env.reset()
        r_cumulative_return = 0
        r_discounted_cumulative_return = 0
        b_episode_over = False
        ld_data = []

# finish
env.close()
writer.close()

