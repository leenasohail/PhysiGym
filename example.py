import gymnasium as gym
from embedding import physicell
import matplotlib.pyplot as plt
import numpy as np
import os
import pcdl
import physigym  # import the Gymnasium PhysiCell bridge module
import random
import shutil


#############
# run tests #
#############
print(os.getcwd())
print("\nUNITTEST run test ...")
os.chdir("../PhysiCell")
# set variables
i_cell_target = 64

# load PhysiCell Gymnasium environment
env = gym.make(
    "physigym/ModelPhysiCellEnv-v0",
    # settingxml='config/PhysiCell_settings.xml',
    # render_mode='rgb_array',
    # render_fps=10
)


class DummyModelWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, variable_name: str):
        super().__init__(env)
        self.cell_count_target = int(self.x_root.xpath("//cell_count_target")[0].text)
        if not isinstance(variable_name, str):
            raise ValueError(
                f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
            )

        self.variable_name = variable_name

    def reset(self, seed=None, options={}):
        o_observation, d_info = self.env.reset(seed=seed, options=options)
        o_observation = o_observation.astype(float) / self.cell_count_target
        return o_observation[0], d_info

    def step(self, r_dose: float):
        d_action = {self.variable_name: np.array([r_dose])}
        o_observation, r_reward, b_terminated, b_truncated, d_info = self.env.step(
            d_action
        )
        o_observation = o_observation.astype(float) / self.cell_count_target
        return o_observation[0], r_reward, b_terminated, b_truncated, d_info

    @property
    def action_space(self):
        return self.env.action_space[self.variable_name]


# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False

    # Algorithm specific arguments
    env_id: str = "physigym/ModelPhysiCellEnv-v0"
    """the id of the environment"""
    wrapper: bool = True
    """use wrapper"""
    total_timesteps: int = int(1e7)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.LazyLinear(
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, buffer_size, batch_size):
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = np.empty((self.buffer_size, state_dim), dtype=np.float32)
        self.next_state = np.empty((self.buffer_size, state_dim), dtype=np.float32)
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, state, action, reward, next_state, done):
        self.state[self.buffer_index] = state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_state[self.buffer_index] = next_state
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """
        Sample a batch of experiences from the replay buffer.
        """
        batch_size = self.batch_size
        # Ensure there are enough samples in the buffer
        assert self.full or (
            self.buffer_index > batch_size
        ), "Buffer does not have enough samples"

        # Generate random indices for sampling
        sample_index = np.random.randint(
            0, self.capacity if self.full else self.buffer_index, batch_size
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = gym.make(id=args.env_id)
    if args.wrapper:
        env = gym.wrappers.RecordEpisodeStatistics(DummyModelWrapper(env, "drug_dose"))

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf2 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    qf2_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    env.observation_space.dtype = np.float32

    rb = ReplayBuffer(
        state_dim=np.array(env.observation_space.shape).prod(),
        action_dim=np.array(env.observation_space.shape).prod(),
        device=device,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step <= args.learning_starts:
            actions = np.array(env.action_space.sample())
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor([obs]).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(env.action_space.low, env.action_space.high)
                )
        print(f"Action:{actions}")
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations or truncations

        rb.add(obs, actions, rewards, next_obs, done)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample()
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data["action"], device=device) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (
                    target_actor(data["next_state"]) + clipped_noise
                ).clamp(env.action_space.low[0], env.action_space.high[0])
                qf1_next_target = qf1_target(data["next_state"], next_state_actions)
                qf2_next_target = qf2_target(data["next_state"], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data["reward"].flatten() + (
                    1 - data["done"].flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data["state"], data["action"]).view(-1)
            qf2_a_values = qf2(data["state"], data["action"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data["state"], actor(data["state"])).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
            writer.add_scalar("env/reward_value", rewards, global_step)
            writer.add_scalar("env/number_of_cells", physicell.get_cell(), global_step)
            writer.add_scalar("env/drug_dose", actions, global_step)

        if done:
            obs, _ = env.reset(seed=args.seed)
    env.close()
    writer.close()
