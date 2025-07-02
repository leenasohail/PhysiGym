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

import numpy as np
import torch
from tensordict import TensorDict

class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.
    Stores o_observations, a_actions, r_rewards, next o_observations, and b_episode_over flags.
    """

    def __init__(
            self,
            o_observation_dim,
            a_action_dim,
            device,
            buffer_size,
            batch_size,
            o_observation_type=np.float32,
        ):
        """
        Initializes the replay buffer.

        Parameters:
        - o_observation_dim tuple(int): Dimensionality of the o_observation space.
        - a_action_dim tuple(int): Dimensionality of the a_action space.
        - device (torch.device): Device where tensors should be stored.
        - buffer_size (int): Maximum size of the replay buffer.
        - batch_size (int): Number of samples per batch.
        - o_observation_type (numpy dtype, optional): Data type of the o_observation representation (default: np.float32).
        """
        self.device = device
        self.buffer_size = int(buffer_size)

        self.o_observation = np.empty((self.buffer_size, *o_observation_dim), dtype=o_observation_type)
        self.o_observation_next = np.empty((self.buffer_size, *o_observation_dim), dtype=o_observation_type)
        self.a_action = np.empty((self.buffer_size, *a_action_dim), dtype=np.float32)
        self.r_reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.b_episode_over = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the current number of stored experiences in the buffer.
        """
        return self.buffer_size if self.full else self.buffer_index

    def add(self, o_observation, a_action, o_observation_next, r_reward, b_episode_over):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        - o_observation (np.ndarray): Current o_observation.
        - a_action (np.ndarray): Action taken.
        - o_observation_next (np.ndarray): Next o_observation after taking the a_action.
        - r_reward (float): Reward received.
        - b_episode_over (bool): Whether the episode has ended.
        """
        self.o_observation[self.buffer_index] = o_observation
        self.a_action[self.buffer_index] = a_action
        self.r_reward[self.buffer_index] = r_reward
        self.o_observation_next[self.buffer_index] = o_observation_next
        self.b_episode_over[self.buffer_index] = b_episode_over

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
        - TensorDict containing sampled o_observations, a_actions, r_rewards, next o_observations, and b_episode_over flags.
        """
        batch_size = self.batch_size

        # Ensure there are enough samples in the buffer
        assert self.full or (self.buffer_index > batch_size), "Buffer does not have enough samples"

        # Generate random indices for sampling
        sample_index = np.random.randint(0, self.buffer_size if self.full else self.buffer_index, batch_size)

        # Convert indices to tensors and gather the sampled experiences
        o_observation = torch.as_tensor(self.o_observation[sample_index]).float()
        a_action = torch.as_tensor(self.a_action[sample_index])
        o_observation_next = torch.as_tensor(self.o_observation_next[sample_index]).float()
        r_reward = torch.as_tensor(self.r_reward[sample_index])
        b_episode_over = torch.as_tensor(self.b_episode_over[sample_index])

        # Create a dictionary of the sampled experiences
        sample = TensorDict(
            {
                "o_observation": o_observation,
                "a_action": a_action,
                "o_observation_next": o_observation_next,
                "r_reward": r_reward,
                "b_episode_over": b_episode_over,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample
