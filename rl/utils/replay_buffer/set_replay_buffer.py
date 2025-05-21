import torch
import numpy as np
import pickle
import pandas as pd
from tensordict import TensorDict
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange


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
        - state_dim (int): Dimensionality of the state space.
        - action_dim (int): Dimensionality of the action space.
        - device (torch.device): Device where tensors should be stored.
        - buffer_size (int): Maximum size of the replay buffer.
        - batch_size (int): Number of samples per batch.
        - state_type (numpy dtype, optional): Data type of the state representation (default: np.float32).
        """
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = np.empty((self.buffer_size, state_dim), dtype=state_type)
        self.next_state = np.empty((self.buffer_size, state_dim), dtype=state_type)
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
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


@njit(parallel=True)
def colorize(x_valid, y_valid, types_valid, type_to_color_array, o_observation):
    for i in prange(len(x_valid)):
        type_idx = types_valid[i]
        if 0 <= type_idx < type_to_color_array.shape[0]:
            r, g, b = type_to_color_array[type_idx]
        else:
            r, g, b = 0, 0, 0

        x = x_valid[i]
        y = y_valid[i]

        # Numba doesn't like writing to global memory from multiple threads,
        # but we assume no collisions here because each (x, y) is unique.
        o_observation[0, x, y] = r
        o_observation[1, x, y] = g
        o_observation[2, x, y] = b


# --- Your class ---
class MinimalImgReplayBuffer(object):
    """
    A lighter replay buffer that only stores (x, y, type_int) points per state,
    and reconstructs images when sampling.
    """

    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        buffer_size: int,
        batch_size: int,
        height: int,
        width: int,
        x_min: int,
        y_min: int,
        type_to_color: dict,
        image_gray: bool,
        num_workers: int = 12,
    ):
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = [None] * self.buffer_size
        self.next_state = [None] * self.buffer_size
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.x_min = x_min
        self.y_min = y_min
        self.image_gray = image_gray
        self.num_workers = num_workers

        # --- Precompute the type_to_color_array for fast lookup ---
        self.type_to_color_array = self._make_type_to_color_array(type_to_color)

    def _make_type_to_color_array(self, type_to_color):
        max_type = max(type_to_color.keys())
        color_array = np.zeros((max_type + 1, 3), dtype=np.uint8)
        for t, color in type_to_color.items():
            color_array[t] = np.array(color, dtype=np.uint8)
        return color_array

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, df_cell, action, reward, next_df_cell, done, type_to_int):
        state_array = self.extract_minimal_array(df_cell, type_to_int)
        next_state_array = self.extract_minimal_array(next_df_cell, type_to_int)

        self.state[self.buffer_index] = state_array
        self.next_state[self.buffer_index] = next_state_array
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def extract_minimal_array(self, df_cell, type_to_int):
        x = df_cell["x"].to_numpy()
        y = df_cell["y"].to_numpy()
        type_labels = df_cell["type"].map(type_to_int).to_numpy()

        minimal_array = np.stack([x, y, type_labels], axis=1)
        return minimal_array

    def sample(self):
        batch_size = self.batch_size
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        state_list = [self.state[i] for i in sample_index]
        next_state_list = [self.next_state[i] for i in sample_index]

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        done = torch.as_tensor(self.done[sample_index], device=self.device)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            if self.image_gray:
                state_images = list(
                    executor.map(self.minimal_array_to_grayscale, state_list)
                )
                next_state_images = list(
                    executor.map(self.minimal_array_to_grayscale, next_state_list)
                )
            else:
                state_images = list(
                    executor.map(self.minimal_array_to_image, state_list)
                )
                next_state_images = list(
                    executor.map(self.minimal_array_to_image, next_state_list)
                )

        state_tensor = torch.from_numpy(np.stack(state_images)).float().to(self.device)
        next_state_tensor = (
            torch.from_numpy(np.stack(next_state_images)).float().to(self.device)
        )

        sample = TensorDict(
            {
                "state": state_tensor,
                "action": action,
                "reward": reward,
                "next_state": next_state_tensor,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample

    def minimal_array_to_image(self, state_array):
        x = state_array[:, 0]
        y = state_array[:, 1]
        type_int = state_array[:, 2].astype(int)

        x_norm = (x - self.x_min).astype(int)
        y_norm = (y - self.y_min).astype(int)

        o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)

        valid_mask = (
            (0 <= x_norm)
            & (x_norm < self.height)
            & (0 <= y_norm)
            & (y_norm < self.width)
        )

        x_valid = x_norm[valid_mask]
        y_valid = y_norm[valid_mask]
        types_valid = type_int[valid_mask]

        # --- Fast colorization using Numba ---
        colorize(x_valid, y_valid, types_valid, self.type_to_color_array, o_observation)

        return o_observation

    def minimal_array_to_grayscale(self, state_array):
        color_image = self.minimal_array_to_image(state_array)
        grayscale_image = np.dot(
            color_image.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140]
        ).astype(np.uint8)
        return grayscale_image[np.newaxis, :, :]
