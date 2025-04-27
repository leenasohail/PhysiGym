import torch
import numpy as np
import pyarrow as pa
import pandas as pd
from tensordict import TensorDict
from concurrent.futures import ThreadPoolExecutor


class ImgReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences,
    where states and next states are stored as serialized DataFrames
    and converted to image representations when sampled.
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
        color_mapping: dict,
        image_gray: bool,
        num_workers: int = 4,
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
        self.color_mapping = color_mapping
        self.image_gray = image_gray
        self.num_workers = num_workers

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    @staticmethod
    def serialize(df: pd.DataFrame):
        return pa.serialize(df).to_buffer()

    @staticmethod
    def deserialize(buffer):
        return pa.deserialize(buffer)

    def add(self, df_cell, action, reward, next_df_cell, done):
        self.state[self.buffer_index] = self.serialize(df_cell)
        self.next_state[self.buffer_index] = self.serialize(next_df_cell)
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        batch_size = self.batch_size
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        # Deserialize DataFrames
        state_df_list = [self.deserialize(self.state[i]) for i in sample_index]
        next_state_df_list = [
            self.deserialize(self.next_state[i]) for i in sample_index
        ]

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        done = torch.as_tensor(self.done[sample_index], device=self.device)

        # Parallel image reconstruction
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            if self.image_gray:
                state_images = list(executor.map(self.df_to_grayscale, state_df_list))
                next_state_images = list(
                    executor.map(self.df_to_grayscale, next_state_df_list)
                )
            else:
                state_images = list(executor.map(self.df_to_image, state_df_list))
                next_state_images = list(
                    executor.map(self.df_to_image, next_state_df_list)
                )

        # Stack images and create tensors
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

    def df_to_image(self, df_cell: pd.DataFrame) -> np.ndarray:
        """
        Converts a DataFrame into a color image (3, height, width).
        """
        x = df_cell["x"].to_numpy()
        y = df_cell["y"].to_numpy()

        # Normalize coordinates
        x_normalized = (x - self.x_min).astype(int)
        y_normalized = (y - self.y_min).astype(int)

        # Compute color
        df_cell = df_cell.copy()
        df_cell["color"] = df_cell["type"].map(
            lambda t: self.color_mapping.get(t, (0, 0, 0))
        )
        df_cell["color"] = df_cell.apply(
            lambda row: [0, 0, 0] if row["dead"] != 0.0 else row["color"], axis=1
        )

        colors = np.stack(df_cell["color"].to_numpy(), axis=0)  # (N, 3)

        o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)

        # Vectorized assignment
        valid_mask = (
            (0 <= x_normalized)
            & (x_normalized < self.height)
            & (0 <= y_normalized)
            & (y_normalized < self.width)
        )

        x_valid = x_normalized[valid_mask]
        y_valid = y_normalized[valid_mask]
        colors_valid = colors[valid_mask]

        o_observation[:, x_valid, y_valid] = colors_valid.T

        return o_observation

    def df_to_grayscale(self, df_cell: pd.DataFrame) -> np.ndarray:
        """
        Converts a DataFrame into a grayscale image (1, height, width).
        """
        color_image = self.df_to_image(df_cell)  # (3, H, W)

        # Move channels last temporarily
        grayscale_image = np.dot(
            color_image.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140]
        ).astype(np.uint8)
        return grayscale_image[np.newaxis, :, :]  # Shape (1, height, width)
